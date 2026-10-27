"""Streaming session writer (contracts/SESSION_FORMAT.md, ADR-0003/0004/0007).

Lifecycle: the constructor creates `<root>/sessions/<session_id>/` and writes `manifest.json`
with `status = RECORDING`; `finalize()` closes every stream, computes checksums of all
non-image files and atomically rewrites the manifest as `COMPLETE` or `ABORTED`.

Durability: images and JSON files are written atomically (tmp -> fsync -> rename); image
directories are fsynced once at finalization. JSONL streams are append-only (flushed per line,
fsynced at most every `jsonl_fsync_interval_s` and on close) so a crash keeps the log; their
checksums enter the manifest only at finalization.

Thread safety: `write_images` may be called concurrently (writer pool); `record_frame`,
`log_robot`, `log_event` are serialized internally. `record_frame` reorders records so that
`frames.jsonl` is sorted by `frame_id` regardless of writer completion order.
"""

from __future__ import annotations

import os
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from mvsr.core import codec
from mvsr.core.codec import SCHEMA_VERSION
from mvsr.core.errors import SessionFormatError
from mvsr.core.types import (
    CAMERA_IDS,
    SESSION_ID_RE,
    CalibrationBundle,
    EventType,
    FrameStatus,
    SessionStatus,
)
from mvsr.io import images
from mvsr.io.atomic import (
    atomic_write_bytes,
    atomic_write_json,
    fsync_dir,
    sha256_bytes,
    sha256_file,
)
from mvsr.io.records import EventRecord, FileRef, FrameRecord, RobotRecord, dumps_line

MANIFEST_NAME = "manifest.json"
EVENTS_NAME = "events.jsonl"
ROBOT_NAME = "robot.jsonl"
FRAMES_NAME = "frames.jsonl"


def is_image_path(rel: str) -> bool:
    parts = rel.split("/")
    return (
        len(parts) == 4
        and parts[0] == "cameras"
        and parts[2] in ("color", "depth")
        and parts[3].endswith(".png")
    )


def checksummed_files(session_dir: Path) -> dict[str, str]:
    """sha256 of every file except the manifest, per-frame images and `derived/`."""
    out: dict[str, str] = {}
    for path in sorted(session_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(session_dir).as_posix()
        if rel == MANIFEST_NAME or rel.startswith("derived/") or is_image_path(rel):
            continue
        out[rel] = sha256_file(path)
    return out


class JsonlAppender:
    """Thread-safe append-only JSONL file (one sorted-key JSON object per line)."""

    def __init__(self, path: Path, fsync_interval_s: float = 1.0) -> None:
        self.path = path
        self._f = open(path, "x", encoding="utf-8")
        self._lock = threading.Lock()
        self._interval = fsync_interval_s
        self._last_sync = time.monotonic()
        self.count = 0
        self.closed = False

    def append(self, obj: dict[str, Any]) -> None:
        line = dumps_line(obj) + "\n"
        with self._lock:
            if self.closed:
                raise SessionFormatError(f"{self.path.name} is closed")
            self._f.write(line)
            self._f.flush()
            self.count += 1
            now = time.monotonic()
            if now - self._last_sync >= self._interval:
                os.fsync(self._f.fileno())
                self._last_sync = now

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            self._f.flush()
            os.fsync(self._f.fileno())
            self._f.close()
            self.closed = True


class _CameraStream:
    def __init__(self, session_dir: Path, camera_id: str, fsync_interval_s: float) -> None:
        self.camera_id = camera_id
        base = session_dir / "cameras" / camera_id
        (base / "color").mkdir(parents=True)
        (base / "depth").mkdir(parents=True)
        self.dir = base
        self.frames = JsonlAppender(base / FRAMES_NAME, fsync_interval_s)
        self.lock = threading.Lock()
        self.pending: dict[int, FrameRecord] = {}
        self.next_id = 0
        self.status_counts: Counter[str] = Counter()
        self.bytes_written = 0

    def flush_ready(self) -> None:
        while self.next_id in self.pending:
            rec = self.pending.pop(self.next_id)
            self.frames.append(rec.to_dict())
            self.status_counts[rec.status.value] += 1
            self.next_id += 1


class SessionWriter:
    def __init__(
        self,
        data_root: Path,
        session_id: str,
        manifest: dict[str, Any],
        *,
        png_compression: int = images.DEFAULT_PNG_COMPRESSION,
        fsync_images: bool = True,
        jsonl_fsync_interval_s: float = 1.0,
    ) -> None:
        if not SESSION_ID_RE.match(session_id):
            raise SessionFormatError(f"invalid session_id {session_id!r}")
        self.session_id = session_id
        self.dir = Path(data_root) / "sessions" / session_id
        if self.dir.exists():
            raise FileExistsError(f"session directory already exists: {self.dir}")
        self.dir.mkdir(parents=True)
        (self.dir / "calibration").mkdir()
        (self.dir / "cameras").mkdir()
        self._png_level = int(png_compression)
        self._fsync_images = bool(fsync_images)
        self._fsync_interval = float(jsonl_fsync_interval_s)
        self._lock = threading.Lock()
        self._manifest: dict[str, Any] = dict(manifest)
        self._manifest.update(
            schema_version=SCHEMA_VERSION,
            session_id=session_id,
            status=SessionStatus.RECORDING.value,
        )
        self._manifest.setdefault("cameras", {})
        self._cams: dict[str, _CameraStream] = {}
        self._robot = JsonlAppender(self.dir / ROBOT_NAME, self._fsync_interval)
        self._robot_counts: Counter[str] = Counter()
        self._events = JsonlAppender(self.dir / EVENTS_NAME, self._fsync_interval)
        self._finalized = False
        self._write_manifest()

    # ------------------------------------------------------------------ setup
    @property
    def manifest(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._manifest)

    def _write_manifest(self) -> None:
        atomic_write_json(self.dir / MANIFEST_NAME, self._manifest)

    def update_manifest(self, **fields: Any) -> None:
        """Merge top-level manifest keys (status stays RECORDING until `finalize`)."""
        with self._lock:
            self._check_open()
            fields.pop("status", None)
            self._manifest.update(fields)
            self._write_manifest()

    def write_file(self, rel_path: str, data: bytes) -> str:
        """Atomically write a non-image file inside the session; returns its sha256."""
        self._check_open()
        target = self.dir / rel_path
        if not target.resolve().is_relative_to(self.dir.resolve()):
            raise SessionFormatError(f"path escapes the session directory: {rel_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        return atomic_write_bytes(target, data)

    def add_camera(
        self,
        camera_id: str,
        bundle: CalibrationBundle,
        realsense_info: dict[str, Any],
        manifest_entry: dict[str, Any],
    ) -> None:
        if camera_id not in CAMERA_IDS or bundle.camera_id != camera_id:
            raise SessionFormatError(f"bad camera_id {camera_id!r} / bundle {bundle.camera_id!r}")
        with self._lock:
            self._check_open()
            if camera_id in self._cams:
                raise SessionFormatError(f"camera {camera_id} already added")
            atomic_write_json(
                self.dir / "calibration" / f"{camera_id}.json", codec.to_json_dict(bundle)
            )
            atomic_write_json(
                self.dir / "calibration" / f"{camera_id}_realsense.json", realsense_info
            )
            self._cams[camera_id] = _CameraStream(self.dir, camera_id, self._fsync_interval)
            entry = dict(manifest_entry)
            entry.update(calibration_id=bundle.calibration_id, depth_units_m=bundle.depth_units_m)
            self._manifest["cameras"][camera_id] = entry
            self._write_manifest()

    def _check_open(self) -> None:
        if self._finalized:
            raise SessionFormatError("session writer already finalized")

    def _cam(self, camera_id: str) -> _CameraStream:
        try:
            return self._cams[camera_id]
        except KeyError as exc:
            raise SessionFormatError(f"camera {camera_id} was not added") from exc

    # ------------------------------------------------------------------ streaming
    def write_images(
        self, camera_id: str, frame_id: int, color_rgb: np.ndarray, depth_raw: np.ndarray
    ) -> tuple[FileRef, FileRef]:
        """Encode + atomically write one RGB-D pair (thread-safe). Raises OSError on disk errors."""
        cam = self._cam(camera_id)
        color_ref = self._write_png(
            cam, "color", frame_id, images.encode_png_rgb(color_rgb, self._png_level), color_rgb
        )
        depth_ref = self._write_png(
            cam, "depth", frame_id, images.encode_png_u16(depth_raw, self._png_level), depth_raw
        )
        return color_ref, depth_ref

    def _write_png(
        self, cam: _CameraStream, kind: str, frame_id: int, data: bytes, arr: np.ndarray
    ) -> FileRef:
        rel = f"cameras/{cam.camera_id}/{kind}/{frame_id:06d}.png"
        if self._fsync_images:
            digest = atomic_write_bytes(self.dir / rel, data, sync_dir=False)
        else:
            digest = _write_no_fsync(self.dir / rel, data)
        with cam.lock:
            cam.bytes_written += len(data)
        return FileRef(rel, digest, images.raw_sha256(arr))

    def record_frame(self, camera_id: str, record: FrameRecord) -> None:
        """Queue one frames.jsonl record; records are written in `frame_id` order."""
        cam = self._cam(camera_id)
        with cam.lock:
            if record.frame_id < cam.next_id or record.frame_id in cam.pending:
                raise SessionFormatError(f"{camera_id}: duplicate frame_id {record.frame_id}")
            cam.pending[record.frame_id] = record
            cam.flush_ready()

    def pending_records(self, camera_id: str) -> int:
        cam = self._cam(camera_id)
        with cam.lock:
            return len(cam.pending)

    def camera_bytes(self, camera_id: str) -> int:
        cam = self._cam(camera_id)
        with cam.lock:
            return cam.bytes_written

    def log_robot(self, record: RobotRecord) -> None:
        self._robot.append(record.to_dict())
        with self._lock:
            self._robot_counts["failures" if record.error is not None else "samples"] += 1

    def log_event(
        self, etype: EventType, payload: dict[str, Any] | None = None, t_host_ns: int | None = None
    ) -> None:
        rec = EventRecord(
            time.monotonic_ns() if t_host_ns is None else t_host_ns, etype, payload or {}
        )
        self._events.append(rec.to_dict())

    # ------------------------------------------------------------------ finalize
    def finalize(self, status: SessionStatus, **manifest_fields: Any) -> dict[str, Any]:
        """Close streams, checksum files and write the final manifest (idempotent: once only)."""
        if status is SessionStatus.RECORDING:
            raise SessionFormatError("final status must be COMPLETE or ABORTED")
        with self._lock:
            self._check_open()
            self._finalized = True
        counts: dict[str, Any] = {"cameras": {}, "robot": {}}
        for camera_id, cam in self._cams.items():
            with cam.lock:
                missing = sorted(cam.pending)
                # Records after a gap cannot be ordered; persist them so verify reports the gap.
                for fid in missing:
                    rec = cam.pending.pop(fid)
                    cam.frames.append(rec.to_dict())
                    cam.status_counts[rec.status.value] += 1
                cam.frames.close()
                written = cam.status_counts[FrameStatus.WRITTEN.value]
                seen = sum(cam.status_counts.values())
                counts["cameras"][camera_id] = {
                    "seen": seen,
                    "written": written,
                    "dropped": seen - written,
                    "by_status": {s.value: cam.status_counts[s.value] for s in FrameStatus},
                    "png_bytes": cam.bytes_written,
                }
                if missing:
                    status = SessionStatus.ABORTED
                    manifest_fields.setdefault("errors", []).append(
                        f"{camera_id}: frame records missing before frame_id {missing[0]}"
                    )
            for kind in ("color", "depth"):
                fsync_dir(cam.dir / kind)
        self._robot.close()
        self._events.close()
        counts["robot"] = {
            "samples": self._robot_counts["samples"],
            "failures": self._robot_counts["failures"],
        }
        with self._lock:
            self._manifest.update(manifest_fields)
            self._manifest["counts"] = counts
            self._manifest["status"] = status.value
            self._manifest["files"] = checksummed_files(self.dir)
            self._write_manifest()
            fsync_dir(self.dir)
            return dict(self._manifest)

    @property
    def finalized(self) -> bool:
        return self._finalized


def _write_no_fsync(path: Path, data: bytes) -> str:
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "xb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return sha256_bytes(data)
