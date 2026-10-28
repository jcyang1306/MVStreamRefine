"""Session reader (contracts/SESSION_FORMAT.md).

- Opening a session validates `schema_version` and (by default) the sha256 of every
  manifest-listed file; mismatches raise `ChecksumMismatchError`.
- `read_frame` verifies the image checksums from `frames.jsonl` and converts depth to meters
  exactly once via `mvsr.core.depth.depth_raw_to_m` (ADR-0003).
- No clock mapping, pose pairing or interpolation happens here (P1).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from mvsr.core import codec
from mvsr.core.codec import SCHEMA_VERSION
from mvsr.core.depth import depth_raw_to_m
from mvsr.core.errors import ChecksumMismatchError, SchemaVersionError, SessionFormatError
from mvsr.core.types import CalibrationBundle, ClockDomain, FrameStatus, FrameTiming, RobotSample
from mvsr.devices.rm65 import xyzrpy_to_T_base_tcp
from mvsr.io import images
from mvsr.io.atomic import sha256_bytes, sha256_file
from mvsr.io.records import EventRecord, FrameRecord, RobotRecord
from mvsr.io.session_writer import EVENTS_NAME, FRAMES_NAME, MANIFEST_NAME, ROBOT_NAME

#: `FrameTiming.method` for frames read without a clock model (mapping is P1 S4).
UNMAPPED_TIMING_METHOD = "raw_unmapped"


@dataclass(frozen=True, eq=False)
class RawFrame:
    """One recorded RGB-D frame with raw timing; arrays are non-writeable."""

    camera_id: str
    frame_id: int
    color_rgb_u8: npt.NDArray[np.uint8]
    depth_raw_u16: npt.NDArray[np.uint16]
    depth_units_m: float
    depth_m_f32: npt.NDArray[np.float32]
    valid_depth_mask: npt.NDArray[np.bool_]
    timing: FrameTiming
    record: FrameRecord


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SessionFormatError(f"{path.name}:{lineno}: invalid JSON ({exc})") from exc
            if not isinstance(obj, dict):
                raise SessionFormatError(f"{path.name}:{lineno}: expected an object")
            out.append(obj)
    return out


def load_manifest(session_dir: Path) -> dict[str, Any]:
    path = session_dir / MANIFEST_NAME
    if not path.is_file():
        raise SessionFormatError(f"{session_dir}: no {MANIFEST_NAME}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise SessionFormatError("manifest must be a JSON object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise SchemaVersionError(
            f"manifest schema_version {manifest.get('schema_version')!r} != {SCHEMA_VERSION}"
        )
    return manifest


def _freeze(arr: np.ndarray) -> np.ndarray:
    arr.flags.writeable = False
    return arr


def robot_sample_from_record(r: RobotRecord) -> RobotSample | None:
    """`RobotSample` of a successful read (HOST_MONOTONIC); None for a failed read."""
    if r.error is not None or r.tcp_xyzrpy is None:
        return None
    xyzrpy = np.asarray(r.tcp_xyzrpy, dtype=np.float64)
    return RobotSample(
        seq=r.seq,
        t_host_ns=r.t_host_ns,
        clock_domain=ClockDomain.HOST_MONOTONIC,
        T_base_tcp=xyzrpy_to_T_base_tcp(xyzrpy),
        joints_rad=None if r.joints_rad is None else np.asarray(r.joints_rad, dtype=np.float64),
        raw_xyzrpy=xyzrpy,
        source=r.source,  # type: ignore[arg-type]
        read_latency_ns=r.read_latency_ns,
    )


class SessionReader:
    def __init__(self, session_dir: str | Path, *, verify_checksums: bool = True) -> None:
        self.dir = Path(session_dir)
        self.manifest = load_manifest(self.dir)
        self.session_id: str = self.manifest["session_id"]
        self.status: str = self.manifest["status"]
        if verify_checksums:
            self.verify_listed_files()
        self._records: dict[str, dict[int, FrameRecord]] = {}

    # ------------------------------------------------------------------ metadata
    def verify_listed_files(self) -> None:
        for rel, expected in sorted((self.manifest.get("files") or {}).items()):
            path = self.dir / rel
            if not path.is_file():
                raise SessionFormatError(f"manifest lists missing file {rel}")
            actual = sha256_file(path)
            if actual != expected:
                raise ChecksumMismatchError(f"{rel}: sha256 {actual} != manifest {expected}")

    @property
    def camera_ids(self) -> list[str]:
        return sorted(self.manifest.get("cameras", {}))

    def calibration(self, camera_id: str) -> CalibrationBundle:
        path = self.dir / "calibration" / f"{camera_id}.json"
        bundle = codec.from_json(CalibrationBundle, path.read_text(encoding="utf-8"))
        cam = self.manifest["cameras"][camera_id]
        if bundle.calibration_id != cam["calibration_id"]:
            raise SessionFormatError(
                f"{camera_id}: calibration_id {bundle.calibration_id} != manifest {cam['calibration_id']}"
            )
        if bundle.depth_units_m != cam["depth_units_m"]:
            raise SessionFormatError(
                f"{camera_id}: depth_units_m differs between bundle and manifest"
            )
        return bundle

    def depth_units_m(self, camera_id: str) -> float:
        return float(self.manifest["cameras"][camera_id]["depth_units_m"])

    # ------------------------------------------------------------------ frames
    def frame_records(self, camera_id: str) -> list[FrameRecord]:
        if camera_id not in self._records:
            path = self.dir / "cameras" / camera_id / FRAMES_NAME
            recs: dict[int, FrameRecord] = {}
            for d in read_jsonl(path):
                rec = FrameRecord.from_dict(d, f"{camera_id}/{FRAMES_NAME}")
                if rec.frame_id in recs:
                    raise SessionFormatError(f"{camera_id}: duplicate frame_id {rec.frame_id}")
                recs[rec.frame_id] = rec
            self._records[camera_id] = recs
        return [self._records[camera_id][k] for k in sorted(self._records[camera_id])]

    def written_frame_ids(self, camera_id: str) -> list[int]:
        return [
            r.frame_id for r in self.frame_records(camera_id) if r.status is FrameStatus.WRITTEN
        ]

    def frame_record(self, camera_id: str, frame_id: int) -> FrameRecord:
        self.frame_records(camera_id)
        try:
            return self._records[camera_id][frame_id]
        except KeyError as exc:
            raise SessionFormatError(f"{camera_id}: no record for frame_id {frame_id}") from exc

    def read_frame(self, camera_id: str, frame_id: int, *, verify: bool = True) -> RawFrame:
        rec = self.frame_record(camera_id, frame_id)
        if rec.status is not FrameStatus.WRITTEN or rec.color is None or rec.depth is None:
            raise SessionFormatError(
                f"{camera_id} frame {frame_id} was not written ({rec.status.value})"
            )
        color_bytes = (self.dir / rec.color.path).read_bytes()
        depth_bytes = (self.dir / rec.depth.path).read_bytes()
        if verify:
            for ref, data in ((rec.color, color_bytes), (rec.depth, depth_bytes)):
                actual = sha256_bytes(data)
                if actual != ref.sha256:
                    raise ChecksumMismatchError(f"{ref.path}: sha256 {actual} != {ref.sha256}")
        color = images.decode_png_rgb(color_bytes)
        depth = images.decode_png_u16(depth_bytes)
        if verify:
            for ref, arr in ((rec.color, color), (rec.depth, depth)):
                if ref.raw_sha256 is not None and images.raw_sha256(arr) != ref.raw_sha256:
                    raise ChecksumMismatchError(
                        f"{ref.path}: decoded pixels differ from recorded raw hash"
                    )
        units = self.depth_units_m(camera_id)
        t = rec.timing
        timing = FrameTiming(
            t_host_arrival_ns=rec.t_host_arrival_ns,
            rs_timestamp_ms=t.rs_timestamp_ms,
            rs_timestamp_domain=t.rs_timestamp_domain,
            rs_sensor_timestamp_us=t.rs_sensor_timestamp_us,
            rs_frame_timestamp_us=t.rs_frame_timestamp_us,
            exposure_us=t.exposure_us,
            rs_frame_number=t.rs_frame_number,
            method=UNMAPPED_TIMING_METHOD,
        )
        return RawFrame(
            camera_id=camera_id,
            frame_id=frame_id,
            color_rgb_u8=_freeze(color),
            depth_raw_u16=_freeze(depth),
            depth_units_m=units,
            depth_m_f32=_freeze(depth_raw_to_m(depth, units)),
            valid_depth_mask=_freeze(depth != 0),
            timing=timing,
            record=rec,
        )

    # ------------------------------------------------------------------ robot / events
    def robot_records(self) -> list[RobotRecord]:
        return [RobotRecord.from_dict(d) for d in read_jsonl(self.dir / ROBOT_NAME)]

    def robot_samples(self) -> list[RobotSample]:
        """Successful robot reads as `RobotSample` (failed reads are skipped; see robot_records)."""
        return [s for r in self.robot_records() if (s := robot_sample_from_record(r)) is not None]

    def events(self) -> list[EventRecord]:
        return [EventRecord.from_dict(d) for d in read_jsonl(self.dir / EVENTS_NAME)]
