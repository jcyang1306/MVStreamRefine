"""`mvsr verify-session`: integrity check of one recorded session (I-P0 gate criteria).

Errors (session invalid): manifest not COMPLETE / malformed, checksum mismatch, unlisted or
missing files, non-contiguous or duplicate `frame_id`, WRITTEN frame without images, counts
disagreeing with records, calibration inconsistent with the manifest, robot `seq` gaps,
events not bracketed by START/STOP, and (full mode) decoded pixels differing from the
recorded raw hash or depth meters not equal to `raw * depth_units_m`.
`--p0-gate` additionally requires wrist frames written, robot samples, and (if head was
recorded) head frames written. Capture length, fps and robot Hz are measured, not gated.
Warnings do not fail the check (e.g. hand-eye copy differs from the reference checksum).
"""

from __future__ import annotations

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from mvsr.calib.handeye import REFERENCE_HANDEYE_SHA256
from mvsr.core.errors import MvsrError
from mvsr.core.types import SESSION_ID_RE, EventType, FrameStatus, SessionStatus
from mvsr.io.atomic import sha256_file
from mvsr.io.records import FRAME_TIMING_KEYS, FrameRecord
from mvsr.io.session_reader import SessionReader, load_manifest, read_jsonl
from mvsr.io.session_writer import FRAMES_NAME, checksummed_files

REQUIRED_MANIFEST_KEYS = (
    "schema_version",
    "session_id",
    "status",
    "created_utc",
    "host",
    "software",
    "robot",
    "cameras",
    "clock",
    "scenario",
    "counts",
    "disk",
    "files",
)
REQUIRED_CAMERA_KEYS = (
    "model",
    "serial",
    "firmware",
    "mount",
    "calibration_id",
    "depth_units_m",
    "color_codec",
    "resolution",
    "fps",
    "formats",
    "preset",
    "emitter",
    "exposure_mode",
    "post_filters",
)


@dataclass
class VerifyReport:
    session_dir: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    images_checked: bool = False

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_dir": self.session_dir,
            "ok": self.ok,
            "images_checked": self.images_checked,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }

    def format(self) -> str:
        lines = [
            f"verify-session {self.session_dir}: {'OK' if self.ok else 'FAILED'}"
            f" (images {'checked' if self.images_checked else 'NOT checked (--fast)'})"
        ]
        for k, v in sorted(self.stats.items()):
            lines.append(f"  {k}: {v}")
        lines += [f"  ERROR: {e}" for e in self.errors[:200]]
        if len(self.errors) > 200:
            lines.append(f"  ... {len(self.errors) - 200} more errors")
        lines += [f"  WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def _check_files(d: Path, manifest: dict[str, Any], rep: VerifyReport) -> None:
    listed: dict[str, str] = manifest.get("files") or {}
    actual = checksummed_files(d)
    for rel, sha in sorted(listed.items()):
        if rel not in actual:
            rep.errors.append(f"listed file missing: {rel}")
        elif actual[rel] != sha:
            rep.errors.append(f"checksum mismatch: {rel}")
    for rel in sorted(set(actual) - set(listed)):
        rep.errors.append(f"file not listed in manifest: {rel}")
    for path in d.rglob("*.tmp"):
        rep.errors.append(f"stray temporary file: {path.relative_to(d).as_posix()}")


def _check_camera(
    reader: SessionReader, cid: str, manifest: dict[str, Any], rep: VerifyReport
) -> list[FrameRecord]:
    cam = manifest["cameras"][cid]
    for key in REQUIRED_CAMERA_KEYS:
        if key not in cam:
            rep.errors.append(f"{cid}: manifest camera entry missing '{key}'")
    try:
        bundle = reader.calibration(cid)
        w, h = cam.get("resolution", [None, None])
        if (bundle.width, bundle.height) != (w, h):
            rep.errors.append(
                f"{cid}: calibration {bundle.width}x{bundle.height} != manifest {w}x{h}"
            )
        he = reader.dir / "calibration" / "handeye_tf.txt"
        he_sha = sha256_file(he) if he.is_file() else None
        if he_sha is None:
            rep.errors.append("calibration/handeye_tf.txt missing")
        elif bundle.source_files.get("calibration/handeye_tf.txt") != he_sha:
            rep.errors.append(
                f"{cid}: bundle source_files hand-eye sha differs from the session copy"
            )
        elif he_sha != REFERENCE_HANDEYE_SHA256:
            rep.warnings.append("hand-eye copy is not the 2026-10-12 reference (ADR-0007)")
    except (MvsrError, OSError, KeyError, ValueError) as exc:
        rep.errors.append(f"{cid}: calibration invalid: {exc}")
    try:
        records = reader.frame_records(cid)
    except (MvsrError, OSError) as exc:
        rep.errors.append(f"{cid}: frames.jsonl invalid: {exc}")
        return []
    ids = [r.frame_id for r in records]
    if ids != list(range(len(ids))):
        missing = sorted(set(range(max(ids, default=-1) + 1)) - set(ids))
        rep.errors.append(f"{cid}: frame_id not contiguous from 0 (missing {missing[:10]}...)")
    file_order = [r["frame_id"] for r in _raw_lines(reader, cid)]
    if file_order != sorted(file_order):
        rep.warnings.append(f"{cid}: frames.jsonl not ordered by frame_id")
    arrivals = [r.t_host_arrival_ns for r in records]
    if any(b <= a for a, b in pairwise(arrivals)):
        rep.errors.append(f"{cid}: t_host_arrival_ns not strictly increasing")
    referenced: set[str] = set()
    pattern = re.compile(rf"^cameras/{re.escape(cid)}/(color|depth)/(\d{{6}})\.png$")
    for r in records:
        if r.status is FrameStatus.WRITTEN:
            assert r.color is not None and r.depth is not None
            for kind, ref in (("color", r.color), ("depth", r.depth)):
                m = pattern.match(ref.path)
                if not m or m.group(1) != kind or int(m.group(2)) != r.frame_id:
                    rep.errors.append(f"{cid} frame {r.frame_id}: bad {kind} path {ref.path}")
                if not (reader.dir / ref.path).is_file():
                    rep.errors.append(f"{cid} frame {r.frame_id}: missing {ref.path}")
                referenced.add(ref.path)
    on_disk = {
        p.relative_to(reader.dir).as_posix() for p in (reader.dir / "cameras" / cid).rglob("*.png")
    }
    for rel in sorted(on_disk - referenced):
        rep.errors.append(f"{cid}: image not referenced by frames.jsonl: {rel}")
    by_status = Counter(r.status.value for r in records)
    counts = (manifest.get("counts") or {}).get("cameras", {}).get(cid, {})
    written = by_status[FrameStatus.WRITTEN.value]
    if counts.get("seen") != len(records) or counts.get("written") != written:
        rep.errors.append(
            f"{cid}: manifest counts {counts.get('seen')}/{counts.get('written')} != records {len(records)}/{written}"
        )
    for status in FrameStatus:
        if (counts.get("by_status") or {}).get(status.value, 0) != by_status[status.value]:
            rep.errors.append(f"{cid}: manifest by_status[{status.value}] != records")
    rep.stats[f"{cid}.frames_seen"] = len(records)
    rep.stats[f"{cid}.frames_written"] = written
    rep.stats[f"{cid}.written_ratio"] = round(written / len(records), 6) if records else None
    rep.stats[f"{cid}.drops"] = {k: v for k, v in sorted(by_status.items()) if k != "WRITTEN"}
    coverage = {}
    for key in FRAME_TIMING_KEYS:
        n = sum(getattr(r.timing, key) is not None for r in records)
        coverage[key] = round(n / len(records), 4) if records else None
    rep.stats[f"{cid}.timestamp_field_coverage"] = coverage
    return records


def _raw_lines(reader: SessionReader, cid: str) -> list[dict[str, Any]]:
    return read_jsonl(reader.dir / "cameras" / cid / FRAMES_NAME)


def _check_images(
    reader: SessionReader,
    cid: str,
    records: list[FrameRecord],
    manifest: dict[str, Any],
    rep: VerifyReport,
    workers: int,
) -> None:
    w, h = manifest["cameras"][cid]["resolution"]
    units = reader.depth_units_m(cid)

    def check(fid: int) -> str | None:
        try:
            f = reader.read_frame(cid, fid, verify=True)
        except (MvsrError, OSError) as exc:
            return f"{cid} frame {fid}: {exc}"
        if f.color_rgb_u8.shape != (h, w, 3) or f.depth_raw_u16.shape != (h, w):
            return f"{cid} frame {fid}: image shape {f.color_rgb_u8.shape}/{f.depth_raw_u16.shape} != {w}x{h}"
        if (
            f.record.color is None
            or f.record.color.raw_sha256 is None
            or f.record.depth is None
            or f.record.depth.raw_sha256 is None
        ):
            return f"{cid} frame {fid}: no raw_sha256; lossless round trip not provable"
        expected = (f.depth_raw_u16.astype(np.float64) * units).astype(np.float32)
        if not np.array_equal(f.depth_m_f32, expected) or not np.array_equal(
            f.valid_depth_mask, f.depth_raw_u16 != 0
        ):
            return f"{cid} frame {fid}: depth meters != raw * depth_units_m"
        return None

    ids = [r.frame_id for r in records if r.status is FrameStatus.WRITTEN]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        problems = [p for p in pool.map(check, ids) if p is not None]
    rep.errors.extend(problems)
    rep.stats[f"{cid}.images_verified"] = len(ids) - len(problems)


def _check_robot(reader: SessionReader, manifest: dict[str, Any], rep: VerifyReport) -> None:
    try:
        recs = reader.robot_records()
    except (MvsrError, OSError) as exc:
        rep.errors.append(f"robot.jsonl invalid: {exc}")
        return
    seqs = [r.seq for r in recs]
    if seqs != list(range(len(seqs))):
        rep.errors.append("robot.jsonl seq not contiguous from 0")
    times = [r.t_host_ns for r in recs]
    if any(b < a for a, b in pairwise(times)):
        rep.errors.append("robot.jsonl t_host_ns decreasing")
    expected_source = (manifest.get("robot") or {}).get("state_source")
    bad = {r.source for r in recs} - {expected_source}
    if bad:
        rep.errors.append(f"robot.jsonl sources {sorted(bad)} != manifest {expected_source}")
    ok = sum(r.error is None for r in recs)
    counts = (manifest.get("counts") or {}).get("robot", {})
    if counts.get("samples") != ok or counts.get("failures") != len(recs) - ok:
        rep.errors.append("manifest robot counts disagree with robot.jsonl")
    try:
        reader.robot_samples()
    except (MvsrError, ValueError) as exc:
        rep.errors.append(f"robot sample conversion failed: {exc}")
    rep.stats["robot.samples"] = ok
    rep.stats["robot.failures"] = len(recs) - ok
    good = [r.t_host_ns for r in recs if r.error is None]
    if len(good) > 1:
        rep.stats["robot.mean_rate_hz"] = round((len(good) - 1) / ((good[-1] - good[0]) / 1e9), 2)


def _check_events(reader: SessionReader, rep: VerifyReport) -> None:
    try:
        events = reader.events()
    except (MvsrError, OSError) as exc:
        rep.errors.append(f"events.jsonl invalid: {exc}")
        return
    types = [e.type for e in events]
    if not types or types[0] is not EventType.START or types[-1] is not EventType.STOP:
        rep.errors.append("events.jsonl must start with START and end with STOP")
    if types.count(EventType.START) != 1 or types.count(EventType.STOP) != 1:
        rep.errors.append("events.jsonl must contain exactly one START and one STOP")
    times = [e.t_host_ns for e in events]
    if any(b < a for a, b in pairwise(times)):
        rep.warnings.append("events.jsonl timestamps not monotonic")
    rep.stats["events"] = dict(Counter(t.value for t in types))


def apply_p0_gate(rep: VerifyReport, manifest: dict[str, Any]) -> None:
    """Completeness for one operator-stopped take. No duration, fps, or robot-Hz targets."""
    cameras = manifest.get("cameras") or {}
    wrist_written = int(rep.stats.get("wrist.frames_written") or 0)
    if "wrist" not in cameras or wrist_written < 1:
        rep.errors.append("P0-gate: wrist.frames_written must be > 0")
    if int(rep.stats.get("robot.samples") or 0) < 1:
        rep.errors.append("P0-gate: robot.samples must be > 0")
    if "head" in cameras and int(rep.stats.get("head.frames_written") or 0) < 1:
        rep.errors.append("P0-gate: head was recorded but frames_written is 0")
    rep.stats["p0_gate"] = {
        "wrist_frames_written": wrist_written,
        "head_frames_written": int(rep.stats.get("head.frames_written") or 0)
        if "head" in cameras
        else None,
        "robot_samples": int(rep.stats.get("robot.samples") or 0),
        "duration_not_gated": True,
        "fps_not_gated": True,
        "robot_hz_not_gated": True,
    }


def verify_session(
    session_dir: str | Path,
    *,
    check_images: bool = True,
    workers: int = 8,
    p0_gate: bool = False,
) -> VerifyReport:
    d = Path(session_dir)
    rep = VerifyReport(str(d), images_checked=check_images)
    try:
        manifest = load_manifest(d)
    except (MvsrError, OSError, ValueError) as exc:
        rep.errors.append(f"manifest unreadable: {exc}")
        return rep
    for key in REQUIRED_MANIFEST_KEYS:
        if key not in manifest:
            rep.errors.append(f"manifest missing '{key}'")
    if manifest.get("status") != SessionStatus.COMPLETE.value:
        rep.errors.append(f"manifest status is {manifest.get('status')!r}, not COMPLETE")
    sid = manifest.get("session_id", "")
    if not SESSION_ID_RE.match(str(sid)) or d.name != sid:
        rep.errors.append(f"session_id {sid!r} invalid or != directory name {d.name!r}")
    _check_files(d, manifest, rep)
    reader = SessionReader(d, verify_checksums=False)
    cameras = manifest.get("cameras") or {}
    if "wrist" not in cameras:
        rep.errors.append("manifest has no wrist camera")
    for cid in sorted(cameras):
        records = _check_camera(reader, cid, manifest, rep)
        if check_images and records:
            _check_images(reader, cid, records, manifest, rep, workers)
    _check_robot(reader, manifest, rep)
    _check_events(reader, rep)
    if p0_gate:
        apply_p0_gate(rep, manifest)
    rep.stats["status"] = manifest.get("status")
    return rep
