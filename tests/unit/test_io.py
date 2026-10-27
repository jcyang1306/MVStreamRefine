"""Session IO, calibration parsing and robot conversion (F3/F4 analytic + format tests)."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mvsr.calib.bundle import HANDEYE_SESSION_PATH, build_bundle
from mvsr.calib.handeye import REFERENCE_HANDEYE_SHA256, load_handeye, parse_handeye_text
from mvsr.core.errors import ChecksumMismatchError, SchemaVersionError, SessionFormatError
from mvsr.core.types import EventType, FrameStatus, SessionStatus
from mvsr.devices.rm65 import xyzrpy_to_rotation, xyzrpy_to_T_base_tcp
from mvsr.io import images
from mvsr.io.atomic import atomic_write_bytes, sha256_file
from mvsr.io.inspect import format_report, inspect_session
from mvsr.io.records import FrameRecord, RawTiming, RobotRecord
from mvsr.io.registry import load_registry, register_session, resolve_fixture
from mvsr.io.session_reader import SessionReader
from mvsr.io.session_writer import SessionWriter
from mvsr.io.verify import verify_session

SID = "20261028T101500_CW20175_a3f1"
H, W = 12, 16


# --------------------------------------------------------------------------- atomic writes


def test_atomic_write_and_sha(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    digest = atomic_write_bytes(p, b"hello")
    assert p.read_bytes() == b"hello"
    assert digest == hashlib.sha256(b"hello").hexdigest() == sha256_file(p)
    big = os.urandom(3 * (1 << 20) + 7)
    atomic_write_bytes(p, big)
    assert sha256_file(p) == hashlib.sha256(big).hexdigest()


@pytest.mark.parametrize("fail_at", ["fsync", "replace"])
def test_atomic_write_interruption_keeps_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_at: str
) -> None:
    p = tmp_path / "manifest.json"
    atomic_write_bytes(p, b"old")

    def boom(*args: object) -> None:
        raise OSError("simulated crash")

    monkeypatch.setattr(os, fail_at, boom)
    with pytest.raises(OSError, match="simulated"):
        atomic_write_bytes(p, b"new content")
    monkeypatch.undo()
    assert p.read_bytes() == b"old"
    assert list(tmp_path.glob("*.tmp")) == []


# --------------------------------------------------------------------------- PNG


def test_png_lossless_and_channel_order() -> None:
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    rgb[0, 0] = (255, 0, 0)
    depth = rng.integers(0, 65536, (H, W), dtype=np.uint16)
    for level in (0, 1, 9):
        c = images.encode_png_rgb(rgb, level)
        d = images.encode_png_u16(depth, level)
        np.testing.assert_array_equal(images.decode_png_rgb(c), rgb)
        np.testing.assert_array_equal(images.decode_png_u16(d), depth)
    # The file is a standard RGB PNG: OpenCV (BGR) sees red in the last channel.
    bgr = cv2.imdecode(np.frombuffer(images.encode_png_rgb(rgb), np.uint8), cv2.IMREAD_UNCHANGED)
    assert tuple(bgr[0, 0]) == (0, 0, 255)
    with pytest.raises(SessionFormatError):
        images.decode_png_u16(images.encode_png_rgb(rgb))


# --------------------------------------------------------------------------- hand-eye + robot


def test_handeye_reference_file(reference_handeye_path: Path) -> None:
    he = load_handeye(reference_handeye_path)
    assert he.sha256 == REFERENCE_HANDEYE_SHA256
    assert (he.T_tcp_cam.dst, he.T_tcp_cam.src) == ("tcp", "cam")
    assert he.T_base_head is not None and (he.T_base_head.dst, he.T_base_head.src) == (
        "base",
        "head",
    )
    np.testing.assert_array_equal(
        he.T_tcp_cam.matrix[0],
        [0.7309865907111534, 0.5929286867088821, 0.33777829515551216, -0.06366091630072569],
    )
    np.testing.assert_array_equal(
        he.T_tcp_cam.t, [-0.06366091630072569, -0.055732728637328574, 0.09396108001466855]
    )
    np.testing.assert_array_equal(he.T_base_head.t, [-0.13905919, 0.01150721, 0.44054436])


def test_handeye_checksum_and_malformed(tmp_path: Path, reference_handeye_path: Path) -> None:
    changed = tmp_path / "h.txt"
    changed.write_text(reference_handeye_path.read_text().replace("0.0939", "0.0940"))
    with pytest.raises(ChecksumMismatchError):
        load_handeye(changed)
    load_handeye(changed, expected_sha256=None)
    with pytest.raises(SessionFormatError):
        parse_handeye_text("wrist_cam2:\n[1, 2, 3]\n")
    with pytest.raises(SessionFormatError):
        parse_handeye_text("junk\nwrist_cam2:\n" + "1 " * 16)
    with pytest.raises(SessionFormatError):
        parse_handeye_text("a:\n" + "0 " * 16 + "\na:\n" + "0 " * 16)


def test_xyzrpy_matches_scipy_extrinsic_xyz() -> None:
    rng = np.random.default_rng(2)
    for _ in range(200):
        rpy = rng.uniform(-np.pi, np.pi, 3)
        np.testing.assert_allclose(
            xyzrpy_to_rotation(rpy), Rotation.from_euler("xyz", rpy).as_matrix(), atol=1e-13
        )
    T = xyzrpy_to_T_base_tcp([0.3, -0.1, 0.4, 3.1, 0.05, -1.2])
    assert (T.dst, T.src) == ("base", "tcp")
    np.testing.assert_array_equal(T.t, [0.3, -0.1, 0.4])


# --------------------------------------------------------------------------- writer / reader


def make_session(
    root: Path, handeye_path: Path, n: int = 12, drop: tuple[int, ...] = (4, 9)
) -> tuple[Path, dict[int, tuple[np.ndarray, np.ndarray]]]:
    he = load_handeye(handeye_path)
    bundle = build_bundle(
        camera_id="wrist",
        camera_serial="323622271240",
        camera_model="Intel RealSense D405",
        width=W,
        height=H,
        fx=20.0,
        fy=20.0,
        cx=7.5,
        cy=5.5,
        distortion_model="inverse_brown_conrady",
        distortion=[0.0] * 5,
        depth_units_m=1e-4,
        handeye=he,
    )
    manifest = {
        "created_utc": "2026-10-28T02:15:00+00:00",
        "host": "CW20175",
        "software": {"version": "test"},
        "robot": {"model": "RM65", "state_source": "poll"},
        "clock": {},
        "scenario": {"tags": ["SMOKE"]},
        "disk": {},
    }
    w = SessionWriter(root, SID, manifest)
    w.write_file(HANDEYE_SESSION_PATH, handeye_path.read_bytes())
    entry = {
        k: None
        for k in (
            "model",
            "serial",
            "firmware",
            "mount",
            "color_codec",
            "fps",
            "formats",
            "preset",
            "emitter",
            "exposure_mode",
        )
    }
    entry.update(resolution=[W, H], post_filters=[])
    w.add_camera("wrist", bundle, {"device": {}}, entry)
    w.log_event(EventType.START, {})
    rng = np.random.default_rng(3)
    data = {}
    for fid in range(n):
        data[fid] = (
            rng.integers(0, 256, (H, W, 3), dtype=np.uint8),
            rng.integers(0, 60000, (H, W), dtype=np.uint16),
        )

    def write(fid: int) -> None:
        timing = RawTiming(1.0 + fid, "global_time", 100 + fid, 200 + fid, 8000.0, fid)
        if fid in drop:
            w.record_frame(
                "wrist",
                FrameRecord(
                    fid,
                    FrameStatus.DROPPED_QUEUE_FULL,
                    1000 + fid,
                    timing,
                    drop_detail="writer_queue",
                ),
            )
            return
        refs = w.write_images("wrist", fid, *data[fid])
        w.record_frame("wrist", FrameRecord(fid, FrameStatus.WRITTEN, 1000 + fid, timing, *refs))

    threads = [threading.Thread(target=write, args=(f,)) for f in reversed(range(n))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for seq in range(5):
        err = "timeout" if seq == 2 else None
        w.log_robot(
            RobotRecord(
                seq,
                900 + seq,
                "poll",
                None if err else [0.0] * 6,
                None if err else [0.3, 0, 0.4, 3.1, 0, 0],
                1000,
                err,
            )
        )
    w.log_event(EventType.STOP, {"reason": "test"})
    w.finalize(SessionStatus.COMPLETE, duration_s=1.0)
    return w.dir, data


def test_writer_reader_roundtrip(tmp_path: Path, reference_handeye_path: Path) -> None:
    d, data = make_session(tmp_path, reference_handeye_path)
    rep = verify_session(d)
    assert rep.ok, rep.errors
    assert rep.stats["wrist.frames_seen"] == 12 and rep.stats["wrist.frames_written"] == 10
    r = SessionReader(d)
    assert [x.frame_id for x in r.frame_records("wrist")] == list(range(12))
    lines = (d / "cameras/wrist/frames.jsonl").read_text().splitlines()
    assert [json.loads(x)["frame_id"] for x in lines] == list(range(12))  # reordered on write
    assert r.written_frame_ids("wrist") == [i for i in range(12) if i not in (4, 9)]
    f = r.read_frame("wrist", 5)
    np.testing.assert_array_equal(f.color_rgb_u8, data[5][0])
    np.testing.assert_array_equal(f.depth_raw_u16, data[5][1])
    np.testing.assert_array_equal(
        f.depth_m_f32, (data[5][1].astype(np.float64) * 1e-4).astype(np.float32)
    )
    np.testing.assert_array_equal(f.valid_depth_mask, data[5][1] != 0)
    assert not f.depth_m_f32.flags.writeable
    assert f.timing.rs_sensor_timestamp_us == 105 and f.timing.method == "raw_unmapped"
    with pytest.raises(SessionFormatError):
        r.read_frame("wrist", 4)
    assert r.calibration("wrist").calibration_id == r.manifest["cameras"]["wrist"]["calibration_id"]
    samples = r.robot_samples()
    assert [s.seq for s in samples] == [0, 1, 3, 4]
    assert r.manifest["counts"]["robot"] == {"samples": 4, "failures": 1}
    report = inspect_session(d)
    assert report["cameras"]["wrist"]["written_ratio"] == pytest.approx(10 / 12)
    assert report["robot"]["failures"] == 1
    assert "session" in format_report(report)


def test_checksum_mismatch_detected(tmp_path: Path, reference_handeye_path: Path) -> None:
    d, _ = make_session(tmp_path, reference_handeye_path)
    png = d / "cameras/wrist/depth/000005.png"
    blob = bytearray(png.read_bytes())
    blob[-20] ^= 0xFF
    png.write_bytes(bytes(blob))
    with pytest.raises(ChecksumMismatchError):
        SessionReader(d).read_frame("wrist", 5)
    assert not verify_session(d).ok
    (d / "robot.jsonl").write_text((d / "robot.jsonl").read_text() + "\n")
    with pytest.raises(ChecksumMismatchError):
        SessionReader(d)


def test_verify_detects_format_violations(tmp_path: Path, reference_handeye_path: Path) -> None:
    d, _ = make_session(tmp_path, reference_handeye_path)
    (d / "notes.txt").write_text("unlisted")
    (d / "cameras/wrist/color/000099.png").write_bytes(b"x")
    rep = verify_session(d, check_images=False)
    joined = "\n".join(rep.errors)
    assert "not listed" in joined and "not referenced" in joined


def test_verify_detects_frame_gap(tmp_path: Path, reference_handeye_path: Path) -> None:
    d, _ = make_session(tmp_path, reference_handeye_path)
    p = d / "cameras/wrist/frames.jsonl"
    lines = [x for x in p.read_text().splitlines() if json.loads(x)["frame_id"] != 7]
    p.write_text("\n".join(lines) + "\n")
    m = json.loads((d / "manifest.json").read_text())
    m["files"]["cameras/wrist/frames.jsonl"] = sha256_file(p)
    (d / "manifest.json").write_text(json.dumps(m))
    rep = verify_session(d, check_images=False)
    assert any("not contiguous" in e for e in rep.errors)


def test_unfinalized_session_fails_verify(tmp_path: Path, reference_handeye_path: Path) -> None:
    w = SessionWriter(tmp_path, SID, {"created_utc": "x"})
    assert json.loads((w.dir / "manifest.json").read_text())["status"] == "RECORDING"
    rep = verify_session(w.dir, check_images=False)
    assert any("not COMPLETE" in e for e in rep.errors)
    with pytest.raises(FileExistsError):
        SessionWriter(tmp_path, SID, {})


def test_manifest_schema_version_rejected(tmp_path: Path, reference_handeye_path: Path) -> None:
    d, _ = make_session(tmp_path, reference_handeye_path)
    m = json.loads((d / "manifest.json").read_text())
    m["schema_version"] = 2
    (d / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(SchemaVersionError):
        SessionReader(d)
    assert not verify_session(d).ok


def test_writer_rejects_unknown_camera_and_late_writes(tmp_path: Path) -> None:
    w = SessionWriter(tmp_path, SID, {})
    with pytest.raises(SessionFormatError):
        w.record_frame("wrist", FrameRecord(0, FrameStatus.DROPPED_OPERATOR, 1, RawTiming()))
    w.finalize(SessionStatus.ABORTED)
    with pytest.raises(SessionFormatError):
        w.log_event(EventType.ANNOTATION, {})
    with pytest.raises(SessionFormatError):
        FrameRecord(0, FrameStatus.WRITTEN, 1, RawTiming())  # WRITTEN needs file refs


def test_registry_roundtrip(tmp_path: Path, reference_handeye_path: Path) -> None:
    d, _ = make_session(tmp_path, reference_handeye_path)
    reg = tmp_path / "registry.yaml"
    reg.write_text("# header comment\n[]\n")
    entry = register_session(d, used_by=["P0-gate"], gt="none", registry=reg)
    entries = load_registry(reg)
    assert entries == [entry] and reg.read_text().startswith("# header comment")
    assert resolve_fixture(entry, tmp_path) == d
    with pytest.raises(ChecksumMismatchError):
        resolve_fixture({**entry, "manifest_sha256": "0" * 64}, tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_fixture({**entry, "session_id": "20200101T000000_x_0000"}, tmp_path)
