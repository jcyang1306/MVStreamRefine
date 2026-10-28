"""ReplaySource and the sensors report on a small hand-built session (P1 S5/S6).

Timing values are hand-written so the expected mid-exposure time and robot pose of every
frame are known exactly; this tests code paths, not sensor accuracy (ADR-0002).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mvsr.calib.bundle import HANDEYE_SESSION_PATH, build_bundle
from mvsr.calib.handeye import load_handeye
from mvsr.cli import main
from mvsr.core.errors import SessionFormatError
from mvsr.core.types import EventType, FrameStatus, SessionStatus, SyncStatus, deep_equal
from mvsr.io.atomic import sha256_file
from mvsr.io.records import FrameRecord, RawTiming, RobotRecord
from mvsr.io.session_writer import SessionWriter
from mvsr.sensors.clock import METHOD_GLOBAL_SENSOR
from mvsr.sensors.config import load_sensors_config
from mvsr.sensors.replay import ReplaySource
from mvsr.sensors.report import i0_decision, session_report, write_run
from mvsr.sensors.source import EndOfStream

SID = "20261028T120000_CW20175_b1c2"
R0, M0 = 1_793_000_000_000_000_000, 2_000_000_000_000_000
T_FRAME0 = M0 + 1_000_000_000
H, W = 12, 16
DROPPED = (3,)
FAILED_ROBOT_MS = (110, 120, 130)  # bracket 100..140 ms > 25 ms gap limit
ROBOT_MS = range(-50, 251, 10)


def frame_mid(i: int) -> int:
    return T_FRAME0 + i * 33_333_333


def tcp_x(t_ns: int) -> float:
    return 0.3 + 0.001 * (t_ns - T_FRAME0) / 10_000_000


def make_session(root: Path, handeye_path: Path, n: int = 10) -> Path:
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
        "created_utc": "2026-10-28T04:00:00+00:00",
        "host": "CW20175",
        "software": {"version": "test"},
        "robot": {"model": "RM65", "state_source": "poll"},
        "clock": {
            "start": {"realtime_ns": R0, "monotonic_ns": M0, "uncertainty_ns": 100},
            "end": {"realtime_ns": R0 + 10**10, "monotonic_ns": M0 + 10**10, "uncertainty_ns": 100},
        },
        "scenario": {"tags": ["R-STATIC"], "object": "test", "gt_type": None, "notes": ""},
        "disk": {},
    }
    w = SessionWriter(root, SID, manifest)
    w.write_file(HANDEYE_SESSION_PATH, handeye_path.read_bytes())
    entry: dict[str, Any] = {
        k: None
        for k in (
            "model",
            "serial",
            "firmware",
            "color_codec",
            "fps",
            "formats",
            "preset",
            "emitter",
            "exposure_mode",
        )
    }
    entry.update(mount="wrist", resolution=[W, H], post_filters=[])
    w.add_camera("wrist", bundle, {"device": {}}, entry)
    w.log_event(EventType.START, {}, t_host_ns=M0 + 900_000_000)
    rng = np.random.default_rng(7)
    for i in range(n):
        mid = frame_mid(i)
        sensor = 5_000_000 + (mid - T_FRAME0) // 1000
        timing = RawTiming(
            rs_timestamp_ms=(R0 + (mid - M0) + 10_000_000) / 10**6,
            rs_timestamp_domain="global_time",
            rs_sensor_timestamp_us=sensor,
            rs_frame_timestamp_us=sensor + 10_000,
            exposure_us=20_000.0,
            rs_frame_number=i,
        )
        arrival = mid + 28_000_000 + (i % 3) * 1_000_000
        if i in DROPPED:
            w.record_frame(
                "wrist",
                FrameRecord(
                    i, FrameStatus.DROPPED_QUEUE_FULL, arrival, timing, drop_detail="writer_queue"
                ),
            )
            continue
        color = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        depth = rng.integers(1, 5000, (H, W), dtype=np.uint16)
        refs = w.write_images("wrist", i, color, depth)
        w.record_frame("wrist", FrameRecord(i, FrameStatus.WRITTEN, arrival, timing, *refs))
    for seq, ms in enumerate(ROBOT_MS):
        t = T_FRAME0 + ms * 1_000_000
        ok = ms not in FAILED_ROBOT_MS
        w.log_robot(
            RobotRecord(
                seq,
                t,
                "poll",
                [0.0] * 6 if ok else None,
                [tcp_x(t), 0.0, 0.4, 3.1, 0.0, 0.0] if ok else None,
                1_000_000,
                None if ok else "rm_get_current_arm_state ret=-2",
            )
        )
    w.log_event(EventType.STOP, {"reason": "test"}, t_host_ns=M0 + 1_400_000_000)
    w.finalize(SessionStatus.COMPLETE, duration_s=0.5)
    return w.dir


@pytest.fixture
def session(tmp_path: Path, reference_handeye_path: Path) -> Path:
    return make_session(tmp_path, reference_handeye_path)


def test_replay_frames_timestamps_and_priors(session: Path) -> None:
    src = ReplaySource(session)
    frames = list(src)
    assert [f.header.frame_id for f in frames] == [0, 1, 2, 4, 5, 6, 7, 8, 9]
    status = {f.header.frame_id: f.pose_sync.status for f in frames}
    assert status[4] is SyncStatus.GAP_TOO_LARGE
    assert status[8] is SyncStatus.OUT_OF_RANGE and status[9] is SyncStatus.OUT_OF_RANGE
    for f in frames:
        i = f.header.frame_id
        assert abs(f.header.timestamp_ns - frame_mid(i)) <= 1_000
        assert f.timing.method == METHOD_GLOBAL_SENSOR
        assert f.header.calibration_id == src.bundle.calibration_id
        if f.pose_sync.status is SyncStatus.INTERPOLATED:
            assert f.T_base_tcp is not None and f.T_base_cam_prior is not None
            assert abs(f.T_base_tcp.t[0] - tcp_x(f.header.timestamp_ns)) < 1e-9
            assert f.T_base_cam_prior.allclose(f.T_base_tcp @ src.bundle.T_tcp_cam, atol=0)  # type: ignore[operator]
        else:
            assert f.T_base_tcp is None and f.T_base_cam_prior is None
    assert {f.header.frame_id: f.source_meta["not_written_before"] for f in frames}[4] == 1
    assert all(f.source_meta["rgbd"] == "UNCHECKED" for f in frames)  # no depth timing recorded
    end = src.next()
    assert isinstance(end, EndOfStream)
    assert (end.frames_emitted, end.not_written, end.not_emitted) == (9, 1, 0)


def test_replay_is_deterministic(session: Path) -> None:
    a, b = list(ReplaySource(session)), list(ReplaySource(session))
    assert len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b, strict=True))
    assert deep_equal(ReplaySource(session).read(5), a[4])
    subset = ReplaySource(session, frame_ids=[5]).read(5)
    assert subset.source_meta["not_written_before"] == 1  # frame 3 lies before 5 in the subset


def test_handeye_override_changes_prior_and_leaves_the_session(session: Path) -> None:
    from mvsr.calib.handeye import HandEye
    from mvsr.core.se3 import PoseSE3

    he = load_handeye(session / "calibration" / "handeye_tf.txt", expected_sha256=None)
    shifted = HandEye(
        T_tcp_cam=PoseSE3.from_rt(
            he.T_tcp_cam.R,
            np.add(np.array(he.T_tcp_cam.t), np.array([0.01, 0.0, 0.0])),
            "tcp",
            "cam",
        ),
        T_base_head=he.T_base_head,
        sha256="ab" * 32,
        source_path="shifted",
        raw_text="",
    )
    before = (session / "calibration" / "wrist.json").read_bytes()
    base = ReplaySource(session).read(0)
    over = ReplaySource(session, handeye=shifted).read(0)
    assert base.T_base_cam_prior is not None and over.T_base_cam_prior is not None
    assert not base.T_base_cam_prior.allclose(over.T_base_cam_prior, atol=1e-4)
    assert over.header.calibration_id != base.header.calibration_id
    assert (session / "calibration" / "wrist.json").read_bytes() == before


def test_replay_rejects_bad_requests(session: Path) -> None:
    with pytest.raises(SessionFormatError, match="not WRITTEN"):
        ReplaySource(session, frame_ids=[3])
    with pytest.raises(SessionFormatError, match="no camera"):
        ReplaySource(session, "head")
    with pytest.raises(ValueError):
        ReplaySource(session, speed=0)


def test_replay_speed_paces_emission(session: Path) -> None:
    src = ReplaySource(session, speed=2.0)
    t0 = time.monotonic()
    n = sum(1 for _ in src)
    span_s = (frame_mid(9) - frame_mid(0)) / 1e9
    assert n == 9 and time.monotonic() - t0 >= 0.9 * span_s / 2.0


def test_sensors_report_run_and_i0_without_board(session: Path, tmp_path: Path) -> None:
    cfg = load_sensors_config()
    rep, bundles = session_report(session, cfg)
    assert rep["board_check"] is None and bundles == {}
    w = rep["cameras"]["wrist"]
    assert w["frames"] == {
        "records": 10,
        "written": 9,
        "emitted": 9,
        "by_status": {"DROPPED_QUEUE_FULL": 1, "WRITTEN": 9},
        "unlogged_frame_ids": 0,
    }
    assert w["rgbd"]["written_by_status"] == {"UNCHECKED": 9} and w["rgbd"]["valid_ratio"] == 0.0
    assert w["sync"]["by_status"] == {"GAP_TOO_LARGE": 1, "INTERPOLATED": 6, "OUT_OF_RANGE": 2}
    assert rep["robot"]["samples"] == len(ROBOT_MS) - 3 and rep["robot"]["failures"] == 3
    decision = i0_decision([rep], cfg)
    assert decision["decision"] == "BLOCKED" and "r_calib_session" in decision["failed_hard"]
    run = write_run([rep], bundles, decision, cfg, tmp_path)
    manifest = json.loads((run / "manifest.json").read_text())
    assert run.name.split("_")[1] == "CW20175"
    for rel, sha in manifest["files"].items():
        assert sha256_file(run / rel) == sha
    assert manifest["inputs"] == [{"session_id": SID, "manifest_sha256": rep["manifest_sha256"]}]


def test_cli_replay_digest_is_stable(
    session: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["replay", str(session)]) == 0
    first = json.loads(capsys.readouterr().out)
    assert main(["replay", SID, "--data-root", str(tmp_path), "--no-verify"]) == 0
    second = json.loads(capsys.readouterr().out)
    assert first["frames"] == 9 and first["output_sha256"] == second["output_sha256"]
    assert main(["sensors-report", str(session), "--data-root", str(tmp_path), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["i0"] is None and Path(out["run_dir"]).is_dir()
    assert main(["i0-gate", str(session), "--data-root", str(tmp_path)]) == 1
