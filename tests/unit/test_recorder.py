"""Recorder control flow with SDK test doubles (ADR-0002: error paths only, never accuracy/timing)."""

from __future__ import annotations

import json
import signal
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fakes.fake_rm import FakeRM
from fakes.fake_rs import DeviceSpec, FakeRS
from mvsr.core.errors import DeviceError, DiskBudgetError
from mvsr.io.session_reader import SessionReader
from mvsr.io.session_writer import SessionWriter
from mvsr.io.verify import verify_session
from mvsr.record.config import RecordConfig, load_record_config
from mvsr.record.recorder import Recorder, RecordRequest

WRIST, HEAD = "323622271240", "244622070961"


@pytest.fixture
def config() -> RecordConfig:
    cfg = load_record_config()
    cams = {cid: replace(c, width=32, height=24) for cid, c in cfg.cameras.items()}
    return replace(
        cfg,
        cameras=cams,
        capture=replace(
            cfg.capture,
            warmup_frames=3,
            first_frame_timeout_ms=2000,
            frame_timeout_ms=200,
            max_consecutive_timeouts=3,
        ),
        robot=replace(cfg.robot, poll_hz=200.0, abort_after_failure_s=0.5),
        disk=replace(cfg.disk, reserve_gb=0.0, est_bytes_per_frame={"wrist": 1e3, "head": 1e3}),
        monitor=replace(cfg.monitor, status_interval_s=0.5),
    )


def request(tmp_path: Path, **kw: Any) -> RecordRequest:
    base: dict[str, Any] = dict(
        cameras=("wrist",),
        scenario=("SMOKE",),
        object_name="test-object",
        duration_s=0.6,
        data_root=tmp_path,
    )
    base.update(kw)
    return RecordRequest(**base)


def run(
    config: RecordConfig,
    req: RecordRequest,
    rs: FakeRS | None = None,
    rm: FakeRM | None = None,
    **kw: Any,
) -> Any:
    rs = rs or FakeRS(
        DeviceSpec(WRIST), DeviceSpec(HEAD, name="Intel RealSense D435", depth_scale=1e-3)
    )
    kw.setdefault("install_signal_handlers", False)
    return Recorder(config, req, rs_module=rs, rm_sdk=rm or FakeRM(), **kw).run()


def sessions(tmp_path: Path) -> list[Path]:
    d = tmp_path / "sessions"
    return sorted(d.iterdir()) if d.exists() else []


def test_duration_stop_complete_and_verifies(config: RecordConfig, tmp_path: Path) -> None:
    res = run(config, request(tmp_path))
    assert res.status == "COMPLETE" and res.stop_reason == "duration"
    rep = verify_session(res.session_dir)
    assert rep.ok, rep.errors
    r = SessionReader(res.session_dir)
    recs = r.frame_records("wrist")
    assert len(recs) > 20 and all(x.status.value == "WRITTEN" for x in recs)
    assert recs[0].timing.rs_sensor_timestamp_us is not None
    assert recs[0].timing.rs_timestamp_domain == "global_time"
    assert recs[0].depth_timing is not None
    assert r.manifest["cameras"]["wrist"]["depth_units_m"] == 1e-4
    assert r.manifest["cameras"]["wrist"]["global_time_enabled"] == {
        "Stereo Module": True,
        "RGB Camera": True,
    }
    assert r.manifest["robot"]["state_source"] == "poll"
    assert r.manifest["counts"]["robot"]["samples"] > 10
    assert [e.type.value for e in r.events()][0 :: len(r.events()) - 1] == ["START", "STOP"]
    perf = r.manifest["performance"]
    assert perf["cameras"]["wrist"]["warmup_frames_discarded"] >= 3
    assert (Path(res.session_dir) / "calibration/handeye_tf.txt").read_bytes() == (
        Path(config.handeye_path())
    ).read_bytes()


def test_ctrl_c_finalizes_manifest(config: RecordConfig, tmp_path: Path) -> None:
    req = request(tmp_path, duration_s=None)
    timer = threading.Timer(0.7, lambda: signal.raise_signal(signal.SIGINT))
    timer.start()
    try:
        res = run(config, req, install_signal_handlers=True)
    finally:
        timer.cancel()
    assert res.status == "COMPLETE" and res.stop_reason == "operator"
    assert verify_session(res.session_dir).ok
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_device_disconnect_aborts_with_complete_records(
    config: RecordConfig, tmp_path: Path
) -> None:
    rs = FakeRS(DeviceSpec(WRIST, disconnect_after=40))
    res = run(config, request(tmp_path, duration_s=5.0), rs=rs)
    assert res.status == "ABORTED" and res.stop_reason == "error"
    assert any("No device connected" in e for e in res.errors)
    rep = verify_session(res.session_dir)
    assert rep.errors == ["manifest status is 'ABORTED', not COMPLETE"]
    events = [e.type.value for e in SessionReader(res.session_dir).events()]
    assert "ERROR" in events and events[-1] == "STOP"


def test_frame_timeouts_treated_as_disconnect(config: RecordConfig, tmp_path: Path) -> None:
    rs = FakeRS(DeviceSpec(WRIST, no_frames=True))
    with pytest.raises(DeviceError):
        run(config, request(tmp_path), rs=rs)
    assert sessions(tmp_path) == []


def test_wrong_depth_unit_refuses_start(config: RecordConfig, tmp_path: Path) -> None:
    rs = FakeRS(DeviceSpec(WRIST, depth_scale=1e-3))
    with pytest.raises(DeviceError, match="depth unit"):
        run(config, request(tmp_path), rs=rs)
    assert sessions(tmp_path) == []


def test_wrong_serial_refuses_start(config: RecordConfig, tmp_path: Path) -> None:
    with pytest.raises(DeviceError, match="not found"):
        run(config, request(tmp_path), rs=FakeRS(DeviceSpec("999")))


def test_robot_connect_failure(config: RecordConfig, tmp_path: Path) -> None:
    with pytest.raises(DeviceError, match="RM65"):
        run(config, request(tmp_path), rm=FakeRM(connect_ok=False))
    assert sessions(tmp_path) == []


def test_robot_read_failures_logged(config: RecordConfig, tmp_path: Path) -> None:
    rm = FakeRM(fail=lambda n: n % 3 == 1)
    res = run(config, request(tmp_path), rm=rm)
    assert res.status == "COMPLETE"
    r = SessionReader(res.session_dir)
    recs = r.robot_records()
    failed = [x for x in recs if x.error is not None]
    assert failed and all(x.tcp_xyzrpy is None and "ret=-2" in x.error for x in failed)
    assert [x.seq for x in recs] == list(range(len(recs)))
    ok = [x for x in recs if x.error is None]
    assert (
        ok[0].joints_rad is not None
        and abs(ok[0].joints_rad[1] - (-20.0 * 3.141592653589793 / 180)) < 1e-12
    )
    assert r.manifest["counts"]["robot"]["failures"] == len(failed)
    assert "WARNING" in [e.type.value for e in r.events()]
    assert verify_session(res.session_dir).ok
    assert rm.arms[0].deleted


def test_robot_persistent_failure_aborts(config: RecordConfig, tmp_path: Path) -> None:
    res = run(config, request(tmp_path, duration_s=5.0), rm=FakeRM(fail=lambda n: True))
    assert res.status == "ABORTED"
    assert any("robot: no valid state" in e for e in res.errors)


def test_full_queue_drops_are_logged(
    config: RecordConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = replace(
        config,
        capture=replace(config.capture, process_queue_size=1, writer_queue_size=1),
        cameras={"wrist": replace(config.cameras["wrist"], writer_threads=1)},
    )
    original = SessionWriter.write_images

    def slow(self: SessionWriter, *args: Any) -> Any:
        time.sleep(0.05)
        return original(self, *args)

    monkeypatch.setattr(SessionWriter, "write_images", slow)
    res = run(cfg, request(tmp_path, duration_s=0.8))
    assert res.status == "COMPLETE"
    recs = SessionReader(res.session_dir).frame_records("wrist")
    full = [x for x in recs if x.status.value == "DROPPED_QUEUE_FULL"]
    assert full and {x.drop_detail for x in full} <= {"process_queue", "writer_queue"}
    assert [x.frame_id for x in recs] == list(range(len(recs)))
    assert verify_session(res.session_dir).ok


def test_incomplete_framesets_logged(config: RecordConfig, tmp_path: Path) -> None:
    rs = FakeRS(DeviceSpec(WRIST, incomplete_every=5))
    res = run(config, request(tmp_path), rs=rs)
    recs = SessionReader(res.session_dir).frame_records("wrist")
    inc = [x for x in recs if x.status.value == "DROPPED_INCOMPLETE"]
    assert inc and all(x.drop_detail == "color=True depth=False" for x in inc)
    assert verify_session(res.session_dir).ok


def test_disk_reserve_stops_cleanly(config: RecordConfig, tmp_path: Path) -> None:
    cfg = replace(config, disk=replace(config.disk, reserve_gb=1.0, check_interval_s=0.05))
    t0 = time.monotonic()

    def usage(path: str) -> SimpleNamespace:
        return SimpleNamespace(free=int(1e12 if time.monotonic() - t0 < 0.5 else 0.5e9))

    res = run(cfg, request(tmp_path, duration_s=10.0), disk_usage=usage)
    assert res.status == "COMPLETE" and res.stop_reason == "disk_reserve"
    events = SessionReader(res.session_dir).events()
    assert any(e.type.value == "WARNING" and "disk reserve" in e.payload["message"] for e in events)
    assert verify_session(res.session_dir).ok


def test_disk_budget_refuses_start(config: RecordConfig, tmp_path: Path) -> None:
    cfg = replace(config, disk=replace(config.disk, reserve_gb=20.0))
    with pytest.raises(DiskBudgetError):
        run(cfg, request(tmp_path), disk_usage=lambda p: SimpleNamespace(free=int(10e9)))
    assert sessions(tmp_path) == []


def test_disk_write_error_aborts(
    config: RecordConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}
    original = SessionWriter.write_images

    def failing(self: SessionWriter, *args: Any) -> Any:
        calls["n"] += 1
        if calls["n"] > 5:
            raise OSError(28, "No space left on device")
        return original(self, *args)

    monkeypatch.setattr(SessionWriter, "write_images", failing)
    res = run(config, request(tmp_path, duration_s=3.0))
    assert res.status == "ABORTED"
    recs = SessionReader(res.session_dir).frame_records("wrist")
    assert any(x.status.value == "DROPPED_DISK" for x in recs)
    assert [x.frame_id for x in recs] == list(range(len(recs)))


def test_wrist_and_head_head_started_first(config: RecordConfig, tmp_path: Path) -> None:
    rs = FakeRS(DeviceSpec(WRIST), DeviceSpec(HEAD, name="Intel RealSense D435", depth_scale=1e-3))
    res = run(config, request(tmp_path, cameras=("wrist", "head")), rs=rs)
    assert rs.start_order == [HEAD, WRIST]
    assert res.status == "COMPLETE"
    r = SessionReader(res.session_dir)
    assert r.camera_ids == ["head", "wrist"]
    head = r.calibration("head")
    assert head.mount == "fixed" and head.T_base_head is not None and head.depth_units_m == 1e-3
    assert r.calibration("wrist").T_tcp_cam is not None
    assert r.manifest["cameras"]["head"]["fps"] == 15
    rep = verify_session(res.session_dir)
    assert rep.ok, rep.errors


def test_request_validation(tmp_path: Path) -> None:
    for kw in (
        {"cameras": ("head",)},
        {"scenario": ("R-FOO",)},
        {"object_name": " "},
        {"duration_s": 0.0},
        {"start_paused": True},
    ):
        with pytest.raises(ValueError):
            request(tmp_path, **kw)


def test_manifest_keeps_config_snapshot(config: RecordConfig, tmp_path: Path) -> None:
    res = run(config, request(tmp_path, notes="hello", gt_type="constant_T_base_obj"))
    m = json.loads((Path(res.session_dir) / "manifest.json").read_text())
    assert m["scenario"] == {
        "tags": ["SMOKE"],
        "object": "test-object",
        "gt_type": "constant_T_base_obj",
        "notes": "hello",
    }
    assert m["recorder"]["config_sha256"] == config.source_sha256
    assert m["clock"]["start"]["uncertainty_ns"] >= 0 and "end" in m["clock"]
    assert m["disk"]["free_bytes_end"] > 0
