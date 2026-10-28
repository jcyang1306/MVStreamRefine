"""LiveSource control flow with SDK test doubles (ADR-0002: error paths only, never timing evidence)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from fakes.fake_rm import FakeRM
from fakes.fake_rs import DeviceSpec, FakeRS
from mvsr.core.errors import DeviceError
from mvsr.core.types import SensorFrame, SyncStatus
from mvsr.record.config import RecordConfig, load_record_config
from mvsr.sensors.clock import METHOD_GLOBAL_SENSOR
from mvsr.sensors.config import SensorsConfig, load_sensors_config
from mvsr.sensors.live import LiveSource
from mvsr.sensors.source import EndOfStream

WRIST = "323622271240"


@pytest.fixture
def configs() -> tuple[RecordConfig, SensorsConfig]:
    rc = load_record_config()
    rc = replace(
        rc,
        cameras={"wrist": replace(rc.cameras["wrist"], width=32, height=24)},
        robot=replace(rc.robot, poll_hz=200.0, poll_retry_delay_s=0.0),
    )
    sc = load_sensors_config()
    sc = replace(
        sc,
        live=replace(sc.live, warmup_frames=2, first_frame_timeout_ms=2000, frame_timeout_ms=300),
    )
    return rc, sc


def test_live_frames_are_timestamped_and_synced(
    configs: tuple[RecordConfig, SensorsConfig],
) -> None:
    rc, sc = configs
    rm = FakeRM()
    with LiveSource(rc, sc, rs_module=FakeRS(DeviceSpec(WRIST, period_s=0.02)), rm_sdk=rm) as src:
        items = [src.next() for _ in range(8)]
    frames = [f for f in items if isinstance(f, SensorFrame)]
    assert len(frames) == 8
    ids = [f.header.frame_id for f in frames]
    assert ids == sorted(ids) and len(set(ids)) == 8
    assert all(f.timing.method == METHOD_GLOBAL_SENSOR for f in frames)
    assert all(f.header.timestamp_ns < f.timing.t_host_arrival_ns for f in frames)
    synced = [f for f in frames if f.pose_sync.status is SyncStatus.INTERPOLATED]
    assert synced and all(f.T_base_cam_prior is not None for f in synced)
    assert all(f.T_base_cam_prior is None for f in frames if f not in synced)
    assert isinstance(src.next(), EndOfStream)
    assert rm.arms[0].deleted


def test_live_counts_incomplete_framesets(configs: tuple[RecordConfig, SensorsConfig]) -> None:
    rc, sc = configs
    rs = FakeRS(DeviceSpec(WRIST, period_s=0.005, incomplete_every=3))
    with LiveSource(rc, sc, rs_module=rs, rm_sdk=FakeRM()) as src:
        frames = [src.next() for _ in range(6)]
    assert all(isinstance(f, SensorFrame) for f in frames)
    skipped = sum(f.source_meta["not_written_before"] for f in frames if isinstance(f, SensorFrame))
    last = frames[-1]
    assert isinstance(last, SensorFrame) and skipped == last.header.frame_id + 1 - len(frames)
    assert skipped >= 2


def test_live_start_failures_raise_and_release(configs: tuple[RecordConfig, SensorsConfig]) -> None:
    rc, sc = configs
    with pytest.raises(DeviceError, match="RM65"):
        LiveSource(
            rc, sc, rs_module=FakeRS(DeviceSpec(WRIST)), rm_sdk=FakeRM(connect_ok=False)
        ).start()
    rm = FakeRM()
    with pytest.raises(DeviceError, match="not found"):
        LiveSource(rc, sc, rs_module=FakeRS(DeviceSpec("999")), rm_sdk=rm).start()
    assert rm.arms[0].deleted
    with pytest.raises(DeviceError, match="timeouts"):
        LiveSource(
            rc, sc, rs_module=FakeRS(DeviceSpec(WRIST, no_frames=True)), rm_sdk=FakeRM()
        ).start()
