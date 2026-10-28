"""Live SensorFrames on the rig (P1 S1). Run on CW20175 only:

docker compose run --rm rig pytest -q tests/hardware --run-hardware
"""

from __future__ import annotations

import itertools

import pytest

from mvsr.core.types import SensorFrame, SyncStatus
from mvsr.sensors.clock import METHOD_GLOBAL_SENSOR
from mvsr.sensors.live import LiveSource

pytestmark = pytest.mark.hardware


def test_live_source_emits_synced_mid_exposure_frames() -> None:
    with LiveSource() as src:
        items = [src.next() for _ in range(90)]
    frames = [f for f in items if isinstance(f, SensorFrame)]
    assert len(frames) == 90
    ts = [f.header.timestamp_ns for f in frames]
    assert all(b > a for a, b in itertools.pairwise(ts))
    assert all(f.timing.method == METHOD_GLOBAL_SENSOR for f in frames)
    latency_ms = [(f.timing.t_host_arrival_ns - f.header.timestamp_ns) / 1e6 for f in frames]
    print("arrival - mid-exposure ms", min(latency_ms), max(latency_ms))
    assert all(0.0 < v < 200.0 for v in latency_ms)
    synced = sum(f.pose_sync.status is SyncStatus.INTERPOLATED for f in frames)
    print("interpolated", synced, "/", len(frames))
    assert synced >= 0.9 * len(frames)
