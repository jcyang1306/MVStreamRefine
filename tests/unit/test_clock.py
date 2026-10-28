"""Clock mapping (P1 S4): counters, envelope fit, realtime map, per-frame methods (analytic)."""

from __future__ import annotations

import numpy as np
import pytest

from mvsr.core.errors import ClockModelError
from mvsr.core.types import FrameStatus
from mvsr.io.records import FrameRecord, RawTiming
from mvsr.sensors.clock import (
    METHOD_GLOBAL_FRAME,
    METHOD_GLOBAL_HALF_EXPOSURE,
    METHOD_GLOBAL_SENSOR,
    RealtimeMap,
    arrival_method,
    build_stream_clock,
    counter_delta_us,
    envelope_method,
    fit_lower_envelope,
    frame_timestamp_ns,
    global_mid_exposure_ns,
    unwrap_counter,
)
from mvsr.sensors.config import load_sensors_config

CFG = load_sensors_config().clock
P = 1 << 32
R0 = 1_793_000_000_000_000_000  # CLOCK_REALTIME ns (multiple of 1 ms)
M0 = 2_000_000_000_000_000  # CLOCK_MONOTONIC ns
RT = RealtimeMap.from_pairs({"realtime_ns": R0, "monotonic_ns": M0, "uncertainty_ns": 120})


def test_counter_delta_and_unwrap() -> None:
    assert counter_delta_us(5, P - 5, P) == 10
    assert counter_delta_us(P - 5, 5, P) == -10
    assert unwrap_counter([P - 20, P - 10, 3, 13], P) == [P - 20, P - 10, P + 3, P + 13]
    assert unwrap_counter([7, 7, 8], P) == [7, 7, 8]
    with pytest.raises(ClockModelError, match="backwards"):
        unwrap_counter([100, 90], P)
    with pytest.raises(ClockModelError, match="jumped"):
        unwrap_counter([0, P // 2 + 10], P)


def test_lower_envelope_recovers_minimum_latency_line() -> None:
    rate, offset = 1.0 - 68e-6, 5_000_000
    k = np.arange(1067, dtype=np.int64)
    x = k * 37_500_000  # 68 ppm of 37.5 ms = 2550 ns exactly, so min-latency points are collinear
    lat = np.where(k % 10 == 0, 0, 1_000_000 + (k * 7919) % 9_000_000)
    y = offset + x - 2550 * k + lat
    dev, host = (x + 3_000_000_000_000_000).tolist(), (y + M0).tolist()
    fit = fit_lower_envelope(dev, host)
    assert abs(fit.rate - rate) < 1e-9 and abs(fit.rate_ppm + 68.0) < 1e-3
    mapped = np.array([fit.map_ns(d) for d in dev])
    assert np.all(np.asarray(host) - mapped >= -1)
    assert np.all(np.abs((np.asarray(host) - mapped)[lat == 0]) <= 1)
    assert fit_lower_envelope(dev[::-1], host[::-1]) == fit  # order independent


def test_lower_envelope_degenerate_inputs() -> None:
    single = fit_lower_envelope([10], [50])
    assert single.rate == 1.0 and single.map_ns(10) == 50
    with pytest.raises(ClockModelError):
        fit_lower_envelope([], [])


def test_realtime_map_interpolates_offset_and_keeps_nanoseconds() -> None:
    m = RealtimeMap.from_pairs(
        {"realtime_ns": R0, "monotonic_ns": M0},
        {"realtime_ns": R0 + 100_000_000_000, "monotonic_ns": M0 + 100_000_000_100},
    )
    assert m.drift_ns == -100
    assert m.to_monotonic_ns(R0) == M0
    assert m.to_monotonic_ns(R0 + 100_000_000_000) == M0 + 100_000_000_100
    assert m.to_monotonic_ns(R0 + 50_000_000_000) == M0 + 50_000_000_050
    assert m.to_monotonic_ns(R0 - 1_000) == M0 - 1_000  # held constant outside the pairs
    assert m.ms_to_monotonic_ns((R0 + 50_000_250_000) / 10**6) == M0 + 50_000_250_050
    assert RealtimeMap.from_manifest({}) is None
    with pytest.raises(ClockModelError):
        RealtimeMap.from_pairs({"realtime_ns": R0})


def _timing(**kw: object) -> RawTiming:
    base: dict[str, object] = dict(
        rs_timestamp_ms=(R0 + 10_000_000) / 10**6,
        rs_timestamp_domain="global_time",
        rs_sensor_timestamp_us=1_000,
        rs_frame_timestamp_us=13_626,
        exposure_us=25_252.0,
        rs_frame_number=1,
    )
    base.update(kw)
    return RawTiming(**base)  # type: ignore[arg-type]


def test_global_mid_exposure_methods() -> None:
    frame_mono = M0 + 10_000_000
    assert global_mid_exposure_ns(_timing(), RT, P) == (
        frame_mono - 12_626_000,
        METHOD_GLOBAL_SENSOR,
    )
    wrap = _timing(rs_sensor_timestamp_us=P - 100, rs_frame_timestamp_us=26)
    assert global_mid_exposure_ns(wrap, RT, P) == (frame_mono - 126_000, METHOD_GLOBAL_SENSOR)
    half = _timing(rs_sensor_timestamp_us=None)
    assert global_mid_exposure_ns(half, RT, P) == (
        frame_mono - 12_626_000,
        METHOD_GLOBAL_HALF_EXPOSURE,
    )
    bare = _timing(rs_sensor_timestamp_us=None, exposure_us=None)
    assert global_mid_exposure_ns(bare, RT, P) == (frame_mono, METHOD_GLOBAL_FRAME)
    assert global_mid_exposure_ns(_timing(rs_timestamp_domain="hardware_clock"), RT, P) is None
    assert global_mid_exposure_ns(_timing(), None, P) is None
    lat = CFG.fallback_latency_ns
    assert frame_timestamp_ns(_timing(), 0, None, CFG) == (-lat, arrival_method(lat))


def _records(
    n: int, *, global_time: bool = True, device: bool = True
) -> tuple[list[FrameRecord], list[int]]:
    """Frames 33.3 ms apart; arrival = mid-exposure + 28 ms + (0..4 ms) jitter."""
    recs, mids = [], []
    for i in range(n):
        mid = M0 + 1_000_000_000 + i * 33_333_000
        sensor = (500 + (mid - M0) // 1000) % P
        timing = RawTiming(
            rs_timestamp_ms=(R0 + (mid - M0) + 12_000_000) / 10**6 if global_time else None,
            rs_timestamp_domain="global_time" if global_time else "hardware_clock",
            rs_sensor_timestamp_us=sensor if device else None,
            rs_frame_timestamp_us=(sensor + 12_000) % P if device else None,
            exposure_us=24_000.0 if device else None,
            rs_frame_number=i,
        )
        arrival = mid + 28_000_000 + (i % 5) * 1_000_000
        recs.append(FrameRecord(i, FrameStatus.DROPPED_QUEUE_FULL, arrival, timing))
        mids.append(mid)
    return recs, mids


def test_stream_clock_prefers_global_time_and_reports_transfer_latency() -> None:
    recs, mids = _records(60)
    clock = build_stream_clock("wrist", recs[::-1], RT, CFG)
    for r, mid in zip(recs, mids, strict=True):
        ts, method = clock.timestamp(r.frame_id)
        assert method == METHOD_GLOBAL_SENSOR and abs(ts - mid) <= 1_000
    rep = clock.report
    assert rep["methods"] == {METHOD_GLOBAL_SENSOR: 60} and rep["non_increasing_timestamps"] == 0
    assert abs(rep["envelope_minus_global"]["offset_ms"] - 28.0) < 0.002
    assert rep["envelope_minus_global"]["abs_residual_ms"]["max"] < 0.002
    assert abs(rep["arrival_minus_timestamp_ms"]["median"] - 30.0) < 0.01
    assert abs(rep["envelope"]["rate_ppm"]) < 1.0
    with pytest.raises(ClockModelError):
        clock.timestamp(999)


def test_stream_clock_fallbacks_declare_their_latency() -> None:
    lat = CFG.fallback_latency_ns
    recs, mids = _records(30, global_time=False)
    clock = build_stream_clock("wrist", recs, RT, CFG)
    ts, method = clock.timestamp(0)
    assert method == envelope_method(lat) and abs(ts - mids[0]) <= 1_000
    recs, _ = _records(30, global_time=False, device=False)
    clock = build_stream_clock("wrist", recs, None, CFG)
    assert clock.timestamp(3) == (recs[3].t_host_arrival_ns - lat, arrival_method(lat))
    assert clock.envelope is None and clock.report["envelope_minus_global"] is None


def test_device_clock_rate_outside_tolerance_raises() -> None:
    recs = []
    for i in range(20):
        t = RawTiming(
            rs_sensor_timestamp_us=int(i * 33_333 * 1.002), rs_timestamp_domain="hardware_clock"
        )
        recs.append(FrameRecord(i, FrameStatus.DROPPED_QUEUE_FULL, M0 + i * 33_333_000, t))
    with pytest.raises(ClockModelError, match="ppm"):
        build_stream_clock("wrist", recs, None, CFG)
