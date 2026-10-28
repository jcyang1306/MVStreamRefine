"""Colour/depth exposure pairing check (hand-written timings; tests code, not sensors)."""

from __future__ import annotations

from mvsr.core.types import FrameStatus
from mvsr.io.records import FrameRecord, RawTiming
from mvsr.sensors.rgbd import OnlinePairCheck, RgbdPair, check_pairs

P = 1 << 32


def _rec(i: int, color_us: int, depth_us: int | None, number: int) -> FrameRecord:
    color = RawTiming(rs_sensor_timestamp_us=color_us, rs_frame_number=number)
    depth = (
        None
        if depth_us is None
        else RawTiming(rs_sensor_timestamp_us=depth_us, rs_frame_number=number + 14)
    )
    return FrameRecord(i, FrameStatus.DROPPED_QUEUE_FULL, 1000 + i, color, depth_timing=depth)


def _stream() -> list[FrameRecord]:
    """In sync: depth 230 us before colour. Frame 4 repeats colour #3, frame 6 has depth 33 ms old."""
    recs = []
    for i in range(10):
        number = 3 if i == 4 else i
        c = 1_000_000 + number * 33_333
        d = 1_000_000 + i * 33_333 - 230
        if i == 6:
            d -= 33_333
        recs.append(_rec(i, c, d, number))
    recs.append(_rec(10, P - 10, 40, 10))  # wraps the counter; dt = +50 us, in sync
    recs.append(_rec(11, 2_000_000, None, 11))
    return recs


def test_check_pairs_flags_repeats_and_mismatches() -> None:
    checks, report = check_pairs(_stream(), 5.0, P)
    status = {fid: c.status for fid, c in checks.items()}
    assert status[4] is RgbdPair.REPEATED_COLOR
    assert status[6] is RgbdPair.COLOR_DEPTH_MISMATCH and checks[6].dt_us == -33_563
    assert status[10] is RgbdPair.OK and checks[10].dt_us == 50
    assert status[11] is RgbdPair.UNCHECKED
    assert [f for f, s in status.items() if s is RgbdPair.OK] == [0, 1, 2, 3, 5, 7, 8, 9, 10]
    assert report["expected_color_depth_dt_ms"] == -0.23
    assert report["by_status"] == {
        "COLOR_DEPTH_MISMATCH": 1,
        "OK": 9,
        "REPEATED_COLOR": 1,
        "UNCHECKED": 1,
    }
    disabled, _ = check_pairs(_stream(), None, P)
    assert {c.status for c in disabled.values()} == {RgbdPair.UNCHECKED}


def test_online_check_needs_history_then_applies_the_same_rule() -> None:
    online = OnlinePairCheck(5.0, P, window=50)
    statuses = []
    for i in range(30):
        color = RawTiming(rs_sensor_timestamp_us=i * 33_333, rs_frame_number=i)
        shift = -33_333 if i == 20 else 0
        depth = RawTiming(rs_sensor_timestamp_us=i * 33_333 - 230 + shift, rs_frame_number=i)
        statuses.append(online.check(color, depth).status)
    assert statuses[:10] == [RgbdPair.UNCHECKED] * 10
    assert statuses[20] is RgbdPair.COLOR_DEPTH_MISMATCH
    assert all(s is RgbdPair.OK for i, s in enumerate(statuses[10:], 10) if i != 20)
    repeat = online.check(RawTiming(rs_sensor_timestamp_us=29 * 33_333, rs_frame_number=29), None)
    assert repeat.status is RgbdPair.REPEATED_COLOR
