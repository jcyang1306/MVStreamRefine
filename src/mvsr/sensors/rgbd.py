"""Colour/depth exposure pairing check (P1; found on the 2026-10-28 sessions).

librealsense can deliver a frameset whose colour and depth come from different exposures:
the D405 wrist stream repeats the previous colour frame with a new depth frame, or pairs a
colour frame with a depth frame one or more intervals older (14-15 % of wrist frames on the
dual-camera sessions). Such a frame is not a registered RGB-D observation.

Per frame `dt = depth_sensor_timestamp - colour_sensor_timestamp` (device µs, both recorded).
A stream has a constant in-sync offset (D405 about -0.2 ms; D435 -32.4 ms because colour has
no mid-exposure stamp), so a frame is `COLOR_DEPTH_MISMATCH` when `|dt - median(dt)|` exceeds
`rgbd.max_color_depth_dt_ms`, and `REPEATED_COLOR` when its colour frame number equals the
previous record's. Frames without depth timing are `UNCHECKED`.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from mvsr.io.records import FrameRecord, RawTiming
from mvsr.sensors.clock import counter_delta_us


class RgbdPair(str, Enum):
    OK = "OK"
    REPEATED_COLOR = "REPEATED_COLOR"
    COLOR_DEPTH_MISMATCH = "COLOR_DEPTH_MISMATCH"
    UNCHECKED = "UNCHECKED"


REJECTED = (RgbdPair.REPEATED_COLOR, RgbdPair.COLOR_DEPTH_MISMATCH)


def color_depth_dt_us(color: RawTiming, depth: RawTiming | None, period_us: int) -> int | None:
    if (
        depth is None
        or color.rs_sensor_timestamp_us is None
        or depth.rs_sensor_timestamp_us is None
    ):
        return None
    return counter_delta_us(depth.rs_sensor_timestamp_us, color.rs_sensor_timestamp_us, period_us)


@dataclass(frozen=True)
class RgbdCheck:
    status: RgbdPair
    dt_us: int | None


def check_pairs(
    records: Sequence[FrameRecord], max_dt_ms: float | None, period_us: int
) -> tuple[dict[int, RgbdCheck], dict[str, Any]]:
    """Pairing status per frame_id of one stream (`max_dt_ms=None` disables the check)."""
    recs = sorted(records, key=lambda r: r.frame_id)
    dts = {r.frame_id: color_depth_dt_us(r.timing, r.depth_timing, period_us) for r in recs}
    known = [d for d in dts.values() if d is not None]
    expected = float(np.median(known)) if known else None
    out: dict[int, RgbdCheck] = {}
    prev_number: int | None = None
    for r in recs:
        dt = dts[r.frame_id]
        number = r.timing.rs_frame_number
        if max_dt_ms is None or (dt is None and number is None):
            status = RgbdPair.UNCHECKED
        elif number is not None and number == prev_number:
            status = RgbdPair.REPEATED_COLOR
        elif dt is None or expected is None:
            status = RgbdPair.UNCHECKED
        elif abs(dt - expected) > max_dt_ms * 1000.0:
            status = RgbdPair.COLOR_DEPTH_MISMATCH
        else:
            status = RgbdPair.OK
        out[r.frame_id] = RgbdCheck(status, dt)
        prev_number = number if number is not None else prev_number
    report = {
        "max_color_depth_dt_ms": max_dt_ms,
        "expected_color_depth_dt_ms": None if expected is None else round(expected / 1e3, 4),
        "by_status": dict(sorted(Counter(c.status.value for c in out.values()).items())),
    }
    return out, report


class OnlinePairCheck:
    """The same rule online (live source): median over the last `window` in-range offsets."""

    def __init__(self, max_dt_ms: float | None, period_us: int, window: int = 200) -> None:
        self.max_dt_ms = max_dt_ms
        self.period_us = period_us
        self._dts: deque[int] = deque(maxlen=window)
        self._prev_number: int | None = None

    def check(self, color: RawTiming, depth: RawTiming | None) -> RgbdCheck:
        dt = color_depth_dt_us(color, depth, self.period_us)
        number = color.rs_frame_number
        repeated = number is not None and number == self._prev_number
        self._prev_number = number if number is not None else self._prev_number
        if self.max_dt_ms is None:
            return RgbdCheck(RgbdPair.UNCHECKED, dt)
        if repeated:
            return RgbdCheck(RgbdPair.REPEATED_COLOR, dt)
        if dt is None:
            return RgbdCheck(RgbdPair.UNCHECKED, dt)
        if len(self._dts) < 10:
            self._dts.append(dt)
            return RgbdCheck(RgbdPair.UNCHECKED, dt)
        expected = float(np.median(np.asarray(self._dts)))
        if abs(dt - expected) > self.max_dt_ms * 1000.0:
            return RgbdCheck(RgbdPair.COLOR_DEPTH_MISMATCH, dt)
        self._dts.append(dt)
        return RgbdCheck(RgbdPair.OK, dt)
