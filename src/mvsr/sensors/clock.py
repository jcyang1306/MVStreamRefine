"""Camera clocks -> HOST_MONOTONIC mid-exposure time (ADR-0004, subplans/01_SENSORS.md S4).

Methods, in order of preference (the one used is `FrameTiming.method` of every frame):

- `rs_global_sensor`: RealSense global time (`rs_timestamp_ms`, domain `global_time`) is
  librealsense's regression of the device clock onto CLOCK_REALTIME at the frame timestamp
  (readout start). It is converted to HOST_MONOTONIC with the session's realtime/monotonic
  pairs and moved back to mid-exposure by `rs_frame_timestamp_us - rs_sensor_timestamp_us`
  (= exposure / 2 on the D405). It excludes the frame-transfer latency that host arrival has.
- `rs_global_half_exposure`: same, without a sensor timestamp: minus `exposure_us / 2`.
- `rs_global_frame`: same, neither available: the frame timestamp itself (not mid-exposure).
- `envelope_minus_<L>ns`: no global time: minimum-latency envelope of host arrival vs. the
  device mid-exposure clock, minus a declared constant latency `L` (`clock.fallback_latency_ns`).
- `arrival_minus_<L>ns`: no device clock at all: arrival minus `L`.

The envelope (the line below all `(device, arrival)` points with the largest mean, i.e. the
lower convex hull edge spanning the mean device time) is always fitted when possible; its
shape residual against global time is the reported "camera timestamp mapping residual", and
its rate is the device clock drift.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Protocol

import numpy as np

from mvsr.core.errors import ClockModelError
from mvsr.io.records import FrameRecord
from mvsr.sensors.config import ClockConfig
from mvsr.sensors.stats import summary

GLOBAL_DOMAIN = "global_time"
METHOD_GLOBAL_SENSOR = "rs_global_sensor"
METHOD_GLOBAL_HALF_EXPOSURE = "rs_global_half_exposure"
METHOD_GLOBAL_FRAME = "rs_global_frame"
#: Methods whose timestamp is a mid-exposure estimate (`rs_global_frame` is not).
MID_EXPOSURE_METHODS = (METHOD_GLOBAL_SENSOR, METHOD_GLOBAL_HALF_EXPOSURE)


def envelope_method(latency_ns: int) -> str:
    return f"envelope_minus_{latency_ns}ns"


def arrival_method(latency_ns: int) -> str:
    return f"arrival_minus_{latency_ns}ns"


class RawClock(Protocol):
    """Raw clock fields shared by `RawTiming` (recorder) and `FrameTiming` (contract)."""

    @property
    def rs_timestamp_ms(self) -> float | None: ...
    @property
    def rs_timestamp_domain(self) -> str | None: ...
    @property
    def rs_sensor_timestamp_us(self) -> int | None: ...
    @property
    def rs_frame_timestamp_us(self) -> int | None: ...
    @property
    def exposure_us(self) -> float | None: ...


# --------------------------------------------------------------------------- realtime map


@dataclass(frozen=True)
class RealtimeMap:
    """CLOCK_REALTIME -> CLOCK_MONOTONIC from `(realtime_ns, monotonic_ns)` pairs.

    The offset is interpolated linearly between the start and end pair (NTP slew) and held
    constant outside them. Measured offset change over a 45 s session: 6 ns.
    """

    rt0_ns: int
    off0_ns: int
    rt1_ns: int
    off1_ns: int
    uncertainty_ns: int

    @classmethod
    def from_pairs(
        cls, start: Mapping[str, Any], end: Mapping[str, Any] | None = None
    ) -> RealtimeMap:
        e = end if end is not None else start
        try:
            rt0, m0 = int(start["realtime_ns"]), int(start["monotonic_ns"])
            rt1, m1 = int(e["realtime_ns"]), int(e["monotonic_ns"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ClockModelError(f"bad clock pair {start!r} / {e!r}") from exc
        if rt1 < rt0:
            raise ClockModelError("clock.end realtime is before clock.start")
        unc = max(int(start.get("uncertainty_ns") or 0), int(e.get("uncertainty_ns") or 0))
        return cls(rt0, rt0 - m0, rt1, rt1 - m1, unc)

    @classmethod
    def from_manifest(cls, clock: Mapping[str, Any] | None) -> RealtimeMap | None:
        if not clock or not clock.get("start"):
            return None
        return cls.from_pairs(clock["start"], clock.get("end"))

    def to_monotonic_ns(self, rt_ns: int) -> int:
        if self.rt1_ns == self.rt0_ns:
            off = self.off0_ns
        else:
            alpha = min(1.0, max(0.0, (rt_ns - self.rt0_ns) / (self.rt1_ns - self.rt0_ns)))
            off = self.off0_ns + round(alpha * (self.off1_ns - self.off0_ns))
        return rt_ns - off

    def ms_to_monotonic_ns(self, rt_ms: float) -> int:
        """Realtime milliseconds (float, as librealsense reports) -> monotonic ns."""
        base_ms = self.rt0_ns // 1_000_000
        rt_ns = base_ms * 1_000_000 + round((rt_ms - base_ms) * 1e6)
        return self.to_monotonic_ns(rt_ns)

    @property
    def drift_ns(self) -> int:
        return self.off1_ns - self.off0_ns


# --------------------------------------------------------------------------- device counter


def counter_delta_us(later: int, earlier: int, period_us: int) -> int:
    """`later - earlier` on a wrapping counter, in `(-period/2, period/2]`."""
    d = (int(later) - int(earlier)) % period_us
    return d - period_us if d > period_us // 2 else d


def unwrap_counter(values: Sequence[int], period_us: int) -> list[int]:
    """Unwrap a non-decreasing wrapping counter; a backwards step (device reset) raises."""
    out: list[int] = []
    wraps = 0
    prev: int | None = None
    for v in values:
        v = int(v)
        if prev is not None:
            step = v - prev
            if step < -(period_us // 2):
                wraps += 1
            elif step < 0:
                raise ClockModelError(f"device counter went backwards ({prev} -> {v} us): reset?")
            elif step > period_us // 2:
                raise ClockModelError(f"device counter jumped forward by {step} us")
        out.append(v + wraps * period_us)
        prev = v
    return out


# --------------------------------------------------------------------------- envelope


@dataclass(frozen=True)
class ClockFit:
    """Minimum-latency envelope `t_host = host0 + offset + rate * (t_dev - dev0)` (ns)."""

    dev0_ns: int
    host0_ns: int
    offset_ns: float
    rate: float
    n_points: int
    hull_points: int

    def map_ns(self, t_dev_ns: int) -> int:
        return self.host0_ns + round(self.offset_ns + self.rate * (int(t_dev_ns) - self.dev0_ns))

    @property
    def rate_ppm(self) -> float:
        return (self.rate - 1.0) * 1e6


def fit_lower_envelope(dev_ns: Sequence[int], host_ns: Sequence[int]) -> ClockFit:
    """Line below every `(dev, host)` point that maximizes the mean of the line values.

    That LP optimum lies on the lower convex hull at the mean device time, so the fit is the
    hull edge spanning it (exact integer hull, deterministic). Host time minus the line is the
    per-frame latency above the minimum (>= 0).
    """
    if len(dev_ns) != len(host_ns) or not dev_ns:
        raise ClockModelError("envelope fit needs matching, non-empty point lists")
    pts = sorted(zip((int(x) for x in dev_ns), (int(y) for y in host_ns), strict=True))
    x0, y0 = pts[0]
    rel = [(x - x0, y - y0) for x, y in pts]
    hull: list[tuple[int, int]] = []
    for p in rel:
        while len(hull) >= 2:
            (ox, oy), (ax, ay) = hull[-2], hull[-1]
            if (ax - ox) * (p[1] - oy) - (ay - oy) * (p[0] - ox) <= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    xm = sum(x for x, _ in rel) / len(rel)
    rate, offset = 1.0, float(min(y - x for x, y in rel))
    for (x1, y1), (x2, y2) in pairwise(hull):
        if x2 > x1 and x1 <= xm <= x2:
            rate = (y2 - y1) / (x2 - x1)
            offset = y1 - rate * x1
            break
    return ClockFit(x0, y0, float(offset), float(rate), len(rel), len(hull))


# --------------------------------------------------------------------------- per frame


def global_mid_exposure_ns(
    t: RawClock, rt_map: RealtimeMap | None, period_us: int
) -> tuple[int, str] | None:
    """Mid-exposure HOST_MONOTONIC time from RealSense global time, or None if unavailable."""
    if rt_map is None or t.rs_timestamp_domain != GLOBAL_DOMAIN or t.rs_timestamp_ms is None:
        return None
    frame_ns = rt_map.ms_to_monotonic_ns(float(t.rs_timestamp_ms))
    if t.rs_frame_timestamp_us is not None and t.rs_sensor_timestamp_us is not None:
        back_us = counter_delta_us(t.rs_frame_timestamp_us, t.rs_sensor_timestamp_us, period_us)
        return frame_ns - back_us * 1000, METHOD_GLOBAL_SENSOR
    if t.exposure_us is not None:
        return frame_ns - round(float(t.exposure_us) * 500.0), METHOD_GLOBAL_HALF_EXPOSURE
    return frame_ns, METHOD_GLOBAL_FRAME


def frame_timestamp_ns(
    t: RawClock, t_host_arrival_ns: int, rt_map: RealtimeMap | None, cfg: ClockConfig
) -> tuple[int, str]:
    """Online mapping (live source): global time if available, else declared arrival fallback."""
    g = global_mid_exposure_ns(t, rt_map, cfg.device_counter_period_us)
    if g is not None:
        return g
    lat = cfg.fallback_latency_ns
    return int(t_host_arrival_ns) - lat, arrival_method(lat)


def _device_mid_ns(records: Sequence[FrameRecord], period_us: int) -> list[int | None]:
    """Unwrapped device mid-exposure time (ns) per record; None without a device clock."""
    counters: list[tuple[int, int, int]] = []  # (index, counter_us, ns to subtract)
    for i, r in enumerate(records):
        t = r.timing
        if t.rs_sensor_timestamp_us is not None:
            counters.append((i, t.rs_sensor_timestamp_us, 0))
        elif t.rs_frame_timestamp_us is not None and t.exposure_us is not None:
            counters.append((i, t.rs_frame_timestamp_us, round(float(t.exposure_us) * 500.0)))
    out: list[int | None] = [None] * len(records)
    unwrapped = unwrap_counter([c for _, c, _ in counters], period_us)
    for (i, _, sub), u in zip(counters, unwrapped, strict=True):
        out[i] = u * 1000 - sub
    return out


@dataclass(frozen=True)
class StreamClock:
    """Per-frame `(timestamp_ns, method)` of one recorded stream plus its clock report."""

    camera_id: str
    timestamps: Mapping[int, tuple[int, str]]
    envelope: ClockFit | None
    report: Mapping[str, Any]

    def timestamp(self, frame_id: int) -> tuple[int, str]:
        try:
            return self.timestamps[frame_id]
        except KeyError as exc:
            raise ClockModelError(f"{self.camera_id}: no clock for frame_id {frame_id}") from exc


def build_stream_clock(
    camera_id: str,
    records: Sequence[FrameRecord],
    rt_map: RealtimeMap | None,
    cfg: ClockConfig,
) -> StreamClock:
    """Timestamps for every record (written or dropped) of one stream, in frame_id order."""
    recs = sorted(records, key=lambda r: r.frame_id)
    period = cfg.device_counter_period_us
    dev = _device_mid_ns(recs, period)
    pairs = [(d, r.t_host_arrival_ns) for d, r in zip(dev, recs, strict=True) if d is not None]
    envelope: ClockFit | None = None
    if len(pairs) >= cfg.min_envelope_points:
        envelope = fit_lower_envelope([p[0] for p in pairs], [p[1] for p in pairs])
        if abs(envelope.rate_ppm) > cfg.max_rate_error_ppm:
            raise ClockModelError(
                f"{camera_id}: device clock rate {envelope.rate_ppm:+.0f} ppm exceeds "
                f"{cfg.max_rate_error_ppm:g} ppm"
            )
    lat = cfg.fallback_latency_ns
    stamps: dict[int, tuple[int, str]] = {}
    residual_ns: list[int] = []
    for d, r in zip(dev, recs, strict=True):
        g = global_mid_exposure_ns(r.timing, rt_map, period)
        env = envelope.map_ns(d) if envelope is not None and d is not None else None
        if g is not None:
            stamps[r.frame_id] = g
            if env is not None and g[1] in MID_EXPOSURE_METHODS:
                residual_ns.append(env - g[0])
        elif env is not None:
            stamps[r.frame_id] = (env - lat, envelope_method(lat))
        else:
            stamps[r.frame_id] = (r.t_host_arrival_ns - lat, arrival_method(lat))
    ts = np.array([stamps[r.frame_id][0] for r in recs], dtype=np.int64)
    arrival = np.array([r.t_host_arrival_ns for r in recs], dtype=np.int64)
    report: dict[str, Any] = {
        "n_frames": len(recs),
        "methods": dict(sorted(Counter(m for _, m in stamps.values()).items())),
        "non_increasing_timestamps": int((np.diff(ts) <= 0).sum()) if ts.size > 1 else 0,
        "arrival_minus_timestamp_ms": summary(arrival - ts, 1e-6),
        "envelope": None
        if envelope is None
        else {
            "rate_ppm": round(envelope.rate_ppm, 3),
            "n_points": envelope.n_points,
            "hull_points": envelope.hull_points,
        },
        "realtime_map": None
        if rt_map is None
        else {"drift_ns": rt_map.drift_ns, "pair_uncertainty_ns": rt_map.uncertainty_ns},
    }
    if residual_ns:
        r_arr = np.asarray(residual_ns, dtype=np.float64)
        med = float(np.median(r_arr))
        report["envelope_minus_global"] = {
            "offset_ms": round(med / 1e6, 4),
            "abs_residual_ms": summary(np.abs(r_arr - med), 1e-6),
        }
    else:
        report["envelope_minus_global"] = None
    return StreamClock(camera_id, stamps, envelope, report)
