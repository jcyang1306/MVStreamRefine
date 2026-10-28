"""`mvsr inspect-session`: measured summary of one session (I-P0 performance evidence).

Reports per camera: frames seen/written/dropped by reason, written ratio, host-arrival
interval statistics, device frame-number gaps (frames the SDK never delivered), raw timestamp
field coverage and domains; robot: sample rate, interval jitter, read latency, failures;
events and the recorder's own performance block from the manifest. Reading only metadata,
no images, so it is fast; integrity is `verify-session`'s job.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from mvsr.io.records import FRAME_TIMING_KEYS
from mvsr.io.session_reader import SessionReader


def _stats_ms(values_ns: np.ndarray) -> dict[str, float] | None:
    if values_ns.size == 0:
        return None
    v = values_ns.astype(np.float64) / 1e6
    return {
        "n": int(v.size),
        "mean_ms": round(float(v.mean()), 4),
        "median_ms": round(float(np.median(v)), 4),
        "p95_ms": round(float(np.percentile(v, 95)), 4),
        "max_ms": round(float(v.max()), 4),
        "std_ms": round(float(v.std()), 4),
    }


def _camera_report(reader: SessionReader, cid: str) -> dict[str, Any]:
    cam = reader.manifest["cameras"][cid]
    recs = reader.frame_records(cid)
    by_status = Counter(r.status.value for r in recs)
    written = by_status.get("WRITTEN", 0)
    arrivals = np.array([r.t_host_arrival_ns for r in recs], dtype=np.int64)
    intervals = np.diff(arrivals)
    span_s = float(arrivals[-1] - arrivals[0]) / 1e9 if arrivals.size > 1 else 0.0
    numbers = [r.timing.rs_frame_number for r in recs if r.timing.rs_frame_number is not None]
    gaps = (
        np.diff(np.array(numbers, dtype=np.int64)) if len(numbers) > 1 else np.array([], np.int64)
    )
    coverage = {
        k: round(sum(getattr(r.timing, k) is not None for r in recs) / len(recs), 4)
        if recs
        else None
        for k in FRAME_TIMING_KEYS
    }
    depth_cov = None
    if recs and any(r.depth_timing is not None for r in recs):
        depth_cov = {
            k: round(
                sum(
                    r.depth_timing is not None and getattr(r.depth_timing, k) is not None
                    for r in recs
                )
                / len(recs),
                4,
            )
            for k in FRAME_TIMING_KEYS
        }
    sensor = np.array(
        [
            r.timing.rs_sensor_timestamp_us
            for r in recs
            if r.timing.rs_sensor_timestamp_us is not None
        ],
        dtype=np.int64,
    )
    exposure = [r.timing.exposure_us for r in recs if r.timing.exposure_us is not None]
    return {
        "model": cam.get("model"),
        "serial": cam.get("serial"),
        "nominal_fps": cam.get("fps"),
        "resolution": cam.get("resolution"),
        "depth_units_m": cam.get("depth_units_m"),
        "calibration_id": cam.get("calibration_id"),
        "frames_seen": len(recs),
        "frames_written": written,
        "written_ratio": round(written / len(recs), 6) if recs else None,
        "drops_by_reason": {k: v for k, v in sorted(by_status.items()) if k != "WRITTEN"},
        "arrival_fps": round((arrivals.size - 1) / span_s, 3) if span_s > 0 else None,
        "arrival_interval": _stats_ms(intervals),
        "t_host_arrival_first_ns": int(arrivals[0]) if arrivals.size else None,
        "t_host_arrival_last_ns": int(arrivals[-1]) if arrivals.size else None,
        "capture_span_s": round(span_s, 6) if arrivals.size > 1 else None,
        "device_frame_number_gaps": {
            "frames_missing": int(np.clip(gaps - 1, 0, None).sum()) if gaps.size else 0,
            "max_gap": int(gaps.max()) if gaps.size else None,
            "non_increasing": int((gaps <= 0).sum()) if gaps.size else 0,
        },
        "sensor_timestamp_interval": _stats_ms(np.diff(sensor) * 1000) if sensor.size > 1 else None,
        "timestamp_field_coverage": coverage,
        "depth_timestamp_field_coverage": depth_cov,
        "timestamp_domains": dict(Counter(str(r.timing.rs_timestamp_domain) for r in recs)),
        "exposure_us_median": float(np.median(exposure)) if exposure else None,
        "global_time_enabled": cam.get("global_time_enabled"),
        "png_bytes": (reader.manifest.get("counts") or {})
        .get("cameras", {})
        .get(cid, {})
        .get("png_bytes"),
    }


def _robot_report(reader: SessionReader) -> dict[str, Any]:
    recs = reader.robot_records()
    good = [r for r in recs if r.error is None]
    t = np.array([r.t_host_ns for r in good], dtype=np.int64)
    span_s = float(t[-1] - t[0]) / 1e9 if t.size > 1 else 0.0
    lat = np.array(
        [r.read_latency_ns for r in good if r.read_latency_ns is not None], dtype=np.int64
    )
    errors = Counter(r.error for r in recs if r.error is not None)
    return {
        "source": (reader.manifest.get("robot") or {}).get("state_source"),
        "nominal_rate_hz": (reader.manifest.get("robot") or {}).get("nominal_rate_hz"),
        "samples": len(good),
        "failures": len(recs) - len(good),
        "failure_messages": dict(errors.most_common(5)),
        "rate_hz": round((t.size - 1) / span_s, 3) if span_s > 0 else None,
        "interval": _stats_ms(np.diff(t)),
        "read_latency": _stats_ms(lat),
    }


def _measured_from_cameras(cameras: dict[str, Any]) -> dict[str, Any]:
    """Duration and frame counts from the records, independent of any requested --duration."""
    firsts = [
        c["t_host_arrival_first_ns"]
        for c in cameras.values()
        if c.get("t_host_arrival_first_ns") is not None
    ]
    lasts = [
        c["t_host_arrival_last_ns"]
        for c in cameras.values()
        if c.get("t_host_arrival_last_ns") is not None
    ]
    span = round((max(lasts) - min(firsts)) / 1e9, 6) if firsts and lasts else None
    return {
        "duration_s": span,
        "cameras": {
            cid: {
                "frames_seen": c["frames_seen"],
                "frames_written": c["frames_written"],
                "arrival_fps": c["arrival_fps"],
                "capture_span_s": c["capture_span_s"],
            }
            for cid, c in cameras.items()
        },
    }


def inspect_session(session_dir: str | Path) -> dict[str, Any]:
    reader = SessionReader(session_dir, verify_checksums=False)
    m = reader.manifest
    events = reader.events()
    cameras = {cid: _camera_report(reader, cid) for cid in reader.camera_ids}
    return {
        "session_id": m.get("session_id"),
        "status": m.get("status"),
        "created_utc": m.get("created_utc"),
        "host": m.get("host"),
        "software": m.get("software"),
        "scenario": m.get("scenario"),
        "duration_s": m.get("duration_s"),
        "measured": _measured_from_cameras(cameras),
        "stop_reason": m.get("stop_reason"),
        "errors": m.get("errors"),
        "cameras": cameras,
        "robot": _robot_report(reader),
        "events": {
            "counts": dict(Counter(e.type.value for e in events)),
            "notable": [
                {"t_host_ns": e.t_host_ns, "type": e.type.value, "payload": e.payload}
                for e in events
                if e.type.value in ("WARNING", "ERROR", "ANNOTATION", "PAUSE", "RESUME")
            ][:50],
        },
        "disk": m.get("disk"),
        "performance": m.get("performance"),
        "clock": m.get("clock"),
    }


def format_report(r: dict[str, Any]) -> str:
    sc = r.get("scenario") or {}
    out = [
        f"session {r['session_id']}  status={r['status']}  stop={r.get('stop_reason')}",
        f"  created {r.get('created_utc')} on {r.get('host')}  "
        f"measured {(r.get('measured') or {}).get('duration_s')} s "
        f"(manifest {r.get('duration_s')} s)",
        f"  scenario {sc.get('tags')} object={sc.get('object')!r} gt={sc.get('gt_type')} notes={sc.get('notes')!r}",
        f"  software {r.get('software')}",
    ]
    if r.get("errors"):
        out.append(f"  errors: {r['errors']}")
    for cid, c in r["cameras"].items():
        out += [
            f"  [{cid}] {c['model']} {c['serial']} {c['resolution']} @{c['nominal_fps']} fps  units={c['depth_units_m']}",
            f"    frames seen={c['frames_seen']} written={c['frames_written']} ratio={c['written_ratio']} drops={c['drops_by_reason']}",
            f"    arrival fps={c['arrival_fps']} interval={c['arrival_interval']}",
            f"    device frame-number gaps={c['device_frame_number_gaps']}",
            f"    timestamp coverage={c['timestamp_field_coverage']} domains={c['timestamp_domains']}",
        ]
    rb = r["robot"]
    out += [
        f"  [robot] source={rb['source']} samples={rb['samples']} failures={rb['failures']} rate={rb['rate_hz']} Hz",
        f"    interval={rb['interval']}",
        f"    read latency={rb['read_latency']}",
        f"  events {r['events']['counts']}",
        f"  disk {r.get('disk')}",
        f"  performance {r.get('performance')}",
    ]
    return "\n".join(out)
