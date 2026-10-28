"""Sensors report (P1 S4/S6) and the I0 gate decision (INTEGRATION_PLAN.md).

`session_report` replays one session's clocks and robot pairing (no pixels needed) and, for
R-CALIB sessions, runs the board check (`calib_check`). `i0_decision` applies
`configs/sensors.yaml: gate`. `write_run` stores everything under
`$MVSR_DATA_ROOT/runs/<run_id>/` (manifest with sha256 of every file); raw sessions are
never modified (SESSION_FORMAT.md).
"""

from __future__ import annotations

import socket
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mvsr import paths
from mvsr.core import codec
from mvsr.core.codec import SCHEMA_VERSION
from mvsr.core.types import CalibrationBundle, FrameStatus, SyncStatus, make_session_id
from mvsr.io.atomic import atomic_write_json, sha256_file
from mvsr.io.session_writer import MANIFEST_NAME
from mvsr.sensors.calib_check import check_board_session
from mvsr.sensors.config import SensorsConfig
from mvsr.sensors.replay import ReplaySource
from mvsr.sensors.rgbd import RgbdPair
from mvsr.sensors.stats import histogram, summary
from mvsr.software import software_info

ROBOT_INTERVAL_EDGES_MS = (0, 5, 10, 15, 20, 25, 30, 40, 50)
DECISIONS = ("PASS", "PASS-WITH-LIMITS", "BLOCKED")
#: Scope limits that hold for every I0 run on this rig (owner decisions, ADR-0004 §7).
SCOPE_LIMITS = (
    "dt_robot_cam_ns = 0: camera-robot time offset not estimated (limited viewpoints/speeds)",
    "board spread and hand-eye residual hold only inside the recorded motion envelope",
    "robot poll stamp = request midpoint; controller-internal latency is part of the untested offset",
    "head D435 colour has no exposure metadata; its sensor timestamp is used as mid-exposure",
    "frames whose colour and depth come from different exposures are not emitted (counted)",
)


def stream_report(src: ReplaySource) -> dict[str, Any]:
    """Frame accounting, clock report and robot sync over every WRITTEN frame of one stream."""
    by_status = Counter(r.status.value for r in src.records)
    ids = [r.frame_id for r in src.records]
    missing = sorted(set(range(max(ids, default=-1) + 1)) - set(ids))
    sync: Counter[str] = Counter()
    gaps_ms: list[float] = []
    for fid in src.frame_ids:
        ts, _ = src.clock.timestamp(fid)
        _, info = src.robot.pose_at(
            ts, src.config.sync.max_interp_gap_ns, src.bundle.dt_robot_cam_ns
        )
        sync[info.status.value] += 1
        if info.status is SyncStatus.INTERPOLATED and info.gap_before_ns is not None:
            gaps_ms.append(((info.gap_before_ns or 0) + (info.gap_after_ns or 0)) / 1e6)
    n = len(src.frame_ids)
    written = [r.frame_id for r in src.records if r.status is FrameStatus.WRITTEN]
    pair_written = Counter(src.rgbd[f].status.value for f in written)
    ok = pair_written.get(RgbdPair.OK.value, 0)
    return {
        "calibration_id": src.bundle.calibration_id,
        "dt_robot_cam_ns": src.bundle.dt_robot_cam_ns,
        "frames": {
            "records": len(src.records),
            "written": len(written),
            "emitted": n,
            "by_status": dict(sorted(by_status.items())),
            "unlogged_frame_ids": len(missing),
        },
        "rgbd": {
            **src.rgbd_report,
            "written_by_status": dict(sorted(pair_written.items())),
            "valid_ratio": round(ok / len(written), 6) if written else None,
        },
        "clock": dict(src.clock.report),
        "sync": {
            "max_interp_gap_ms": src.config.sync.max_interp_gap_ns / 1e6,
            "by_status": dict(sorted(sync.items())),
            "interpolated_ratio": round(sync[SyncStatus.INTERPOLATED.value] / n, 6) if n else None,
            "bracket_ms": summary(gaps_ms),
        },
    }


def robot_report(src: ReplaySource) -> dict[str, Any]:
    recs = src.reader.robot_records()
    good = [r for r in recs if r.error is None]
    t = np.array(sorted(r.t_host_ns for r in good), dtype=np.int64)
    iv = np.diff(t) / 1e6 if t.size > 1 else np.array([])
    lat = [r.read_latency_ns for r in good if r.read_latency_ns is not None]
    return {
        "source": (src.reader.manifest.get("robot") or {}).get("state_source"),
        "samples": len(good),
        "failures": len(recs) - len(good),
        "failure_messages": dict(Counter(r.error for r in recs if r.error).most_common(5)),
        "rate_hz": round((t.size - 1) / ((t[-1] - t[0]) / 1e9), 3) if t.size > 1 else None,
        "interval_ms": summary(iv),
        "interval_histogram_ms": histogram(iv, ROBOT_INTERVAL_EDGES_MS),
        "read_latency_ms": summary(lat, 1e-6),
    }


def session_report(
    session_dir: str | Path,
    cfg: SensorsConfig,
    *,
    square_mm: float | None = None,
    board: bool | None = None,
    handeye: Any = None,
) -> tuple[dict[str, Any], dict[str, CalibrationBundle]]:
    """Report of one session; `board=None` runs the board check only for R-CALIB sessions.

    `handeye` replaces the session's extrinsic for this report only.
    """
    wrist = ReplaySource(session_dir, "wrist", config=cfg, handeye=handeye)
    m = wrist.reader.manifest
    scenario = m.get("scenario") or {}
    cams: dict[str, Any] = {"wrist": stream_report(wrist)}
    for cid in wrist.reader.camera_ids:
        if cid != "wrist":
            cams[cid] = stream_report(ReplaySource(session_dir, cid, config=cfg, handeye=handeye))
    rep: dict[str, Any] = {
        "session_id": wrist.session_id,
        "host": m.get("host"),
        "manifest_sha256": sha256_file(Path(wrist.reader.dir) / MANIFEST_NAME),
        "scenario": scenario,
        "cameras": cams,
        "robot": robot_report(wrist),
        "board_check": None,
    }
    bundles: dict[str, CalibrationBundle] = {}
    do_board = board if board is not None else "R-CALIB" in (scenario.get("tags") or [])
    if do_board:
        rep["board_check"], bundles = check_board_session(wrist.reader.dir, square_mm, cfg, handeye)
    return rep, bundles


# --------------------------------------------------------------------------- gate


def _check(name: str, ok: bool, value: Any, target: str, hard: bool) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "value": value, "target": target, "hard": hard}


def i0_decision(reports: list[dict[str, Any]], cfg: SensorsConfig) -> dict[str, Any]:
    """PASS / PASS-WITH-LIMITS (a soft target missed) / BLOCKED (a hard check failed)."""
    g = cfg.gate
    checks: list[dict[str, Any]] = []
    calib = [r for r in reports if r.get("board_check")]
    checks.append(_check("r_calib_session", bool(calib), len(calib), ">= 1 R-CALIB session", True))
    for r in reports:
        sid = r["session_id"]
        for cid, c in r["cameras"].items():
            unlogged = c["frames"]["unlogged_frame_ids"]
            checks.append(
                _check(f"{sid}/{cid}/unlogged_frames", unlogged == 0, unlogged, "== 0", True)
            )
            res = (c["clock"].get("envelope_minus_global") or {}).get("abs_residual_ms")
            p95 = None if res is None else res["p95"]
            checks.append(
                _check(
                    f"{sid}/{cid}/clock_residual_p95_ms",
                    p95 is not None and p95 <= g.max_clock_residual_p95_ms,
                    p95,
                    f"<= {g.max_clock_residual_p95_ms}",
                    False,
                )
            )
        ratio = r["cameras"]["wrist"]["sync"]["interpolated_ratio"]
        checks.append(
            _check(
                f"{sid}/wrist/interpolated_ratio",
                ratio is not None and ratio >= g.min_interpolated_ratio,
                ratio,
                f">= {g.min_interpolated_ratio}",
                True,
            )
        )
        valid = r["cameras"]["wrist"]["rgbd"]["valid_ratio"]
        checks.append(
            _check(
                f"{sid}/wrist/rgbd_valid_ratio",
                valid is not None and valid >= g.min_rgbd_valid_ratio,
                valid,
                f">= {g.min_rgbd_valid_ratio}",
                False,
            )
        )
    for r in calib:
        sid = r["session_id"]
        w = r["board_check"]["cameras"]["wrist"]
        n = w.get("frames_with_prior", 0)
        checks.append(
            _check(
                f"{sid}/board_frames", n >= g.min_board_frames, n, f">= {g.min_board_frames}", True
            )
        )
        sp = w.get("interpolated") or {}
        t95 = (sp.get("translation_mm") or {}).get("p95")
        r95 = (sp.get("rotation_deg") or {}).get("p95")
        h95 = (w.get("handeye_reproj_rms_px") or {}).get("p95")
        checks.append(
            _check(
                f"{sid}/board_spread_p95_mm",
                t95 is not None and t95 <= g.max_board_spread_p95_mm,
                t95,
                f"<= {g.max_board_spread_p95_mm}",
                False,
            )
        )
        checks.append(
            _check(
                f"{sid}/board_spread_p95_deg",
                r95 is not None and r95 <= g.max_board_spread_p95_deg,
                r95,
                f"<= {g.max_board_spread_p95_deg}",
                False,
            )
        )
        checks.append(
            _check(
                f"{sid}/handeye_reproj_p95_px",
                h95 is not None and h95 <= g.max_handeye_reproj_p95_px,
                h95,
                f"<= {g.max_handeye_reproj_p95_px}",
                False,
            )
        )
    hard_fail = [c for c in checks if c["hard"] and not c["ok"]]
    soft_fail = [c for c in checks if not c["hard"] and not c["ok"]]
    decision = "BLOCKED" if hard_fail else ("PASS-WITH-LIMITS" if soft_fail else "PASS")
    return {
        "gate": "I0",
        "decision": decision,
        "checks": checks,
        "failed_hard": [c["name"] for c in hard_fail],
        "missed_targets": [c["name"] for c in soft_fail],
        "scope_limits": list(SCOPE_LIMITS),
        "targets_are_placeholders": True,
    }


# --------------------------------------------------------------------------- run directory


def write_run(
    reports: list[dict[str, Any]],
    bundles: dict[str, dict[str, CalibrationBundle]],
    decision: dict[str, Any] | None,
    cfg: SensorsConfig,
    root: Path | None = None,
) -> Path:
    hosts = sorted({str(r["host"]) for r in reports if r.get("host")})
    run_id = make_session_id(hosts[0] if len(hosts) == 1 else socket.gethostname())
    d = (root or paths.data_root()) / "runs" / run_id
    d.mkdir(parents=True)
    body = {
        "schema_version": SCHEMA_VERSION,
        "type": "SensorsReport",
        "run_id": run_id,
        "sessions": reports,
        "i0": decision,
    }
    files = {"sensors_report.json": atomic_write_json(d / "sensors_report.json", body)}
    for sid, per_cam in sorted(bundles.items()):
        (d / "calibration" / sid).mkdir(parents=True, exist_ok=True)
        for cid, b in sorted(per_cam.items()):
            rel = f"calibration/{sid}/{cid}.json"
            files[rel] = atomic_write_json(d / rel, codec.to_json_dict(b))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "type": "P1Run",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "processing_host": socket.gethostname(),
        "software": software_info(),
        "inputs": [
            {"session_id": r["session_id"], "manifest_sha256": r["manifest_sha256"]}
            for r in reports
        ],
        "config": {"path": cfg.source_path, "sha256": cfg.source_sha256, "values": cfg.snapshot()},
        "files": files,
    }
    atomic_write_json(d / MANIFEST_NAME, manifest)
    return d


# --------------------------------------------------------------------------- text


def _s(stats: dict[str, Any] | None, *keys: str) -> str:
    if not stats:
        return "n/a"
    return " ".join(f"{k}={stats[k]:g}" for k in keys)


def format_report(
    reports: list[dict[str, Any]], decision: dict[str, Any] | None, run_dir: Path | None
) -> str:
    out: list[str] = []
    for r in reports:
        sc = r.get("scenario") or {}
        out.append(
            f"session {r['session_id']}  scenario={sc.get('tags')} object={sc.get('object')!r}"
        )
        for cid, c in r["cameras"].items():
            ck, sy, fr, rg = c["clock"], c["sync"], c["frames"], c["rgbd"]
            emg = ck.get("envelope_minus_global") or {}
            env = ck.get("envelope") or {}
            out += [
                f"  [{cid}] frames emitted={fr['emitted']} written={fr['written']}/{fr['records']} unlogged={fr['unlogged_frame_ids']} methods={ck['methods']}",
                f"    rgbd pairing (written): {rg['written_by_status']} valid={rg['valid_ratio']} in-sync offset={rg['expected_color_depth_dt_ms']} ms",
                f"    clock: drift={env.get('rate_ppm')} ppm, transfer latency={emg.get('offset_ms')} ms, "
                f"residual {_s(emg.get('abs_residual_ms'), 'median', 'p95', 'max')} ms",
                f"    sync: {sy['by_status']} interpolated={sy['interpolated_ratio']} bracket {_s(sy['bracket_ms'], 'median', 'p95', 'max')} ms",
            ]
        rb = r["robot"]
        out.append(
            f"  [robot] samples={rb['samples']} failures={rb['failures']} rate={rb['rate_hz']} Hz interval {_s(rb['interval_ms'], 'median', 'p95', 'max')} ms"
        )
        bc = r.get("board_check")
        if bc:
            w = bc["cameras"]["wrist"]
            sp = w.get("interpolated") or {}
            nr = w.get("nearest") or {}
            d = w.get("depth") or {}
            out += [
                f"  [board wrist] accepted={w['accepted_frames']}/{w['frames_checked']} with_prior={w['frames_with_prior']} rejected={w['rejected']} pnp {_s(w['pnp_reproj_rms_px'], 'median', 'p95')} px",
                f"    T_base_marker spread interp: {_s(sp.get('translation_mm'), 'median', 'p95', 'max')} mm, {_s(sp.get('rotation_deg'), 'median', 'p95', 'max')} deg",
                f"    T_base_marker spread nearest: {_s(nr.get('translation_mm'), 'median', 'p95', 'max')} mm",
                f"    hand-eye reproj {_s(w.get('handeye_reproj_rms_px'), 'median', 'p95', 'max')} px",
                f"    depth at {_s(d.get('distance_m'), 'min', 'max')} m: bias {_s(d.get('bias_mm'), 'median', 'p95')} mm, "
                f"|plane dist| median {_s(d.get('plane_abs_median_mm'), 'median')} / p90 {_s(d.get('plane_abs_p90_mm'), 'median')} mm "
                f"(rms {_s(d.get('plane_rms_mm'), 'median')}), pitch scale {d.get('pitch_scale_vs_nominal')}",
            ]
            h = bc["cameras"].get("head")
            if h:
                vs = h.get("vs_wrist_ref") or {}
                hs = (h.get("spread") or {}).get("translation_mm")
                out.append(
                    f"  [board head] accepted={h['accepted_frames']}/{h['frames_checked']} spread {_s(hs, 'median', 'p95')} mm; vs wrist ref {vs.get('translation_mm')} mm / {vs.get('rotation_deg')} deg"
                )
    if decision:
        out.append(f"I0 decision: {decision['decision']}")
        for c in decision["checks"]:
            flag = "ok " if c["ok"] else ("FAIL" if c["hard"] else "miss")
            out.append(
                f"  {flag} {c['name']} = {c['value']} (target {c['target']}{', hard' if c['hard'] else ''})"
            )
        out.append("  scope limits: " + "; ".join(decision["scope_limits"]))
    if run_dir is not None:
        out.append(f"run written to {run_dir}")
    return "\n".join(out)
