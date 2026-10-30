"""Data-derived provisional parameters (owner decision 2026-10-30).

Every function states its selection rule; `mvsr derive-params` runs them on registered
sessions and writes the evidence to `data/runs/`. Values are provisional: the basis sessions
and the recompute condition are kept in `configs/provisional_params.yaml` (`tuning.registry`).

Sensors (P1):
- `depth_sigma`: per R-CALIB board frame, robust plane noise sigma_f = median|d| / 0.6745
  (d = distance of measured board points to their best-fit plane) at the median board
  distance z_f; fit sigma(z) = b z^2 (stereo model, a = 0) by least squares. Valid only in
  the measured distance range.
- `max_interp_gap`: ceil of the 99.9th percentile of successful robot sample intervals
  (failed reads still widen brackets and give GAP_TOO_LARGE).
- `rgbd_limits`: per camera, deviation of depth-minus-colour sensor time from the stream
  median; frames below half a frame interval are in sync; limit = min(interval / 4,
  max(1 ms, 10 x p99.9 of the in-sync deviation)).
Observation (P2, R-STATIC take with a mask revision):
- `mask_erode_px` = max(1, ceil(-p5 of the silhouette offset)): covers depth steps that lie
  inside the colour mask; positive offsets (depth fattening) are outside the mask anyway.
- `edge_jump_rel` = p99.5 of relative 4-neighbour depth differences on the mask interior
  (smooth object surface), i.e. at most 0.5 % of smooth pixels are taken for edges.
- correspondence thresholds from reference-true matches (robot reference, static object)
  under the per-pair rigid model: `ransac_rigid_m` = p95 3D residual, `ransac_reproj_px` =
  p95 reprojection error of one-sided matches, `sigma_px` = 1.4826 x median per-axis error,
  `ransac_epipolar_px` = 1.96 x `sigma_px` (Sampson distance of a true match ~ N(0, sigma_px)).
  The truth labels use `benchmark.true_match_m/px`, so a p95 close to those caps is bounded
  by them; recheck with a finer reference (R-MARKER).
- sweeps (`fb_max_px`, `safe_erode_px`, `fast_threshold`, `max_descriptor_drift_bits`):
  one parameter at a time; keep values with median tracked keypoints >= 2 x the I1 minimum,
  pick the lowest median reprojection drift at age <= 30 frames (ties: current value).
  `safe_erode_px` is then raised to `mask_erode_px`: keypoint depth exists only inside the
  eroded geometry mask.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from mvsr.calib.intrinsics import project_cam_points, undistort_pixels
from mvsr.io.session_reader import SessionReader
from mvsr.observation.benchmark import STATIC_GT, iter_observations, match_truth, run_benchmark
from mvsr.observation.config import ObservationConfig
from mvsr.observation.correspondence import KeypointView, match_observations
from mvsr.observation.depth import erode, silhouette_offsets_px
from mvsr.sensors.board import BoardSpec
from mvsr.sensors.calib_check import observe_stream, square_mm_from_notes
from mvsr.sensors.config import SensorsConfig
from mvsr.sensors.replay import ReplaySource
from mvsr.sensors.rgbd import color_depth_dt_us
from mvsr.sensors.stats import summary

MAD_TO_SIGMA = 1.4826
HALF_NORMAL_MEDIAN = 0.6745


def _q(v: Sequence[float] | np.ndarray, p: float) -> float:
    return float(np.percentile(np.asarray(v, dtype=np.float64), p))


# --------------------------------------------------------------------------- sensors


def fit_quadratic_sigma(z: np.ndarray, sigma: np.ndarray) -> dict[str, Any]:
    """Least squares of sigma = b z^2 (a = 0) and, for comparison, sigma = a + b z^2."""
    z2 = np.asarray(z, dtype=np.float64) ** 2
    s = np.asarray(sigma, dtype=np.float64)
    b = float(np.sum(s * z2) / np.sum(z2 * z2))
    A = np.stack([np.ones_like(z2), z2], axis=1)
    (a2, b2), *_ = np.linalg.lstsq(A, s, rcond=None)
    resid = s - b * z2
    return {
        "b_per_m": b,
        "rms_residual_m": float(np.sqrt(np.mean(resid * resid))),
        "free_fit": {"a_m": float(a2), "b_per_m": float(b2)},
        "z_range_m": [float(np.min(z)), float(np.max(z))],
        "n_frames": int(s.size),
    }


def depth_sigma(calib_session: str | Path, cfg: SensorsConfig) -> dict[str, Any]:
    src = ReplaySource(calib_session, "wrist", config=cfg)
    notes = (src.reader.manifest.get("scenario") or {}).get("notes")
    sq = cfg.board.square_mm or square_mm_from_notes(notes)
    if sq is None:
        raise ValueError("board square size unknown (notes square_mm=)")
    spec = BoardSpec(cfg.board.inner_cols, cfg.board.inner_rows, sq / 1e3)
    obs, _, _ = observe_stream(src, spec, cfg)
    used = [o.depth for o in obs if o.depth.get("used")]
    z = np.array([d["distance_m"] for d in used])
    sig = np.array([d["plane_abs_median_mm"] for d in used]) / 1e3 / HALF_NORMAL_MEDIAN
    fit = fit_quadratic_sigma(z, sig)
    zm = float(np.median(z))
    fit.update(
        session_id=src.session_id,
        sigma_at_median_z_mm=round(fit["b_per_m"] * zm * zm * 1e3, 3),
        median_z_m=round(zm, 4),
        frame_sigma_mm=summary(sig, 1e3),
    )
    return fit


def robot_intervals_ms(session_dirs: Sequence[str | Path]) -> np.ndarray:
    out: list[np.ndarray] = []
    for d in session_dirs:
        t = np.array(
            sorted(
                r.t_host_ns
                for r in SessionReader(d, verify_checksums=False).robot_records()
                if r.error is None
            )
        )
        out.append(np.diff(t) / 1e6)
    return np.concatenate(out) if out else np.zeros(0)


def max_interp_gap(session_dirs: Sequence[str | Path]) -> dict[str, Any]:
    iv = robot_intervals_ms(session_dirs)
    p999 = _q(iv, 99.9)
    value_ms = max(math.ceil(p999), math.ceil(2 * float(np.median(iv))))
    return {
        "value_ns": int(value_ms * 1_000_000),
        "intervals_ms": {
            "p50": round(float(np.median(iv)), 3),
            "p99": round(_q(iv, 99), 3),
            "p99.9": round(p999, 3),
            "max": round(float(iv.max()), 3),
            "n": int(iv.size),
        },
        "beyond_value": int((iv > value_ms).sum()),
    }


def rgbd_limits(session_dirs: Sequence[str | Path], cfg: SensorsConfig) -> dict[str, Any]:
    per: dict[str, dict[str, list[float]]] = {}
    for d in session_dirs:
        r = SessionReader(d, verify_checksums=False)
        for cid in r.camera_ids:
            recs = r.frame_records(cid)
            dts = [
                x
                for x in (
                    color_depth_dt_us(x.timing, x.depth_timing, cfg.clock.device_counter_period_us)
                    for x in recs
                )
                if x is not None
            ]
            if not dts:
                continue
            dev = np.abs(np.asarray(dts, dtype=np.float64) - float(np.median(dts))) / 1e3
            fps = float((r.manifest.get("cameras") or {}).get(cid, {}).get("fps") or 30)
            acc = per.setdefault(cid, {"dev": [], "interval": []})
            acc["dev"].extend(dev.tolist())
            acc["interval"].append(1000.0 / fps)
    out: dict[str, Any] = {}
    for cid, acc in per.items():
        dev = np.asarray(acc["dev"])
        interval = float(np.median(acc["interval"]))
        sync = dev[dev < interval / 2]
        p999 = _q(sync, 99.9) if sync.size else 0.0
        lim = min(interval / 4, max(1.0, 10 * p999))
        out[cid] = {
            "value_ms": round(lim, 2),
            "frame_interval_ms": round(interval, 2),
            "in_sync_dev_ms": {
                "p50": round(float(np.median(sync)), 4),
                "p99.9": round(p999, 4),
                "max": round(float(sync.max()), 4),
            }
            if sync.size
            else None,
            "mismatched": int((dev >= interval / 2).sum()),
            "ambiguous": int(((dev > lim) & (dev < interval / 2)).sum()),
            "n": int(dev.size),
        }
    return out


# --------------------------------------------------------------------------- observation


def observation_statistics(
    session_dir: str | Path,
    revision: str,
    cfg: ObservationConfig,
    sensors_cfg: SensorsConfig | None = None,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    src = ReplaySource(session_dir, "wrist", config=sensors_cfg)
    bundle = src.bundle
    static = (src.reader.manifest.get("scenario") or {}).get("gt_type") == STATIC_GT
    offsets: list[float] = []
    jumps: list[float] = []
    res3: list[float] = []
    reproj: list[float] = []
    axis: list[float] = []
    window: list[tuple[KeypointView, Any]] = []
    rng = np.random.default_rng(0)
    gap, stride = 5, cfg.benchmark.pair_stride
    for n, (frame, obs, _) in enumerate(
        iter_observations(session_dir, revision, cfg, sensors_cfg, limit=limit)
    ):
        if n % 10 == 0 and obs.mask_binary.any():
            offsets.extend(
                silhouette_offsets_px(
                    obs.mask_binary,
                    frame.depth_m_f32,
                    frame.valid_depth_mask,
                    cfg.depth.edge_jump_rel,
                ).tolist()
            )
            inner = erode(obs.mask_binary, 8) & frame.valid_depth_mask
            z = frame.depth_m_f32
            for dz, ok in (
                (
                    np.abs(z[:, 1:] - z[:, :-1]) / np.maximum(z[:, :-1], 1e-6),
                    inner[:, 1:] & inner[:, :-1],
                ),
                (
                    np.abs(z[1:, :] - z[:-1, :]) / np.maximum(z[:-1, :], 1e-6),
                    inner[1:, :] & inner[:-1, :],
                ),
            ):
                v = dz[ok]
                if v.size > 5000:
                    v = v[rng.choice(v.size, 5000, replace=False)]
                jumps.extend(v.tolist())
        view = KeypointView.of(obs)
        prior = frame.T_base_cam_prior
        if static:
            for pv, pp in window:
                if (
                    view.frame_id - pv.frame_id != gap
                    or pv.frame_id % stride
                    or pp is None
                    or prior is None
                ):
                    continue
                T = prior.inverse() @ pp
                for method in ("track", "orb"):
                    cs, diag = match_observations(
                        pv, view, bundle, cfg.correspondence, cfg.depth, method
                    )
                    if diag.rigid_R is None:
                        continue
                    R, t = diag.rigid_R, diag.rigid_t
                    truth = match_truth(cs, T, bundle, cfg)
                    ps, pt = (
                        cs.src_point_cam_m.astype(np.float64),
                        cs.tgt_point_cam_m.astype(np.float64),
                    )
                    ds, dt = np.isfinite(ps[:, 0]), np.isfinite(pt[:, 0])
                    both = (truth == 1) & ds & dt
                    if both.any():
                        q = ps[both] @ R.T + t
                        res3.extend(np.linalg.norm(q - pt[both], axis=1).tolist())
                        ut = undistort_pixels(
                            cs.tgt_uv[both], bundle.K, bundle.distortion_model, bundle.distortion
                        )
                        axis.extend(np.abs(project_cam_points(q, bundle.K) - ut).ravel().tolist())
                    one = (truth == 1) & ds & ~dt
                    if one.any():
                        q = ps[one] @ R.T + t
                        good = q[:, 2] > 0
                        ut = undistort_pixels(
                            cs.tgt_uv[one][good],
                            bundle.K,
                            bundle.distortion_model,
                            bundle.distortion,
                        )
                        reproj.extend(
                            np.linalg.norm(
                                project_cam_points(q[good], bundle.K) - ut, axis=1
                            ).tolist()
                        )
        window.append((view, prior))
        window = window[-(gap + 1) :]
    off = np.asarray(offsets)
    sigma_px = (
        round(max(0.05, round(MAD_TO_SIGMA * float(np.median(axis)) * 20) / 20), 2)
        if axis
        else None
    )
    out: dict[str, Any] = {
        "silhouette_offset_px": {
            "p5": round(_q(off, 5), 3),
            "p50": round(_q(off, 50), 3),
            "p95": round(_q(off, 95), 3),
            "n": int(off.size),
        }
        if off.size
        else None,
        "mask_erode_px": max(1, math.ceil(-_q(off, 5))) if off.size else None,
        "interior_rel_jump": {
            "p99": round(_q(jumps, 99), 5),
            "p99.5": round(_q(jumps, 99.5), 5),
            "n": len(jumps),
        }
        if jumps
        else None,
        "edge_jump_rel": round(_q(jumps, 99.5), 3) if jumps else None,
        "true_match_residual_mm": summary(res3, 1e3),
        "ransac_rigid_m": round(math.ceil(_q(res3, 95) * 2000) / 2000, 4) if res3 else None,
        "true_one_sided_reproj_px": summary(reproj),
        "ransac_reproj_px": round(math.ceil(_q(reproj, 95) * 10) / 10, 1) if reproj else None,
        "sigma_px": sigma_px,
        "ransac_epipolar_px": None
        if sigma_px is None
        else round(math.ceil(1.96 * sigma_px * 10) / 10, 1),
    }
    return out


SWEEPS: dict[str, tuple[str, str, list[Any]]] = {
    "tracks.fb_max_px": ("tracks", "fb_max_px", [0.5, 1.0, 2.0]),
    "features.safe_erode_px": ("features", "safe_erode_px", [4, 6, 10, 14]),
    "features.fast_threshold": ("features", "fast_threshold", [5, 7, 10, 14]),
    "tracks.max_descriptor_drift_bits": ("tracks", "max_descriptor_drift_bits", [48, 64, 80, None]),
}


def sweep(
    session_dir: str | Path,
    revision: str,
    cfg: ObservationConfig,
    key: str,
    *,
    limit: int,
    sensors_cfg: SensorsConfig | None = None,
) -> dict[str, Any]:
    section, field_name, values = SWEEPS[key]
    base = replace(
        cfg, benchmark=replace(cfg.benchmark, frame_gaps=[5, 30], reproducibility_frames=2)
    )
    current = getattr(getattr(cfg, section), field_name)
    rows = []
    for v in values:
        c = replace(base, **{section: replace(getattr(base, section), **{field_name: v})})
        rep = run_benchmark(session_dir, revision, c, sensors_cfg, limit=limit)
        drift = (rep["tracks"]["reprojection_drift_px"].get("<=30") or {}).get("median")
        tracked = (rep["keypoints"]["tracked_per_frame"] or {}).get("median")
        resid = (rep["inlier_residual_mm"].get("gap_5/track") or {}).get("median")
        rows.append(
            {
                "value": v,
                "drift_age30_median_px": drift,
                "tracked_median": tracked,
                "gap5_track_residual_median_mm": resid,
            }
        )
    need = 2 * cfg.gate.min_tracked_keypoints
    ok = [
        r
        for r in rows
        if r["tracked_median"] is not None
        and r["tracked_median"] >= need
        and r["drift_age30_median_px"] is not None
    ]
    best = (
        min(ok, key=lambda r: (round(r["drift_age30_median_px"], 1), r["value"] != current))
        if ok
        else None
    )
    return {
        "key": key,
        "current": current,
        "min_tracked": need,
        "rows": rows,
        "value": None if best is None else best["value"],
    }
