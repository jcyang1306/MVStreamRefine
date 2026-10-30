"""Observation quality benchmark (P2 O6) and the I1 gate (INTEGRATION_PLAN.md).

The session is replayed in order through `ObservationBuilder` with a fixed mask revision.
Reference geometry (R-STATIC only, object fixed in `base`): for frames i < j with robot
priors, `T_cj_ci = inv(T_base_cam_j) @ T_base_cam_i` maps object points from camera i to
camera j. It carries the hand-eye + robot error measured in P1 (board spread p95 ≈ 4 mm), so
"true match" thresholds are set above it (`benchmark.true_match_m/px`).

Per evaluated pair (gaps `benchmark.frame_gaps`, every `pair_stride`-th source frame):
- mask agreement: sampled geometry points of i projected into j that land inside mask j;
- match precision / recall for `track` and `orb` candidates, judged by the reference.
Per track: life length and reprojection drift of its first depth-valid point by age.
Per frame: mask status/area/score, depth coverage, keypoints and track counts, and every
10th frame a depth/colour edge agreement (registration check).
Pairs are evaluated in a sliding window of `max(frame_gaps)` frames, so memory is bounded.
"""

from __future__ import annotations

import hashlib
import socket
from collections import Counter, deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from mvsr import paths
from mvsr.calib.intrinsics import project_cam_points, rays_for_pixels, undistort_pixels
from mvsr.core.codec import SCHEMA_VERSION
from mvsr.core.se3 import PoseSE3
from mvsr.core.types import ObjectObservation, SensorFrame, make_session_id
from mvsr.io.atomic import atomic_write_json, sha256_file
from mvsr.io.session_writer import MANIFEST_NAME
from mvsr.observation.builder import ObservationBuilder
from mvsr.observation.config import ObservationConfig
from mvsr.observation.correspondence import KeypointView, match_observations
from mvsr.observation.depth import silhouette_offsets_px
from mvsr.observation.masks import MaskCache, MaskStatus
from mvsr.sensors.config import SensorsConfig
from mvsr.sensors.replay import ReplaySource
from mvsr.sensors.stats import summary
from mvsr.software import software_info

AGE_BINS = (1, 5, 15, 30, 100)
SURVIVAL_AT = (5, 15, 30, 100)
STATIC_GT = "constant_T_base_obj"
I1_SCOPE_LIMITS = (
    "scope (owner 2026-10-30): object always in view, slow motion; occlusion, fast motion and "
    "missing-depth scenes are not validated",
    "mask_prob not stored (binary mask + per-frame score only)",
    "reference transforms carry hand-eye/robot error (P1 board spread p95 ~4 mm)",
    "only ORB + KLT; learned extractors/matchers have no adapter yet",
    "frames before the prompt frame are not segmented",
)


def observation_digest(obs: ObjectObservation) -> str:
    h = hashlib.sha256()
    for arr in (
        np.packbits(obs.mask_binary),
        np.packbits(obs.geometry_mask),
        obs.pixel_uv,
        obs.points_cam_m,
        obs.keypoints_uv,
        obs.keypoint_depth_m,
        obs.descriptors,
        obs.track_ids,
    ):
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(obs.mask_revision.encode())
    return h.hexdigest()


def iter_observations(
    session_dir: str | Path,
    revision: str,
    cfg: ObservationConfig,
    sensors_cfg: SensorsConfig | None = None,
    *,
    limit: int | None = None,
) -> Iterator[tuple[SensorFrame, ObjectObservation, ObservationBuilder]]:
    src = ReplaySource(session_dir, "wrist", config=sensors_cfg)
    masks = MaskCache(src.reader.dir, revision, "wrist")
    builder = ObservationBuilder(masks, src.bundle, cfg)
    ids = [f for f in src.frame_ids if f in masks]
    for fid in ids[:limit]:
        frame = src.read(fid)
        yield frame, builder.process(frame), builder


@dataclass
class _Keep:
    view: KeypointView
    prior: PoseSE3 | None
    mask: npt.NDArray[np.bool_]
    samples: npt.NDArray[np.float64]


@dataclass
class _TrackLife:
    first: int
    last: int
    seen: int = 1
    anchor_base: npt.NDArray[np.float64] | None = None
    anchor_frame: int | None = None


@dataclass
class _Acc:
    frames: list[dict[str, Any]] = field(default_factory=list)
    edge: list[float] = field(default_factory=list)
    edge_points: list[float] = field(default_factory=list)
    residual_mm: dict[str, list[float]] = field(default_factory=dict)
    pairs: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    drift: dict[str, list[float]] = field(default_factory=dict)
    lives: dict[int, _TrackLife] = field(default_factory=dict)


def _ref(prior_i: PoseSE3 | None, prior_j: PoseSE3 | None) -> PoseSE3 | None:
    if prior_i is None or prior_j is None:
        return None
    return prior_j.inverse() @ prior_i


def match_truth(
    cs: Any, T: PoseSE3, bundle: Any, cfg: ObservationConfig
) -> npt.NDArray[np.float64]:
    """Per match: 1 true, 0 false, NaN not evaluable (no depth on either side)."""
    b = cfg.benchmark
    k = cs.inlier.shape[0]
    out = np.full(k, np.nan)
    if k == 0:
        return out
    ps = cs.src_point_cam_m.astype(np.float64)
    pt = cs.tgt_point_cam_m.astype(np.float64)
    ds, dt = np.isfinite(ps[:, 0]), np.isfinite(pt[:, 0])
    K = bundle.K
    q = np.full((k, 3), np.nan)
    if ds.any():
        q[ds] = T.transform_points(ps[ds])
    both = ds & dt
    out[both] = (np.linalg.norm(q[both] - pt[both], axis=1) <= b.true_match_m).astype(float)
    one = ds & ~dt & (q[:, 2] > 0)
    if one.any():
        ut = undistort_pixels(cs.tgt_uv[one], K, bundle.distortion_model, bundle.distortion)
        out[one] = (
            np.linalg.norm(project_cam_points(q[one], K) - ut, axis=1) <= b.true_match_px
        ).astype(float)
    other = dt & ~ds
    if other.any():
        back = T.inverse().transform_points(pt[other])
        good = back[:, 2] > 0
        idx = np.nonzero(other)[0][good]
        if idx.size:
            us = undistort_pixels(cs.src_uv[idx], K, bundle.distortion_model, bundle.distortion)
            out[idx] = (
                np.linalg.norm(project_cam_points(back[good], K) - us, axis=1) <= b.true_match_px
            ).astype(float)
    return out


def _mask_agreement(keep_i: _Keep, keep_j: _Keep, T: PoseSE3, bundle: Any) -> float | None:
    if keep_i.samples.shape[0] == 0:
        return None
    q = T.transform_points(keep_i.samples)
    q = q[q[:, 2] > 0]
    if q.shape[0] == 0:
        return None
    uv = project_cam_points(q, bundle.K)
    h, w = keep_j.mask.shape
    ui, vi = np.rint(uv[:, 0]).astype(int), np.rint(uv[:, 1]).astype(int)
    inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    if not inside.any():
        return None
    return float(keep_j.mask[vi[inside], ui[inside]].mean())


def _add(acc: _Acc, key: str, name: str, value: float | None) -> None:
    if value is None or not np.isfinite(value):
        return
    acc.pairs.setdefault(key, {}).setdefault(name, []).append(float(value))


def run_benchmark(
    session_dir: str | Path,
    revision: str,
    cfg: ObservationConfig,
    sensors_cfg: SensorsConfig | None = None,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    src = ReplaySource(session_dir, "wrist", config=sensors_cfg)
    bundle = src.bundle
    masks = MaskCache(src.reader.dir, revision, "wrist")
    scenario = src.reader.manifest.get("scenario") or {}
    static = scenario.get("gt_type") == STATIC_GT
    gaps = sorted(set(cfg.benchmark.frame_gaps))
    window: deque[_Keep] = deque(maxlen=max(gaps) + 1)
    rng = np.random.default_rng(0)
    acc = _Acc()
    digests: list[str] = []
    n_repro = cfg.benchmark.reproducibility_frames
    for n, (frame, obs, builder) in enumerate(
        iter_observations(session_dir, revision, cfg, sensors_cfg, limit=limit)
    ):
        fid = frame.header.frame_id
        q = obs.quality
        acc.frames.append(
            {
                "frame_id": fid,
                "mask_status": q["mask_status"],
                "mask_area_px": q["mask_area_px"],
                "mask_score": q["mask_score"],
                "valid_depth_ratio": q["valid_depth_ratio"],
                "geometry_ratio": q["geometry_ratio"],
                "n_keypoints": q["n_keypoints"],
                "keypoints_with_depth": q["keypoints_with_depth"],
                "tracks_new": q["tracks_new"],
                "tracks_tracked": q["tracks_tracked"],
                "tracks_occluded": q["tracks_occluded"],
            }
        )
        if n < n_repro:
            digests.append(observation_digest(obs))
        if n % 10 == 0 and q["mask_status"] != MaskStatus.EMPTY.value:
            off = silhouette_offsets_px(
                obs.mask_binary, frame.depth_m_f32, frame.valid_depth_mask, cfg.depth.edge_jump_rel
            )
            if off.size:
                acc.edge.append(float(np.median(off)))
                acc.edge_points.extend(off.tolist())
        prior = frame.T_base_cam_prior
        pts = obs.points_cam_m.astype(np.float64)
        if pts.shape[0] > cfg.benchmark.mask_sample_points:
            pts = pts[
                np.sort(rng.choice(pts.shape[0], cfg.benchmark.mask_sample_points, replace=False))
            ]
        keep = _Keep(KeypointView.of(obs), prior, np.asarray(obs.mask_binary), pts)
        tr = builder.last_tracks
        if tr is not None:
            for tid, uv, z in zip(
                tr.track_ids.tolist(), obs.keypoints_uv, obs.keypoint_depth_m, strict=True
            ):
                life = acc.lives.get(tid)
                if life is None:
                    life = acc.lives[tid] = _TrackLife(fid, fid)
                else:
                    life.last, life.seen = fid, life.seen + 1
                if not static or prior is None:
                    continue
                if life.anchor_base is None and np.isfinite(z):
                    pc = rays_for_pixels(bundle, uv[None, :]) * float(z)
                    life.anchor_base, life.anchor_frame = prior.transform_points(pc)[0], fid
                elif life.anchor_base is not None and life.anchor_frame is not None:
                    pc = prior.inverse().transform_points(life.anchor_base[None, :])
                    if pc[0, 2] > 0:
                        pred = project_cam_points(pc, bundle.K)[0]
                        obs_uv = undistort_pixels(
                            uv[None, :], bundle.K, bundle.distortion_model, bundle.distortion
                        )[0]
                        age = fid - life.anchor_frame
                        b = next((f"<={a}" for a in AGE_BINS if age <= a), f">{AGE_BINS[-1]}")
                        acc.drift.setdefault(b, []).append(float(np.linalg.norm(pred - obs_uv)))
        if static:
            for prev in window:
                gap = fid - prev.view.frame_id
                if gap not in gaps or prev.view.frame_id % cfg.benchmark.pair_stride:
                    continue
                T = _ref(prev.prior, prior)
                if T is None:
                    continue
                key = f"gap_{gap}"
                _add(acc, key, "mask_agreement", _mask_agreement(prev, keep, T, bundle))
                for method in ("track", "orb"):
                    cs, _ = match_observations(
                        prev.view, keep.view, bundle, cfg.correspondence, cfg.depth, method
                    )
                    truth = match_truth(cs, T, bundle, cfg)
                    ev = np.isfinite(truth)
                    inl = cs.inlier & ev
                    tp = float(((truth == 1) & cs.inlier).sum())
                    _add(acc, key, f"{method}_candidates", float(cs.inlier.shape[0]))
                    _add(acc, key, f"{method}_inliers", float(cs.inlier.sum()))
                    if inl.any():
                        _add(acc, key, f"{method}_precision", tp / float(inl.sum()))
                    true_n = float(((truth == 1) & ev).sum())
                    if true_n > 0:
                        _add(acc, key, f"{method}_recall", tp / true_n)
                    both = (
                        cs.inlier
                        & np.isfinite(cs.src_point_cam_m[:, 0])
                        & np.isfinite(cs.tgt_point_cam_m[:, 0])
                    )
                    if both.any():
                        moved = T.transform_points(cs.src_point_cam_m[both].astype(np.float64))
                        res = np.linalg.norm(moved - cs.tgt_point_cam_m[both], axis=1) * 1e3
                        acc.residual_mm.setdefault(f"{key}/{method}", []).extend(res.tolist())
        window.append(keep)
    repro_ok = digests == [
        observation_digest(o)
        for _, o, _ in iter_observations(
            session_dir, revision, cfg, sensors_cfg, limit=len(digests)
        )
    ]
    return _report(src, masks, acc, digests, repro_ok, static, scenario, cfg)


def _report(
    src: ReplaySource,
    masks: MaskCache,
    acc: _Acc,
    digests: list[str],
    repro_ok: bool,
    static: bool,
    scenario: dict[str, Any],
    cfg: ObservationConfig,
) -> dict[str, Any]:
    f = acc.frames
    emitted = list(src.frame_ids)
    prompt = (masks.revision_meta.get("params") or {}).get("prompt") or {}
    start = prompt.get("frame_id", emitted[0] if emitted else 0)
    expected = [i for i in emitted if i >= start]
    status = Counter(x["mask_status"] for x in f)
    non_empty = [x for x in f if x["mask_status"] != MaskStatus.EMPTY.value]
    lives = list(acc.lives.values())
    life_len = [lv.last - lv.first + 1 for lv in lives]
    pairs = {
        key: {name: summary(vals) for name, vals in sorted(metrics.items())}
        for key, metrics in sorted(acc.pairs.items(), key=lambda kv: int(kv[0].split("_")[1]))
    }
    return {
        "session_id": src.session_id,
        "host": src.reader.manifest.get("host"),
        "manifest_sha256": sha256_file(Path(src.reader.dir) / MANIFEST_NAME),
        "scenario": scenario,
        "mask_revision": masks.revision,
        "mask_revision_meta": {
            k: masks.revision_meta.get(k)
            for k in ("params", "by_status", "frames", "fps", "runtime")
        },
        "reference_available": static,
        "frames": {
            "emitted": len(emitted),
            "expected_from_prompt": len(expected),
            "observed": len(f),
            "mask_coverage": round(len(f) / len(expected), 6) if expected else None,
            "mask_status": dict(sorted(status.items())),
        },
        "mask": {
            "area_px": summary([x["mask_area_px"] for x in non_empty]),
            "score": summary([x["mask_score"] for x in non_empty if x["mask_score"] is not None]),
        },
        "depth": {
            "valid_depth_ratio": summary([x["valid_depth_ratio"] for x in non_empty]),
            "geometry_ratio": summary([x["geometry_ratio"] for x in non_empty]),
            "silhouette_offset_px_per_frame": summary(acc.edge),
            "silhouette_offset_px_points": summary(acc.edge_points),
        },
        "keypoints": {
            "per_frame": summary([x["n_keypoints"] for x in non_empty]),
            "with_depth_per_frame": summary([x["keypoints_with_depth"] for x in non_empty]),
            "tracked_per_frame": summary([x["tracks_tracked"] for x in non_empty]),
            "new_per_frame": summary([x["tracks_new"] for x in non_empty]),
        },
        "tracks": {
            "count": len(lives),
            "life_frames": summary(life_len),
            "survival": {
                f">={k}": round(sum(1 for v in life_len if v >= k) / len(life_len), 6)
                if life_len
                else None
                for k in SURVIVAL_AT
            },
            "reprojection_drift_px": {
                k: summary(v) for k, v in sorted(acc.drift.items(), key=lambda kv: _age_key(kv[0]))
            },
        },
        "pairs": pairs,
        "inlier_residual_mm": {k: summary(v) for k, v in sorted(acc.residual_mm.items())},
        "reproducibility": {
            "frames": len(digests),
            "identical": repro_ok,
            "first_digest": digests[0] if digests else None,
        },
        "config_sha256": cfg.source_sha256,
    }


def _age_key(label: str) -> float:
    return float(label.lstrip("<=>")) + (0.5 if label.startswith(">") else 0.0)


# --------------------------------------------------------------------------- gate


def _check(name: str, ok: bool, value: Any, target: str, hard: bool) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "value": value, "target": target, "hard": hard}


def i1_decision(reports: list[dict[str, Any]], cfg: ObservationConfig) -> dict[str, Any]:
    """R-HARD is out of scope (owner decision 2026-10-30): object always visible, slow motion."""
    g = cfg.gate
    checks: list[dict[str, Any]] = []
    checks.append(_check("sessions", bool(reports), len(reports), ">= 1 session used_by I1", True))
    for r in reports:
        sid = r["session_id"]
        cov = r["frames"]["mask_coverage"]
        checks.append(
            _check(
                f"{sid}/mask_coverage",
                cov is not None and cov >= g.min_mask_coverage,
                cov,
                f">= {g.min_mask_coverage}",
                True,
            )
        )
        rep = r["reproducibility"]["identical"]
        checks.append(
            _check(
                f"{sid}/reproducible",
                rep,
                r["reproducibility"]["frames"],
                "identical observation digests",
                True,
            )
        )
        vd = (r["depth"]["valid_depth_ratio"] or {}).get("median")
        checks.append(
            _check(
                f"{sid}/valid_depth_ratio_median",
                vd is not None and vd >= g.min_valid_depth_ratio,
                vd,
                f">= {g.min_valid_depth_ratio}",
                False,
            )
        )
        kp = (r["keypoints"]["tracked_per_frame"] or {}).get("median")
        checks.append(
            _check(
                f"{sid}/tracked_keypoints_median",
                kp is not None and kp >= g.min_tracked_keypoints,
                kp,
                f">= {g.min_tracked_keypoints}",
                False,
            )
        )
        if r["reference_available"]:
            g5 = r["pairs"].get("gap_5") or {}
            ma = (g5.get("mask_agreement") or {}).get("median")
            checks.append(
                _check(
                    f"{sid}/mask_agreement_gap5",
                    ma is not None and ma >= g.min_mask_agreement,
                    ma,
                    f">= {g.min_mask_agreement}",
                    False,
                )
            )
            pr = (g5.get("track_precision") or {}).get("median")
            checks.append(
                _check(
                    f"{sid}/track_precision_gap5",
                    pr is not None and pr >= g.min_match_precision,
                    pr,
                    f">= {g.min_match_precision}",
                    False,
                )
            )
    hard = [c["name"] for c in checks if c["hard"] and not c["ok"]]
    soft = [c["name"] for c in checks if not c["hard"] and not c["ok"]]
    return {
        "gate": "I1",
        "decision": "BLOCKED" if hard else ("PASS-WITH-LIMITS" if soft else "PASS"),
        "checks": checks,
        "failed_hard": hard,
        "missed_targets": soft,
        "scope_limits": list(I1_SCOPE_LIMITS),
        "targets_are_placeholders": True,
    }


def write_run(
    reports: list[dict[str, Any]],
    decision: dict[str, Any] | None,
    cfg: ObservationConfig,
    root: Path | None = None,
) -> Path:
    hosts = sorted({str(r["host"]) for r in reports if r.get("host")})
    run_id = make_session_id(hosts[0] if len(hosts) == 1 else socket.gethostname())
    d = (root or paths.data_root()) / "runs" / run_id
    d.mkdir(parents=True)
    body = {
        "schema_version": SCHEMA_VERSION,
        "type": "ObservationReport",
        "run_id": run_id,
        "sessions": reports,
        "i1": decision,
    }
    files = {"observation_report.json": atomic_write_json(d / "observation_report.json", body)}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "type": "P2Run",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "processing_host": socket.gethostname(),
        "software": software_info(),
        "inputs": [
            {
                "session_id": r["session_id"],
                "manifest_sha256": r["manifest_sha256"],
                "mask_revision": r["mask_revision"],
            }
            for r in reports
        ],
        "config": {"path": cfg.source_path, "sha256": cfg.source_sha256, "values": cfg.snapshot()},
        "files": files,
    }
    atomic_write_json(d / MANIFEST_NAME, manifest)
    return d
