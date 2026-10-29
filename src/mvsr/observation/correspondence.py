"""2D-2D / 3D-3D correspondences between two observations (P2 O5).

Candidates: shared track ids (`method="track"`) or ORB descriptors (`"orb"`: Hamming,
Lowe ratio, mutual best, `max_hamming`). Screening (no pose is output; P2 never estimates
the object pose, the model exists only to reject outliers):
- with >= `min_model_matches` pairs that have depth on both sides, a rigid model
  `p_tgt = R p_src + t` is fitted by deterministic RANSAC (seeded) + Kabsch refit;
  both-depth pairs pass with a 3D residual `<= ransac_rigid_m`, one-sided pairs with a
  reprojection error `<= ransac_epipolar_px`, no-depth pairs with a Sampson distance
  `<= ransac_epipolar_px` under `E = [t]x R`;
- otherwise the essential matrix (OpenCV RANSAC, normalized coordinates) screens all pairs;
- fewer than 8 pairs and no rigid model: every pair is rejected `NO_MODEL`.
Rejected matches stay in the set with a reason (`RIGID_RESIDUAL`, `REPROJECTION`,
`EPIPOLAR`, `NO_MODEL`); points without depth are NaN rows; `sigma_m` is NaN unless both
sides have depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np
import numpy.typing as npt

from mvsr.calib.intrinsics import project_cam_points, rays_for_pixels, undistort_normalized
from mvsr.core.types import CalibrationBundle, CorrespondenceSet, ObjectObservation
from mvsr.observation.config import CorrespondenceConfig, DepthConfig
from mvsr.observation.depth import depth_sigma_m

F64 = npt.NDArray[np.float64]
IDX = npt.NDArray[np.intp]


class KeypointSource(Protocol):
    """What matching needs from an observation (`ObjectObservation` or `KeypointView`)."""

    @property
    def frame_id(self) -> int: ...
    @property
    def keypoints_uv(self) -> npt.NDArray[np.float32]: ...
    @property
    def keypoint_depth_m(self) -> npt.NDArray[np.float32]: ...
    @property
    def track_ids(self) -> npt.NDArray[np.int64]: ...
    @property
    def descriptors(self) -> np.ndarray: ...
    @property
    def descriptor_type(self) -> str: ...


@dataclass(frozen=True, eq=False)
class KeypointView:
    """Keypoint part of an `ObjectObservation` (kept by the benchmark instead of full frames)."""

    frame_id: int
    keypoints_uv: npt.NDArray[np.float32]
    keypoint_depth_m: npt.NDArray[np.float32]
    track_ids: npt.NDArray[np.int64]
    descriptors: np.ndarray
    descriptor_type: str

    @classmethod
    def of(cls, obs: ObjectObservation) -> KeypointView:
        return cls(
            obs.header.frame_id,
            obs.keypoints_uv,
            obs.keypoint_depth_m,
            obs.track_ids,
            obs.descriptors,
            obs.descriptor_type,
        )


def kabsch(P: F64, Q: F64) -> tuple[F64, F64]:
    """R, t minimizing sum |R p + t - q|^2 (proper rotation)."""
    cp, cq = P.mean(axis=0), Q.mean(axis=0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T))) or 1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cq - R @ cp


def candidates_by_track(src: KeypointSource, tgt: KeypointSource) -> tuple[IDX, IDX]:
    pos = {int(t): k for k, t in enumerate(tgt.track_ids) if t >= 0}
    i = [k for k, t in enumerate(src.track_ids) if int(t) in pos]
    j = [pos[int(src.track_ids[k])] for k in i]
    return np.asarray(i, dtype=np.intp), np.asarray(j, dtype=np.intp)


def candidates_by_descriptor(
    src: KeypointSource, tgt: KeypointSource, cfg: CorrespondenceConfig
) -> tuple[IDX, IDX]:
    if src.descriptor_type != tgt.descriptor_type:
        raise ValueError(f"descriptor types differ: {src.descriptor_type} vs {tgt.descriptor_type}")
    if len(src.descriptors) < 2 or len(tgt.descriptors) < 2:
        return np.zeros(0, np.intp), np.zeros(0, np.intp)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    a, b = np.asarray(src.descriptors), np.asarray(tgt.descriptors)
    fwd = bf.knnMatch(a, b, k=2)
    back = {m[0].queryIdx: m[0].trainIdx for m in bf.knnMatch(b, a, k=1) if m}
    ii, jj = [], []
    for m in fwd:
        if len(m) < 2:
            continue
        best, second = m
        if best.distance > cfg.max_hamming or best.distance > cfg.ratio * second.distance:
            continue
        if back.get(best.trainIdx) != best.queryIdx:
            continue
        ii.append(best.queryIdx)
        jj.append(best.trainIdx)
    order = np.argsort(np.asarray(ii, dtype=np.intp), kind="stable")
    return np.asarray(ii, dtype=np.intp)[order], np.asarray(jj, dtype=np.intp)[order]


def _points(obs: KeypointSource, idx: IDX, bundle: CalibrationBundle) -> F64:
    z = obs.keypoint_depth_m[idx].astype(np.float64)
    if idx.size == 0:
        return np.zeros((0, 3))
    return rays_for_pixels(bundle, obs.keypoints_uv[idx]) * z[:, None]


def ransac_rigid(
    P: F64, Q: F64, cfg: CorrespondenceConfig
) -> tuple[F64, F64, npt.NDArray[np.bool_]] | None:
    n = P.shape[0]
    if n < max(3, cfg.min_model_matches):
        return None
    rng = np.random.default_rng(cfg.seed)
    best: npt.NDArray[np.bool_] | None = None
    for _ in range(cfg.ransac_iterations):
        s = rng.choice(n, 3, replace=False)
        if np.linalg.matrix_rank(P[s] - P[s].mean(axis=0), tol=1e-6) < 2:
            continue
        R, t = kabsch(P[s], Q[s])
        inl = np.linalg.norm(P @ R.T + t - Q, axis=1) <= cfg.ransac_rigid_m
        if best is None or int(inl.sum()) > int(best.sum()):
            best = inl
    if best is None or int(best.sum()) < 3:
        return None
    R, t = kabsch(P[best], Q[best])
    inl = np.linalg.norm(P @ R.T + t - Q, axis=1) <= cfg.ransac_rigid_m
    if int(inl.sum()) < 3:
        return None
    R, t = kabsch(P[inl], Q[inl])
    return R, t, np.linalg.norm(P @ R.T + t - Q, axis=1) <= cfg.ransac_rigid_m


def _hat(v: F64) -> F64:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)


def sampson_px(E: F64, xs: F64, xt: F64, focal: float) -> F64:
    """Sampson distance (pixels) of normalized correspondences under `xt^T E xs = 0`."""
    a = np.concatenate([xs, np.ones((xs.shape[0], 1))], axis=1)
    b = np.concatenate([xt, np.ones((xt.shape[0], 1))], axis=1)
    Ea, Etb = a @ E.T, b @ E
    num = np.einsum("ij,ij->i", b, Ea) ** 2
    den = Ea[:, 0] ** 2 + Ea[:, 1] ** 2 + Etb[:, 0] ** 2 + Etb[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-18)) * focal


@dataclass(frozen=True)
class MatchDiagnostics:
    candidates: int
    model: str  # rigid | essential | none
    rigid_R: F64 | None
    rigid_t: F64 | None
    inliers: int

    def to_dict(self) -> dict[str, Any]:
        return {"candidates": self.candidates, "model": self.model, "inliers": self.inliers}


def match_observations(
    src: KeypointSource,
    tgt: KeypointSource,
    bundle: CalibrationBundle,
    cfg: CorrespondenceConfig,
    depth_cfg: DepthConfig,
    method: str = "track",
) -> tuple[CorrespondenceSet, MatchDiagnostics]:
    if method == "track":
        i, j = candidates_by_track(src, tgt)
    elif method == "orb":
        i, j = candidates_by_descriptor(src, tgt, cfg)
    else:
        raise ValueError(f"unknown correspondence method {method!r}")
    k = i.size
    zs = src.keypoint_depth_m[i].astype(np.float64)
    zt = tgt.keypoint_depth_m[j].astype(np.float64)
    ds, dt = np.isfinite(zs), np.isfinite(zt)
    Ps = np.full((k, 3), np.nan)
    Pt = np.full((k, 3), np.nan)
    Ps[ds] = _points(src, i[ds], bundle)
    Pt[dt] = _points(tgt, j[dt], bundle)
    uv_s = src.keypoints_uv[i].astype(np.float64)
    uv_t = tgt.keypoints_uv[j].astype(np.float64)
    K = bundle.K
    focal = float(0.5 * (K[0, 0] + K[1, 1]))
    xs = (
        undistort_normalized(uv_s, K, bundle.distortion_model, bundle.distortion)
        if k
        else np.zeros((0, 2))
    )
    xt = (
        undistort_normalized(uv_t, K, bundle.distortion_model, bundle.distortion)
        if k
        else np.zeros((0, 2))
    )
    inlier = np.zeros(k, dtype=bool)
    reason: list[str | None] = ["NO_MODEL"] * k
    both = ds & dt
    fit = ransac_rigid(Ps[both], Pt[both], cfg) if k else None
    model, R_out, t_out = "none", None, None
    if fit is not None:
        R, t, _ = fit
        model, R_out, t_out = "rigid", R, t
        res3 = np.linalg.norm(Ps @ R.T + t - Pt, axis=1)
        for idx in np.nonzero(both)[0]:
            ok = res3[idx] <= cfg.ransac_rigid_m
            inlier[idx], reason[idx] = ok, None if ok else "RIGID_RESIDUAL"
        only_s = ds & ~dt
        if only_s.any():
            q = Ps[only_s] @ R.T + t
            good = q[:, 2] > 0
            err = np.full(q.shape[0], np.inf)
            if good.any():
                pred = project_cam_points(q[good], K)
                ut = np.stack(
                    [
                        K[0, 0] * xt[only_s][good, 0] + K[0, 2],
                        K[1, 1] * xt[only_s][good, 1] + K[1, 2],
                    ],
                    axis=1,
                )
                err[good] = np.linalg.norm(pred - ut, axis=1)
            for idx, e in zip(np.nonzero(only_s)[0], err, strict=True):
                ok = e <= cfg.ransac_epipolar_px
                inlier[idx], reason[idx] = ok, None if ok else "REPROJECTION"
        only_t = dt & ~ds
        if only_t.any():
            q = (Pt[only_t] - t) @ R
            good = q[:, 2] > 0
            err = np.full(q.shape[0], np.inf)
            if good.any():
                pred = project_cam_points(q[good], K)
                us = np.stack(
                    [
                        K[0, 0] * xs[only_t][good, 0] + K[0, 2],
                        K[1, 1] * xs[only_t][good, 1] + K[1, 2],
                    ],
                    axis=1,
                )
                err[good] = np.linalg.norm(pred - us, axis=1)
            for idx, e in zip(np.nonzero(only_t)[0], err, strict=True):
                ok = e <= cfg.ransac_epipolar_px
                inlier[idx], reason[idx] = ok, None if ok else "REPROJECTION"
        none = ~ds & ~dt
        if none.any() and np.linalg.norm(t) > 1e-9:
            d = sampson_px(_hat(t) @ R, xs[none], xt[none], focal)
            for idx, e in zip(np.nonzero(none)[0], d, strict=True):
                ok = e <= cfg.ransac_epipolar_px
                inlier[idx], reason[idx] = ok, None if ok else "EPIPOLAR"
    elif k >= 8:
        E, m = cv2.findEssentialMat(
            xs,
            xt,
            focal=1.0,
            pp=(0.0, 0.0),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=cfg.ransac_epipolar_px / focal,
        )
        if E is not None and E.shape == (3, 3) and m is not None:
            model = "essential"
            ok_all = m.reshape(-1).astype(bool)
            for idx in range(k):
                inlier[idx], reason[idx] = bool(ok_all[idx]), None if ok_all[idx] else "EPIPOLAR"
    sigma_m = np.full(k, np.nan, dtype=np.float32)
    sigma_m[both] = depth_sigma_m(np.maximum(zs[both], zt[both]), depth_cfg).astype(np.float32)
    tracks = (
        src.track_ids[i]
        if method == "track"
        else np.where(src.track_ids[i] == tgt.track_ids[j], src.track_ids[i], -1)
    )
    cs = CorrespondenceSet(
        src_frame_id=src.frame_id,
        tgt_frame_id=tgt.frame_id,
        src_uv=uv_s.astype(np.float32),
        tgt_uv=uv_t.astype(np.float32),
        src_point_cam_m=Ps.astype(np.float32),
        tgt_point_cam_m=Pt.astype(np.float32),
        sigma_px=np.full(k, cfg.sigma_px, dtype=np.float32),
        sigma_m=sigma_m,
        track_ids=np.asarray(tracks, dtype=np.int64),
        method=f"{method}+{model}",
        inlier=inlier,
        rejection_reason=reason,
    )
    return cs, MatchDiagnostics(k, model, R_out, t_out, int(inlier.sum()))
