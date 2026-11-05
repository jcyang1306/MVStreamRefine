"""Deterministic point-to-plane ICP used by P3 T4.

The model is expressed in the immutable object frame and the target observation in the
current camera frame. The optimized transform is therefore `T_cam_obj`. Correspondences
are nearest neighbours under a bounded distance; no pose is returned when the geometry is
insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from mvsr.core.se3 import PoseSE3, se3_exp, so3_log
from mvsr.tracking.config import IcpConfig

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class IcpResult:
    T_cam_obj: PoseSE3
    correspondences: int
    overlap: float
    rmse_m: float
    hessian_min_eig: float
    correction_m: float
    correction_deg: float
    iterations: int
    covariance: F64


def _sample(points: F64, normals: F64, maximum: int) -> tuple[F64, F64]:
    if points.shape[0] <= maximum:
        return points, normals
    idx = np.linspace(0, points.shape[0] - 1, maximum, dtype=np.intp)
    return points[idx], normals[idx]


def _clean(points: F64, normals: F64) -> tuple[F64, F64]:
    lengths = np.linalg.norm(normals, axis=1)
    good = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1) & (lengths > 1e-9)
    return points[good], normals[good] / lengths[good, None]


def _nearest(source: F64, index: Any) -> tuple[npt.NDArray[np.intp], F64]:
    if source.size == 0:
        return np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.float64)
    idx, squared = index.knnSearch(source.astype(np.float32), 1, params={})
    return idx[:, 0].astype(np.intp), np.sqrt(squared[:, 0].astype(np.float64))


def _delta_metrics(before: PoseSE3, after: PoseSE3) -> tuple[float, float]:
    delta = after.matrix @ np.linalg.inv(before.matrix)
    return float(np.linalg.norm(delta[:3, 3])), float(
        np.degrees(np.linalg.norm(so3_log(delta[:3, :3])))
    )


def point_to_plane_icp(
    model_points_obj: npt.ArrayLike,
    model_normals_obj: npt.ArrayLike,
    target_points_cam: npt.ArrayLike,
    target_normals_cam: npt.ArrayLike,
    initial: PoseSE3,
    cfg: IcpConfig,
) -> IcpResult | None:
    """Refine `initial=T_cam_obj`; return None when no level has enough correspondences."""
    initial.expect_frames("cam", "obj")
    mp = np.asarray(model_points_obj, dtype=np.float64)
    mn = np.asarray(model_normals_obj, dtype=np.float64)
    tp = np.asarray(target_points_cam, dtype=np.float64)
    tn = np.asarray(target_normals_cam, dtype=np.float64)
    if (
        mp.ndim != 2
        or tp.ndim != 2
        or mp.shape[1:] != (3,)
        or tp.shape[1:] != (3,)
        or mn.shape != mp.shape
        or tn.shape != tp.shape
    ):
        return None
    mp, mn = _clean(mp, mn)
    tp, tn = _clean(tp, tn)
    if min(mp.shape[0], tp.shape[0]) < cfg.min_correspondences:
        return None
    mp, mn = _sample(mp, mn, cfg.max_points)
    tp, tn = _sample(tp, tn, cfg.max_points)
    pose = initial
    total_iterations = 0
    last: tuple[int, float, float] | None = None
    last_covariance: F64 | None = None
    last_source_count = mp.shape[0]
    cos_limit = float(np.cos(np.radians(cfg.normal_angle_deg)))
    levels = len(cfg.iterations)
    for level, (n_iter, max_dist) in enumerate(
        zip(cfg.iterations, cfg.max_correspondence_m, strict=True)
    ):
        divisor = 2 ** (levels - level - 1)
        count = max(cfg.min_correspondences, cfg.max_points // divisor)
        level_mp, level_mn = _sample(mp, mn, count)
        level_tp, level_tn = _sample(tp, tn, count)
        index = cv2.flann_Index(  # type: ignore[attr-defined]
            level_tp.astype(np.float32), {"algorithm": 0}
        )
        for _ in range(n_iter):
            q = pose.transform_points(level_mp)
            nq = level_mn @ pose.R.T
            idx, dist = _nearest(q, index)
            if idx.size == 0:
                break
            p, n = level_tp[idx], level_tn[idx]
            good = (dist <= max_dist) & (np.einsum("ij,ij->i", nq, n) >= cos_limit)
            if int(good.sum()) < cfg.min_correspondences:
                break
            qg, pg, ng = q[good], p[good], n[good]
            residual = np.einsum("ij,ij->i", ng, qg - pg)
            absolute = np.abs(residual)
            weights = np.ones_like(residual)
            outside = absolute > cfg.huber_delta_m
            weights[outside] = cfg.huber_delta_m / absolute[outside]
            J = np.concatenate([ng, np.cross(qg, ng)], axis=1)
            sw = np.sqrt(weights)
            A, b = J * sw[:, None], -residual * sw
            H = A.T @ A
            eig = np.linalg.eigvalsh(H / max(1, len(residual)))
            try:
                delta = np.linalg.solve(H + 1e-9 * np.eye(6), A.T @ b)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(delta)):
                break
            pose = PoseSE3(se3_exp(delta) @ pose.matrix, "cam", "obj")
            total_iterations += 1
            rmse = float(np.sqrt(np.mean(residual * residual)))
            last = int(good.sum()), rmse, float(eig[0])
            last_source_count = level_mp.shape[0]
            last_covariance = np.linalg.pinv(H + 1e-9 * np.eye(6)) * rmse * rmse
            last_covariance = 0.5 * (last_covariance + last_covariance.T)
            if float(np.linalg.norm(delta[:3])) < 1e-5 and float(np.linalg.norm(delta[3:])) < 1e-4:
                break
    if last is None or last_covariance is None:
        return None
    n_corr, rmse, min_eig = last
    pose = pose.with_covariance(last_covariance)
    correction_m, correction_deg = _delta_metrics(initial, pose)
    return IcpResult(
        T_cam_obj=pose,
        correspondences=n_corr,
        overlap=n_corr / max(1, last_source_count),
        rmse_m=rmse,
        hessian_min_eig=min_eig,
        correction_m=correction_m,
        correction_deg=correction_deg,
        iterations=total_iterations,
        covariance=last_covariance,
    )
