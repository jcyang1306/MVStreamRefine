"""Depth preprocessing inside the object mask (P2 O2).

`geometry_mask` = mask eroded by `depth.mask_erode_px`
  ∩ valid depth within `[min_depth_m, max_depth_m]`
  ∩ not a discontinuity: all 4 neighbours valid and `|z - z_n| <= edge_jump_rel * z`
    (removes hole borders, object/background mixing and flying pixels)
  ∩ not noisy: 3x3 std of valid depth `<= noise_k * sigma(z)`, `sigma(z) = a + b z²`.
Holes are never filled; a pixel without measured depth produces no point.
Points: `p_cam = z * [x, y, 1]` with undistorted rays of the colour intrinsics (depth is
registered to colour). Normals: cross product of central differences on the organized
cloud, oriented toward the camera; defined for every geometry pixel because its 4
neighbours are valid by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from mvsr.calib.intrinsics import ray_grid
from mvsr.core.types import CalibrationBundle
from mvsr.observation.config import DepthConfig

BOOL = npt.NDArray[np.bool_]
F32 = npt.NDArray[np.float32]


def depth_sigma_m(z: npt.ArrayLike, cfg: DepthConfig) -> npt.NDArray[np.float64]:
    zz = np.asarray(z, dtype=np.float64)
    out: npt.NDArray[np.float64] = cfg.sigma_a_m + cfg.sigma_b_per_m * zz * zz
    return out


def erode(mask: BOOL, radius: int) -> BOOL:
    if radius <= 0:
        return np.asarray(mask, dtype=bool)
    k = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    out: BOOL = cv2.erode(mask.astype(np.uint8), k, borderValue=0).astype(bool)
    return out


def discontinuity(z: F32, in_range: BOOL, rel: float) -> BOOL:
    """Pixels with an invalid 4-neighbour or a relative depth jump above `rel`."""
    zp = np.pad(z, 1, constant_values=0.0)
    vp = np.pad(in_range, 1, constant_values=False)
    edge = np.zeros(z.shape, dtype=bool)
    for dz, dv in (
        (zp[:-2, 1:-1], vp[:-2, 1:-1]),
        (zp[2:, 1:-1], vp[2:, 1:-1]),
        (zp[1:-1, :-2], vp[1:-1, :-2]),
        (zp[1:-1, 2:], vp[1:-1, 2:]),
    ):
        edge |= ~dv | (np.abs(z - dz) > rel * z)
    return edge


def local_std(z: F32, valid: BOOL) -> F32:
    """3x3 standard deviation over valid pixels (0 where fewer than 2 are valid)."""
    vf = valid.astype(np.float32)
    zf = np.where(valid, z, 0.0).astype(np.float32)
    n = cv2.boxFilter(vf, -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
    s1 = cv2.boxFilter(zf, -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
    s2 = cv2.boxFilter(zf * zf, -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n
        var = s2 / n - mean * mean
    std = np.sqrt(np.clip(np.nan_to_num(var), 0.0, None)).astype(np.float32)
    std[n < 2] = 0.0
    return std


@dataclass(frozen=True, eq=False)
class DepthGeometry:
    geometry_mask: BOOL
    pixel_uv: npt.NDArray[np.int32]
    points_cam_m: F32
    normals_cam: F32
    stats: dict[str, Any]


def depth_geometry(
    depth_m: F32,
    valid_depth: BOOL,
    mask: BOOL,
    bundle: CalibrationBundle,
    cfg: DepthConfig,
) -> DepthGeometry:
    z = np.asarray(depth_m, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    eroded = erode(mask, cfg.mask_erode_px)
    in_range = valid_depth & (z >= cfg.min_depth_m) & (z <= cfg.max_depth_m)
    edge = discontinuity(z, in_range, cfg.edge_jump_rel)
    noisy = local_std(z, in_range) > cfg.noise_k * depth_sigma_m(z, cfg)
    geometry = eroded & in_range & ~edge & ~noisy
    gx, gy = ray_grid(bundle)
    v, u = np.nonzero(geometry)
    if cfg.point_stride > 1:
        v, u = v[:: cfg.point_stride], u[:: cfg.point_stride]
    zz = z[v, u]
    pts = np.stack([gx[v, u] * zz, gy[v, u] * zz, zz], axis=1).astype(np.float32)

    def cloud(vv: npt.NDArray[np.intp], uu: npt.NDArray[np.intp]) -> npt.NDArray[np.float64]:
        zn = z[vv, uu].astype(np.float64)
        return np.stack([gx[vv, uu] * zn, gy[vv, uu] * zn, zn], axis=1)

    du = cloud(v, u + 1) - cloud(v, u - 1) if v.size else np.zeros((0, 3))
    dv = cloud(v + 1, u) - cloud(v - 1, u) if v.size else np.zeros((0, 3))
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 0)
    flip = np.einsum("ij,ij->i", n, pts.astype(np.float64)) > 0
    n[flip] *= -1.0
    area = int(mask.sum())
    stats = {
        "mask_area_px": area,
        "valid_depth_ratio": round(float((mask & valid_depth).sum()) / area, 6) if area else 0.0,
        "in_range_ratio": round(float((mask & in_range).sum()) / area, 6) if area else 0.0,
        "geometry_px": int(geometry.sum()),
        "geometry_ratio": round(float(geometry.sum()) / area, 6) if area else 0.0,
        "removed_erosion_px": int((mask & ~eroded).sum()),
        "removed_edge_px": int((eroded & in_range & edge).sum()),
        "removed_noise_px": int((eroded & in_range & ~edge & noisy).sum()),
        "n_points": int(pts.shape[0]),
    }
    return DepthGeometry(
        geometry_mask=geometry,
        pixel_uv=np.stack([u, v], axis=1).astype(np.int32),
        points_cam_m=pts,
        normals_cam=n.astype(np.float32),
        stats=stats,
    )


def sample_depth(depth_m: F32, geometry_mask: BOOL, uv: npt.ArrayLike) -> F32:
    """Depth at the nearest pixel of each `(M, 2)` keypoint; NaN outside `geometry_mask`."""
    p = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    h, w = depth_m.shape
    out = np.full(p.shape[0], np.nan, dtype=np.float32)
    if p.size == 0:
        return out
    ui = np.rint(p[:, 0]).astype(int)
    vi = np.rint(p[:, 1]).astype(int)
    inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    ok = np.zeros(p.shape[0], dtype=bool)
    ok[inside] = geometry_mask[vi[inside], ui[inside]]
    out[ok] = depth_m[vi[ok], ui[ok]]
    return out


def silhouette_offsets_px(
    mask: BOOL, depth_m: F32, valid_depth: BOOL, rel: float, *, max_px: int = 8, every: int = 4
) -> npt.NDArray[np.float64]:
    """Signed offset (px) of the depth step from the colour mask boundary (registration check).

    Along the outward normal of every `every`-th point of the mask's outer contour, depth is
    sampled at -max_px..+max_px; the largest valid jump `> rel * z` going farther outward (an
    occluding silhouette, object in front) gives the offset of the step midpoint. Positive =
    the depth step lies outside the colour mask. Points without such a step are skipped.
    """
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.zeros(0)
    c = max(cnts, key=len).reshape(-1, 2).astype(np.float64)
    if c.shape[0] < 16:
        return np.zeros(0)
    tang = np.roll(c, -3, axis=0) - np.roll(c, 3, axis=0)
    nrm = np.stack([tang[:, 1], -tang[:, 0]], axis=1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    h, w = mask.shape
    probe = np.rint(c + 3.0 * nrm).astype(int)
    ok = (probe[:, 0] >= 0) & (probe[:, 0] < w) & (probe[:, 1] >= 0) & (probe[:, 1] < h)
    inward = np.zeros(c.shape[0], dtype=bool)
    inward[ok] = mask[probe[ok, 1], probe[ok, 0]]
    nrm[inward] *= -1.0
    steps = np.arange(-max_px, max_px + 1, dtype=np.float64)
    out: list[float] = []
    for k in range(0, c.shape[0], every):
        pos = np.rint(c[k] + steps[:, None] * nrm[k]).astype(int)
        inside = (pos[:, 0] >= 0) & (pos[:, 0] < w) & (pos[:, 1] >= 0) & (pos[:, 1] < h)
        z = np.full(steps.shape[0], np.nan)
        pv, pu = pos[inside, 1], pos[inside, 0]
        z[inside] = np.where(valid_depth[pv, pu], depth_m[pv, pu], np.nan)
        dz = z[1:] - z[:-1]
        jump = np.isfinite(dz) & (dz > rel * np.fmin(z[1:], z[:-1]))
        if not jump.any():
            continue
        j = int(np.argmax(np.where(jump, dz, -np.inf)))
        out.append(0.5 * (steps[j] + steps[j + 1]))
    return np.asarray(out, dtype=np.float64)
