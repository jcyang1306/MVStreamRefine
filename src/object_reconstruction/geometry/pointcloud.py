"""Depth back-projection and point-cloud transforms (PLAN Phase 1)."""

from __future__ import annotations

import numpy as np

from ..data.models import CameraIntrinsics


def depth_to_pointcloud(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_scale: float,
    mask: np.ndarray | None = None,
    depth_min_m: float | None = None,
    depth_max_m: float | None = None,
    stride: int = 1,
    rgb: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Back-project a depth image into an (N, 3) camera-frame point cloud in meters.

    depth_scale converts raw depth units to meters (1000.0 for uint16 mm).
    Returns (points, colors); colors is None unless rgb is given.
    """
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError(f"depth shape {depth.shape} does not match intrinsics")
    depth_m = depth.astype(np.float64) / float(depth_scale)

    valid = depth_m > 0
    if depth_min_m is not None:
        valid &= depth_m >= depth_min_m
    if depth_max_m is not None:
        valid &= depth_m <= depth_max_m
    if mask is not None:
        valid &= mask.astype(bool)
    if stride > 1:
        keep = np.zeros_like(valid)
        keep[::stride, ::stride] = True
        valid &= keep

    v, u = np.nonzero(valid)
    z = depth_m[v, u]
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    points = np.column_stack([x, y, z])

    colors = rgb[v, u] if rgb is not None else None
    return points, colors


def transform_points(points: np.ndarray, T_dst_src: np.ndarray) -> np.ndarray:
    """Apply p_dst = T_dst_src @ p_src to an (N, 3) point array."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    return points @ T_dst_src[:3, :3].T + T_dst_src[:3, 3]
