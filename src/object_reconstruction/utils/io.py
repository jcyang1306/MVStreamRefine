"""Minimal geometry file output (ASCII PLY), no Open3D dependency."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_ply_points(
    path: str | Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
) -> None:
    """Write an (N, 3) float point cloud, optionally with (N, 3) uint8 colors."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    has_color = colors is not None
    if has_color:
        colors = np.asarray(colors)
        if colors.shape != points.shape:
            raise ValueError("colors must match points shape")
        colors = colors.astype(np.uint8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if has_color:
            for (x, y, z), (r, g, b) in zip(points, colors):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
        else:
            for x, y, z in points:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
