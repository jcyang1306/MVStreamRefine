"""Loaders for the plain-text 3x3 K matrices in data/intrinsic/."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.models import CameraIntrinsics


def load_intrinsics_matrix(path: str | Path) -> np.ndarray:
    K = np.loadtxt(path, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"{path}: expected 3x3 K matrix, got {K.shape}")
    if not np.allclose(K[2], [0.0, 0.0, 1.0]):
        raise ValueError(f"{path}: bottom row of K must be [0, 0, 1], got {K[2]}")
    return K


def intrinsics_from_matrix(K: np.ndarray, width: int, height: int) -> CameraIntrinsics:
    intr = CameraIntrinsics(
        width=width,
        height=height,
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
    )
    intr.validate()
    return intr
