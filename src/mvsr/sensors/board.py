"""Checkerboard detection and pose for R-CALIB ground truth (P1 S2).

Board frame `marker` (DATA_CONTRACTS.md frame names): origin at the first inner corner,
x along a row of `inner_cols` corners, y down the rows, z = x × y; corner
`k = r * inner_cols + c` sits at `(c * s, r * s, 0)`.

Corner order is ambiguous in OpenCV (on R-CALIB the classic and SB detectors return opposite
orders on every frame), so it is canonicalized from the image alone:
1. handedness: image(x) × image(y) > 0, i.e. the board is seen from the front and z points
   away from the camera;
2. colour: the square spanned by corners (0,0)-(1,1) is darker than the square at the
   opposite end. Both squares have opposite colours because `inner_cols + inner_rows` is odd.

Intrinsics follow librealsense: `inverse_brown_conrady` coefficients map distorted normalized
coordinates straight to undistorted ones (`rs2_deproject_pixel_to_point`, tangential terms as
OpenCV); `brown_conrady` is OpenCV's forward model and is inverted iteratively. PnP and all
reprojection residuals are computed on undistorted pixels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from mvsr.calib.intrinsics import undistort_normalized, undistort_pixels
from mvsr.core.errors import ContractValidationError
from mvsr.core.se3 import PoseSE3

__all__ = [
    "BoardSpec",
    "detect_corners",
    "solve_board_pose",
    "undistort_normalized",
    "undistort_pixels",
]

F64 = npt.NDArray[np.float64]
MARKER_FRAME = "marker"
#: `cv2.findChessboardCornersSB` flags (subpixel "accuracy" mode, exhaustive search).
SB_FLAGS = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY


@dataclass(frozen=True)
class BoardSpec:
    inner_cols: int
    inner_rows: int
    square_m: float

    def __post_init__(self) -> None:
        if min(self.inner_cols, self.inner_rows) < 2:
            raise ContractValidationError("board needs >= 2 inner corners per side")
        if (self.inner_cols + self.inner_rows) % 2 == 0:
            raise ContractValidationError("inner_cols + inner_rows must be odd (180 deg ambiguity)")
        if not np.isfinite(self.square_m) or self.square_m <= 0:
            raise ContractValidationError(f"square_m must be > 0, got {self.square_m}")

    @property
    def n_corners(self) -> int:
        return self.inner_cols * self.inner_rows

    def object_points(self) -> F64:
        r, c = np.mgrid[0 : self.inner_rows, 0 : self.inner_cols]
        pts = np.stack([c.ravel(), r.ravel(), np.zeros(r.size)], axis=1).astype(np.float64)
        out: F64 = pts * self.square_m
        return out


def project_points(T_cam_marker: PoseSE3, points_marker: F64, K: F64) -> F64:
    """Undistorted pixel projection of marker-frame points (no distortion model applied)."""
    p = T_cam_marker.transform_points(points_marker)
    if np.any(p[:, 2] <= 0):
        raise ContractValidationError("points behind the camera")
    out: F64 = np.stack(
        [K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2], K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]], axis=1
    )
    return out


# --------------------------------------------------------------------------- detection


def _cell_intensity(gray: npt.NDArray[np.uint8], quad: F64) -> float:
    center = quad.mean(axis=0)
    side = float(np.linalg.norm(quad[1] - quad[0]))
    w = max(3, int(side * 0.4) | 1)
    patch = cv2.getRectSubPix(gray, (w, w), (float(center[0]), float(center[1])))
    return float(patch.mean())


def canonical_order(grid: F64, gray: npt.NDArray[np.uint8]) -> F64:
    """Reorder a `(rows, cols, 2)` corner grid to the module's canonical orientation."""
    g = np.asarray(grid, dtype=np.float64)
    xd = (g[:, -1] - g[:, 0]).mean(axis=0)
    yd = (g[-1, :] - g[0, :]).mean(axis=0)
    if xd[0] * yd[1] - xd[1] * yd[0] < 0:
        g = g[::-1, :]
    first = _cell_intensity(gray, np.array([g[0, 0], g[0, 1], g[1, 1], g[1, 0]]))
    last = _cell_intensity(gray, np.array([g[-1, -1], g[-1, -2], g[-2, -2], g[-2, -1]]))
    if first > last:
        g = g[::-1, ::-1]
    return np.ascontiguousarray(g)


def detect_corners(gray: npt.NDArray[np.uint8], spec: BoardSpec) -> F64 | None:
    """Canonically ordered inner corners `(rows * cols, 2)` in distorted pixels, or None."""
    found, corners = cv2.findChessboardCornersSB(
        gray, (spec.inner_cols, spec.inner_rows), flags=SB_FLAGS
    )
    if not found or corners is None or len(corners) != spec.n_corners:
        return None
    grid = corners.reshape(spec.inner_rows, spec.inner_cols, 2).astype(np.float64)
    out: F64 = canonical_order(grid, gray).reshape(-1, 2)
    return out


@dataclass(frozen=True, eq=False)
class BoardPose:
    T_cam_marker: PoseSE3
    corners_px: F64
    corners_undist_px: F64
    reproj_rms_px: float


def solve_board_pose(
    corners_px: F64,
    spec: BoardSpec,
    K: F64,
    model: str,
    coeffs: npt.ArrayLike,
    camera_frame: str,
) -> BoardPose | None:
    """IPPE + LM refinement on undistorted corners; None when the solution is not in front."""
    obj = spec.object_points()
    und = undistort_pixels(corners_px, K, model, coeffs)
    ok, rvec, tvec = cv2.solvePnP(obj, und.reshape(-1, 1, 2), K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, und.reshape(-1, 1, 2), K, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    if t[2] <= 0 or float(R[:, 2] @ t) <= 0:
        return None
    pose = PoseSE3.from_rt(R, t, camera_frame, MARKER_FRAME)
    proj = project_points(pose, obj, K)
    rms = float(np.sqrt(np.mean(np.sum((proj - und) ** 2, axis=1))))
    return BoardPose(pose, np.asarray(corners_px, dtype=np.float64), und, rms)


# --------------------------------------------------------------------------- poses


def mean_pose(poses: Sequence[PoseSE3]) -> PoseSE3:
    """Component-wise median translation and chordal L2 mean rotation (deterministic)."""
    if not poses:
        raise ContractValidationError("mean_pose needs at least one pose")
    dst, src = poses[0].dst, poses[0].src
    if any(p.dst != dst or p.src != src for p in poses):
        raise ContractValidationError("mean_pose needs poses with identical frames")
    t = np.median(np.stack([p.t for p in poses]), axis=0)
    M = np.sum(np.stack([p.R for p in poses]), axis=0)
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    return PoseSE3.from_rt(U @ D @ Vt, t, dst, src)


def rotation_angle_deg(Ra: F64, Rb: F64) -> float:
    c = (float(np.trace(Ra.T @ Rb)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(min(1.0, max(-1.0, c)))))
