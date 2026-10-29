"""Pixel <-> ray conversion for the registered colour image (RealSense distortion models).

`inverse_brown_conrady` coefficients map distorted normalized coordinates straight to
undistorted ones (librealsense `rs2_deproject_pixel_to_point`, tangential terms as OpenCV);
`brown_conrady` is OpenCV's forward model and is inverted iteratively. Rays are
`[x, y, 1]` in the camera optical frame, so `p_cam = depth_m * ray` (COORDINATES_AND_TIMING.md).
"""

from __future__ import annotations

import threading

import cv2
import numpy as np
import numpy.typing as npt

from mvsr.core.errors import ContractValidationError
from mvsr.core.types import CalibrationBundle

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]


def undistort_normalized(uv: npt.ArrayLike, K: F64, model: str, coeffs: npt.ArrayLike) -> F64:
    """Pixels `(N, 2)` -> undistorted normalized image coordinates `(N, 2)`."""
    px = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (px[:, 0] - cx) / fx
    y = (px[:, 1] - cy) / fy
    k = np.asarray(coeffs, dtype=np.float64)
    if not np.any(k) or model == "none":
        return np.stack([x, y], axis=1)
    if model == "inverse_brown_conrady":
        k1, k2, p1, p2, k3 = k
        r2 = x * x + y * y
        f = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        ux = x * f + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        uy = y * f + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)
        return np.stack([ux, uy], axis=1)
    if model == "brown_conrady":
        und = cv2.undistortPoints(px.reshape(-1, 1, 2), K, k)
        out: F64 = und.reshape(-1, 2).astype(np.float64)
        return out
    raise ContractValidationError(
        f"unsupported distortion model {model!r} with non-zero coefficients"
    )


def undistort_pixels(uv: npt.ArrayLike, K: F64, model: str, coeffs: npt.ArrayLike) -> F64:
    n = undistort_normalized(uv, K, model, coeffs)
    out: F64 = np.stack([K[0, 0] * n[:, 0] + K[0, 2], K[1, 1] * n[:, 1] + K[1, 2]], axis=1)
    return out


def rays_for_pixels(bundle: CalibrationBundle, uv: npt.ArrayLike) -> F64:
    """`(N, 3)` rays `[x, y, 1]` for pixel coordinates of `bundle`'s image."""
    n = undistort_normalized(uv, bundle.K, bundle.distortion_model, bundle.distortion)
    return np.concatenate([n, np.ones((n.shape[0], 1))], axis=1)


_GRID_CACHE: dict[str, tuple[F32, F32]] = {}
_GRID_LOCK = threading.Lock()


def ray_grid(bundle: CalibrationBundle) -> tuple[F32, F32]:
    """Undistorted normalized `(x, y)` for every pixel centre, each `(H, W)` float32 (cached)."""
    key = bundle.calibration_id
    with _GRID_LOCK:
        hit = _GRID_CACHE.get(key)
    if hit is not None:
        return hit
    v, u = np.mgrid[0 : bundle.height, 0 : bundle.width]
    uv = np.stack([u.ravel(), v.ravel()], axis=1).astype(np.float64)
    n = undistort_normalized(uv, bundle.K, bundle.distortion_model, bundle.distortion)
    gx = n[:, 0].reshape(bundle.height, bundle.width).astype(np.float32)
    gy = n[:, 1].reshape(bundle.height, bundle.width).astype(np.float32)
    gx.flags.writeable = False
    gy.flags.writeable = False
    with _GRID_LOCK:
        _GRID_CACHE[key] = (gx, gy)
    return gx, gy


def project_cam_points(points_cam: npt.ArrayLike, K: F64) -> F64:
    """Undistorted pixel projection of camera-frame points `(N, 3)`; z must be > 0."""
    p = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    if np.any(p[:, 2] <= 0):
        raise ContractValidationError("points behind the camera")
    out: F64 = np.stack(
        [K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2], K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]], axis=1
    )
    return out
