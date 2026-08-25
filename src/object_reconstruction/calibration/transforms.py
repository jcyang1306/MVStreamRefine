"""Rigid-transform utilities for the single wrist-camera system.

Confirmed conventions (capture side, 2026-08-19):
    pose 7D           = x, y, z, qx, qy, qz, qw  (quaternion order xyzw)
    pose semantics    = T_base_tcp (robot TCP pose in base frame)
    handeye wrist_cam2 = T_tcp_cam (wrist camera pose in TCP frame)
    WORLD = robot base frame

    T_world_cam = T_base_tcp @ T_tcp_cam
"""

from __future__ import annotations

import numpy as np

SUPPORTED_POSE_SEMANTICS = "T_base_tcp"
SUPPORTED_QUATERNION_ORDER = "xyzw"
SUPPORTED_HANDEYE_CONVENTION = "wrist_cam2=T_tcp_cam"


def quaternion_xyzw_to_rotation(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {q.shape}")
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError(f"quaternion norm {norm} is invalid")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose7d_to_matrix(pose: np.ndarray, quaternion_order: str = "xyzw") -> np.ndarray:
    if quaternion_order != SUPPORTED_QUATERNION_ORDER:
        raise NotImplementedError(f"unsupported quaternion order: {quaternion_order}")
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"pose must have shape (7,), got {pose.shape}")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_xyzw_to_rotation(pose[3:])
    T[:3, 3] = pose[:3]
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    validate_transform(T)
    R = T[:3, :3]
    t = T[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def validate_transform(T: np.ndarray, tolerance: float = 1e-5) -> None:
    T = np.asarray(T)
    if T.shape != (4, 4):
        raise ValueError(f"transform must be 4x4, got {T.shape}")
    if not np.isfinite(T).all():
        raise ValueError("transform contains NaN/inf")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=tolerance):
        raise ValueError(f"bottom row must be [0,0,0,1], got {T[3]}")
    R = T[:3, :3]
    if abs(np.linalg.det(R) - 1.0) > 1e-3:
        raise ValueError(f"det(R) = {np.linalg.det(R)} is not +1")
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-3):
        raise ValueError("rotation part is not orthonormal")


def compose_T_world_cam(
    T_base_tcp: np.ndarray,
    T_tcp_cam: np.ndarray,
) -> np.ndarray:
    """WORLD = robot base: T_world_cam = T_base_tcp @ T_tcp_cam."""
    for T in (T_base_tcp, T_tcp_cam):
        validate_transform(T)
    return T_base_tcp @ T_tcp_cam


def check_supported_conventions(
    pose_semantics: str, quaternion_order: str, handeye_convention: str
) -> None:
    """Fail fast on any convention this code was not written for."""
    normalized = handeye_convention.replace(" ", "")
    if (
        pose_semantics != SUPPORTED_POSE_SEMANTICS
        or quaternion_order != SUPPORTED_QUATERNION_ORDER
        or normalized != SUPPORTED_HANDEYE_CONVENTION
    ):
        raise NotImplementedError(
            "unsupported convention combination: "
            f"pose_semantics={pose_semantics!r}, quaternion_order={quaternion_order!r}, "
            f"handeye_convention={handeye_convention!r}; "
            f"this build supports only ({SUPPORTED_POSE_SEMANTICS!r}, "
            f"{SUPPORTED_QUATERNION_ORDER!r}, {SUPPORTED_HANDEYE_CONVENTION!r})"
        )
