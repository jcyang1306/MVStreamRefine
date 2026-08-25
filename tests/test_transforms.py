import numpy as np
import pytest

from object_reconstruction.calibration.transforms import (
    check_supported_conventions,
    compose_T_world_cam,
    invert_transform,
    pose7d_to_matrix,
    quaternion_xyzw_to_rotation,
    validate_transform,
)


def rigid(axis, angle_rad, translation):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    half = angle_rad / 2.0
    q = np.array([*(axis * np.sin(half)), np.cos(half)])
    T = np.eye(4)
    T[:3, :3] = quaternion_xyzw_to_rotation(q)
    T[:3, 3] = translation
    return T


def test_identity_quaternion():
    T = pose7d_to_matrix(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]))
    assert np.allclose(T[:3, :3], np.eye(3))
    assert np.allclose(T[:3, 3], [1.0, 2.0, 3.0])


def test_known_rotation_90deg_about_z():
    # xyzw quaternion for +90 deg about z: (0, 0, sin45, cos45)
    s = np.sqrt(0.5)
    T = pose7d_to_matrix(np.array([0, 0, 0, 0, 0, s, s]))
    p = T[:3, :3] @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(p, [0.0, 1.0, 0.0], atol=1e-12)


def test_inverse_roundtrip():
    T = rigid([0.3, -0.5, 0.8], 1.1, [0.2, -0.4, 0.9])
    assert np.allclose(T @ invert_transform(T), np.eye(4), atol=1e-12)
    assert np.allclose(invert_transform(T) @ T, np.eye(4), atol=1e-12)


def test_composition_formula():
    T_base_tcp = rigid([1, 1, 0], -0.7, [0.5, -0.1, 0.2])
    T_tcp_cam = rigid([0, 1, 0], 0.2, [0.02, 0.03, -0.05])
    T = compose_T_world_cam(T_base_tcp, T_tcp_cam)
    expected = T_base_tcp @ T_tcp_cam
    assert np.allclose(T, expected, atol=1e-12)
    validate_transform(T)


def test_validate_rejects_scaled_rotation():
    T = np.eye(4)
    T[:3, :3] *= 2.0
    with pytest.raises(ValueError):
        validate_transform(T)


def test_unsupported_conventions_fail_fast():
    with pytest.raises(NotImplementedError):
        check_supported_conventions("T_tcp_base", "xyzw", "wrist_cam2=T_tcp_cam")
    with pytest.raises(NotImplementedError):
        check_supported_conventions("T_base_tcp", "wxyz", "wrist_cam2=T_tcp_cam")
    # Confirmed combination passes.
    check_supported_conventions(
        "T_base_tcp", "xyzw", "wrist_cam2=T_tcp_cam"
    )
