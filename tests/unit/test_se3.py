"""Analytic tests of mvsr.core.se3 (code tests with seeded/hand values, not datasets; ADR-0002)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.spatial.transform import Rotation, Slerp

from mvsr.core import se3
from mvsr.core.errors import ContractValidationError, FrameMismatchError, InvalidPoseError
from mvsr.core.se3 import PoseSE3

RNG = np.random.default_rng(20261027)

# Angles exercising every branch: exact zero, series region, branch thresholds, near pi, pi.
EDGE_ANGLES = [
    0.0,
    1e-12,
    1e-8,
    1e-5,
    1e-3,
    0.0099,
    0.0101,
    0.5,
    2.0,
    math.pi - 0.0101,
    math.pi - 0.0099,
    math.pi - 1e-4,
    math.pi - 1e-7,
    math.pi - 1e-10,
    math.pi,
]


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def random_axis() -> np.ndarray:
    return unit(RNG.normal(size=3))


def random_pose(dst: str = "a", src: str = "b", scale: float = 1.0) -> PoseSE3:
    phi = random_axis() * RNG.uniform(0, math.pi)
    return PoseSE3.from_rt(se3.so3_exp(phi), RNG.normal(size=3) * scale, dst, src)


vec3 = st.lists(st.floats(-1, 1, allow_nan=False), min_size=3, max_size=3).filter(
    lambda v: np.linalg.norm(v) > 1e-9
)
angle = st.floats(0.0, math.pi - 1e-9, allow_nan=False)


# --------------------------------------------------------------------------- hat / vee


def test_hat_vee_roundtrip_and_cross_product() -> None:
    w, v = RNG.normal(size=3), RNG.normal(size=3)
    np.testing.assert_allclose(se3.hat3(w) @ v, np.cross(w, v), atol=1e-15)
    np.testing.assert_array_equal(se3.vee3(se3.hat3(w)), w)
    xi = RNG.normal(size=6)
    np.testing.assert_array_equal(se3.vee6(se3.hat6(xi)), xi)


# --------------------------------------------------------------------------- SO(3)


@pytest.mark.parametrize("theta", EDGE_ANGLES)
def test_so3_exp_log_roundtrip_edge_angles(theta: float) -> None:
    for _ in range(20):
        phi = random_axis() * theta
        R = se3.so3_exp(phi)
        se3.validate_rotation(R, atol=1e-12)
        np.testing.assert_allclose(R, Rotation.from_rotvec(phi).as_matrix(), atol=1e-12)
        back = se3.so3_log(R)
        if theta < math.pi - 1e-6:
            np.testing.assert_allclose(back, phi, atol=1e-9)
        else:  # +phi and -phi are the same rotation at pi
            np.testing.assert_allclose(se3.so3_exp(back), R, atol=1e-9)
            assert abs(np.linalg.norm(back) - theta) < 1e-7


@settings(max_examples=300, deadline=None)
@given(vec3, angle)
def test_so3_log_exp_property(axis: list[float], theta: float) -> None:
    phi = unit(np.array(axis)) * theta
    R = se3.so3_exp(phi)
    np.testing.assert_allclose(se3.so3_exp(se3.so3_log(R)), R, atol=1e-10)
    np.testing.assert_allclose(se3.so3_log(R), phi, atol=1e-6 if theta > math.pi - 1e-3 else 1e-10)


def test_so3_log_matches_scipy_random() -> None:
    for R in Rotation.random(200, random_state=1).as_matrix():
        expected = Rotation.from_matrix(R).as_rotvec()
        np.testing.assert_allclose(se3.so3_log(R), expected, atol=1e-9)


@pytest.mark.parametrize("theta", [0.0, 1e-7, 1e-3, 0.0099, 0.0101, 0.7, 2.5, math.pi - 1e-3])
def test_so3_left_jacobian_numerical(theta: float) -> None:
    phi = random_axis() * theta
    J = se3.so3_left_jacobian(phi)
    eps = 1e-6
    num = np.zeros((3, 3))
    R0 = se3.so3_exp(phi)
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        num[:, i] = (
            se3.so3_log(se3.so3_exp(phi + d) @ R0.T) - se3.so3_log(se3.so3_exp(phi - d) @ R0.T)
        ) / (2 * eps)
    np.testing.assert_allclose(J, num, atol=1e-7)
    np.testing.assert_allclose(se3.so3_left_jacobian_inv(phi) @ J, np.eye(3), atol=1e-10)


# --------------------------------------------------------------------------- SE(3)


@pytest.mark.parametrize("theta", EDGE_ANGLES)
def test_se3_exp_log_roundtrip(theta: float) -> None:
    for _ in range(10):
        xi = np.concatenate([RNG.normal(size=3), random_axis() * theta])
        T = se3.se3_exp(xi)
        se3.validate_rigid(T, atol=1e-12)
        back = se3.se3_log(T)
        np.testing.assert_allclose(se3.se3_exp(back), T, atol=1e-9)
        if theta < math.pi - 1e-6:
            np.testing.assert_allclose(back, xi, atol=1e-8)


def test_se3_exp_is_not_decoupled() -> None:
    xi = np.array([1.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2])
    T = se3.se3_exp(xi)
    # V @ rho for a 90 deg rotation about z: (sin/th, (1-cos)/th, 0)
    np.testing.assert_allclose(T[:3, 3], [2 / math.pi, 2 / math.pi, 0.0], atol=1e-12)


def test_se3_exp_matches_matrix_exponential() -> None:
    from scipy.linalg import expm

    for _ in range(20):
        xi = RNG.normal(size=6)
        np.testing.assert_allclose(se3.se3_exp(xi), expm(se3.hat6(xi)), atol=1e-10)


@pytest.mark.parametrize("theta", [0.0, 1e-6, 1e-3, 0.0099, 0.0101, 0.4, 1.5, 3.0])
def test_se3_left_jacobian_numerical(theta: float) -> None:
    xi = np.concatenate([RNG.normal(size=3), random_axis() * theta])
    J = se3.se3_left_jacobian(xi)
    T0inv = np.linalg.inv(se3.se3_exp(xi))
    eps = 1e-6
    num = np.zeros((6, 6))
    for i in range(6):
        d = np.zeros(6)
        d[i] = eps
        plus = se3.se3_log(se3.se3_exp(xi + d) @ T0inv)
        minus = se3.se3_log(se3.se3_exp(xi - d) @ T0inv)
        num[:, i] = (plus - minus) / (2 * eps)
    np.testing.assert_allclose(J, num, atol=2e-7)
    np.testing.assert_allclose(se3.se3_left_jacobian_inv(xi) @ J, np.eye(6), atol=1e-9)


def test_adjoint_identity() -> None:
    for _ in range(20):
        T = random_pose().matrix
        xi = RNG.normal(size=6) * 0.3
        lhs = T @ se3.se3_exp(xi) @ np.linalg.inv(T)
        np.testing.assert_allclose(lhs, se3.se3_exp(se3.se3_adjoint(T) @ xi), atol=1e-10)


# --------------------------------------------------------------------------- PoseSE3


def test_compose_order_and_points() -> None:
    T_a_b, T_b_c = random_pose("a", "b"), random_pose("b", "c")
    T_a_c = T_a_b @ T_b_c
    assert (T_a_c.dst, T_a_c.src) == ("a", "c")
    p_c = RNG.normal(size=(5, 3))
    np.testing.assert_allclose(
        T_a_c.transform_points(p_c), T_a_b.transform_points(T_b_c.transform_points(p_c)), atol=1e-12
    )
    T_c_d = random_pose("c", "d")
    assert ((T_a_b @ T_b_c) @ T_c_d).allclose(T_a_b @ (T_b_c @ T_c_d), atol=1e-12)


def test_contract_chain_base_cam_obj() -> None:
    T_base_tcp = random_pose("base", "tcp")
    T_tcp_cam = random_pose("tcp", "cam", 0.1)
    T_base_obj = random_pose("base", "obj")
    T_base_cam = T_base_tcp @ T_tcp_cam
    T_cam_obj = T_base_cam.inverse() @ T_base_obj
    assert (T_cam_obj.dst, T_cam_obj.src) == ("cam", "obj")
    assert (T_base_cam @ T_cam_obj).allclose(T_base_obj, atol=1e-12)


def test_frame_mismatch_raises() -> None:
    with pytest.raises(FrameMismatchError):
        _ = random_pose("a", "b") @ random_pose("c", "d")
    with pytest.raises(FrameMismatchError):
        random_pose("cam", "obj").expect_frames("obj", "cam")
    with pytest.raises(FrameMismatchError):
        se3.interpolate_decoupled(random_pose("a", "b"), random_pose("a", "c"), 0.5)


def test_inverse() -> None:
    T = random_pose("a", "b")
    inv = T.inverse()
    assert (inv.dst, inv.src) == ("b", "a")
    np.testing.assert_allclose((T @ inv).matrix, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(inv.matrix, np.linalg.inv(T.matrix), atol=1e-12)


@pytest.mark.parametrize(
    "matrix",
    [
        np.diag([1.0, 1.0, -1.0, 1.0]),  # reflection, det -1
        np.diag([1.0, 1.0, 1.001, 1.0]),  # not orthonormal
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0.1, 1]], float),  # last row
        np.full((4, 4), np.nan),
        np.eye(3),
    ],
)
def test_invalid_pose_rejected(matrix: np.ndarray) -> None:
    with pytest.raises(InvalidPoseError):
        PoseSE3(matrix, "a", "b")


def test_pose_frames_and_arrays_frozen() -> None:
    with pytest.raises(ContractValidationError):
        PoseSE3(np.eye(4), "", "b")
    m = np.eye(4)
    T = PoseSE3(m, "a", "b")
    m[0, 3] = 5.0
    assert T.t[0] == 0.0  # copied
    with pytest.raises(ValueError):
        T.matrix[0, 3] = 1.0


def test_orthonormal_tolerance_is_1e6() -> None:
    R = se3.so3_exp([0.1, 0.2, 0.3])
    PoseSE3.from_rt(R * (1 + 3e-7), [0, 0, 0], "a", "b")  # |R^T R - I| 6e-7, |det - 1| 9e-7
    with pytest.raises(InvalidPoseError):
        PoseSE3.from_rt(R * (1 + 2e-6), [0, 0, 0], "a", "b")


# --------------------------------------------------------------------------- covariance


def spd(scale: float = 1e-4) -> np.ndarray:
    A = RNG.normal(size=(6, 6)) * scale
    return A @ A.T + np.eye(6) * scale**2


def test_left_perturbation_maps_with_adjoint_exactly() -> None:
    A, B = random_pose("a", "b"), random_pose("b", "c")
    d = RNG.normal(size=6) * 1e-2
    perturbed = A.matrix @ se3.se3_exp(d) @ B.matrix
    delta_c = se3.se3_log(perturbed @ np.linalg.inv((A @ B).matrix))
    np.testing.assert_allclose(delta_c, se3.se3_adjoint(A.matrix) @ d, atol=1e-12)
    inv_pert = np.linalg.inv(se3.se3_exp(d) @ A.matrix)
    delta_inv = se3.se3_log(inv_pert @ A.matrix)
    np.testing.assert_allclose(delta_inv, -se3.se3_adjoint(np.linalg.inv(A.matrix)) @ d, atol=1e-12)


def test_covariance_propagation() -> None:
    SA, SB = spd(), spd()
    A = random_pose("a", "b").with_covariance(SA)
    B = random_pose("b", "c").with_covariance(SB)
    Ad = se3.se3_adjoint(A.matrix)
    C = A @ B
    assert C.covariance is not None
    np.testing.assert_allclose(C.covariance, SA + Ad @ SB @ Ad.T, atol=1e-15)
    inv = A.inverse()
    Adi = se3.se3_adjoint(np.linalg.inv(A.matrix))
    assert inv.covariance is not None
    np.testing.assert_allclose(inv.covariance, Adi @ SA @ Adi.T, atol=1e-15)
    # never copied across frames unchanged
    assert not np.allclose(inv.covariance, SA)


def test_covariance_monte_carlo_composition() -> None:
    """Seeded sampling check of the first-order propagation (code test, not accuracy evidence)."""
    SB = spd(1e-3)
    A = random_pose("a", "b")
    B = random_pose("b", "c").with_covariance(SB)
    Cbar = (A @ B).matrix
    L = np.linalg.cholesky(SB)
    rng = np.random.default_rng(7)
    deltas = []
    for _ in range(4000):
        d = L @ rng.normal(size=6)
        Ci = A.matrix @ se3.se3_exp(d) @ B.matrix
        deltas.append(se3.se3_log(Ci @ np.linalg.inv(Cbar)))
    emp = np.cov(np.array(deltas).T)
    Ad = se3.se3_adjoint(A.matrix)
    expected = Ad @ SB @ Ad.T
    assert np.max(np.abs(emp - expected)) < 0.1 * np.max(np.abs(expected))


def test_covariance_none_semantics() -> None:
    A = random_pose("a", "b").with_covariance(spd())
    B = random_pose("b", "c")
    assert (A @ B).covariance is None
    with pytest.raises(ContractValidationError):
        A.with_covariance(-np.eye(6))
    with pytest.raises(ContractValidationError):
        bad = np.eye(6)
        bad[0, 1] = 1.0
        A.with_covariance(bad)


# --------------------------------------------------------------------------- quaternions


def test_quaternion_roundtrip_and_scipy() -> None:
    for R in Rotation.random(300, random_state=3).as_matrix():
        q = se3.rotation_to_quat_xyzw(R)
        assert q[3] >= 0.0
        ref = Rotation.from_matrix(R).as_quat()  # scipy uses xyzw
        ref = ref if ref[3] >= 0 else -ref
        np.testing.assert_allclose(q, ref, atol=1e-12)
        np.testing.assert_allclose(se3.quat_xyzw_to_rotation(q), R, atol=1e-12)


def test_quaternion_sign_canonicalization() -> None:
    np.testing.assert_array_equal(
        se3.canonical_quat_xyzw([0.1, 0.2, 0.3, -0.927]), [-0.1, -0.2, -0.3, 0.927]
    )
    # w == 0 (180 deg): first non-zero xyz made positive
    np.testing.assert_array_equal(
        se3.canonical_quat_xyzw([0.0, -0.6, 0.8, 0.0]), [0.0, 0.6, -0.8, 0.0]
    )
    q = se3.rotation_to_quat_xyzw(np.diag([-1.0, 1.0, -1.0]))  # 180 deg about y
    np.testing.assert_allclose(q, [0.0, 1.0, 0.0, 0.0], atol=1e-15)
    with pytest.raises(ContractValidationError):
        se3.quat_xyzw_to_rotation([0, 0, 0, 2.0])


# --------------------------------------------------------------------------- interpolation


def test_interpolation_endpoints_and_midpoints() -> None:
    a, b = random_pose("base", "tcp"), random_pose("base", "tcp")
    for fn in (se3.interpolate_decoupled, se3.interpolate_geodesic):
        assert fn(a, b, 0.0).allclose(a, atol=1e-12)
        assert fn(a, b, 1.0).allclose(b, atol=1e-9)
        with pytest.raises(ContractValidationError):
            fn(a, b, 1.0001)
        with pytest.raises(ContractValidationError):
            fn(a, b, -0.1)
    m = se3.interpolate_decoupled(a, b, 0.3)
    slerp = Slerp([0, 1], Rotation.from_matrix([a.R, b.R]))
    np.testing.assert_allclose(m.R, slerp([0.3]).as_matrix()[0], atol=1e-10)
    np.testing.assert_allclose(m.t, 0.7 * a.t + 0.3 * b.t, atol=1e-15)
    g = se3.interpolate_geodesic(a, b, 0.5)
    half = se3.se3_log(np.linalg.inv(a.matrix) @ b.matrix) * 0.5
    np.testing.assert_allclose(se3.se3_log(np.linalg.inv(a.matrix) @ g.matrix), half, atol=1e-9)
    # decoupled and geodesic differ for coupled motion (both named separately on purpose)
    assert not np.allclose(m.t, se3.interpolate_geodesic(a, b, 0.3).t, atol=1e-6)


def test_transform_points_shape_check() -> None:
    with pytest.raises(ContractValidationError):
        random_pose().transform_points(np.zeros((3,)))
