"""Checkerboard helpers (P1 S2): orientation, undistortion, PnP, pose averaging (analytic).

The painted image below only exercises corner ordering and detection code paths; it is not
accuracy evidence (ADR-0002).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from mvsr.core.errors import ContractValidationError
from mvsr.core.se3 import PoseSE3, so3_exp
from mvsr.sensors.board import (
    BoardSpec,
    canonical_order,
    detect_corners,
    mean_pose,
    project_points,
    rotation_angle_deg,
    solve_board_pose,
    undistort_normalized,
    undistort_pixels,
)
from mvsr.sensors.calib_check import square_mm_from_notes

SPEC = BoardSpec(11, 8, 0.015)
K = np.array([[650.0, 0.0, 640.0], [0.0, 650.0, 360.0], [0.0, 0.0, 1.0]])
CELL, U0, V0 = 30, 200, 150


def _painted(first_dark: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Fronto-parallel 12x9-square board painted in pixels; returns (gray, corner grid)."""
    img = np.full((720, 1280), 255, dtype=np.uint8)
    for i in range(-1, SPEC.inner_cols):
        for j in range(-1, SPEC.inner_rows):
            dark = (i + j) % 2 == (0 if first_dark else 1)
            if dark:
                u, v = U0 + i * CELL, V0 + j * CELL
                img[v : v + CELL, u : u + CELL] = 20
    r, c = np.mgrid[0 : SPEC.inner_rows, 0 : SPEC.inner_cols]
    grid = np.stack([U0 + c * CELL, V0 + r * CELL], axis=2).astype(np.float64)
    return img, grid


def test_board_spec_validation_and_object_points() -> None:
    with pytest.raises(ContractValidationError):
        BoardSpec(10, 8, 0.015)
    with pytest.raises(ContractValidationError):
        BoardSpec(11, 8, 0.0)
    obj = SPEC.object_points()
    assert obj.shape == (88, 3)
    np.testing.assert_allclose(obj[1] - obj[0], [0.015, 0, 0])
    np.testing.assert_allclose(obj[11] - obj[0], [0, 0.015, 0])


def test_canonical_order_resolves_reversal_and_mirror() -> None:
    img, g = _painted()
    for variant in (g, g[::-1, ::-1], g[::-1, :], g[:, ::-1]):
        np.testing.assert_array_equal(canonical_order(variant, img), g)
    img2, g2 = _painted(first_dark=False)
    np.testing.assert_array_equal(canonical_order(g2, img2), g2[::-1, ::-1])


def test_detect_corners_returns_canonical_grid() -> None:
    img, g = _painted()
    corners = detect_corners(img, SPEC)
    assert corners is not None
    np.testing.assert_allclose(corners, g.reshape(-1, 2), atol=1.0)
    assert detect_corners(np.full_like(img, 128), SPEC) is None


def test_inverse_brown_conrady_follows_librealsense_formula() -> None:
    coeffs = [-0.049, 0.060, 0.0009, 0.0008, -0.021]
    uv = np.array([[640.0, 360.0], [1200.0, 650.0], [30.0, 40.0]])
    n = undistort_normalized(uv, K, "inverse_brown_conrady", coeffs)
    x, y = (uv[2, 0] - 640.0) / 650.0, (uv[2, 1] - 360.0) / 650.0
    k1, k2, p1, p2, k3 = coeffs
    r2 = x * x + y * y
    f = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    np.testing.assert_allclose(n[0], [0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(
        n[2],
        [
            x * f + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * f + 2 * p2 * x * y + p1 * (r2 + 2 * y * y),
        ],
    )
    np.testing.assert_array_equal(undistort_pixels(uv, K, "inverse_brown_conrady", [0.0] * 5), uv)
    with pytest.raises(ContractValidationError):
        undistort_normalized(uv, K, "kannala_brandt4", coeffs)


def test_brown_conrady_inverts_the_forward_model() -> None:
    coeffs = np.array([-0.05, 0.06, 0.001, 0.0008, -0.02])
    pts = np.array([[0.1, -0.05, 0.4], [-0.15, 0.1, 0.35]])
    distorted, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, coeffs)
    und = undistort_pixels(distorted.reshape(-1, 2), K, "brown_conrady", coeffs)
    ideal = (K @ (pts / pts[:, 2:3]).T).T[:, :2]
    np.testing.assert_allclose(und, ideal, atol=1e-2)


def test_solve_board_pose_round_trip() -> None:
    R = so3_exp([0.4, -0.2, 0.3])
    truth = PoseSE3.from_rt(R, [-0.05, -0.03, 0.35], "cam", "marker")
    px = project_points(truth, SPEC.object_points(), K)
    pose = solve_board_pose(px, SPEC, K, "inverse_brown_conrady", [0.0] * 5, "cam")
    assert pose is not None and pose.reproj_rms_px < 1e-6
    assert pose.T_cam_marker.allclose(truth, atol=1e-8)
    assert (pose.T_cam_marker.dst, pose.T_cam_marker.src) == ("cam", "marker")


def test_mean_pose_and_rotation_angle() -> None:
    a = PoseSE3.from_rt(so3_exp([0, 0, 0.01]), [0.0, 0.0, 0.0], "base", "marker")
    b = PoseSE3.from_rt(so3_exp([0, 0, -0.01]), [0.002, 0.0, 0.0], "base", "marker")
    c = PoseSE3.from_rt(np.eye(3), [0.001, 0.0, 0.0], "base", "marker")
    m = mean_pose([a, b, c])
    assert rotation_angle_deg(m.R, np.eye(3)) < 1e-9
    np.testing.assert_allclose(m.t, [0.001, 0.0, 0.0])
    assert abs(rotation_angle_deg(a.R, b.R) - math.degrees(0.02)) < 1e-9
    with pytest.raises(ContractValidationError):
        mean_pose([a, PoseSE3.identity("cam", "marker")])


def test_square_size_from_operator_notes() -> None:
    assert square_mm_from_notes("square_mm=15") == 15.0
    assert square_mm_from_notes("lighting ok; square_mm = 14.5 ") == 14.5
    assert square_mm_from_notes("") is None and square_mm_from_notes(None) is None
