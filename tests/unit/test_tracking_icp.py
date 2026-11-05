from __future__ import annotations

from dataclasses import replace

import numpy as np

from mvsr.core.se3 import PoseSE3, se3_exp
from mvsr.tracking.config import load_tracking_config
from mvsr.tracking.icp import point_to_plane_icp


def test_point_to_plane_icp_recovers_known_pose() -> None:
    rng = np.random.default_rng(42)
    points = rng.uniform([-0.05, -0.04, -0.03], [0.05, 0.04, 0.03], (600, 3))
    normals = rng.normal(size=(600, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    truth = PoseSE3(se3_exp([0.008, -0.004, 0.006, 0.015, -0.02, 0.01]), "cam", "obj")
    target = truth.transform_points(points)
    target_normals = normals @ truth.R.T
    initial = PoseSE3(
        se3_exp([0.003, -0.002, 0.001, 0.005, -0.004, 0.003]) @ truth.matrix,
        "cam",
        "obj",
    )
    cfg = replace(
        load_tracking_config().icp,
        max_points=600,
        min_correspondences=100,
        normal_angle_deg=89.0,
    )
    result = point_to_plane_icp(points, normals, target, target_normals, initial, cfg)
    assert result is not None
    assert result.correspondences > 500
    np.testing.assert_allclose(result.T_cam_obj.matrix, truth.matrix, atol=2e-4)
    assert result.rmse_m < 1e-4


def test_icp_rejects_insufficient_geometry() -> None:
    cfg = load_tracking_config().icp
    eye = PoseSE3.identity("cam", "obj")
    assert (
        point_to_plane_icp(
            np.zeros((5, 3)),
            np.zeros((5, 3)),
            np.zeros((5, 3)),
            np.zeros((5, 3)),
            eye,
            cfg,
        )
        is None
    )


def test_icp_filters_nonfinite_and_zero_normals() -> None:
    cfg = replace(load_tracking_config().icp, min_correspondences=6)
    points = np.arange(30, dtype=np.float64).reshape(10, 3) / 1000
    normals = np.tile([0.0, 0.0, 1.0], (10, 1))
    normals[:4] = 0.0
    points[4] = np.nan
    result = point_to_plane_icp(
        points,
        normals,
        np.nan_to_num(points),
        normals,
        PoseSE3.identity("cam", "obj"),
        cfg,
    )
    assert result is None
