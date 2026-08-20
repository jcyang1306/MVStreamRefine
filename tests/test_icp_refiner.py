"""Task 9 tests. Parameter parsing and safety-gate logic run everywhere; the
real point-to-plane ICP tests require open3d and are skipped on the dev
machine (pytest.importorskip), where only the CUDA container ships open3d."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from object_reconstruction.registration.icp_refiner import (
    ICPParams,
    ICPRefiner,
    correction_magnitudes,
)
from object_reconstruction.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def rotation_z(deg: float) -> np.ndarray:
    T = np.eye(4)
    rad = np.radians(deg)
    T[:2, :2] = [[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]]
    return T


# -- params and gates (no open3d required) -----------------------------------

def test_params_from_repo_offline_yaml():
    config = load_config(REPO_ROOT / "configs" / "offline.yaml")
    params = ICPParams.from_config(config)
    assert params.enabled is False
    assert params.max_correspondence_distance_m == 0.01
    assert params.min_fitness == 0.4
    assert params.max_rmse_m == 0.008
    assert params.max_translation_correction_m == 0.02
    assert params.max_rotation_correction_deg == 5.0


def test_params_defaults_and_validation():
    assert ICPParams.from_config({}) == ICPParams()
    with pytest.raises(ValueError, match="icp.min_fitness"):
        ICPParams.from_config({"icp": {"min_fitness": 0.0}})
    with pytest.raises(ValueError, match="icp.max_iterations"):
        ICPParams.from_config({"icp": {"max_iterations": 0}})


def test_correction_magnitudes():
    T_init = np.eye(4)
    T_refined = rotation_z(3.0)
    T_refined[:3, 3] = [0.01, 0.0, 0.0]
    translation, rotation = correction_magnitudes(T_init, T_refined)
    assert translation == pytest.approx(0.01, abs=1e-9)
    assert rotation == pytest.approx(3.0, abs=1e-6)
    # Identical poses: zero correction.
    assert correction_magnitudes(T_refined, T_refined) == pytest.approx((0.0, 0.0))


@pytest.mark.parametrize(
    "fitness,rmse,dt,dr,expected",
    [
        (0.9, 0.002, 0.005, 1.0, True),    # all gates pass
        (0.3, 0.002, 0.005, 1.0, False),   # fitness too low
        (0.9, 0.020, 0.005, 1.0, False),   # rmse too high
        (0.9, 0.002, 0.050, 1.0, False),   # translation correction too large
        (0.9, 0.002, 0.005, 10.0, False),  # rotation correction too large
    ],
)
def test_safety_gates(fitness, rmse, dt, dr, expected):
    params = ICPParams()
    assert params.accepts(fitness, rmse, dt, dr) is expected


def test_missing_open3d_raises_actionable_error():
    if importlib.util.find_spec("open3d") is not None:
        pytest.skip("open3d is installed; the lazy-import error path is unreachable")
    with pytest.raises(RuntimeError, match="CUDA container"):
        ICPRefiner(ICPParams())


# -- real ICP on synthetic geometry (open3d required) ------------------------

def make_object_points(n: int = 4000, seed: int = 0) -> np.ndarray:
    """Random points on three faces of a 10 cm box (gives ICP full constraints)."""
    rng = np.random.default_rng(seed)
    per_face = n // 3
    u, v = rng.uniform(0, 0.1, (2, per_face))
    top = np.column_stack([u, v, np.full(per_face, 0.1)])
    front = np.column_stack([u, np.zeros(per_face), v])
    side = np.column_stack([np.zeros(per_face), u, v])
    return np.vstack([top, front, side])


def small_perturbation() -> np.ndarray:
    T = rotation_z(1.0)
    T[:3, 3] = [0.003, -0.002, 0.001]
    return T


def test_refine_recovers_small_pose_error():
    pytest.importorskip("open3d")
    model_world = make_object_points()
    T_world_cam_true = rotation_z(30.0)
    T_world_cam_true[:3, 3] = [0.2, 0.1, 0.4]
    # Camera observes the object perfectly at the true pose.
    source_cam = (model_world - T_world_cam_true[:3, 3]) @ T_world_cam_true[:3, :3]
    # The robot reports a slightly wrong pose.
    T_robot = T_world_cam_true @ small_perturbation()

    refiner = ICPRefiner(ICPParams(voxel_downsample_m=0.002))
    result = refiner.refine(source_cam, model_world, T_robot)

    assert result.success
    assert result.fitness > 0.8
    # The refined pose must be much closer to the truth than the robot pose.
    dt_before, dr_before = correction_magnitudes(T_world_cam_true, T_robot)
    dt_after, dr_after = correction_magnitudes(
        T_world_cam_true, result.T_world_cam_refined
    )
    assert dt_after < dt_before / 2
    assert dr_after < dr_before / 2


def test_refine_falls_back_when_gates_reject():
    pytest.importorskip("open3d")
    model_world = make_object_points()
    T_world_cam_true = np.eye(4)
    T_world_cam_true[:3, 3] = [0.0, 0.0, 0.3]
    source_cam = (model_world - T_world_cam_true[:3, 3]) @ T_world_cam_true[:3, :3]
    T_robot = T_world_cam_true @ small_perturbation()

    # Impossible gates: any correction is rejected -> robot pose fallback.
    params = ICPParams(min_fitness=1.1, voxel_downsample_m=0.002)
    result = ICPRefiner(params).refine(source_cam, model_world, T_robot)

    assert not result.success
    assert result.reason
    np.testing.assert_allclose(result.T_world_cam_refined, T_robot)


def test_refine_falls_back_on_too_few_points():
    pytest.importorskip("open3d")
    tiny = make_object_points(n=30)
    result = ICPRefiner(ICPParams()).refine(tiny, make_object_points(), np.eye(4))
    assert not result.success
    assert "points" in result.reason
    np.testing.assert_allclose(result.T_world_cam_refined, np.eye(4))
