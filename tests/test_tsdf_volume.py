"""Task 5 tests. Parameter parsing and fail-fast logic run everywhere; the
synthetic integration tests require open3d and are skipped on the dev machine
(pytest.importorskip), where only the CUDA container ships open3d."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from object_reconstruction.data.models import CameraIntrinsics
from object_reconstruction.fusion.tsdf_volume import (
    TSDFParams,
    TSDFVolume,
    check_device_string,
)
from object_reconstruction.utils.config import load_config

CONFIG = {
    "tsdf": {
        "device": "CPU:0",
        "voxel_size_m": 0.002,
        "block_resolution": 16,
        "block_count": 50000,
        "trunc_voxel_multiplier": 4.0,
        "depth_max_m": 1.2,
        "weight_threshold": 3.0,
    },
    "depth": {"scale": 1000.0},
}


# -- parameter parsing (no open3d required) ---------------------------------

def test_from_config_reads_all_fields():
    params = TSDFParams.from_config(CONFIG)
    assert params.device == "CPU:0"
    assert params.voxel_size_m == 0.002
    assert params.block_resolution == 16
    assert params.block_count == 50000
    assert params.trunc_voxel_multiplier == 4.0
    assert params.depth_max_m == 1.2
    assert params.weight_threshold == 3.0
    assert params.depth_scale == 1000.0


def test_from_config_accepts_repo_offline_yaml():
    config = load_config(Path(__file__).resolve().parent.parent / "configs" / "offline.yaml")
    params = TSDFParams.from_config(config)
    assert params.device == "CPU:0"
    assert params.depth_scale == 1000.0


def test_from_config_defaults_for_missing_keys():
    params = TSDFParams.from_config({"tsdf": {}, "depth": {"scale": 1000.0}})
    assert params == TSDFParams()


def test_null_depth_scale_forbidden():
    with pytest.raises(ValueError, match="depth.scale"):
        TSDFParams.from_config({"tsdf": {}, "depth": {"scale": None}})


@pytest.mark.parametrize("device", ["cuda:0", "CPU", "GPU:0", "CUDA:", "", None])
def test_invalid_device_strings_rejected(device):
    with pytest.raises(ValueError, match="tsdf.device"):
        check_device_string(device)


def test_valid_device_strings_accepted():
    assert check_device_string("CPU:0") == "CPU:0"
    assert check_device_string("CUDA:1") == "CUDA:1"


@pytest.mark.parametrize(
    "overrides",
    [
        {"voxel_size_m": 0.0},
        {"block_resolution": -16},
        {"block_count": 0},
        {"trunc_voxel_multiplier": -1.0},
        {"depth_max_m": 0.0},
        {"depth_scale": 0.0},
        {"weight_threshold": -1.0},
    ],
)
def test_invalid_parameter_values_rejected(overrides):
    config = {"tsdf": dict(CONFIG["tsdf"]), "depth": dict(CONFIG["depth"])}
    for key, value in overrides.items():
        if key == "depth_scale":
            config["depth"]["scale"] = value
        else:
            config["tsdf"][key] = value
    with pytest.raises(ValueError):
        TSDFParams.from_config(config)


def test_missing_open3d_raises_actionable_error():
    if importlib.util.find_spec("open3d") is not None:
        pytest.skip("open3d is installed; the lazy-import error path is unreachable")
    with pytest.raises(RuntimeError, match="CUDA container"):
        TSDFVolume(TSDFParams())


# -- synthetic integration (open3d required) ---------------------------------

def make_plane_frame(width=64, height=48, depth_mm=500):
    """Flat plane at depth_mm in front of a small pinhole camera."""
    rgb = np.full((height, width, 3), 200, dtype=np.uint8)
    depth = np.full((height, width), depth_mm, dtype=np.uint16)
    intrinsics = CameraIntrinsics(
        width=width, height=height, fx=60.0, fy=60.0, cx=32.0, cy=24.0
    )
    return rgb, depth, intrinsics


def small_params(**overrides):
    base = dict(
        device="CPU:0",
        voxel_size_m=0.005,
        block_resolution=8,
        block_count=2000,
        trunc_voxel_multiplier=4.0,
        depth_max_m=1.2,
        weight_threshold=1.0,
        depth_scale=1000.0,
    )
    base.update(overrides)
    return TSDFParams(**base)


def test_integrate_plane_and_extract_point_cloud(tmp_path):
    pytest.importorskip("open3d")
    volume = TSDFVolume(small_params())
    rgb, depth, intrinsics = make_plane_frame(depth_mm=500)
    for _ in range(3):
        volume.integrate(rgb, depth, intrinsics, T_world_cam=np.eye(4))
    assert volume.integration_count == 3

    points = volume.extract_point_cloud().point.positions.numpy()
    assert len(points) > 0
    # Surface points must sit on the plane, within ~2 voxels of z = 0.5 m.
    assert np.abs(points[:, 2] - 0.5).max() < 2 * volume.params.voxel_size_m + 1e-6

    saved = volume.save(tmp_path / "plane.ply")
    assert saved.is_file() and saved.stat().st_size > 0


def test_integrate_rejects_bad_inputs():
    pytest.importorskip("open3d")
    volume = TSDFVolume(small_params())
    rgb, depth, intrinsics = make_plane_frame()
    with pytest.raises(ValueError, match="depth must be uint16"):
        volume.integrate(rgb, depth.astype(np.float32), intrinsics, np.eye(4))
    with pytest.raises(ValueError, match="rgb must be uint8"):
        volume.integrate(rgb.astype(np.float64), depth, intrinsics, np.eye(4))


def test_extract_before_integration_fails():
    pytest.importorskip("open3d")
    volume = TSDFVolume(small_params())
    with pytest.raises(RuntimeError, match="nothing has been integrated"):
        volume.extract_point_cloud()


def test_cuda_request_fails_fast_on_cpu_only_build():
    o3d = pytest.importorskip("open3d")
    if o3d.core.cuda.is_available():
        pytest.skip("CUDA-enabled Open3D build; the fail-fast path is unreachable")
    with pytest.raises(RuntimeError, match="no.*CUDA support"):
        TSDFVolume(small_params(device="CUDA:0"))
