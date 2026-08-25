"""Single-camera OfflinePipeline tests on real frames with fake geometry."""

from pathlib import Path

import numpy as np
import pytest

from object_reconstruction.data.offline_source import OfflineFrameSource
from object_reconstruction.fusion.keyframe_selector import KeyframeSelector
from object_reconstruction.pipeline.offline_pipeline import OfflinePipeline
from object_reconstruction.registration.icp_refiner import ICPResult
from object_reconstruction.segmentation.mask_cache import MaskCache
from object_reconstruction.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
FRAMES = 6


class FakePointCloud:
    def __init__(self, n_points):
        self.points = list(range(n_points))


class FakeTSDF:
    def __init__(self):
        self.calls = []
        self.saved_point_clouds = []

    def integrate(self, rgb, depth, intrinsics, T_world_cam):
        assert rgb.dtype == np.uint8 and depth.dtype == np.uint16
        self.calls.append(np.asarray(T_world_cam))

    def extract_point_cloud(self):
        return FakePointCloud(n_points=len(self.calls) * 10)

    def save_point_cloud(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake ply")
        self.saved_point_clouds.append(path)
        return path

    def block_count(self):
        return len(self.calls) * 7


class FakeRefiner:
    OFFSET = np.array([0.001, 0.0, 0.0])

    def __init__(self, accept=True):
        self.accept = accept
        self.calls = []

    def refine(self, source_points, model_points, T_init):
        self.calls.append((np.asarray(source_points), np.asarray(model_points)))
        T_init = np.asarray(T_init, dtype=np.float64)
        if not self.accept:
            return ICPResult(
                success=False,
                T_world_cam_refined=T_init.copy(),
                fitness=0.1,
                inlier_rmse=0.02,
                translation_correction_m=0.0,
                rotation_correction_deg=0.0,
                reason="safety gates rejected the correction",
            )
        T_refined = T_init.copy()
        T_refined[:3, 3] += self.OFFSET
        return ICPResult(
            success=True,
            T_world_cam_refined=T_refined,
            fitness=0.9,
            inlier_rmse=0.002,
            translation_correction_m=0.001,
            rotation_correction_deg=0.0,
        )


class FakeViewer:
    def __init__(self, quit_after=None):
        self.updates = []
        self.quit_after = quit_after
        self.closed = False

    def update(self, point_cloud, T_world_cam):
        self.updates.append((len(point_cloud.points), np.asarray(T_world_cam)))
        return self.quit_after is None or len(self.updates) < self.quit_after

    def close(self):
        self.closed = True


@pytest.fixture(scope="module")
def config():
    cfg = load_config(REPO_ROOT / "configs" / "offline.yaml")
    cfg["dataset"]["root"] = str(REPO_ROOT / "data")
    cfg["output"] = {
        "root": "./output",
        "save_keyframes": False,
        "save_debug": False,
    }
    return cfg


@pytest.fixture(scope="module")
def source(config):
    return OfflineFrameSource.from_config(config)


@pytest.fixture()
def mask_cache(tmp_path, source):
    cache = MaskCache(tmp_path / "masks")
    mask = np.zeros((720, 1280), dtype=bool)
    mask[200:520, 400:880] = True
    for position in range(FRAMES):
        cache.save(source.indices[position], mask)
    return cache


def test_single_camera_keyframe_loop(config, source, mask_cache):
    tsdf = FakeTSDF()
    stats = OfflinePipeline(
        tsdf, KeyframeSelector.from_config(config), config
    ).run(source, mask_cache, max_frames=FRAMES, verbose=False)

    assert stats["frames_processed"] == FRAMES
    assert stats["missing_mask_frames"] == []
    assert stats["keyframes"] >= 1
    assert stats["keyframes"] + stats["rejected"] == FRAMES
    assert len(stats["keyframe_indices"]) == stats["keyframes"]
    assert len(tsdf.calls) == stats["keyframes"]
    assert all(abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-6 for T in tsdf.calls)
    assert all(not np.allclose(T, np.eye(4)) for T in tsdf.calls)


def test_viewer_snapshots_and_debug_records(config, source, mask_cache, tmp_path):
    cfg = dict(config)
    cfg["visualization"] = {"enabled": True, "update_every_keyframes": 1}
    cfg["output"] = {
        "root": str(tmp_path / "out"),
        "save_keyframes": True,
        "save_debug": True,
    }
    tsdf = FakeTSDF()
    viewer = FakeViewer()
    stats = OfflinePipeline(
        tsdf, KeyframeSelector.from_config(cfg), cfg, viewer=viewer
    ).run(source, mask_cache, max_frames=FRAMES, verbose=False)

    keyframes = stats["keyframes"]
    assert keyframes >= 1
    assert len(stats["debug_records"]) == keyframes
    record = stats["debug_records"][0]
    assert record["keyframe_id"] == 1
    assert np.asarray(record["T_world_cam"]).shape == (4, 4)
    assert record["mask_area_px"] > 0
    assert record["tsdf_blocks"] > 0
    assert record["point_count"] is not None
    assert len(stats["snapshots"]) == keyframes
    assert all(Path(path).exists() for path in stats["snapshots"])
    assert len(viewer.updates) == keyframes
    assert viewer.closed


def test_viewer_quit_stops_early(config, source, mask_cache):
    cfg = dict(config)
    cfg["visualization"] = {"enabled": True, "update_every_keyframes": 1}
    viewer = FakeViewer(quit_after=1)
    stats = OfflinePipeline(
        FakeTSDF(),
        KeyframeSelector.from_config(cfg),
        cfg,
        viewer=viewer,
    ).run(source, mask_cache, max_frames=FRAMES, verbose=False)
    assert stats["quit_requested"]
    assert stats["keyframes"] == 1
    assert viewer.closed


def test_icp_refined_pose_used_for_integration(config, source, mask_cache):
    tsdf = FakeTSDF()
    refiner = FakeRefiner(accept=True)
    stats = OfflinePipeline(
        tsdf,
        KeyframeSelector.from_config(config),
        config,
        icp_refiner=refiner,
    ).run(source, mask_cache, max_frames=FRAMES, verbose=False)

    keyframes = stats["keyframes"]
    assert stats["icp_accepted"] == keyframes
    assert stats["icp_fallback"] == 0
    source_points, _ = refiner.calls[0]
    assert source_points.ndim == 2 and source_points.shape[1] == 3
    robot_pose = np.asarray(source.read_packet(0).T_world_cam)
    np.testing.assert_allclose(
        tsdf.calls[0][:3, 3], robot_pose[:3, 3] + FakeRefiner.OFFSET
    )
    record = stats["debug_records"][0]
    assert record["icp"]["accepted"] is True
    assert record["T_world_cam_used"] != record["T_world_cam"]


def test_icp_fallback_keeps_robot_pose(config, source, mask_cache):
    tsdf = FakeTSDF()
    refiner = FakeRefiner(accept=False)
    stats = OfflinePipeline(
        tsdf,
        KeyframeSelector.from_config(config),
        config,
        icp_refiner=refiner,
    ).run(source, mask_cache, max_frames=FRAMES, verbose=False)

    assert stats["icp_fallback"] == stats["keyframes"]
    assert stats["icp_accepted"] == 0
    record = stats["debug_records"][0]
    assert record["icp"]["accepted"] is False
    assert record["T_world_cam_used"] == record["T_world_cam"]


def test_no_refiner_means_no_icp_stats(config, source, mask_cache):
    stats = OfflinePipeline(
        FakeTSDF(), KeyframeSelector.from_config(config), config
    ).run(source, mask_cache, max_frames=FRAMES, verbose=False)
    assert stats["icp_accepted"] == 0 and stats["icp_fallback"] == 0
    assert all(record["icp"] is None for record in stats["debug_records"])


def test_missing_masks_skip_frames(config, source, tmp_path):
    tsdf = FakeTSDF()
    stats = OfflinePipeline(
        tsdf, KeyframeSelector.from_config(config), config
    ).run(
        source,
        MaskCache(tmp_path / "empty_masks"),
        max_frames=3,
        verbose=False,
    )
    assert stats["missing_mask_frames"] == [0, 1, 2]
    assert stats["keyframes"] == 0
    assert tsdf.calls == []
