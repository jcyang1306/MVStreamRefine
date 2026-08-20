"""OfflinePipeline loop test on real frames with a fake TSDF (no open3d)."""

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
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("fake ply")
        self.saved_point_clouds.append(Path(path))
        return Path(path)

    def block_count(self):
        return len(self.calls) * 7


class FakeRefiner:
    """Applies a fixed +1mm x-offset when accepting; robot pose otherwise."""

    OFFSET = np.array([0.001, 0.0, 0.0])

    def __init__(self, accept=True):
        self.accept = accept
        self.calls = []

    def refine(self, source_points, model_points, T_init):
        self.calls.append((np.asarray(source_points), np.asarray(model_points)))
        T_init = np.asarray(T_init, dtype=np.float64)
        if not self.accept:
            return ICPResult(
                success=False, T_world_cam_refined=T_init.copy(),
                fitness=0.1, inlier_rmse=0.02,
                translation_correction_m=0.0, rotation_correction_deg=0.0,
                reason="safety gates rejected the correction",
            )
        T_refined = T_init.copy()
        T_refined[:3, 3] += self.OFFSET
        return ICPResult(
            success=True, T_world_cam_refined=T_refined,
            fitness=0.9, inlier_rmse=0.002,
            translation_correction_m=0.001, rotation_correction_deg=0.0,
        )


class FakeViewer:
    def __init__(self, quit_after=None):
        self.updates = []
        self.quit_after = quit_after
        self.closed = False

    def update(self, point_cloud, T_world_cam2):
        self.updates.append((len(point_cloud.points), np.asarray(T_world_cam2)))
        return self.quit_after is None or len(self.updates) < self.quit_after

    def close(self):
        self.closed = True


@pytest.fixture(scope="module")
def config():
    config = load_config(REPO_ROOT / "configs" / "offline.yaml")
    config["dataset"]["root"] = str(REPO_ROOT / "data")
    # Small cam1 budget so the low-frequency logic is exercised in 6 frames:
    # initial anchor of 2 frames, then one integration every 3 frames.
    config["cam1"] = {"initial_frames": 2, "update_interval_frames": 3,
                      "max_integrations": 30}
    # Task 8 outputs are covered by dedicated tests below.
    config["output"] = {"root": "./output", "save_keyframes": False,
                        "save_debug": False}
    return config


@pytest.fixture(scope="module")
def source(config):
    return OfflineFrameSource.from_config(config)


@pytest.fixture()
def mask_cache(tmp_path, source):
    """Full-object masks for the first FRAMES frames of both cameras."""
    cache = MaskCache(tmp_path / "masks")
    mask = np.zeros((720, 1280), dtype=bool)
    mask[200:520, 400:880] = True
    for position in range(FRAMES):
        index = source.indices[position]
        cache.save(1, index, mask)
        cache.save(2, index, mask)
    return cache


def test_dual_camera_fusion_loop(config, source, mask_cache):
    tsdf = FakeTSDF()
    selector = KeyframeSelector.from_config(config)
    pipeline = OfflinePipeline(tsdf, selector, config)

    stats = pipeline.run(source, mask_cache, max_frames=FRAMES, verbose=False)

    assert stats["frames_processed"] == FRAMES
    assert stats["missing_mask_frames"] == []
    # cam1: frames 0,1 (initial) + frame 4 (interval 3 after frame 1).
    assert stats["cam1_integrations"] == 3
    # cam2: at least the first quality frame becomes a keyframe, and every
    # accepted keyframe integrated exactly once.
    assert stats["cam2_keyframes"] >= 1
    assert stats["cam2_keyframes"] + stats["cam2_rejected"] == FRAMES
    assert len(stats["keyframe_indices"]) == stats["cam2_keyframes"]
    assert len(tsdf.calls) == stats["cam1_integrations"] + stats["cam2_keyframes"]

    # cam1 always integrates at identity; cam2 at its per-frame world pose.
    identity_calls = [T for T in tsdf.calls if np.allclose(T, np.eye(4))]
    moving_calls = [T for T in tsdf.calls if not np.allclose(T, np.eye(4))]
    assert len(identity_calls) == stats["cam1_integrations"]
    assert len(moving_calls) == stats["cam2_keyframes"]
    for T in moving_calls:
        assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-6


def test_viewer_snapshots_and_debug_records(config, source, mask_cache, tmp_path):
    """Task 8: per-keyframe debug records, snapshots and viewer refreshes."""
    cfg = dict(config)
    cfg["visualization"] = {"enabled": True, "update_every_keyframes": 1}
    cfg["output"] = {"root": str(tmp_path / "out"), "save_keyframes": True,
                     "save_debug": True}
    tsdf = FakeTSDF()
    viewer = FakeViewer()
    pipeline = OfflinePipeline(tsdf, KeyframeSelector.from_config(cfg), cfg,
                               viewer=viewer)

    stats = pipeline.run(source, mask_cache, max_frames=FRAMES, verbose=False)

    kf = stats["cam2_keyframes"]
    assert kf >= 1
    # One debug record per keyframe, with the PLAN section 15 fields.
    assert len(stats["debug_records"]) == kf
    record = stats["debug_records"][0]
    assert record["keyframe_id"] == 1
    assert np.asarray(record["T_world_cam2"]).shape == (4, 4)
    assert record["mask_area_px"] > 0
    assert 0.0 <= record["valid_depth_ratio"] <= 1.0
    assert record["tsdf_blocks"] > 0
    assert record["point_count"] is not None  # update_every=1 -> always extracted
    # update_every_keyframes=1: one snapshot and one viewer refresh per keyframe.
    assert len(stats["snapshots"]) == kf
    assert all(Path(p).exists() for p in stats["snapshots"])
    assert Path(stats["snapshots"][0]).name == "model_kf_001.ply"
    assert len(viewer.updates) == kf
    assert viewer.closed
    assert not stats["quit_requested"]


def test_viewer_quit_stops_early(config, source, mask_cache):
    cfg = dict(config)
    cfg["visualization"] = {"enabled": True, "update_every_keyframes": 1}
    viewer = FakeViewer(quit_after=1)
    pipeline = OfflinePipeline(FakeTSDF(), KeyframeSelector.from_config(cfg), cfg,
                               viewer=viewer)

    stats = pipeline.run(source, mask_cache, max_frames=FRAMES, verbose=False)

    assert stats["quit_requested"]
    assert stats["cam2_keyframes"] == 1
    assert stats["frames_processed"] < FRAMES or stats["cam2_rejected"] == 0
    assert viewer.closed


def test_icp_refined_pose_used_for_integration(config, source, mask_cache):
    """Task 9: accepted ICP corrections shift the cam2 integration pose."""
    tsdf = FakeTSDF()
    refiner = FakeRefiner(accept=True)
    pipeline = OfflinePipeline(tsdf, KeyframeSelector.from_config(config), config,
                               icp_refiner=refiner)

    stats = pipeline.run(source, mask_cache, max_frames=FRAMES, verbose=False)

    kf = stats["cam2_keyframes"]
    assert kf >= 1
    assert stats["icp_accepted"] == kf
    assert stats["icp_fallback"] == 0
    assert len(refiner.calls) == kf
    # Refiner receives a camera-frame object cloud and the pre-frame model.
    source_points, model_points = refiner.calls[0]
    assert source_points.ndim == 2 and source_points.shape[1] == 3
    assert len(source_points) > 0
    # cam2 integrations carry the +1mm refined translation vs the robot pose.
    moving_calls = [T for T in tsdf.calls if not np.allclose(T, np.eye(4))]
    robot_poses = [np.asarray(source.read_packet(0).T_world_cam2)]
    np.testing.assert_allclose(
        moving_calls[0][:3, 3], robot_poses[0][:3, 3] + FakeRefiner.OFFSET
    )
    record = stats["debug_records"][0]
    assert record["icp"]["accepted"] is True
    assert record["T_world_cam2_used"] != record["T_world_cam2"]


def test_icp_fallback_keeps_robot_pose(config, source, mask_cache):
    tsdf = FakeTSDF()
    refiner = FakeRefiner(accept=False)
    pipeline = OfflinePipeline(tsdf, KeyframeSelector.from_config(config), config,
                               icp_refiner=refiner)

    stats = pipeline.run(source, mask_cache, max_frames=FRAMES, verbose=False)

    kf = stats["cam2_keyframes"]
    assert kf >= 1
    assert stats["icp_fallback"] == kf
    assert stats["icp_accepted"] == 0
    record = stats["debug_records"][0]
    assert record["icp"]["accepted"] is False
    assert record["icp"]["reason"]
    assert record["T_world_cam2_used"] == record["T_world_cam2"]


def test_no_refiner_means_no_icp_stats(config, source, mask_cache):
    stats = OfflinePipeline(
        FakeTSDF(), KeyframeSelector.from_config(config), config
    ).run(source, mask_cache, max_frames=FRAMES, verbose=False)
    assert stats["icp_accepted"] == 0 and stats["icp_fallback"] == 0
    assert all(r["icp"] is None for r in stats["debug_records"])


def test_missing_masks_skip_frames(config, source, tmp_path):
    tsdf = FakeTSDF()
    pipeline = OfflinePipeline(tsdf, KeyframeSelector.from_config(config), config)
    empty_cache = MaskCache(tmp_path / "empty_masks")

    stats = pipeline.run(source, empty_cache, max_frames=3, verbose=False)

    assert stats["missing_mask_frames"] == [0, 1, 2]
    assert stats["cam1_integrations"] == 0
    assert stats["cam2_keyframes"] == 0
    assert tsdf.calls == []
