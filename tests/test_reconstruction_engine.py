"""ReconstructionEngine tests with synthetic packets and fake geometry."""

import numpy as np
import pytest

from object_reconstruction.data.models import CameraFrame, CameraIntrinsics, FramePacket
from object_reconstruction.fusion.keyframe_selector import KeyframeSelector
from object_reconstruction.pipeline.reconstruction_engine import ReconstructionEngine
from object_reconstruction.registration.icp_refiner import ICPResult

WIDTH, HEIGHT = 64, 48

CONFIG = {
    "depth": {"scale": 1000.0, "min_m": 0.1, "max_m": 2.0},
    "mask": {"erosion_px": 0},
}


class FakeTSDF:
    def __init__(self):
        self.calls = []

    def integrate(self, rgb, depth, intrinsics, T_world_cam):
        assert rgb.dtype == np.uint8 and depth.dtype == np.uint16
        self.calls.append(np.asarray(T_world_cam))

    def extract_point_cloud(self):
        if not self.calls:
            raise RuntimeError("empty model")
        return FakePointCloud(len(self.calls) * 10)

    def block_count(self):
        return len(self.calls) * 7


class FakePointCloud:
    def __init__(self, n_points):
        self.points = [np.zeros(3)] * n_points


class FakeRefiner:
    def __init__(self, accept=True):
        self.accept = accept
        self.calls = 0

    def refine(self, source_points, model_points, T_init):
        self.calls += 1
        T_init = np.asarray(T_init, dtype=np.float64)
        T_refined = T_init.copy()
        if self.accept:
            T_refined[:3, 3] += [0.001, 0.0, 0.0]
        return ICPResult(
            success=self.accept,
            T_world_cam_refined=T_refined,
            fitness=0.9 if self.accept else 0.1,
            inlier_rmse=0.002,
            translation_correction_m=0.001,
            rotation_correction_deg=0.0,
            reason="" if self.accept else "safety gates rejected the correction",
        )


def make_packet(index=0, tx=0.0):
    rgb = np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)
    depth = np.full((HEIGHT, WIDTH), 500, dtype=np.uint16)  # 0.5 m
    T = np.eye(4)
    T[0, 3] = tx
    return FramePacket(
        index=index,
        timestamp=None,
        cam=CameraFrame(
            rgb=rgb,
            depth=depth,
            intrinsics=CameraIntrinsics(WIDTH, HEIGHT, 50.0, 50.0, 32.0, 24.0),
        ),
        T_world_cam=T,
    )


def full_mask():
    return np.ones((HEIGHT, WIDTH), dtype=bool)


def make_engine(icp_refiner=None, min_mask_area_px=10):
    selector = KeyframeSelector(
        translation_m=0.02, rotation_deg=5.0,
        min_mask_area_px=min_mask_area_px, min_valid_depth_ratio=0.5,
    )
    return ReconstructionEngine(FakeTSDF(), selector, CONFIG, icp_refiner=icp_refiner)


def test_first_good_frame_becomes_keyframe():
    engine = make_engine()
    result = engine.process(make_packet(), full_mask())
    assert result.integrated
    assert result.keyframe_id == 1
    assert engine.keyframes == 1
    record = result.debug_record
    assert record["frame_id"] == 0 and record["keyframe_id"] == 1
    assert record["mask_area_px"] == WIDTH * HEIGHT
    assert record["icp"] is None
    assert record["tsdf_blocks"] == 7


def test_static_pose_is_rejected_after_first_keyframe():
    engine = make_engine()
    assert engine.process(make_packet(0), full_mask()).integrated
    result = engine.process(make_packet(1, tx=0.001), full_mask())
    assert not result.integrated
    assert result.keyframe_id is None
    assert engine.keyframes == 1
    # Moving beyond the translation threshold is accepted again.
    assert engine.process(make_packet(2, tx=0.05), full_mask()).integrated


def test_small_mask_is_rejected():
    engine = make_engine(min_mask_area_px=10_000_000)
    result = engine.process(make_packet(), full_mask())
    assert not result.integrated
    assert engine.keyframes == 0


def test_none_pose_forbids_fusion():
    engine = make_engine()
    packet = make_packet()
    packet.T_world_cam = None
    with pytest.raises(RuntimeError, match="unconfirmed"):
        engine.process(packet, full_mask())


def test_icp_skipped_on_empty_model_then_counted():
    refiner = FakeRefiner(accept=True)
    engine = make_engine(icp_refiner=refiner)
    first = engine.process(make_packet(0), full_mask())
    assert first.icp_info == {"skipped": "model is still empty"}
    assert refiner.calls == 0

    second = engine.process(make_packet(1, tx=0.05), full_mask())
    assert second.icp_info["accepted"] is True
    assert refiner.calls == 1
    assert engine.icp_accepted == 1 and engine.icp_fallback == 0
    # The refined pose (not the robot pose) is what was integrated.
    np.testing.assert_allclose(
        engine.tsdf.calls[1][:3, 3], [0.05 + 0.001, 0.0, 0.0]
    )


def test_icp_fallback_keeps_robot_pose():
    engine = make_engine(icp_refiner=FakeRefiner(accept=False))
    engine.process(make_packet(0), full_mask())
    result = engine.process(make_packet(1, tx=0.05), full_mask())
    assert result.icp_info["accepted"] is False
    assert engine.icp_fallback == 1
    np.testing.assert_allclose(engine.tsdf.calls[1][:3, 3], [0.05, 0.0, 0.0])
