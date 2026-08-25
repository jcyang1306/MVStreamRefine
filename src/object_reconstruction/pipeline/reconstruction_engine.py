"""Per-frame reconstruction core shared by the offline and realtime pipelines.

Given a FramePacket (with a confirmed T_world_cam) and an object mask, the
engine runs the same steps in both modes:

    mask preprocessing -> keyframe gate (pose increment + mask quality)
    -> optional ICP refinement (robot pose is the initial guess and fallback)
    -> TSDF integration -> per-keyframe debug record

The TSDF volume, keyframe selector and ICP refiner are injected so the engine
stays free of open3d imports and unit-testable with fakes. Pipelines own the
loop, the mask source (cache vs. live SAM2 tracking) and the viewer refresh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..fusion.keyframe_selector import KeyframeSelector
from ..geometry.pointcloud import depth_to_pointcloud
from ..geometry.rgbd import MaskedRGBD, preprocess_object_rgbd


def point_count(point_cloud: Any) -> int | None:
    try:
        return int(point_cloud.point.positions.shape[0])  # tensor point cloud
    except AttributeError:
        pass
    try:
        return len(point_cloud.points)  # legacy point cloud
    except (AttributeError, TypeError):
        return None


def point_positions(point_cloud: Any) -> np.ndarray:
    try:
        return point_cloud.point.positions.numpy()  # tensor point cloud
    except AttributeError:
        return np.asarray(point_cloud.points)  # legacy point cloud


def format_icp(icp_info: dict[str, Any] | None) -> str:
    """One-line human-readable ICP summary for keyframe log lines."""
    if icp_info is None:
        return ""
    if "skipped" in icp_info:
        return f"  icp=skipped ({icp_info['skipped']})"
    if icp_info["accepted"]:
        return (
            f"  icp=accepted dt={icp_info['translation_correction_m'] * 1000:.1f}mm "
            f"dr={icp_info['rotation_correction_deg']:.2f}deg "
            f"fitness={icp_info['fitness']:.2f}"
        )
    return (
        f"  icp=fallback ({icp_info['reason']}; "
        f"fitness={icp_info['fitness']:.2f} "
        f"rmse={icp_info['inlier_rmse'] * 1000:.1f}mm "
        f"dt={icp_info['translation_correction_m'] * 1000:.1f}mm "
        f"dr={icp_info['rotation_correction_deg']:.2f}deg)"
    )


@dataclass
class EngineResult:
    """Outcome of feeding one (packet, mask) pair to the engine."""

    integrated: bool
    rgbd: MaskedRGBD
    keyframe_id: int | None = None  # 1-based, set when integrated
    T_world_cam_used: np.ndarray | None = None
    icp_info: dict[str, Any] | None = None
    debug_record: dict[str, Any] | None = None


class ReconstructionEngine:
    def __init__(
        self,
        tsdf: Any,
        keyframe_selector: KeyframeSelector,
        config: dict[str, Any],
        icp_refiner: Any = None,
    ) -> None:
        self.tsdf = tsdf
        self.selector = keyframe_selector
        self.config = config
        self.icp_refiner = icp_refiner
        self.depth_scale = float(config["depth"]["scale"])
        self.keyframes = 0
        self.icp_accepted = 0
        self.icp_fallback = 0

    def process(self, packet: Any, mask: np.ndarray) -> EngineResult:
        """Run one frame through the keyframe gate and (maybe) integrate it."""
        if packet.T_world_cam is None:
            raise RuntimeError(
                "T_world_cam is None: conventions are unconfirmed, fusion "
                "is forbidden"
            )
        rgbd = preprocess_object_rgbd(
            packet.cam.rgb, packet.cam.depth, mask, self.config
        )
        if not self.selector.should_add(
            packet.T_world_cam, rgbd.mask_area_px, rgbd.valid_depth_ratio
        ):
            return EngineResult(integrated=False, rgbd=rgbd)

        # ICP must run BEFORE this frame is integrated (PLAN section 16).
        T_used, icp_info = self._refine_pose(packet, rgbd)
        self.tsdf.integrate(
            rgbd.rgb, rgbd.depth, packet.cam.intrinsics, T_world_cam=T_used
        )
        self.keyframes += 1

        record: dict[str, Any] = {
            "frame_id": int(packet.index),
            "keyframe_id": int(self.keyframes),
            "T_world_cam": np.asarray(packet.T_world_cam).tolist(),
            "T_world_cam_used": np.asarray(T_used).tolist(),
            "icp": icp_info,
            "mask_area_px": int(rgbd.mask_area_px),
            "valid_depth_ratio": float(rgbd.valid_depth_ratio),
            "point_count": None,
        }
        if hasattr(self.tsdf, "block_count"):
            record["tsdf_blocks"] = int(self.tsdf.block_count())

        return EngineResult(
            integrated=True,
            rgbd=rgbd,
            keyframe_id=self.keyframes,
            T_world_cam_used=T_used,
            icp_info=icp_info,
            debug_record=record,
        )

    def _refine_pose(
        self, packet: Any, rgbd: MaskedRGBD
    ) -> tuple[np.ndarray, dict[str, Any] | None]:
        """ICP refinement of the robot pose (Task 9); robot pose is the fallback."""
        T_robot = np.asarray(packet.T_world_cam, dtype=np.float64)
        if self.icp_refiner is None:
            return T_robot, None

        try:
            model = self.tsdf.extract_point_cloud()
        except RuntimeError:
            return T_robot, {"skipped": "model is still empty"}
        model_points = point_positions(model)

        # rgbd.depth is already masked and range-filtered by preprocessing.
        source_points, _ = depth_to_pointcloud(
            rgbd.depth, packet.cam.intrinsics, self.depth_scale
        )
        result = self.icp_refiner.refine(source_points, model_points, T_robot)
        info = {
            "accepted": bool(result.success),
            "fitness": float(result.fitness),
            "inlier_rmse": float(result.inlier_rmse),
            "translation_correction_m": float(result.translation_correction_m),
            "rotation_correction_deg": float(result.rotation_correction_deg),
            "reason": result.reason,
        }
        if result.success:
            self.icp_accepted += 1
        else:
            self.icp_fallback += 1
        return np.asarray(result.T_world_cam_refined, dtype=np.float64), info
