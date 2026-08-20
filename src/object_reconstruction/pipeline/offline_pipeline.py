"""First-version dual-camera fusion loop (PLAN sections 11-15, Tasks 6-8).

Per frame packet:
    cam1 (head, fixed = WORLD): low-frequency anchor integration - the first
    ``cam1.initial_frames`` valid frames, then one frame every
    ``cam1.update_interval_frames``, capped at ``cam1.max_integrations``.
    cam2 (wrist, eye-in-hand): integrated only on keyframes accepted by
    KeyframeSelector (pose increment + mask quality gates) at known poses
    T_world_cam2.

Task 8 additions: every ``visualization.update_every_keyframes`` accepted
keyframes the point cloud is extracted once (never per input frame) to
refresh the optional viewer and, when ``output.save_keyframes`` is on, save
an incremental snapshot model_kf_XXX.ply. Per-keyframe debug records (PLAN
section 15) are collected in ``stats["debug_records"]``.

Task 9: when an ICP refiner is injected, each accepted cam2 keyframe is
refined against the model extracted BEFORE integrating that frame (never
register a frame against itself), with the robot pose as the initial guess
and as the fallback whenever the refiner's safety gates reject the result.

The TSDF volume, selector, viewer and refiner are injected so the loop
itself stays free of open3d imports and unit-testable with fakes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..data.offline_source import OfflineFrameSource
from ..fusion.keyframe_selector import KeyframeSelector
from ..geometry.pointcloud import depth_to_pointcloud
from ..geometry.rgbd import preprocess_object_rgbd
from ..segmentation.mask_cache import MaskCache

logger = logging.getLogger(__name__)


def _point_count(point_cloud: Any) -> int | None:
    try:
        return int(point_cloud.point.positions.shape[0])  # tensor point cloud
    except AttributeError:
        pass
    try:
        return len(point_cloud.points)  # legacy point cloud
    except (AttributeError, TypeError):
        return None


def _point_positions(point_cloud: Any) -> np.ndarray:
    try:
        return point_cloud.point.positions.numpy()  # tensor point cloud
    except AttributeError:
        return np.asarray(point_cloud.points)  # legacy point cloud


class OfflinePipeline:
    def __init__(
        self,
        tsdf: Any,
        keyframe_selector: KeyframeSelector,
        config: dict[str, Any],
        viewer: Any = None,
        icp_refiner: Any = None,
    ) -> None:
        self.tsdf = tsdf
        self.selector = keyframe_selector
        self.config = config
        self.viewer = viewer
        self.icp_refiner = icp_refiner
        self.depth_scale = float(config["depth"]["scale"])
        cam1 = config.get("cam1", {})
        self.cam1_initial_frames = int(cam1.get("initial_frames", 10))
        self.cam1_update_interval = int(cam1.get("update_interval_frames", 30))
        self.cam1_max_integrations = int(cam1.get("max_integrations", 30))
        viz = config.get("visualization", {})
        self.update_every_keyframes = max(1, int(viz.get("update_every_keyframes", 5)))
        output = config.get("output", {})
        self.snapshot_dir: Path | None = None
        if bool(output.get("save_keyframes", False)):
            self.snapshot_dir = Path(output.get("root", "./output")) / "pointcloud"

    def run(
        self,
        source: OfflineFrameSource,
        mask_cache: MaskCache,
        max_frames: int | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "frames_processed": 0,
            "cam1_integrations": 0,
            "cam2_keyframes": 0,
            "cam2_rejected": 0,
            "missing_mask_frames": [],
            "keyframe_indices": [],
            "debug_records": [],
            "snapshots": [],
            "quit_requested": False,
            "icp_accepted": 0,
            "icp_fallback": 0,
        }
        last_cam1_position: int | None = None
        keyframe_log = logger.info if verbose else logger.debug

        total = len(source) if max_frames is None else min(max_frames, len(source))
        for position in range(total):
            packet = source.read_packet(position)
            stats["frames_processed"] += 1

            mask1 = mask_cache.get_cam1(packet.index)
            mask2 = mask_cache.get_cam2(packet.index)
            if mask1 is None or mask2 is None:
                stats["missing_mask_frames"].append(packet.index)
                continue

            # -- cam1: fixed view, low-frequency anchor (PLAN section 12) ----
            if self._cam1_due(stats["cam1_integrations"], last_cam1_position, position):
                rgbd1 = preprocess_object_rgbd(
                    packet.cam1.rgb, packet.cam1.depth, mask1, self.config
                )
                if rgbd1.valid_depth_px > 0:
                    self.tsdf.integrate(
                        rgbd1.rgb, rgbd1.depth, packet.cam1.intrinsics,
                        T_world_cam=np.eye(4),
                    )
                    stats["cam1_integrations"] += 1
                    last_cam1_position = position

            # -- cam2: moving view, keyframe-gated (PLAN sections 11/13) -----
            if packet.T_world_cam2 is None:
                raise RuntimeError(
                    "T_world_cam2 is None: conventions are unconfirmed, fusion "
                    "is forbidden"
                )
            rgbd2 = preprocess_object_rgbd(
                packet.cam2.rgb, packet.cam2.depth, mask2, self.config
            )
            if self.selector.should_add(
                packet.T_world_cam2, rgbd2.mask_area_px, rgbd2.valid_depth_ratio
            ):
                # ICP must run BEFORE this frame is integrated (PLAN section 16).
                T_used, icp_info = self._refine_pose(packet, rgbd2, stats)
                self.tsdf.integrate(
                    rgbd2.rgb, rgbd2.depth, packet.cam2.intrinsics,
                    T_world_cam=T_used,
                )
                stats["cam2_keyframes"] += 1
                stats["keyframe_indices"].append(packet.index)
                keyframe_log(
                    "[frame %3d] cam2 keyframe #%d  mask_area=%d  "
                    "valid_depth_ratio=%.3f%s",
                    packet.index, stats["cam2_keyframes"],
                    rgbd2.mask_area_px, rgbd2.valid_depth_ratio,
                    self._format_icp(icp_info),
                )
                record = self._make_debug_record(packet, rgbd2, stats)
                record["T_world_cam2_used"] = np.asarray(T_used).tolist()
                record["icp"] = icp_info
                stats["debug_records"].append(record)
                if not self._refresh(packet, stats, record):
                    stats["quit_requested"] = True
                    logger.info("viewer quit requested; stopping early")
                    break
            else:
                stats["cam2_rejected"] += 1

        if self.viewer is not None:
            self.viewer.close()
        return stats

    def _refine_pose(
        self, packet: Any, rgbd2: Any, stats: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any] | None]:
        """ICP refinement of the robot pose (Task 9); robot pose is the fallback."""
        T_robot = np.asarray(packet.T_world_cam2, dtype=np.float64)
        if self.icp_refiner is None:
            return T_robot, None

        try:
            model = self.tsdf.extract_point_cloud()
        except RuntimeError:
            return T_robot, {"skipped": "model is still empty"}
        model_points = _point_positions(model)

        # rgbd2.depth is already masked and range-filtered by preprocessing.
        source_points, _ = depth_to_pointcloud(
            rgbd2.depth, packet.cam2.intrinsics, self.depth_scale
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
        stats["icp_accepted" if result.success else "icp_fallback"] += 1
        return np.asarray(result.T_world_cam_refined, dtype=np.float64), info

    @staticmethod
    def _format_icp(icp_info: dict[str, Any] | None) -> str:
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

    def _make_debug_record(
        self, packet: Any, rgbd2: Any, stats: dict[str, Any]
    ) -> dict[str, Any]:
        """One debug record per accepted keyframe (PLAN section 15)."""
        record: dict[str, Any] = {
            "frame_id": int(packet.index),
            "keyframe_id": int(stats["cam2_keyframes"]),
            "T_world_cam2": np.asarray(packet.T_world_cam2).tolist(),
            "mask_area_px": int(rgbd2.mask_area_px),
            "valid_depth_ratio": float(rgbd2.valid_depth_ratio),
            "point_count": None,
        }
        if hasattr(self.tsdf, "block_count"):
            record["tsdf_blocks"] = int(self.tsdf.block_count())
        return record

    def _refresh(
        self, packet: Any, stats: dict[str, Any], record: dict[str, Any]
    ) -> bool:
        """Extract once every N keyframes for viewer + snapshot; True = keep going."""
        kf = stats["cam2_keyframes"]
        if kf % self.update_every_keyframes != 0:
            return True
        if self.viewer is None and self.snapshot_dir is None:
            return True
        point_cloud = self.tsdf.extract_point_cloud()
        record["point_count"] = _point_count(point_cloud)
        if self.snapshot_dir is not None:
            path = self.snapshot_dir / f"model_kf_{kf:03d}.ply"
            self.tsdf.save_point_cloud(path)
            stats["snapshots"].append(str(path))
            logger.info("snapshot saved: %s (%s points)", path, record["point_count"])
        if self.viewer is not None:
            return bool(self.viewer.update(point_cloud, packet.T_world_cam2))
        return True

    def _cam1_due(
        self,
        cam1_count: int,
        last_position: int | None,
        position: int,
    ) -> bool:
        if cam1_count >= self.cam1_max_integrations:
            return False
        if cam1_count < self.cam1_initial_frames:
            return True
        return position - last_position >= self.cam1_update_interval
