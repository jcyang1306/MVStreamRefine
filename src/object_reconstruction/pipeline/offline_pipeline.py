"""Single wrist-camera object reconstruction in the robot-base world frame.

Per frame packet:
    cam (wrist, eye-in-hand): integrated only on keyframes accepted by
    KeyframeSelector (pose increment + mask quality gates) at known poses
    T_world_cam, where WORLD is the robot base frame.

Task 8 additions: every ``visualization.update_every_keyframes`` accepted
keyframes the point cloud is extracted once (never per input frame) to
refresh the optional viewer and, when ``output.save_keyframes`` is on, save
an incremental snapshot model_kf_XXX.ply. Per-keyframe debug records (PLAN
section 15) are collected in ``stats["debug_records"]``.

Task 9: when an ICP refiner is injected, each accepted cam keyframe is
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
            "keyframes": 0,
            "rejected": 0,
            "missing_mask_frames": [],
            "keyframe_indices": [],
            "debug_records": [],
            "snapshots": [],
            "quit_requested": False,
            "icp_accepted": 0,
            "icp_fallback": 0,
        }
        keyframe_log = logger.info if verbose else logger.debug

        total = len(source) if max_frames is None else min(max_frames, len(source))
        for position in range(total):
            packet = source.read_packet(position)
            stats["frames_processed"] += 1

            mask = mask_cache.get(packet.index)
            if mask is None:
                stats["missing_mask_frames"].append(packet.index)
                continue

            if packet.T_world_cam is None:
                raise RuntimeError(
                    "T_world_cam is None: conventions are unconfirmed, fusion "
                    "is forbidden"
                )
            rgbd = preprocess_object_rgbd(
                packet.cam.rgb, packet.cam.depth, mask, self.config
            )
            if self.selector.should_add(
                packet.T_world_cam, rgbd.mask_area_px, rgbd.valid_depth_ratio
            ):
                # ICP must run BEFORE this frame is integrated (PLAN section 16).
                T_used, icp_info = self._refine_pose(packet, rgbd, stats)
                self.tsdf.integrate(
                    rgbd.rgb, rgbd.depth, packet.cam.intrinsics,
                    T_world_cam=T_used,
                )
                stats["keyframes"] += 1
                stats["keyframe_indices"].append(packet.index)
                keyframe_log(
                    "[frame %3d] keyframe #%d  mask_area=%d  "
                    "valid_depth_ratio=%.3f%s",
                    packet.index, stats["keyframes"],
                    rgbd.mask_area_px, rgbd.valid_depth_ratio,
                    self._format_icp(icp_info),
                )
                record = self._make_debug_record(packet, rgbd, stats)
                record["T_world_cam_used"] = np.asarray(T_used).tolist()
                record["icp"] = icp_info
                stats["debug_records"].append(record)
                if not self._refresh(packet, stats, record):
                    stats["quit_requested"] = True
                    logger.info("viewer quit requested; stopping early")
                    break
            else:
                stats["rejected"] += 1

        if self.viewer is not None:
            self.viewer.close()
        return stats

    def _refine_pose(
        self, packet: Any, rgbd: Any, stats: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any] | None]:
        """ICP refinement of the robot pose (Task 9); robot pose is the fallback."""
        T_robot = np.asarray(packet.T_world_cam, dtype=np.float64)
        if self.icp_refiner is None:
            return T_robot, None

        try:
            model = self.tsdf.extract_point_cloud()
        except RuntimeError:
            return T_robot, {"skipped": "model is still empty"}
        model_points = _point_positions(model)

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
        self, packet: Any, rgbd: Any, stats: dict[str, Any]
    ) -> dict[str, Any]:
        """One debug record per accepted keyframe (PLAN section 15)."""
        record: dict[str, Any] = {
            "frame_id": int(packet.index),
            "keyframe_id": int(stats["keyframes"]),
            "T_world_cam": np.asarray(packet.T_world_cam).tolist(),
            "mask_area_px": int(rgbd.mask_area_px),
            "valid_depth_ratio": float(rgbd.valid_depth_ratio),
            "point_count": None,
        }
        if hasattr(self.tsdf, "block_count"):
            record["tsdf_blocks"] = int(self.tsdf.block_count())
        return record

    def _refresh(
        self, packet: Any, stats: dict[str, Any], record: dict[str, Any]
    ) -> bool:
        """Extract once every N keyframes for viewer + snapshot; True = keep going."""
        kf = stats["keyframes"]
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
            return bool(self.viewer.update(point_cloud, packet.T_world_cam))
        return True
