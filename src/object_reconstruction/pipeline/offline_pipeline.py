"""Single wrist-camera object reconstruction in the robot-base world frame.

Per frame packet:
    cam (wrist, eye-in-hand): integrated only on keyframes accepted by
    KeyframeSelector (pose increment + mask quality gates) at known poses
    T_world_cam, where WORLD is the robot base frame.

The per-frame work (preprocess -> keyframe gate -> optional ICP -> TSDF
integration -> debug record) lives in ReconstructionEngine, shared with the
realtime pipeline. This class owns the offline loop: mask-cache lookup,
stats aggregation and the every-N-keyframes viewer/snapshot refresh.

Task 8: every ``visualization.update_every_keyframes`` accepted keyframes the
point cloud is extracted once (never per input frame) to refresh the optional
viewer and, when ``output.save_keyframes`` is on, save an incremental snapshot
model_kf_XXX.ply. Per-keyframe debug records (PLAN section 15) are collected
in ``stats["debug_records"]``.

Task 9: when an ICP refiner is injected, each accepted cam keyframe is
refined against the model extracted BEFORE integrating that frame (never
register a frame against itself), with the robot pose as the initial guess
and as the fallback whenever the refiner's safety gates reject the result.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..data.offline_source import OfflineFrameSource
from ..fusion.keyframe_selector import KeyframeSelector
from ..segmentation.mask_cache import MaskCache
from .reconstruction_engine import ReconstructionEngine, format_icp, point_count

logger = logging.getLogger(__name__)


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
        self.config = config
        self.viewer = viewer
        self.engine = ReconstructionEngine(
            tsdf, keyframe_selector, config, icp_refiner=icp_refiner
        )
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

            result = self.engine.process(packet, mask)
            if not result.integrated:
                stats["rejected"] += 1
                continue

            stats["keyframes"] = self.engine.keyframes
            stats["keyframe_indices"].append(packet.index)
            stats["icp_accepted"] = self.engine.icp_accepted
            stats["icp_fallback"] = self.engine.icp_fallback
            keyframe_log(
                "[frame %3d] keyframe #%d  mask_area=%d  "
                "valid_depth_ratio=%.3f%s",
                packet.index, result.keyframe_id,
                result.rgbd.mask_area_px, result.rgbd.valid_depth_ratio,
                format_icp(result.icp_info),
            )
            stats["debug_records"].append(result.debug_record)
            if not self._refresh(packet, stats, result.debug_record):
                stats["quit_requested"] = True
                logger.info("viewer quit requested; stopping early")
                break

        if self.viewer is not None:
            self.viewer.close()
        return stats

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
        record["point_count"] = point_count(point_cloud)
        if self.snapshot_dir is not None:
            path = self.snapshot_dir / f"model_kf_{kf:03d}.ply"
            self.tsdf.save_point_cloud(path)
            stats["snapshots"].append(str(path))
            logger.info("snapshot saved: %s (%s points)", path, record["point_count"])
        if self.viewer is not None:
            return bool(self.viewer.update(point_cloud, packet.T_world_cam))
        return True
