"""First-version dual-camera fusion loop (PLAN sections 11-13, Tasks 6-7).

Per frame packet:
    cam1 (head, fixed = WORLD): low-frequency anchor integration - the first
    ``cam1.initial_frames`` valid frames, then one frame every
    ``cam1.update_interval_frames``, capped at ``cam1.max_integrations``.
    cam2 (wrist, eye-in-hand): integrated only on keyframes accepted by
    KeyframeSelector (pose increment + mask quality gates) at known poses
    T_world_cam2. No ICP in this version.

The TSDF volume and selector are injected so the loop itself stays free of
open3d imports and unit-testable with fakes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..data.offline_source import OfflineFrameSource
from ..fusion.keyframe_selector import KeyframeSelector
from ..geometry.rgbd import preprocess_object_rgbd
from ..segmentation.mask_cache import MaskCache


class OfflinePipeline:
    def __init__(
        self,
        tsdf: Any,
        keyframe_selector: KeyframeSelector,
        config: dict[str, Any],
    ) -> None:
        self.tsdf = tsdf
        self.selector = keyframe_selector
        self.config = config
        cam1 = config.get("cam1", {})
        self.cam1_initial_frames = int(cam1.get("initial_frames", 10))
        self.cam1_update_interval = int(cam1.get("update_interval_frames", 30))
        self.cam1_max_integrations = int(cam1.get("max_integrations", 30))

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
        }
        last_cam1_position: int | None = None

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
                self.tsdf.integrate(
                    rgbd2.rgb, rgbd2.depth, packet.cam2.intrinsics,
                    T_world_cam=packet.T_world_cam2,
                )
                stats["cam2_keyframes"] += 1
                stats["keyframe_indices"].append(packet.index)
                if verbose:
                    print(
                        f"[frame {packet.index:3d}] cam2 keyframe "
                        f"#{stats['cam2_keyframes']}  "
                        f"mask_area={rgbd2.mask_area_px}  "
                        f"valid_depth_ratio={rgbd2.valid_depth_ratio:.3f}"
                    )
            else:
                stats["cam2_rejected"] += 1

        return stats

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
