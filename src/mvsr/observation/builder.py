"""`ObservationBuilder.process(SensorFrame) -> ObjectObservation` (DATA_CONTRACTS.md port, P2).

Stateful over a frame sequence (tracks), so frames must be fed in order; the same frames,
mask revision and `configs/observation.yaml` give identical observations.
- mask: from the offline cache (`MaskCache`); a frame without a cached mask raises.
  `EMPTY` masks give an observation with no geometry and no keypoints (tracks occlude).
- geometry: `depth.depth_geometry` (no invented depth);
- keypoints: visible tracks (`tracks.TrackManager`) inside the mask eroded by
  `features.safe_erode_px`; `keypoint_depth_m` is NaN unless the pixel is in `geometry_mask`.
- `mask_prob` is None (the cache stores the binary mask and a per-frame score).
The latest per-frame track states are in `last_tracks`.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from mvsr.core.errors import MvsrError
from mvsr.core.types import CalibrationBundle, ObjectObservation, SensorFrame
from mvsr.observation.config import ObservationConfig
from mvsr.observation.depth import depth_geometry, erode, sample_depth
from mvsr.observation.features import FeatureExtractor, make_extractor
from mvsr.observation.masks import MaskCache, MaskStatus
from mvsr.observation.tracks import FrameTracks, TrackManager


class ObservationError(MvsrError):
    """A frame cannot be observed (no cached mask, wrong camera)."""


class ObservationBuilder:
    def __init__(
        self,
        masks: MaskCache,
        bundle: CalibrationBundle,
        cfg: ObservationConfig,
        *,
        extractor: FeatureExtractor | None = None,
        instance_id: int = 1,
        verify_masks: bool = True,
    ) -> None:
        if bundle.camera_id != masks.camera_id:
            raise ObservationError(
                f"bundle camera {bundle.camera_id} != mask camera {masks.camera_id}"
            )
        self.masks = masks
        self.bundle = bundle
        self.cfg = cfg
        self.extractor = extractor or make_extractor(cfg.features)
        self.tracks = TrackManager(cfg.tracks, self.extractor)
        self.instance_id = instance_id
        self.verify_masks = verify_masks
        self.last_tracks: FrameTracks | None = None
        self._last_frame_id: int | None = None

    def process(self, frame: SensorFrame) -> ObjectObservation:
        fid = frame.header.frame_id
        if frame.header.camera_id != self.masks.camera_id:
            raise ObservationError(
                f"frame camera {frame.header.camera_id} != mask camera {self.masks.camera_id}"
            )
        if frame.header.calibration_id != self.bundle.calibration_id:
            raise ObservationError(
                f"frame calibration {frame.header.calibration_id} != builder bundle {self.bundle.calibration_id}"
            )
        if self._last_frame_id is not None and fid <= self._last_frame_id:
            raise ObservationError(
                f"frames must be fed in increasing order ({fid} after {self._last_frame_id})"
            )
        if fid not in self.masks:
            raise ObservationError(f"no mask for frame {fid} in revision {self.masks.revision}")
        rec = self.masks.records[fid]
        mask = self.masks.mask(fid, verify=self.verify_masks)
        if mask.shape != frame.depth_raw_u16.shape:
            raise ObservationError(f"mask {mask.shape} != image {frame.depth_raw_u16.shape}")
        empty = rec.status is MaskStatus.EMPTY
        use = np.zeros_like(mask) if empty else mask
        geom = depth_geometry(
            frame.depth_m_f32, frame.valid_depth_mask, use, self.bundle, self.cfg.depth
        )
        gray = np.asarray(cv2.cvtColor(frame.color_rgb_u8, cv2.COLOR_RGB2GRAY), dtype=np.uint8)
        safe = erode(use, self.cfg.features.safe_erode_px)
        tr = self.tracks.update(fid, gray, safe)
        self.last_tracks = tr
        self._last_frame_id = fid
        kp_depth = sample_depth(frame.depth_m_f32, geom.geometry_mask, tr.uv)
        counts = tr.counts()
        quality: dict[str, Any] = {
            "mask_area_px": int(mask.sum()),
            "valid_depth_ratio": geom.stats["valid_depth_ratio"],
            "n_keypoints": int(tr.uv.shape[0]),
            "mask_score": rec.score,
            "mask_status": rec.status.value,
            "geometry_ratio": geom.stats["geometry_ratio"],
            "geometry_px": geom.stats["geometry_px"],
            "keypoints_with_depth": int(np.isfinite(kp_depth).sum()),
            "tracks_new": counts["NEW"],
            "tracks_tracked": counts["TRACKED"],
            "tracks_occluded": counts["OCCLUDED"],
            "tracks_retired": counts["RETIRED"],
            "depth": geom.stats,
        }
        return ObjectObservation(
            header=frame.header,
            instance_id=self.instance_id,
            mask_revision=self.masks.revision,
            mask_binary=mask,
            mask_prob=None,
            geometry_mask=geom.geometry_mask,
            pixel_uv=geom.pixel_uv,
            points_cam_m=geom.points_cam_m,
            normals_cam=geom.normals_cam,
            keypoints_uv=tr.uv,
            keypoint_depth_m=kp_depth,
            descriptors=tr.descriptors,
            descriptor_type=self.extractor.descriptor_type,
            track_ids=tr.track_ids,
            quality=quality,
        )
