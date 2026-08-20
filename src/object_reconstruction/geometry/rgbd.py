"""Masked RGB-D preprocessing (PLAN section 9, Task 4).

Turns a raw camera frame plus a cached SAM mask into fusion-ready RGB-D:

    SAM mask -> optional erosion -> depth range filter -> invalid depth
    filter -> depth[~mask] = 0

Depth stays in its raw uint16 units (millimeters here); TSDF integration
consumes it together with depth.scale. RGB and depth must already be pixel
aligned (verified for this dataset in Task 1), which is asserted via shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MaskedRGBD:
    rgb: np.ndarray          # uint8 H x W x 3, untouched
    depth: np.ndarray        # same dtype as input, zeroed outside object/range
    mask: np.ndarray         # bool H x W, eroded object mask
    mask_area_px: int        # pixels in the eroded mask
    valid_depth_px: int      # mask pixels that survived the depth filters
    valid_depth_ratio: float # valid_depth_px / mask_area_px (0.0 if empty mask)


def erode_mask(mask: np.ndarray, erosion_px: int) -> np.ndarray:
    """Binary erosion with a 3x3 structuring element, applied erosion_px times."""
    eroded = mask.astype(bool)
    for _ in range(erosion_px):
        padded = np.pad(eroded, 1, mode="constant", constant_values=False)
        eroded = (
            padded[:-2, :-2] & padded[:-2, 1:-1] & padded[:-2, 2:]
            & padded[1:-1, :-2] & padded[1:-1, 1:-1] & padded[1:-1, 2:]
            & padded[2:, :-2] & padded[2:, 1:-1] & padded[2:, 2:]
        )
    return eroded


def preprocess_object_rgbd(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    config: dict[str, Any],
) -> MaskedRGBD:
    """Produce object-only RGB-D for TSDF integration.

    config needs mask.erosion_px plus depth.scale / depth.min_m / depth.max_m.
    """
    if depth.shape != rgb.shape[:2] or mask.shape != depth.shape:
        raise ValueError(
            f"rgb {rgb.shape}, depth {depth.shape} and mask {mask.shape} "
            "are not pixel aligned"
        )
    depth_cfg = config["depth"]
    scale = depth_cfg["scale"]
    if scale is None:
        raise ValueError("depth.scale is null; fusion is forbidden (PLAN section 9)")

    eroded = erode_mask(mask, int(config["mask"].get("erosion_px", 0)))

    depth_m = depth.astype(np.float64) / float(scale)
    valid = depth_m > 0
    if depth_cfg.get("min_m") is not None:
        valid &= depth_m >= depth_cfg["min_m"]
    if depth_cfg.get("max_m") is not None:
        valid &= depth_m <= depth_cfg["max_m"]

    keep = eroded & valid
    masked_depth = np.where(keep, depth, 0).astype(depth.dtype)

    mask_area = int(eroded.sum())
    valid_px = int(keep.sum())
    return MaskedRGBD(
        rgb=rgb,
        depth=masked_depth,
        mask=eroded,
        mask_area_px=mask_area,
        valid_depth_px=valid_px,
        valid_depth_ratio=(valid_px / mask_area) if mask_area else 0.0,
    )
