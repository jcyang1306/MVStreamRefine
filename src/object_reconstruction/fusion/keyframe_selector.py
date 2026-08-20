"""Keyframe selection for the moving wrist camera (PLAN section 13, Task 7).

A cam2 frame becomes a keyframe when the camera moved enough since the last
accepted keyframe (translation OR rotation threshold) AND the object mask is
still trustworthy (area and valid-depth-ratio gates). The very first frame
that passes the quality gates is always accepted.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class KeyframeSelector:
    def __init__(
        self,
        translation_m: float = 0.02,
        rotation_deg: float = 5.0,
        min_mask_area_px: int = 1000,
        min_valid_depth_ratio: float = 0.7,
    ) -> None:
        if translation_m <= 0 or rotation_deg <= 0:
            raise ValueError("translation_m and rotation_deg must be > 0")
        self.translation_m = float(translation_m)
        self.rotation_deg = float(rotation_deg)
        self.min_mask_area_px = int(min_mask_area_px)
        self.min_valid_depth_ratio = float(min_valid_depth_ratio)
        self._T_last: np.ndarray | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "KeyframeSelector":
        kf = config["keyframe"]
        return cls(
            translation_m=kf.get("translation_m", 0.02),
            rotation_deg=kf.get("rotation_deg", 5.0),
            min_mask_area_px=kf.get("min_mask_area_px", 1000),
            min_valid_depth_ratio=kf.get("min_valid_depth_ratio", 0.7),
        )

    @staticmethod
    def pose_delta(T_last: np.ndarray, T_current: np.ndarray) -> tuple[float, float]:
        """(translation_m, rotation_deg) of T_delta = inv(T_last) @ T_current."""
        T_delta = np.linalg.inv(T_last) @ T_current
        translation = float(np.linalg.norm(T_delta[:3, 3]))
        # acos argument must be clamped: float error pushes trace past [-1, 3].
        cos_angle = np.clip((np.trace(T_delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        rotation = float(np.degrees(np.arccos(cos_angle)))
        return translation, rotation

    def should_add(
        self,
        T_world_cam: np.ndarray,
        mask_area_px: int,
        valid_depth_ratio: float,
    ) -> bool:
        """Decide keyframe acceptance; accepted poses become the new reference."""
        if mask_area_px < self.min_mask_area_px:
            return False
        if valid_depth_ratio < self.min_valid_depth_ratio:
            return False

        if self._T_last is None:
            self._T_last = np.asarray(T_world_cam, dtype=np.float64).copy()
            return True

        translation, rotation = self.pose_delta(self._T_last, T_world_cam)
        if translation > self.translation_m or rotation > self.rotation_deg:
            self._T_last = np.asarray(T_world_cam, dtype=np.float64).copy()
            return True
        return False

    def reset(self) -> None:
        self._T_last = None
