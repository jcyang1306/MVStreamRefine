"""Core data model shared by offline and realtime pipelines (PLAN section 4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"invalid image size {self.width}x{self.height}")
        if not (np.isfinite(self.fx) and np.isfinite(self.fy) and self.fx > 0 and self.fy > 0):
            raise ValueError(f"invalid focal lengths fx={self.fx} fy={self.fy}")
        if not (0.0 < self.cx < self.width and 0.0 < self.cy < self.height):
            raise ValueError(
                f"principal point ({self.cx}, {self.cy}) outside {self.width}x{self.height}"
            )


@dataclass
class CameraFrame:
    rgb: np.ndarray  # uint8, H x W x 3, RGB order
    depth: np.ndarray  # uint16 (unit defined by depth_scale in config)
    intrinsics: CameraIntrinsics
    mask: np.ndarray | None = None  # bool, H x W


@dataclass
class FramePacket:
    index: int
    # Current offline dataset carries no timestamps; sync is by frame index only.
    timestamp: float | None

    cam: CameraFrame

    # None until pose_semantics / quaternion_order / handeye_convention are
    # confirmed in config; constructing it from unconfirmed conventions is forbidden.
    # WORLD is the robot base frame in the single-camera system.
    T_world_cam: np.ndarray | None

    tcp_pose: np.ndarray | None = None
    raw_pose_7d: np.ndarray | None = None
