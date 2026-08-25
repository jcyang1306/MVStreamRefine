"""RealSense D400 live capture (color-aligned RGB-D), PLAN realtime module.

Mirrors the capture settings of the calibration project: bgr8 color + z16
depth streams, depth aligned to color, intrinsics read from the color
profile, ~1s warmup after start.

Depth is kept as raw uint16 in device units. The device depth scale
(meters per unit, e.g. 0.001) must be consistent with the config's
``depth.scale`` (units per meter, e.g. 1000.0); ``start()`` fails fast on a
mismatch so realtime frames obey the same convention as the offline dataset.

Timestamps are host ``time.monotonic_ns()`` taken when the frame arrives, the
same clock used by RobotPosePoller, so camera/robot sync compares like with
like (at the cost of ~1 frame of transport latency).

``pyrealsense2`` is imported lazily inside ``start()`` so unit tests and
offline tooling run on machines without the RealSense SDK.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .models import CameraIntrinsics

logger = logging.getLogger(__name__)


@dataclass
class RealSenseFrame:
    rgb: np.ndarray  # uint8, H x W x 3, RGB order
    depth: np.ndarray  # uint16, device units (see depth_scale)
    timestamp_ns: int  # host time.monotonic_ns() at arrival


class RealSenseSource:
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        serial: str | None = None,
        warmup_frames: int = 30,
        expected_units_per_meter: float | None = None,
        rs_module: Any = None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.serial = serial
        self.warmup_frames = int(warmup_frames)
        self.expected_units_per_meter = expected_units_per_meter
        self._rs = rs_module  # injectable for tests

        self._pipeline: Any = None
        self._align: Any = None
        self.depth_scale: float | None = None  # meters per depth unit
        self.intrinsics: CameraIntrinsics | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any], **overrides: Any) -> "RealSenseSource":
        cam = config.get("realtime", {}).get("camera", {})
        kwargs: dict[str, Any] = {
            "width": cam.get("width", 1280),
            "height": cam.get("height", 720),
            "fps": cam.get("fps", 30),
            "serial": cam.get("serial"),
            "warmup_frames": cam.get("warmup_frames", 30),
            "expected_units_per_meter": config.get("depth", {}).get("scale"),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    def start(self) -> None:
        rs = self._rs
        if rs is None:
            try:
                import pyrealsense2 as rs  # type: ignore[no-redef]
            except ImportError as exc:
                raise RuntimeError(
                    "pyrealsense2 is not installed; the realtime pipeline needs "
                    "the RealSense SDK (pip install pyrealsense2)"
                ) from exc
            self._rs = rs

        self._pipeline = rs.pipeline()
        rs_config = rs.config()
        if self.serial:
            rs_config.enable_device(str(self.serial))
        rs_config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )
        rs_config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )
        profile = self._pipeline.start(rs_config)
        self._align = rs.align(rs.stream.color)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        self._check_depth_scale()

        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = CameraIntrinsics(
            width=int(intr.width),
            height=int(intr.height),
            fx=float(intr.fx),
            fy=float(intr.fy),
            cx=float(intr.ppx),
            cy=float(intr.ppy),
        )
        self.intrinsics.validate()

        for _ in range(self.warmup_frames):
            self._pipeline.wait_for_frames()
        logger.info(
            "RealSense ready %dx%d @ %d fps  serial=%s  depth_scale=%s m/unit",
            self.width, self.height, self.fps, self.serial, self.depth_scale,
        )

    def _check_depth_scale(self) -> None:
        if self.expected_units_per_meter is None or self.depth_scale is None:
            return
        actual_units_per_meter = 1.0 / self.depth_scale
        expected = float(self.expected_units_per_meter)
        if abs(actual_units_per_meter - expected) > 0.01 * expected:
            raise RuntimeError(
                f"device depth scale {self.depth_scale} m/unit "
                f"({actual_units_per_meter:.1f} units/m) does not match config "
                f"depth.scale={expected} units/m; fusion would be wrong"
            )

    def read(self) -> RealSenseFrame | None:
        """Blocking read of one color-aligned RGB-D frame; None if incomplete."""
        if self._pipeline is None:
            raise RuntimeError("call start() before read()")
        frames = self._pipeline.wait_for_frames()
        timestamp_ns = time.monotonic_ns()
        frames = self._align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None
        bgr = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        return RealSenseFrame(
            rgb=np.ascontiguousarray(bgr[:, :, ::-1]),
            depth=depth.astype(np.uint16, copy=False),
            timestamp_ns=timestamp_ns,
        )

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
