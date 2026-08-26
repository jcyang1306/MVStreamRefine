"""Synchronized live FramePacket source: RealSense frame + nearest robot pose.

Each ``read_packet()`` blocks for one camera frame, looks up the nearest
buffered T_base_tcp on the shared monotonic clock and composes

    T_world_cam = T_base_tcp @ T_tcp_cam        (WORLD = robot base)

with the same hand-eye matrix and convention checks as the offline source.
Frames whose nearest pose is farther than ``max_sync_error_ms`` are dropped
(returns None) and counted, so a stalled poller can't corrupt the fusion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..calibration.handeye import load_handeye_matrices
from ..calibration.transforms import (
    check_supported_conventions,
    compose_T_world_cam,
    validate_transform,
)
from .models import CameraFrame, FramePacket
from .realsense_source import RealSenseSource
from .robot_pose_source import RobotPosePoller

logger = logging.getLogger(__name__)


def load_T_tcp_cam(config: dict[str, Any]) -> np.ndarray:
    """Load the wrist hand-eye matrix with the same fail-fast convention checks."""
    calib = config["calibration"]
    check_supported_conventions(
        calib["pose_semantics"],
        calib["quaternion_order"],
        calib["handeye_convention"],
    )
    handeye = load_handeye_matrices(Path(calib["handeye_path"]))
    try:
        T_tcp_cam = handeye["wrist_cam2"]
    except KeyError as exc:
        raise ValueError(
            f"hand-eye file is missing expected label {exc}; found {sorted(handeye)}"
        ) from exc
    validate_transform(T_tcp_cam)
    return T_tcp_cam


class RealtimeFrameSource:
    def __init__(
        self,
        camera: RealSenseSource,
        pose_poller: RobotPosePoller,
        T_tcp_cam: np.ndarray,
        max_sync_error_ms: float = 50.0,
    ) -> None:
        validate_transform(T_tcp_cam)
        self.camera = camera
        self.pose_poller = pose_poller
        self.T_tcp_cam = np.asarray(T_tcp_cam, dtype=np.float64)
        self.max_sync_error_ms = float(max_sync_error_ms)

        self.frames_emitted = 0
        self.frames_dropped_sync = 0
        self.last_sync_error_ms: float | None = None
        self._next_index = 0

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        camera: RealSenseSource,
        pose_poller: RobotPosePoller,
    ) -> "RealtimeFrameSource":
        sync = config.get("realtime", {}).get("sync", {})
        return cls(
            camera,
            pose_poller,
            T_tcp_cam=load_T_tcp_cam(config),
            max_sync_error_ms=sync.get("max_error_ms", 50.0),
        )

    def start(self) -> None:
        self.camera.start()
        try:
            self.pose_poller.start()
        except Exception:
            self.camera.stop()
            raise

    def stop(self) -> None:
        self.pose_poller.stop()
        self.camera.stop()

    def read_packet(self) -> FramePacket | None:
        """One synchronized packet, or None when the frame had to be dropped."""
        frame = self.camera.read()
        if frame is None:
            return None
        pose = self.pose_poller.pose_at(frame.timestamp_ns, self.max_sync_error_ms)
        if pose is None:
            self.frames_dropped_sync += 1
            self.last_sync_error_ms = None
            return None
        self.last_sync_error_ms = abs(pose.timestamp_ns - frame.timestamp_ns) / 1e6

        T_world_cam = compose_T_world_cam(pose.T_base_tcp, self.T_tcp_cam)
        packet = FramePacket(
            index=self._next_index,
            timestamp=frame.timestamp_ns * 1e-9,
            cam=CameraFrame(
                rgb=frame.rgb,
                depth=frame.depth,
                intrinsics=self.camera.intrinsics,
            ),
            T_world_cam=T_world_cam,
            tcp_pose=pose.T_base_tcp,
        )
        self._next_index += 1
        self.frames_emitted += 1
        return packet
