"""UI-independent RealMan start-to-finish motion state machine.

The controller deliberately stays separate from SAM/TSDF reconstruction. It
executes the configured start move on a worker thread (blocking SDK call while
the UI remains responsive), sends the finish move non-blocking, and detects
arrival from synchronized T_base_tcp samples.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from ..data.robot_pose_source import xyzrpy_to_matrix


class MotionState(str, Enum):
    DISABLED = "DISABLED"
    IDLE = "IDLE"
    MOVING_TO_START = "MOVING_TO_START"
    AT_START = "AT_START"
    SCANNING = "SCANNING"
    PAUSED = "PAUSED"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class MotionConfig:
    enabled: bool
    initial_pose: np.ndarray | None
    final_pose: np.ndarray | None
    velocity: int = 10
    blend_radius: int = 0
    position_tolerance_m: float = 0.008
    rotation_tolerance_deg: float = 2.0
    stable_frames: int = 3

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MotionConfig":
        raw = config.get("realtime", {}).get("robot", {}).get("motion", {})
        enabled = bool(raw.get("enabled", False))

        def pose(name: str) -> np.ndarray | None:
            value = raw.get(name)
            if value is None:
                if enabled:
                    raise ValueError(
                        f"realtime.robot.motion.{name} is required when motion is enabled"
                    )
                return None
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (6,) or not np.isfinite(array).all():
                raise ValueError(
                    f"realtime.robot.motion.{name} must be six finite xyzrpy values"
                )
            return array

        velocity = int(raw.get("velocity", 10))
        blend_radius = int(raw.get("blend_radius", 0))
        position_tolerance_m = float(raw.get("position_tolerance_m", 0.008))
        rotation_tolerance_deg = float(raw.get("rotation_tolerance_deg", 2.0))
        stable_frames = int(raw.get("stable_frames", 3))
        if not 1 <= velocity <= 100:
            raise ValueError("motion.velocity must be in [1, 100]")
        if not 0 <= blend_radius <= 100:
            raise ValueError("motion.blend_radius must be in [0, 100]")
        if position_tolerance_m <= 0 or rotation_tolerance_deg <= 0:
            raise ValueError("motion arrival tolerances must be > 0")
        if stable_frames < 1:
            raise ValueError("motion.stable_frames must be >= 1")
        return cls(
            enabled=enabled,
            initial_pose=pose("initial_pose"),
            final_pose=pose("final_pose"),
            velocity=velocity,
            blend_radius=blend_radius,
            position_tolerance_m=position_tolerance_m,
            rotation_tolerance_deg=rotation_tolerance_deg,
            stable_frames=stable_frames,
        )


class RobotMotionController:
    def __init__(self, robot: Any, config: MotionConfig) -> None:
        self.robot = robot
        self.config = config
        self._state = MotionState.IDLE if config.enabled else MotionState.DISABLED
        self._lock = threading.Lock()
        self._start_thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._stable_count = 0
        self.last_error: str | None = None
        self.position_error_m: float | None = None
        self.rotation_error_deg: float | None = None
        self._T_final = (
            xyzrpy_to_matrix(config.final_pose)
            if config.final_pose is not None
            else None
        )

    @property
    def state(self) -> MotionState:
        with self._lock:
            return self._state

    def _set_state(self, state: MotionState) -> None:
        with self._lock:
            self._state = state

    def move_to_start(self) -> None:
        """Start a blocking SDK move on a worker thread."""
        if not self.config.enabled:
            raise RuntimeError("robot motion is disabled in config")
        if self.state not in (
            MotionState.IDLE,
            MotionState.AT_START,
            MotionState.FINISHED,
            MotionState.ERROR,
        ):
            raise RuntimeError(f"cannot move to start from {self.state.value}")
        if self._start_thread is not None and self._start_thread.is_alive():
            raise RuntimeError("start move is already running")
        self.last_error = None
        self._cancel_event.clear()
        self._set_state(MotionState.MOVING_TO_START)
        self._start_thread = threading.Thread(
            target=self._move_to_start_worker,
            name="robot-move-to-start",
            daemon=True,
        )
        self._start_thread.start()

    def _move_to_start_worker(self) -> None:
        if self._cancel_event.is_set():
            return
        try:
            self.robot.move_linear(
                self.config.initial_pose,
                velocity=self.config.velocity,
                blend_radius=self.config.blend_radius,
                blocking=True,
            )
        except Exception as exc:
            if self._cancel_event.is_set():
                return
            self.last_error = str(exc)
            self._set_state(MotionState.ERROR)
        else:
            if not self._cancel_event.is_set():
                self._set_state(MotionState.AT_START)

    def start_scan(self) -> None:
        if self.state == MotionState.PAUSED:
            self.resume()
            return
        if self.state != MotionState.AT_START:
            raise RuntimeError(f"cannot start scan from {self.state.value}")
        self.robot.move_linear(
            self.config.final_pose,
            velocity=self.config.velocity,
            blend_radius=self.config.blend_radius,
            blocking=False,
        )
        self._cancel_event.clear()
        self._stable_count = 0
        self.last_error = None
        self._set_state(MotionState.SCANNING)

    def pause(self) -> None:
        if self.state != MotionState.SCANNING:
            return
        self.robot.pause()
        self._set_state(MotionState.PAUSED)

    def resume(self) -> None:
        if self.state != MotionState.PAUSED:
            return
        self.robot.resume()
        self._set_state(MotionState.SCANNING)

    def stop(self) -> None:
        """Stop any active command and return to IDLE."""
        if self.state == MotionState.DISABLED:
            return
        self._cancel_event.set()
        thread = self._start_thread
        if thread is not None and thread.is_alive():
            # Give the worker time either to observe cancellation or enter the
            # blocking SDK call and mark robot.motion_active.
            thread.join(0.1)
        if self.robot.motion_active:
            self.robot.stop()
        self._stable_count = 0
        self._set_state(MotionState.IDLE)

    def update_pose(self, T_base_tcp: np.ndarray | None) -> bool:
        """Update arrival gates. Returns True exactly when finish is accepted."""
        if self.state != MotionState.SCANNING or T_base_tcp is None:
            return False
        T = np.asarray(T_base_tcp, dtype=np.float64)
        delta = np.linalg.inv(self._T_final) @ T
        self.position_error_m = float(np.linalg.norm(delta[:3, 3]))
        cos_angle = np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        self.rotation_error_deg = float(np.degrees(np.arccos(cos_angle)))
        inside = (
            self.position_error_m <= self.config.position_tolerance_m
            and self.rotation_error_deg <= self.config.rotation_tolerance_deg
        )
        self._stable_count = self._stable_count + 1 if inside else 0
        if self._stable_count < self.config.stable_frames:
            return False
        self.robot.mark_arrived()
        self._set_state(MotionState.FINISHED)
        return True

    def wait_for_start(self, timeout: float = 2.0) -> None:
        """Testing/CLI helper: join the current start worker."""
        thread = self._start_thread
        if thread is not None:
            thread.join(timeout)

