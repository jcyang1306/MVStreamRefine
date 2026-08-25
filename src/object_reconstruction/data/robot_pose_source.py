"""Background polling of the RealMan arm TCP pose with a timestamped buffer.

``rm_get_current_arm_state`` returns the TCP pose as xyzrpy (meters, radians,
intrinsic-free euler "xyz" as used by the calibration project), semantics
T_base_tcp. A daemon thread polls at a fixed rate and stores (timestamp,
T_base_tcp) pairs on the host ``time.monotonic_ns()`` clock - the same clock
RealSenseSource stamps frames with - so ``pose_at`` can pick the nearest pose
for any camera frame and reject it when the gap exceeds ``max_error_ms``.

The poller takes any ``read_pose() -> 4x4`` callable, keeping it unit-testable
without hardware; ``connect_realman_arm`` builds the real one (lazy import of
the Robotic_Arm SDK).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..calibration.transforms import validate_transform

logger = logging.getLogger(__name__)


def euler_xyz_to_rotation(rpy: np.ndarray) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll), matching scipy Rot.from_euler('xyz')."""
    roll, pitch, yaw = (float(v) for v in rpy)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def xyzrpy_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError(f"xyzrpy pose must have shape (6,), got {pose.shape}")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = euler_xyz_to_rotation(pose[3:])
    T[:3, 3] = pose[:3]
    return T


def connect_realman_arm(
    ip: str = "192.168.1.19", port: int = 8080
) -> Callable[[], np.ndarray]:
    """Connect to the RealMan arm; returns read_pose() -> T_base_tcp (4x4)."""
    try:
        from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
    except ImportError as exc:
        raise RuntimeError(
            "the Robotic_Arm SDK is not installed; the realtime pipeline needs "
            "it to read the TCP pose (pip install Robotic_Arm)"
        ) from exc

    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(ip, port)
    if handle.id <= 0:
        raise RuntimeError(f"failed to connect RealMan arm at {ip}:{port}")
    logger.info("RealMan arm connected %s:%d handle.id=%d", ip, port, handle.id)

    def read_pose() -> np.ndarray:
        ret, state = arm.rm_get_current_arm_state()
        if ret != 0:
            raise RuntimeError(f"rm_get_current_arm_state failed: ret={ret}")
        return xyzrpy_to_matrix(np.asarray(state["pose"], dtype=np.float64))

    return read_pose


@dataclass
class TimedPose:
    timestamp_ns: int
    T_base_tcp: np.ndarray


class RobotPosePoller:
    def __init__(
        self,
        read_pose: Callable[[], np.ndarray],
        poll_hz: float = 50.0,
        buffer_size: int = 1024,
    ) -> None:
        if poll_hz <= 0:
            raise ValueError("poll_hz must be > 0")
        self._read_pose = read_pose
        self.poll_interval_s = 1.0 / float(poll_hz)
        self._buffer: deque[TimedPose] = deque(maxlen=int(buffer_size))
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.read_failures = 0
        self.last_error: str | None = None

    @classmethod
    def from_config(
        cls, config: dict[str, Any], read_pose: Callable[[], np.ndarray]
    ) -> "RobotPosePoller":
        robot = config.get("realtime", {}).get("robot", {})
        return cls(
            read_pose,
            poll_hz=robot.get("poll_hz", 50.0),
            buffer_size=robot.get("buffer_size", 1024),
        )

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("poller already started")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="robot-pose-poller", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            t0 = time.monotonic_ns()
            try:
                T = self._read_pose()
                validate_transform(T)
            except Exception as exc:  # keep polling through transient failures
                self.read_failures += 1
                self.last_error = str(exc)
                logger.warning("robot pose read failed (#%d): %s",
                               self.read_failures, exc)
            else:
                t1 = time.monotonic_ns()
                # The pose was sampled somewhere inside [t0, t1]; use the middle.
                self._append(TimedPose((t0 + t1) // 2, np.asarray(T, np.float64)))
            self._stop_event.wait(self.poll_interval_s)

    def _append(self, pose: TimedPose) -> None:
        with self._lock:
            self._buffer.append(pose)

    def latest(self) -> TimedPose | None:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def pose_at(self, timestamp_ns: int, max_error_ms: float = 50.0) -> TimedPose | None:
        """Nearest buffered pose; None when empty or farther than max_error_ms."""
        with self._lock:
            if not self._buffer:
                return None
            nearest = min(
                self._buffer, key=lambda p: abs(p.timestamp_ns - timestamp_ns)
            )
        if abs(nearest.timestamp_ns - timestamp_ns) > max_error_ms * 1e6:
            return None
        return nearest
