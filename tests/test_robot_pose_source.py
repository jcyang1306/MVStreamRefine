"""RobotPosePoller and xyzrpy conversion tests (no hardware)."""

import time

import numpy as np
import pytest

from object_reconstruction.data.robot_pose_source import (
    RobotPosePoller,
    TimedPose,
    xyzrpy_to_matrix,
)

MS = 1_000_000  # ns


def test_xyzrpy_identity():
    T = xyzrpy_to_matrix(np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], [0.1, -0.2, 0.3])


def test_xyzrpy_matches_scipy_extrinsic_xyz_convention():
    # Rot.from_euler("xyz", [roll, pitch, yaw]) == Rz(yaw) @ Ry(pitch) @ Rx(roll)
    roll, pitch, yaw = 0.3, -0.5, 1.1
    T = xyzrpy_to_matrix(np.array([0.0, 0.0, 0.0, roll, pitch, yaw]))

    def rot(axis, a):
        c, s = np.cos(a), np.sin(a)
        m = {"x": [[1, 0, 0], [0, c, -s], [0, s, c]],
             "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
             "z": [[c, -s, 0], [s, c, 0], [0, 0, 1]]}
        return np.array(m[axis], dtype=np.float64)

    expected = rot("z", yaw) @ rot("y", pitch) @ rot("x", roll)
    np.testing.assert_allclose(T[:3, :3], expected, atol=1e-12)
    assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-12


def test_xyzrpy_rejects_wrong_shape():
    with pytest.raises(ValueError):
        xyzrpy_to_matrix(np.zeros(7))


def make_pose(tx):
    T = np.eye(4)
    T[0, 3] = tx
    return T


def test_pose_at_picks_nearest_and_enforces_max_error():
    poller = RobotPosePoller(read_pose=lambda: np.eye(4), poll_hz=100)
    poller._append(TimedPose(100 * MS, make_pose(0.1)))
    poller._append(TimedPose(200 * MS, make_pose(0.2)))
    poller._append(TimedPose(300 * MS, make_pose(0.3)))

    nearest = poller.pose_at(190 * MS, max_error_ms=50)
    assert nearest.timestamp_ns == 200 * MS
    np.testing.assert_allclose(nearest.T_base_tcp[0, 3], 0.2)

    # 400ms vs the newest pose at 300ms is a 100ms gap > 50ms -> dropped.
    assert poller.pose_at(400 * MS, max_error_ms=50) is None


def test_pose_at_rejects_stale_and_empty():
    poller = RobotPosePoller(read_pose=lambda: np.eye(4), poll_hz=100)
    assert poller.pose_at(0, max_error_ms=50) is None
    poller._append(TimedPose(100 * MS, make_pose(0.1)))
    assert poller.pose_at(200 * MS, max_error_ms=50) is None
    assert poller.pose_at(120 * MS, max_error_ms=50) is not None


def test_polling_thread_collects_and_survives_failures():
    calls = {"n": 0}

    def flaky_read_pose():
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("transient bus error")
        T = np.eye(4)
        T[0, 3] = 0.01 * calls["n"]
        return T

    poller = RobotPosePoller(read_pose=flaky_read_pose, poll_hz=200)
    poller.start()
    time.sleep(0.15)
    poller.stop()

    assert poller.read_failures == 1
    assert "transient" in poller.last_error
    latest = poller.latest()
    assert latest is not None
    assert latest.T_base_tcp[0, 3] > 0
    # Timestamps must be monotonically increasing host-clock values.
    stamps = [p.timestamp_ns for p in poller._buffer]
    assert stamps == sorted(stamps) and len(stamps) >= 2


def test_double_start_is_rejected():
    poller = RobotPosePoller(read_pose=lambda: np.eye(4), poll_hz=200)
    poller.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            poller.start()
    finally:
        poller.stop()
