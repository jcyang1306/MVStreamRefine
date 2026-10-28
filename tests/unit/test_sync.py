"""Robot pose pairing (P1 S4): bracketing interpolation, gap limit, no extrapolation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mvsr.core.errors import ContractValidationError
from mvsr.core.se3 import PoseSE3, interpolate_decoupled
from mvsr.core.types import ClockDomain, RobotSample, SyncStatus
from mvsr.sensors.sync import RobotTrack, interp_method


def _rz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _sample(seq: int, t: int, x: float, yaw: float) -> RobotSample:
    T = PoseSE3.from_rt(_rz(yaw), [x, 0.0, 0.4], "base", "tcp")
    return RobotSample(seq, t, ClockDomain.HOST_MONOTONIC, T, None, None, "poll", None)


GAP = 25_000_000
TRACK = RobotTrack.from_samples(
    [
        _sample(2, 20_000_000, 0.32, 0.2),
        _sample(0, 0, 0.30, 0.0),
        _sample(1, 10_000_000, 0.31, 0.1),
        _sample(3, 60_000_000, 0.36, 0.6),
    ]
)


def test_interpolates_between_bracketing_samples() -> None:
    pose, info = TRACK.pose_at(5_000_000, GAP)
    assert info.status is SyncStatus.INTERPOLATED and info.method == interp_method(GAP, 0)
    assert (info.gap_before_ns, info.gap_after_ns) == (5_000_000, 5_000_000)
    assert (info.robot_seq_before, info.robot_seq_after) == (0, 1)
    expected = interpolate_decoupled(TRACK.samples[0].T_base_tcp, TRACK.samples[1].T_base_tcp, 0.5)
    assert pose is not None and pose == expected
    assert abs(pose.t[0] - 0.305) < 1e-12


def test_exact_sample_time_uses_that_sample() -> None:
    pose, info = TRACK.pose_at(10_000_000, GAP)
    assert info.status is SyncStatus.INTERPOLATED and pose == TRACK.samples[1].T_base_tcp
    assert (info.gap_before_ns, info.gap_after_ns, info.robot_seq_before) == (0, 0, 1)


def test_gap_too_large_and_out_of_range_give_no_pose() -> None:
    pose, info = TRACK.pose_at(40_000_000, GAP)
    assert pose is None and info.status is SyncStatus.GAP_TOO_LARGE
    assert (info.gap_before_ns, info.gap_after_ns, info.robot_seq_before, info.robot_seq_after) == (
        20_000_000,
        20_000_000,
        2,
        3,
    )
    pose, info = TRACK.pose_at(-1, GAP)
    assert pose is None and info.status is SyncStatus.OUT_OF_RANGE and info.gap_after_ns == 1
    pose, info = TRACK.pose_at(60_000_001, GAP)
    assert pose is None and info.status is SyncStatus.OUT_OF_RANGE and info.gap_before_ns == 1
    pose, info = RobotTrack.from_samples([]).pose_at(0, GAP)
    assert pose is None and info.status is SyncStatus.NO_ROBOT


def test_time_offset_shifts_the_query() -> None:
    a, _ = TRACK.pose_at(2_000_000, GAP, dt_robot_cam_ns=3_000_000)
    b, _ = TRACK.pose_at(5_000_000, GAP)
    assert a is not None and b is not None and a == b
    with pytest.raises(ContractValidationError):
        TRACK.pose_at(0, 0)


def test_nearest_baseline() -> None:
    pose, info = TRACK.nearest(13_000_000, GAP)
    assert pose == TRACK.samples[1].T_base_tcp and info.gap_before_ns == 3_000_000
    pose, info = TRACK.nearest(100_000_000, GAP)
    assert pose is None and info.status is SyncStatus.GAP_TOO_LARGE
