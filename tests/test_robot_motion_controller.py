"""RobotMotionController tests with no hardware dependency."""

import numpy as np
import pytest

from object_reconstruction.data.robot_pose_source import xyzrpy_to_matrix
from object_reconstruction.pipeline.robot_motion_controller import (
    MotionConfig,
    MotionState,
    RobotMotionController,
)


class FakeRobot:
    def __init__(self, fail_move=False):
        self.fail_move = fail_move
        self.moves = []
        self.motion_active = False
        self.paused = False
        self.pause_calls = 0
        self.resume_calls = 0
        self.stop_calls = 0
        self.arrived_calls = 0

    def move_linear(self, pose, **kwargs):
        if self.fail_move:
            raise RuntimeError("planned move failed")
        self.moves.append((np.asarray(pose).copy(), kwargs))
        self.motion_active = not kwargs["blocking"]

    def pause(self):
        self.pause_calls += 1
        self.paused = True

    def resume(self):
        self.resume_calls += 1
        self.paused = False

    def stop(self):
        self.stop_calls += 1
        self.motion_active = False
        self.paused = False

    def mark_arrived(self):
        self.arrived_calls += 1
        self.motion_active = False


def config(enabled=True, **overrides):
    raw = {
        "enabled": enabled,
        "initial_pose": [0.1, -0.2, 0.3, 0.0, 0.0, 0.0],
        "final_pose": [0.2, -0.3, 0.4, 0.1, 0.2, 0.3],
        "velocity": 10,
        "blend_radius": 0,
        "position_tolerance_m": 0.008,
        "rotation_tolerance_deg": 2.0,
        "stable_frames": 3,
    }
    raw.update(overrides)
    return {"realtime": {"robot": {"motion": raw}}}


def test_motion_config_disabled_allows_missing_poses():
    cfg = MotionConfig.from_config(
        {"realtime": {"robot": {"motion": {"enabled": False}}}}
    )
    assert not cfg.enabled
    assert cfg.initial_pose is None and cfg.final_pose is None
    controller = RobotMotionController(FakeRobot(), cfg)
    assert controller.state == MotionState.DISABLED


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"initial_pose": None}, "initial_pose"),
        ({"final_pose": [1, 2]}, "final_pose"),
        ({"velocity": 0}, "velocity"),
        ({"blend_radius": 101}, "blend_radius"),
        ({"stable_frames": 0}, "stable_frames"),
    ],
)
def test_motion_config_rejects_unsafe_values(overrides, error):
    with pytest.raises(ValueError, match=error):
        MotionConfig.from_config(config(**overrides))


def test_full_start_scan_and_arrival_sequence():
    robot = FakeRobot()
    cfg = MotionConfig.from_config(config())
    controller = RobotMotionController(robot, cfg)

    controller.move_to_start()
    controller.wait_for_start()
    assert controller.state == MotionState.AT_START
    np.testing.assert_allclose(robot.moves[0][0], cfg.initial_pose)
    assert robot.moves[0][1]["blocking"] is True

    controller.start_scan()
    assert controller.state == MotionState.SCANNING
    assert robot.motion_active
    np.testing.assert_allclose(robot.moves[1][0], cfg.final_pose)
    assert robot.moves[1][1]["blocking"] is False

    T_final = xyzrpy_to_matrix(cfg.final_pose)
    assert not controller.update_pose(T_final)
    assert not controller.update_pose(T_final)
    assert controller.update_pose(T_final)
    assert controller.state == MotionState.FINISHED
    assert robot.arrived_calls == 1 and not robot.motion_active


def test_arrival_requires_both_tolerances_and_consecutive_samples():
    robot = FakeRobot()
    controller = RobotMotionController(robot, MotionConfig.from_config(config()))
    controller.move_to_start()
    controller.wait_for_start()
    controller.start_scan()
    T_final = xyzrpy_to_matrix(controller.config.final_pose)

    far = T_final.copy()
    far[0, 3] += 0.02
    assert not controller.update_pose(T_final)
    assert not controller.update_pose(far)  # resets the stable streak
    assert not controller.update_pose(T_final)
    assert not controller.update_pose(T_final)
    assert controller.update_pose(T_final)


def test_pause_resume_and_stop():
    robot = FakeRobot()
    controller = RobotMotionController(robot, MotionConfig.from_config(config()))
    controller.move_to_start()
    controller.wait_for_start()
    controller.start_scan()

    controller.pause()
    assert controller.state == MotionState.PAUSED
    assert robot.pause_calls == 1
    controller.start_scan()  # G after LOST/ROI correction resumes existing trajectory
    assert controller.state == MotionState.SCANNING
    assert robot.resume_calls == 1
    controller.stop()
    assert controller.state == MotionState.IDLE
    assert robot.stop_calls == 1


def test_start_move_failure_is_reported_as_error_state():
    controller = RobotMotionController(
        FakeRobot(fail_move=True), MotionConfig.from_config(config())
    )
    controller.move_to_start()
    controller.wait_for_start()
    assert controller.state == MotionState.ERROR
    assert "planned move failed" in controller.last_error


def test_scan_requires_start_pose():
    controller = RobotMotionController(
        FakeRobot(), MotionConfig.from_config(config())
    )
    with pytest.raises(RuntimeError, match="cannot start scan"):
        controller.start_scan()

