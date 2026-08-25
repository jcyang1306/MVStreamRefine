"""RealtimeFrameSource sync/composition tests with fake camera and poller."""

from pathlib import Path

import numpy as np
import pytest

from object_reconstruction.calibration.transforms import compose_T_world_cam
from object_reconstruction.data.models import CameraIntrinsics
from object_reconstruction.data.realsense_source import RealSenseFrame
from object_reconstruction.data.realtime_source import (
    RealtimeFrameSource,
    load_T_tcp_cam,
)
from object_reconstruction.data.robot_pose_source import TimedPose

REPO_ROOT = Path(__file__).resolve().parent.parent
WIDTH, HEIGHT = 64, 48
MS = 1_000_000  # ns


class FakeCamera:
    def __init__(self, timestamps_ns):
        self.timestamps = list(timestamps_ns)
        self.intrinsics = CameraIntrinsics(WIDTH, HEIGHT, 50.0, 50.0, 32.0, 24.0)
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def read(self):
        return RealSenseFrame(
            rgb=np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8),
            depth=np.full((HEIGHT, WIDTH), 500, dtype=np.uint16),
            timestamp_ns=self.timestamps.pop(0),
        )


class FakePoller:
    def __init__(self, poses):
        self.poses = poses  # list of TimedPose
        self.started = False
        self.stopped = False
        self.read_failures = 0

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def pose_at(self, timestamp_ns, max_error_ms):
        if not self.poses:
            return None
        nearest = min(self.poses, key=lambda p: abs(p.timestamp_ns - timestamp_ns))
        if abs(nearest.timestamp_ns - timestamp_ns) > max_error_ms * 1e6:
            return None
        return nearest


def make_T_base_tcp(tx):
    T = np.eye(4)
    T[0, 3] = tx
    return T


T_TCP_CAM = np.array(
    [
        [0.0, -1.0, 0.0, 0.01],
        [1.0, 0.0, 0.0, -0.02],
        [0.0, 0.0, 1.0, 0.03],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def test_packet_composes_world_pose_from_nearest_synced_pose():
    camera = FakeCamera(timestamps_ns=[105 * MS])
    poller = FakePoller(
        [TimedPose(90 * MS, make_T_base_tcp(0.9)),
         TimedPose(100 * MS, make_T_base_tcp(0.5))]
    )
    source = RealtimeFrameSource(camera, poller, T_TCP_CAM, max_sync_error_ms=50)
    packet = source.read_packet()

    assert packet is not None
    assert packet.index == 0
    np.testing.assert_allclose(packet.tcp_pose, make_T_base_tcp(0.5))
    np.testing.assert_allclose(
        packet.T_world_cam, compose_T_world_cam(make_T_base_tcp(0.5), T_TCP_CAM)
    )
    assert packet.cam.intrinsics is camera.intrinsics
    assert packet.timestamp == pytest.approx(0.105)
    assert source.frames_emitted == 1
    assert source.last_sync_error_ms == pytest.approx(5.0)


def test_frame_is_dropped_when_no_pose_is_close_enough():
    camera = FakeCamera(timestamps_ns=[500 * MS, 510 * MS])
    poller = FakePoller([TimedPose(100 * MS, make_T_base_tcp(0.5)),
                         TimedPose(505 * MS, make_T_base_tcp(0.7))])
    source = RealtimeFrameSource(camera, poller, T_TCP_CAM, max_sync_error_ms=10)

    poller_backup = poller.poses
    poller.poses = poller_backup[:1]  # only the stale pose exists yet
    assert source.read_packet() is None
    assert source.frames_dropped_sync == 1

    poller.poses = poller_backup
    packet = source.read_packet()
    assert packet is not None and packet.index == 0
    assert source.frames_emitted == 1


def test_packet_indices_increase_monotonically():
    camera = FakeCamera(timestamps_ns=[100 * MS, 110 * MS, 120 * MS])
    poller = FakePoller([TimedPose(105 * MS, make_T_base_tcp(0.1))])
    source = RealtimeFrameSource(camera, poller, T_TCP_CAM, max_sync_error_ms=50)
    indices = [source.read_packet().index for _ in range(3)]
    assert indices == [0, 1, 2]


def test_start_stop_propagate():
    camera = FakeCamera([])
    poller = FakePoller([])
    source = RealtimeFrameSource(camera, poller, T_TCP_CAM)
    source.start()
    source.stop()
    assert camera.started and camera.stopped
    assert poller.started and poller.stopped


def test_load_T_tcp_cam_from_repo_calibration():
    config = {
        "calibration": {
            "handeye_path": str(REPO_ROOT / "data" / "handeye" / "handeye_tf.txt"),
            "pose_semantics": "T_base_tcp",
            "quaternion_order": "xyzw",
            "handeye_convention": "wrist_cam2=T_tcp_cam",
        }
    }
    T = load_T_tcp_cam(config)
    assert T.shape == (4, 4)
    assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-3


def test_load_T_tcp_cam_rejects_unsupported_convention():
    config = {
        "calibration": {
            "handeye_path": str(REPO_ROOT / "data" / "handeye" / "handeye_tf.txt"),
            "pose_semantics": "T_tcp_base",
            "quaternion_order": "xyzw",
            "handeye_convention": "wrist_cam2=T_tcp_cam",
        }
    }
    with pytest.raises(NotImplementedError):
        load_T_tcp_cam(config)
