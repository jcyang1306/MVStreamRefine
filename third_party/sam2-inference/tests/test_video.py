import numpy as np
import pytest

from sam2_inference import TrackingResult, VideoTracker


class FakeVideoPredictor:
    def __init__(self):
        self.reset_count = 0

    def init_state(self, **_kwargs):
        return {"num_frames": 2}

    def add_new_points_or_box(self, **kwargs):
        return kwargs["frame_idx"], [kwargs["obj_id"]], np.ones((1, 1, 5, 6))

    def add_new_mask(self, state, frame_idx, obj_id, mask):
        return frame_idx, [obj_id], mask[None, None].astype(np.float32)

    def propagate_in_video(self, _state, **_kwargs):
        for index in range(2):
            yield index, [7], np.full((1, 1, 5, 6), index + 1.0)

    def reset_state(self, _state):
        self.reset_count += 1


def test_video_lifecycle_and_results(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    predictor = FakeVideoPredictor()
    tracker = VideoTracker(predictor=predictor, device="cpu")
    assert tracker.open(frames) is tracker
    prompt = tracker.add_prompt(0, 7, box=[0, 0, 4, 4])
    assert prompt.masks.shape == (1, 5, 6)
    results = list(tracker.track())
    assert all(isinstance(result, TrackingResult) for result in results)
    assert [result.frame_index for result in results] == [0, 1]
    assert results[0].object_ids == (7,)
    tracker.reset()
    tracker.close()
    assert not tracker.is_open
    assert predictor.reset_count == 2


def test_video_requires_open():
    tracker = VideoTracker(predictor=FakeVideoPredictor(), device="cpu")
    with pytest.raises(RuntimeError, match=r"open\(\)"):
        tracker.add_prompt(0, 1, box=[0, 0, 2, 2])


def test_video_rejects_invalid_frame(tmp_path):
    tracker = VideoTracker(predictor=FakeVideoPredictor(), device="cpu")
    tracker.open(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        tracker.add_prompt(2, 1, box=[0, 0, 2, 2])
