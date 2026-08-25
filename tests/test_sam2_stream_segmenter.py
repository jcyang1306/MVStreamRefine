"""SAM2StreamSegmenter adapter tests with a stub StreamTracker."""

import numpy as np
import pytest

from object_reconstruction.segmentation.sam2_stream_segmenter import SAM2StreamSegmenter

HEIGHT, WIDTH = 48, 64


class StubResult:
    def __init__(self, mask):
        self.masks = [mask]


class StubTracker:
    def __init__(self):
        self.resets = 0
        self.prompts = []
        self.tracked = 0

    def reset(self):
        self.resets += 1

    def add_prompt(self, frame, *, object_id, box):
        self.prompts.append((frame.shape, object_id, box))
        mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
        x1, y1, x2, y2 = (int(v) for v in box)
        mask[y1:y2, x1:x2] = True
        return StubResult(mask)

    def track(self, frame):
        self.tracked += 1
        return StubResult(np.ones((HEIGHT, WIDTH), dtype=np.float32))


@pytest.fixture()
def stub():
    trackers = []

    def factory(checkpoint, model_type, device, memory_frames):
        trackers.append(StubTracker())
        trackers[-1].factory_args = (checkpoint, model_type, device, memory_frames)
        return trackers[-1]

    segmenter = SAM2StreamSegmenter(
        checkpoint="/models/x.pt", model_type="tiny", device="cuda",
        object_id=3, memory_frames=16, tracker_factory=factory,
    )
    return segmenter, trackers


def rgb():
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


def test_start_object_prompts_box_and_returns_bool_mask(stub):
    segmenter, trackers = stub
    mask = segmenter.start_object(rgb(), (10, 5, 30, 25))

    assert len(trackers) == 1
    tracker = trackers[0]
    assert tracker.factory_args == ("/models/x.pt", "tiny", "cuda", 16)
    assert tracker.resets == 1  # reset before every new prompt
    assert tracker.prompts == [((HEIGHT, WIDTH, 3), 3, (10.0, 5.0, 30.0, 25.0))]
    assert mask.dtype == bool and mask.sum() == 20 * 20
    assert segmenter.active


def test_track_requires_active_object(stub):
    segmenter, _ = stub
    with pytest.raises(RuntimeError, match="start_object"):
        segmenter.track(rgb())


def test_track_and_reset_cycle(stub):
    segmenter, trackers = stub
    segmenter.start_object(rgb(), (0, 0, 10, 10))
    mask = segmenter.track(rgb())
    assert mask.dtype == bool and mask.all()
    assert trackers[0].tracked == 1

    segmenter.reset()
    assert not segmenter.active
    assert trackers[0].resets == 2
    with pytest.raises(RuntimeError):
        segmenter.track(rgb())

    # Re-prompting reuses the same (expensive) tracker instance.
    segmenter.start_object(rgb(), (0, 0, 5, 5))
    assert len(trackers) == 1


def test_from_config_reads_sam2_and_tracking_sections():
    captured = {}

    def factory(checkpoint, model_type, device, memory_frames):
        captured.update(checkpoint=checkpoint, model_type=model_type,
                        device=device, memory_frames=memory_frames)
        return StubTracker()

    config = {
        "sam2": {"checkpoint": "/models/y.pt", "model_type": "small", "object_id": 2},
        "devices": {"sam2": "cuda:1"},
        "realtime": {"tracking": {"memory_frames": 12}},
    }
    segmenter = SAM2StreamSegmenter.from_config(config, tracker_factory=factory)
    segmenter.start_object(rgb(), (0, 0, 4, 4))
    assert captured == {"checkpoint": "/models/y.pt", "model_type": "small",
                        "device": "cuda:1", "memory_frames": 12}
    assert segmenter.object_id == 2
