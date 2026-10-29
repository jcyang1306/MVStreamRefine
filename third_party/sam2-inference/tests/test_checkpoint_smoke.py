import os

import numpy as np
import pytest
from PIL import Image

from sam2_inference import ImageSegmenter, StreamTracker, VideoTracker

CHECKPOINT = os.environ.get("SAM2_CHECKPOINT")
MODEL_TYPE = os.environ.get("SAM2_MODEL_TYPE", "tiny")
DEVICE = os.environ.get("SAM2_DEVICE", "auto")

pytestmark = [
    pytest.mark.checkpoint,
    pytest.mark.skipif(not CHECKPOINT, reason="SAM2_CHECKPOINT is not set"),
]


def _frame(offset=0):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[16:48, 16 + offset : 48 + offset] = 255
    return frame


def test_real_checkpoint_image_box():
    result = ImageSegmenter(CHECKPOINT, MODEL_TYPE, DEVICE).segment(
        _frame(), box=[16, 16, 48, 48], multimask=False
    )
    assert result.masks.shape == (1, 64, 64)
    assert result.scores.shape == (1,)


def test_real_checkpoint_jpeg_directory(tmp_path):
    for index in range(2):
        Image.fromarray(_frame(index)).save(tmp_path / f"{index:05d}.jpg")
    tracker = VideoTracker(CHECKPOINT, MODEL_TYPE, DEVICE)
    tracker.open(tmp_path)
    tracker.add_prompt(0, 1, box=[16, 16, 48, 48])
    results = list(tracker.track())
    tracker.close()
    assert [result.frame_index for result in results] == [0, 1]
    assert all(result.masks.shape == (1, 64, 64) for result in results)


def test_real_checkpoint_stream_two_frames():
    tracker = StreamTracker(CHECKPOINT, MODEL_TYPE, DEVICE)
    tracker.add_prompt(_frame(), box=[16, 16, 48, 48])
    result = tracker.track(_frame(1))
    tracker.close()
    assert result.frame_index == 1
    assert result.masks.shape == (1, 64, 64)
