import numpy as np
import pytest

from sam2_inference import ImageSegmenter, SegmentationResult


class FakeImagePredictor:
    def __init__(self):
        self.image = None
        self.reset_count = 0

    def set_image(self, image):
        self.image = image

    def predict(self, **kwargs):
        count = 3 if kwargs["multimask_output"] else 1
        height, width = self.image.shape[:2]
        return (
            np.ones((count, height, width), dtype=bool),
            np.linspace(0.5, 0.9, count, dtype=np.float32),
            np.ones((count, 256, 256), dtype=np.float32),
        )

    def reset_predictor(self):
        self.reset_count += 1


def test_image_result_and_multimask():
    predictor = FakeImagePredictor()
    segmenter = ImageSegmenter(predictor=predictor, device="cpu")
    result = segmenter.segment(
        np.zeros((8, 10, 3), dtype=np.uint8),
        points=[[2, 3]],
        labels=[1],
        box=[1, 1, 9, 7],
        multimask=False,
    )
    assert isinstance(result, SegmentationResult)
    assert result.masks.dtype == np.bool_
    assert result.masks.shape == (1, 8, 10)
    assert result.logits.shape == (1, 256, 256)
    segmenter.reset()
    assert predictor.reset_count == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"points": [[1, 2]]}, "together"),
        ({"points": [[1, 2]], "labels": [2]}, "only 0 or 1"),
        ({"box": [2, 2, 1, 4]}, "x2 > x1"),
        ({}, "provide"),
    ],
)
def test_image_prompt_validation(kwargs, message):
    segmenter = ImageSegmenter(predictor=FakeImagePredictor(), device="cpu")
    with pytest.raises(ValueError, match=message):
        segmenter.segment(np.zeros((4, 4, 3), dtype=np.uint8), **kwargs)


def test_rgb_validation():
    segmenter = ImageSegmenter(predictor=FakeImagePredictor(), device="cpu")
    with pytest.raises(ValueError, match="dtype uint8"):
        segmenter.segment(np.zeros((4, 4, 3)), box=[0, 0, 3, 3])
