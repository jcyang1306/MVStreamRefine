"""Stable image segmentation API."""

from typing import Any, Optional

import numpy as np

from ._common import (
    as_numpy,
    build_image_predictor,
    resolve_device,
    validate_prompts,
    validate_rgb_frame,
)
from .results import SegmentationResult


class ImageSegmenter:
    """Promptable SAM 2.1 image segmenter."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        model_type: str = "tiny",
        device: Optional[str] = "auto",
        *,
        predictor: Any = None,
    ) -> None:
        self.device = resolve_device(device)
        self.predictor = predictor or build_image_predictor(
            checkpoint, model_type, self.device
        )

    def segment(
        self,
        image: np.ndarray,
        *,
        points=None,
        labels=None,
        box=None,
        multimask: bool = True,
    ) -> SegmentationResult:
        """Segment one RGB uint8 image from point and/or box prompts."""
        image = validate_rgb_frame(image)
        validate_prompts(points=points, labels=labels, box=box)
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            point_coords=None if points is None else np.asarray(points, np.float32),
            point_labels=None if labels is None else np.asarray(labels, np.int32),
            box=None if box is None else np.asarray(box, np.float32),
            multimask_output=bool(multimask),
            return_logits=False,
        )
        return SegmentationResult(
            masks=as_numpy(masks).astype(bool, copy=False),
            logits=as_numpy(logits),
            scores=as_numpy(scores),
        )

    def reset(self) -> None:
        """Release cached image embeddings."""
        reset = getattr(self.predictor, "reset_predictor", None)
        if reset is not None:
            reset()
