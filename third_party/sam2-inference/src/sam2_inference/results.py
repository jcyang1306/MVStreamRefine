"""Typed NumPy results returned by the public inference API."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SegmentationResult:
    """Image segmentation output.

    ``masks`` contains thresholded full-resolution masks and ``logits`` contains
    the predictor's low-resolution mask logits for iterative prompting.
    """

    masks: np.ndarray
    logits: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class TrackingResult:
    """Video or stream tracking output at the source frame resolution."""

    frame_index: int
    object_ids: Tuple[int, ...]
    masks: np.ndarray
    logits: np.ndarray
    scores: Optional[np.ndarray] = None
