"""Segmenter protocol (PLAN section 8): reconstruction never talks to SAM directly."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class ObjectSegmenter(Protocol):
    """Produces a bool H x W mask of the tracked object for each RGB frame."""

    def initialize(self, frame_rgb: np.ndarray, prompt: Any) -> np.ndarray:
        """Anchor the object on the first frame and return its mask."""
        ...

    def segment(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Return the object mask for the next frame of the stream."""
        ...
