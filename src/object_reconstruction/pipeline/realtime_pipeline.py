"""Realtime reconstruction state machine (UI-free, unit-testable core).

States:
    PREVIEW       live view only; waiting for an ROI (key B in the UI)
    MASK_CONFIRM  SAM2 tracks the prompted object; user confirms (R) or
                  reselects (B) - nothing is integrated yet
    RUNNING       every synchronized frame is tracked and fed to the shared
                  ReconstructionEngine (keyframe gate -> optional ICP -> TSDF)
    PAUSED        tracking continues so the object is not lost; integration
                  is suspended (P toggles)
    LOST          the mask stayed below min_track_area_px for
                  lost_after_frames consecutive frames; tracking stops and a
                  new ROI (B) is required - the model is kept

Key handling (called by the UI layer with plain characters):
    r confirm mask / resume        p pause / resume
    c clear tracking -> PREVIEW    n new model (fresh engine) -> PREVIEW

ROI selection itself is a UI concern (OpenCV window); the UI calls
``set_roi(rgb, box)`` which (re)prompts the segmenter from any state.
The engine is created via an injected factory so N can rebuild a fresh TSDF;
segmenter and engine are injected, keeping this class free of torch/open3d.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np

from .reconstruction_engine import EngineResult, ReconstructionEngine

logger = logging.getLogger(__name__)


class State(str, Enum):
    PREVIEW = "PREVIEW"
    MASK_CONFIRM = "MASK_CONFIRM"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    LOST = "LOST"


@dataclass
class FrameOutput:
    """Everything the UI needs to render one processed live frame."""

    state: State
    mask: np.ndarray | None = None
    mask_area_px: int = 0
    engine_result: EngineResult | None = None
    message: str | None = None


class RealtimePipeline:
    def __init__(
        self,
        engine_factory: Callable[[], ReconstructionEngine],
        segmenter: Any,
        config: dict[str, Any],
    ) -> None:
        self._engine_factory = engine_factory
        self.engine = engine_factory()
        self.segmenter = segmenter
        tracking = config.get("realtime", {}).get("tracking", {})
        self.min_track_area_px = int(tracking.get("min_mask_area_px", 500))
        self.lost_after_frames = int(tracking.get("lost_after_frames", 10))

        self.state = State.PREVIEW
        self._low_area_streak = 0
        self.frames_seen = 0

    # -- UI events -------------------------------------------------------------

    def set_roi(
        self, rgb: np.ndarray, box: tuple[float, float, float, float]
    ) -> np.ndarray:
        """(Re)prompt the tracker with a first-frame box; -> MASK_CONFIRM."""
        mask = self.segmenter.start_object(rgb, box)
        self._low_area_streak = 0
        self.state = State.MASK_CONFIRM
        logger.info("ROI set (box=%s, mask_area=%d); confirm with R or "
                    "reselect with B", tuple(box), int(mask.sum()))
        return mask

    def handle_key(self, key: str) -> str | None:
        """Apply one hotkey; returns a short status message or None."""
        key = key.lower()
        if key == "r":
            if self.state == State.MASK_CONFIRM:
                self.state = State.RUNNING
                return "mask confirmed; integrating"
            if self.state == State.PAUSED:
                self.state = State.RUNNING
                return "resumed"
        elif key == "p":
            if self.state == State.RUNNING:
                self.state = State.PAUSED
                return "paused (tracking continues, no integration)"
            if self.state == State.PAUSED:
                self.state = State.RUNNING
                return "resumed"
        elif key == "c":
            self.segmenter.reset()
            self._low_area_streak = 0
            self.state = State.PREVIEW
            return "tracking cleared; model kept"
        elif key == "n":
            self.segmenter.reset()
            self.engine = self._engine_factory()
            self._low_area_streak = 0
            self.state = State.PREVIEW
            return "new model started (TSDF + keyframe state reset)"
        return None

    # -- per-frame processing ----------------------------------------------------

    def process(self, packet: Any) -> FrameOutput:
        """Track (and in RUNNING integrate) one synchronized FramePacket."""
        self.frames_seen += 1
        if self.state in (State.PREVIEW, State.LOST):
            return FrameOutput(state=self.state)

        mask = self.segmenter.track(packet.cam.rgb)
        area = int(mask.sum())
        if self._update_lost(area):
            return FrameOutput(
                state=self.state, mask=mask, mask_area_px=area,
                message=f"tracking lost (area<{self.min_track_area_px}px "
                        f"for {self.lost_after_frames} frames); press B",
            )

        result = None
        if self.state == State.RUNNING:
            result = self.engine.process(packet, mask)
        return FrameOutput(
            state=self.state, mask=mask, mask_area_px=area, engine_result=result
        )

    def _update_lost(self, mask_area_px: int) -> bool:
        """Track consecutive low-area frames; switch to LOST at the threshold."""
        if mask_area_px >= self.min_track_area_px:
            self._low_area_streak = 0
            return False
        self._low_area_streak += 1
        if self._low_area_streak < self.lost_after_frames:
            return False
        self.segmenter.reset()
        self.state = State.LOST
        logger.warning("tracking lost after %d consecutive low-area frames",
                       self._low_area_streak)
        return True
