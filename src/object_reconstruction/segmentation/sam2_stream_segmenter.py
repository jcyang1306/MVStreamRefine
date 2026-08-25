"""SAM 2.1 realtime frame-by-frame tracking adapter (realtime pipeline).

Wraps the stable ``sam2_inference.StreamTracker`` API: a first-frame box
prompt starts the object, then every live RGB frame is tracked with bounded
memory. Only the public API is used; ``sam2_inference`` (and torch) are
imported lazily so unit tests run without the GPU stack.

Lifecycle:
    start_object(rgb, box)  new prompt (resets any previous tracking)
    track(rgb)              mask for the next live frame
    reset()                 drop tracking state (model/TSDF untouched)
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def _default_stream_factory(
    checkpoint: str, model_type: str, device: str, memory_frames: int | None
) -> Any:
    try:
        from sam2_inference import StreamTracker
    except ImportError as exc:
        raise RuntimeError(
            "sam2-inference (and torch) are not installed in this environment; "
            "run the realtime pipeline inside the CUDA container"
        ) from exc

    return StreamTracker(
        checkpoint=checkpoint,
        model_type=model_type,
        device=device,
        memory_frames=memory_frames,
    )


class SAM2StreamSegmenter:
    """Tracks one object (object A) across a live RGB stream."""

    def __init__(
        self,
        checkpoint: str,
        model_type: str = "tiny",
        device: str = "cuda",
        object_id: int = 1,
        memory_frames: int | None = None,
        tracker_factory: Callable[..., Any] = _default_stream_factory,
    ) -> None:
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.device = device
        self.object_id = int(object_id)
        self.memory_frames = memory_frames
        self._tracker_factory = tracker_factory
        self._tracker: Any = None
        self.active = False  # True between start_object() and reset()

    @classmethod
    def from_config(cls, config: dict[str, Any], **overrides: Any) -> "SAM2StreamSegmenter":
        sam2 = config["sam2"]
        tracking = config.get("realtime", {}).get("tracking", {})
        kwargs: dict[str, Any] = {
            "checkpoint": sam2["checkpoint"],
            "model_type": sam2.get("model_type", "tiny"),
            "device": config.get("devices", {}).get("sam2", "cuda"),
            "object_id": sam2.get("object_id", 1),
            "memory_frames": tracking.get("memory_frames"),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    def _ensure_tracker(self) -> Any:
        if self._tracker is None:
            self._tracker = self._tracker_factory(
                self.checkpoint, self.model_type, self.device, self.memory_frames
            )
        return self._tracker

    @staticmethod
    def _mask_from(result: Any) -> np.ndarray:
        return np.asarray(result.masks[0], dtype=bool)

    def start_object(
        self, rgb: np.ndarray, box: tuple[float, float, float, float]
    ) -> np.ndarray:
        """(Re)start tracking from a first-frame xyxy box; returns its mask."""
        tracker = self._ensure_tracker()
        tracker.reset()
        result = tracker.add_prompt(
            rgb, object_id=self.object_id, box=tuple(float(v) for v in box)
        )
        self.active = True
        return self._mask_from(result)

    def track(self, rgb: np.ndarray) -> np.ndarray:
        """Mask for the next live frame; requires an active object."""
        if not self.active:
            raise RuntimeError("call start_object() before track()")
        return self._mask_from(self._tracker.track(rgb))

    def reset(self) -> None:
        if self._tracker is not None:
            self._tracker.reset()
        self.active = False
