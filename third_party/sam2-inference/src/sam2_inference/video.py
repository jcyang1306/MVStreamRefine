"""Stable offline video tracking API."""

from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from ._common import (
    as_numpy,
    build_video_predictor,
    resolve_device,
    validate_prompts,
)
from .results import TrackingResult


def _tracking_result(frame_index, object_ids, logits) -> TrackingResult:
    logits_array = as_numpy(logits)
    if logits_array.ndim == 4 and logits_array.shape[1] == 1:
        logits_array = logits_array[:, 0]
    return TrackingResult(
        frame_index=int(frame_index),
        object_ids=tuple(int(obj_id) for obj_id in object_ids),
        masks=logits_array > 0.0,
        logits=logits_array,
    )


class VideoTracker:
    """Stateful SAM 2.1 tracker for MP4 files or numbered JPEG directories."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        model_type: str = "tiny",
        device: Optional[str] = "auto",
        *,
        predictor: Any = None,
    ) -> None:
        self.device = resolve_device(device)
        self.predictor = predictor or build_video_predictor(
            checkpoint, model_type, self.device
        )
        self.state = None

    @property
    def is_open(self) -> bool:
        return self.state is not None

    def open(
        self,
        video_path,
        *,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        async_loading_frames: bool = False,
    ) -> "VideoTracker":
        if self.is_open:
            raise RuntimeError("a video is already open; call close() first")
        path = Path(video_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"video path not found: {path}")
        self.state = self.predictor.init_state(
            video_path=str(path),
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=async_loading_frames,
        )
        return self

    def _require_open(self):
        if self.state is None:
            raise RuntimeError("call open() before adding prompts or tracking")
        return self.state

    def add_prompt(
        self,
        frame_index: int,
        object_id: int,
        *,
        points=None,
        labels=None,
        box=None,
        mask=None,
        clear_old_points: bool = True,
    ) -> TrackingResult:
        state = self._require_open()
        if not isinstance(frame_index, int) or not 0 <= frame_index < state["num_frames"]:
            raise ValueError("frame_index is outside the opened video")
        validate_prompts(points=points, labels=labels, box=box, mask=mask)
        if mask is not None:
            output = self.predictor.add_new_mask(
                state, frame_index, int(object_id), np.asarray(mask, dtype=bool)
            )
        else:
            output = self.predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=frame_index,
                obj_id=int(object_id),
                points=None if points is None else np.asarray(points, np.float32),
                labels=None if labels is None else np.asarray(labels, np.int32),
                box=None if box is None else np.asarray(box, np.float32),
                clear_old_points=clear_old_points,
            )
        return _tracking_result(*output)

    def track(
        self,
        *,
        start_frame_index: Optional[int] = None,
        max_frames: Optional[int] = None,
        reverse: bool = False,
    ) -> Iterator[TrackingResult]:
        state = self._require_open()
        for output in self.predictor.propagate_in_video(
            state,
            start_frame_idx=start_frame_index,
            max_frame_num_to_track=max_frames,
            reverse=reverse,
        ):
            yield _tracking_result(*output)

    def reset(self) -> None:
        if self.state is not None:
            self.predictor.reset_state(self.state)

    def close(self) -> None:
        if self.state is not None:
            self.predictor.reset_state(self.state)
            self.state = None

    def __enter__(self) -> "VideoTracker":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
