"""Real-time frame-by-frame tracking without camera or OpenCV dependencies."""

from collections import OrderedDict
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ._common import (
    build_video_predictor,
    resolve_device,
    validate_prompts,
    validate_rgb_frame,
)
from .results import TrackingResult
from .video import _tracking_result

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class StreamTracker:
    """Bounded-memory SAM 2.1 tracker consuming RGB NumPy frames."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        model_type: str = "tiny",
        device: Optional[str] = "auto",
        *,
        memory_frames: Optional[int] = None,
        predictor: Any = None,
    ) -> None:
        self.device = resolve_device(device)
        self.predictor = predictor or build_video_predictor(
            checkpoint, model_type, self.device
        )
        minimum = int(getattr(self.predictor, "num_maskmem", 7)) + 2
        self.memory_frames = max(minimum, 8) if memory_frames is None else memory_frames
        if not isinstance(self.memory_frames, int) or self.memory_frames < minimum:
            raise ValueError(f"memory_frames must be an integer >= {minimum}")
        self.state = None
        self.frame_index = -1
        self._shape = None
        self._preflight_done = False
        self._mean = torch.tensor(_MEAN, device=self.device)[:, None, None]
        self._std = torch.tensor(_STD, device=self.device)[:, None, None]

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(frame).to(self.device)
        tensor = tensor.permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
        size = int(self.predictor.image_size)
        tensor = F.interpolate(
            tensor, size=(size, size), mode="bilinear", align_corners=False, antialias=True
        )[0]
        return (tensor - self._mean) / self._std

    def _new_state(self, frame: np.ndarray):
        height, width = frame.shape[:2]
        state = {
            "images": [self._preprocess(frame)],
            "num_frames": 1,
            "offload_video_to_cpu": False,
            "offload_state_to_cpu": False,
            "video_height": height,
            "video_width": width,
            "device": torch.device(self.device),
            "storage_device": torch.device(self.device),
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {},
        }
        self.predictor._get_image_feature(state, frame_idx=0, batch_size=1)
        return state

    def add_prompt(
        self,
        frame: Optional[np.ndarray] = None,
        *,
        object_id: int = 1,
        points=None,
        labels=None,
        box=None,
        mask=None,
        clear_old_points: bool = True,
    ) -> TrackingResult:
        """Add or correct an object prompt on the current stream frame."""
        validate_prompts(points=points, labels=labels, box=box, mask=mask)
        if self.state is None:
            if frame is None:
                raise ValueError("the first prompt requires an RGB frame")
            frame = validate_rgb_frame(frame)
            self._shape = frame.shape[:2]
            self.state = self._new_state(frame)
            self.frame_index = 0
        elif frame is not None:
            frame = validate_rgb_frame(frame)
            if frame.shape[:2] != self._shape:
                raise ValueError("all stream frames must keep the initial resolution")

        if mask is not None:
            output = self.predictor.add_new_mask(
                self.state, self.frame_index, int(object_id), np.asarray(mask, dtype=bool)
            )
        else:
            output = self.predictor.add_new_points_or_box(
                inference_state=self.state,
                frame_idx=self.frame_index,
                obj_id=int(object_id),
                points=None if points is None else np.asarray(points, np.float32),
                labels=None if labels is None else np.asarray(labels, np.int32),
                box=None if box is None else np.asarray(box, np.float32),
                clear_old_points=clear_old_points,
            )
        self._preflight_done = False
        return _tracking_result(*output)

    @torch.inference_mode()
    def track(self, frame: np.ndarray) -> TrackingResult:
        """Track all prompted objects on the next RGB frame."""
        if self.state is None:
            raise RuntimeError("call add_prompt() with the first frame before track()")
        frame = validate_rgb_frame(frame)
        if frame.shape[:2] != self._shape:
            raise ValueError("all stream frames must keep the initial resolution")
        if not self._preflight_done:
            self.predictor.propagate_in_video_preflight(self.state)
            for obj_idx in self.state["obj_idx_to_id"]:
                self.state["frames_tracked_per_obj"][obj_idx][self.frame_index] = {
                    "reverse": False
                }
            self._preflight_done = True

        self.state["images"].append(self._preprocess(frame))
        self.state["num_frames"] = len(self.state["images"])
        self.frame_index += 1
        per_object_logits = []
        for obj_idx in self.state["obj_idx_to_id"]:
            output_dict = self.state["output_dict_per_obj"][obj_idx]
            current_out, pred_masks = self.predictor._run_single_frame_inference(
                inference_state=self.state,
                output_dict=output_dict,
                frame_idx=self.frame_index,
                batch_size=1,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
            )
            output_dict["non_cond_frame_outputs"][self.frame_index] = current_out
            self.state["frames_tracked_per_obj"][obj_idx][self.frame_index] = {
                "reverse": False
            }
            per_object_logits.append(pred_masks)

        logits = torch.cat(per_object_logits, dim=0)
        _, logits = self.predictor._get_orig_video_res_output(self.state, logits)
        self._prune()
        return _tracking_result(self.frame_index, self.state["obj_ids"], logits)

    def _prune(self) -> None:
        cutoff = self.frame_index - self.memory_frames
        if cutoff <= 0:
            return
        for index in range(cutoff):
            if index < len(self.state["images"]):
                self.state["images"][index] = None
            self.state["cached_features"].pop(index, None)
        for output in self.state["output_dict_per_obj"].values():
            stale = [
                index
                for index in output["non_cond_frame_outputs"]
                if index < cutoff
            ]
            for index in stale:
                output["non_cond_frame_outputs"].pop(index, None)

    def reset(self) -> None:
        if self.state is not None:
            self.predictor.reset_state(self.state)
        self.state = None
        self.frame_index = -1
        self._shape = None
        self._preflight_done = False

    close = reset
