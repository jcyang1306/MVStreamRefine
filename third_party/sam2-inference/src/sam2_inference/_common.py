"""Shared validation and model construction helpers."""

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

MODEL_CONFIGS = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


def resolve_device(device: Optional[str]) -> str:
    if device not in (None, "auto"):
        return str(device)
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def validate_model_args(checkpoint: Optional[str], model_type: str) -> tuple[str, str]:
    if model_type not in MODEL_CONFIGS:
        choices = ", ".join(MODEL_CONFIGS)
        raise ValueError(f"model_type must be one of: {choices}")
    if checkpoint is None:
        raise ValueError("checkpoint path is required")
    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return str(path), MODEL_CONFIGS[model_type]


def validate_rgb_frame(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a numpy.ndarray")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must have shape (H, W, 3), got {frame.shape}")
    if frame.dtype != np.uint8:
        raise ValueError(f"frame must have dtype uint8, got {frame.dtype}")
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("frame dimensions must be non-zero")
    return np.ascontiguousarray(frame)


def validate_prompts(points=None, labels=None, box=None, mask=None) -> None:
    if (points is None) != (labels is None):
        raise ValueError("points and labels must be provided together")
    if points is None and box is None and mask is None:
        raise ValueError("provide points/labels, box, or mask")
    if mask is not None and (points is not None or box is not None):
        raise ValueError("mask cannot be combined with points or box")
    if points is not None:
        points_array = np.asarray(points)
        labels_array = np.asarray(labels)
        if points_array.ndim != 2 or points_array.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
        if labels_array.shape != (points_array.shape[0],):
            raise ValueError("labels must have shape (N,)")
        if not np.isin(labels_array, (0, 1)).all():
            raise ValueError("point labels must contain only 0 or 1")
    if box is not None:
        box_array = np.asarray(box)
        if box_array.shape != (4,):
            raise ValueError("box must have shape (4,) in xyxy order")
        if box_array[2] <= box_array[0] or box_array[3] <= box_array[1]:
            raise ValueError("box must satisfy x2 > x1 and y2 > y1")
    if mask is not None and np.asarray(mask).ndim != 2:
        raise ValueError("mask must have shape (H, W)")


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def build_image_predictor(
    checkpoint: Optional[str],
    model_type: str,
    device: str,
    builder: Optional[Callable[..., Any]] = None,
):
    checkpoint, config = validate_model_args(checkpoint, model_type)
    if builder is None:
        from sam2.build_sam import build_sam2

        builder = build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    return SAM2ImagePredictor(builder(config, checkpoint, device=device))


def build_video_predictor(
    checkpoint: Optional[str],
    model_type: str,
    device: str,
    builder: Optional[Callable[..., Any]] = None,
):
    checkpoint, config = validate_model_args(checkpoint, model_type)
    if builder is None:
        from sam2.build_sam import build_sam2_video_predictor

        builder = build_sam2_video_predictor
    return builder(config, checkpoint, device=device, vos_optimized=False)
