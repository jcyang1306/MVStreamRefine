"""SAM 2.1 segmentation of a recorded session into the mask cache (P2 O1).

Uses only the public `sam2_inference.StreamTracker` API (package vendored from V1 under
`third_party/sam2-inference`, ADR-0006): the prompt (xyxy box and/or points) is given on one
frame, then every following emitted frame is tracked with bounded memory. torch and
`sam2_inference` are imported lazily, so the rest of P2 and its tests do not need a GPU.

Per frame: `mask = logits > 0`; `score` = mean sigmoid(logit) inside the mask (the video
predictor has no IoU score); status `EMPTY` below `sam2.min_mask_area_px`, `LOW_CONFIDENCE`
below `sam2.low_confidence_score`. Frames before the prompt frame are not segmented.
Input frames are the WRITTEN frames `ReplaySource` emits (colour/depth pairs from the same
exposure), read with checksum verification.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from mvsr.core.errors import ChecksumMismatchError, MvsrError
from mvsr.io.atomic import sha256_file
from mvsr.io.session_writer import MANIFEST_NAME
from mvsr.observation.config import ObservationConfig, Sam2Config
from mvsr.observation.masks import MaskCacheWriter, MaskStatus, mask_revision_id
from mvsr.sensors.config import SensorsConfig
from mvsr.sensors.replay import ReplaySource
from mvsr.software import software_info

OBJECT_ID = 1


class SegmentationError(MvsrError):
    """The segmenter cannot run (missing package/checkpoint) or the prompt is invalid."""


@dataclass(frozen=True)
class MaskPrompt:
    """Initial prompt on `frame_id` (None = first emitted frame). Pixels of the colour image."""

    frame_id: int | None = None
    box_xyxy: tuple[float, float, float, float] | None = None
    points: tuple[tuple[float, float], ...] | None = None
    labels: tuple[int, ...] | None = None
    source: str = "cli"

    def __post_init__(self) -> None:
        if self.box_xyxy is None and self.points is None:
            raise SegmentationError("prompt needs a box and/or points")
        if (self.points is None) != (self.labels is None) or (
            self.points is not None
            and self.labels is not None
            and len(self.points) != len(self.labels)
        ):
            raise SegmentationError("points and labels must be given together with equal length")
        if self.box_xyxy is not None:
            x1, y1, x2, y2 = self.box_xyxy
            if not (x2 > x1 and y2 > y1):
                raise SegmentationError(
                    f"box must satisfy x2 > x1 and y2 > y1, got {self.box_xyxy}"
                )
        if self.labels is not None and any(v not in (0, 1) for v in self.labels):
            raise SegmentationError("point labels must be 0 (background) or 1 (object)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "box_xyxy": None if self.box_xyxy is None else [float(v) for v in self.box_xyxy],
            "points": None
            if self.points is None
            else [[float(a), float(b)] for a, b in self.points],
            "labels": None if self.labels is None else [int(v) for v in self.labels],
            "source": self.source,
        }


TrackerFactory = Callable[[Sam2Config], Any]


def verify_checkpoint(cfg: Sam2Config) -> Path:
    path = cfg.checkpoint_path()
    if not path.is_file():
        raise SegmentationError(
            f"SAM 2.1 checkpoint not found: {path} (copy it from ref/MVStreamRefine/checkpoints/)"
        )
    digest = sha256_file(path)
    if digest != cfg.checkpoint_sha256:
        raise ChecksumMismatchError(
            f"{path}: sha256 {digest} != configured {cfg.checkpoint_sha256}"
        )
    return path


def default_tracker_factory(cfg: Sam2Config) -> Any:
    path = verify_checkpoint(cfg)
    try:
        from sam2_inference import StreamTracker
    except ImportError as exc:
        raise SegmentationError(
            "sam2-inference / torch not installed (rebuild mvsr:cu121)"
        ) from exc
    return StreamTracker(
        str(path), model_type=cfg.model_type, device=cfg.device, memory_frames=cfg.memory_frames
    )


def runtime_info() -> dict[str, Any]:
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        return {
            "torch": str(torch.__version__),
            "cuda_available": torch.cuda.is_available(),
            "gpu": gpu,
        }
    except ImportError:
        return {"torch": None}


def mask_and_score(result: Any) -> tuple[npt.NDArray[np.bool_], float | None]:
    """Object-1 mask and mean in-mask probability from a `TrackingResult`."""
    logits = np.asarray(result.logits[0], dtype=np.float32)
    mask = np.asarray(result.masks[0], dtype=bool)
    if logits.shape != mask.shape:
        raise SegmentationError(f"logits {logits.shape} and mask {mask.shape} differ")
    if not mask.any():
        return mask, None
    inside = logits[mask].astype(np.float64)
    return mask, float(np.mean(1.0 / (1.0 + np.exp(-inside))))


def classify(mask: npt.NDArray[np.bool_], score: float | None, cfg: Sam2Config) -> MaskStatus:
    if int(mask.sum()) < cfg.min_mask_area_px:
        return MaskStatus.EMPTY
    if score is None or score < cfg.low_confidence_score:
        return MaskStatus.LOW_CONFIDENCE
    return MaskStatus.OK


def _ids_digest(ids: Sequence[int]) -> str:
    return hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()[:16]


def segment_session(
    session_dir: str | Path,
    prompt: MaskPrompt,
    cfg: ObservationConfig,
    *,
    sensors_cfg: SensorsConfig | None = None,
    camera_id: str = "wrist",
    tracker_factory: TrackerFactory | None = None,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    """Segment the emitted frames from the prompt frame on; returns the revision metadata.

    The cache goes to `<cache_root or session_dir>/derived/masks/...`.
    """
    src = ReplaySource(session_dir, camera_id, config=sensors_cfg)
    ids = list(src.frame_ids)
    if not ids:
        raise SegmentationError("session has no emitted frames")
    start = ids[0] if prompt.frame_id is None else prompt.frame_id
    if start not in ids:
        raise SegmentationError(f"prompt frame {start} is not an emitted frame of {camera_id}")
    ids = ids[ids.index(start) :]
    if limit is not None:
        ids = ids[:limit]
    s = cfg.sam2
    params = {
        "producer": "sam2_inference.StreamTracker",
        "model_type": s.model_type,
        "checkpoint_sha256": s.checkpoint_sha256,
        "memory_frames": s.memory_frames,
        "min_mask_area_px": s.min_mask_area_px,
        "low_confidence_score": s.low_confidence_score,
        "prompt": {**prompt.to_dict(), "frame_id": start},
        "session_id": src.session_id,
        "session_manifest_sha256": sha256_file(Path(src.reader.dir) / MANIFEST_NAME),
        "camera_id": camera_id,
        "frame_ids_sha": _ids_digest(ids),
        "n_frames": len(ids),
    }
    revision = mask_revision_id(f"sam2.1-{s.model_type}", params)
    factory = tracker_factory or default_tracker_factory
    tracker = factory(s)
    writer = MaskCacheWriter(cache_root or src.reader.dir, camera_id, revision, {"params": params})
    t0 = time.monotonic()
    try:
        for k, fid in enumerate(ids):
            # writable copy: torch.from_numpy on a read-only array is undefined behaviour
            rgb = np.array(src.reader.read_frame(camera_id, fid).color_rgb_u8, copy=True)
            if k == 0:
                result = tracker.add_prompt(
                    rgb,
                    object_id=OBJECT_ID,
                    box=None if prompt.box_xyxy is None else list(prompt.box_xyxy),
                    points=None if prompt.points is None else [list(p) for p in prompt.points],
                    labels=None if prompt.labels is None else list(prompt.labels),
                )
            else:
                result = tracker.track(rgb)
            mask, score = mask_and_score(result)
            writer.add(fid, mask, classify(mask, score, s), score)
            if progress is not None:
                progress(k + 1, len(ids))
    finally:
        reset = getattr(tracker, "reset", None)
        if reset is not None:
            reset()
    wall = time.monotonic() - t0
    return writer.finalize(
        software=software_info(),
        runtime=runtime_info() if tracker_factory is None else {"tracker": "injected"},
        wall_s=round(wall, 3),
        fps=round(len(ids) / wall, 2) if wall > 0 else None,
    )
