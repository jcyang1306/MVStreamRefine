"""SAM 2.1 offline video-tracking adapter (PLAN section 8, Task 3).

Only the stable public API of the in-repo sam2-inference package is used
(``sam2_inference.VideoTracker``); private predictor methods are off limits.
``sam2_inference`` (and torch) are imported lazily so that dataset tooling and
unit tests run on machines without a GPU stack.

``VideoTracker.open()`` requires a directory of purely numeric JPEG names,
while ``data/`` mixes both camera streams; ``stage_jpeg_sequence`` builds a
per-camera symlink staging directory plus a manifest mapping staged positions
back to original frame indices.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class Sam2Prompt:
    """Manual first-frame prompt for object A (MVP: xyxy box on staged frame 0)."""

    object_id: int = 1
    frame_position: int = 0  # position within the staged sequence
    box: tuple[float, float, float, float] | None = None
    points: Sequence[Sequence[float]] | None = None
    labels: Sequence[int] | None = None
    mask: np.ndarray | None = None

    def validate(self) -> None:
        if self.box is None and self.points is None and self.mask is None:
            raise ValueError("prompt needs at least one of box / points / mask")
        if (self.points is None) != (self.labels is None):
            raise ValueError("points and labels must be provided together")


def stage_jpeg_sequence(
    data_root: str | Path,
    prefix: str,
    indices: Sequence[int],
    staging_dir: str | Path,
) -> list[int]:
    """Symlink data/frame-{idx:06d}_{prefix}_color.jpg as {pos:05d}.jpg.

    Returns the manifest (original frame index per staged position) and writes
    it to staging_dir/manifest.json.
    """
    data_root = Path(data_root).resolve()
    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    for stale in staging_dir.glob("*.jpg"):
        stale.unlink()

    manifest: list[int] = []
    for position, index in enumerate(indices):
        source = data_root / f"frame-{index:06d}_{prefix}_color.jpg"
        if not source.is_file():
            raise FileNotFoundError(f"missing staged source image: {source}")
        link = staging_dir / f"{position:05d}.jpg"
        link.symlink_to(source)
        manifest.append(int(index))

    (staging_dir / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def load_staging_manifest(staging_dir: str | Path) -> list[int]:
    return json.loads((Path(staging_dir) / "manifest.json").read_text())


def _default_tracker_factory(
    checkpoint: str, model_type: str, device: str
) -> Any:
    try:
        from sam2_inference import VideoTracker
    except ImportError as exc:
        raise RuntimeError(
            "sam2-inference (and torch) are not installed in this environment; "
            "run mask precomputation inside the CUDA container "
            "(docker compose run --rm reconstruction ...)"
        ) from exc

    return VideoTracker(checkpoint=checkpoint, model_type=model_type, device=device)


def release_gpu_memory() -> None:
    """Free CUDA cache between the two camera streams (PLAN section 8)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class SAM2Segmenter:
    """Runs one camera stream through SAM 2.1 video tracking on staged JPEGs."""

    def __init__(
        self,
        checkpoint: str,
        model_type: str = "tiny",
        device: str = "cuda",
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = False,
        tracker_factory: Callable[[str, str, str], Any] = _default_tracker_factory,
    ) -> None:
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.device = device
        self.offload_video_to_cpu = offload_video_to_cpu
        self.offload_state_to_cpu = offload_state_to_cpu
        self._tracker_factory = tracker_factory

    @classmethod
    def from_config(cls, config: dict[str, Any], **overrides: Any) -> "SAM2Segmenter":
        sam2 = config["sam2"]
        kwargs: dict[str, Any] = {
            "checkpoint": sam2["checkpoint"],
            "model_type": sam2.get("model_type", "tiny"),
            "device": config.get("devices", {}).get("sam2", "cuda"),
            "offload_video_to_cpu": sam2.get("offload_video_to_cpu", True),
            "offload_state_to_cpu": sam2.get("offload_state_to_cpu", False),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    def track_sequence(
        self, staging_dir: str | Path, prompt: Sam2Prompt
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (original_frame_index, bool mask H x W) for every staged frame.

        Video tracking carries no quality scores (result.scores is None);
        downstream quality gating must use mask area / IoU / depth statistics.
        """
        prompt.validate()
        staging_dir = Path(staging_dir)
        manifest = load_staging_manifest(staging_dir)

        tracker = self._tracker_factory(self.checkpoint, self.model_type, self.device)
        try:
            tracker.open(
                staging_dir,
                offload_video_to_cpu=self.offload_video_to_cpu,
                offload_state_to_cpu=self.offload_state_to_cpu,
            )
            tracker.add_prompt(
                prompt.frame_position,
                prompt.object_id,
                box=prompt.box,
                points=prompt.points,
                labels=prompt.labels,
                mask=prompt.mask,
            )
            for result in tracker.track():
                if result.frame_index >= len(manifest):
                    raise RuntimeError(
                        f"tracker produced frame {result.frame_index} beyond the "
                        f"{len(manifest)}-frame staging manifest"
                    )
                mask = np.asarray(result.masks[0], dtype=bool)
                yield manifest[result.frame_index], mask
        finally:
            tracker.close()
