"""SAM2Segmenter adapter tests with a stub tracker (no torch / checkpoint)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from object_reconstruction.segmentation.sam2_segmenter import (
    SAM2Segmenter,
    Sam2Prompt,
    load_staging_manifest,
    stage_jpeg_sequence,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"


def test_stage_jpeg_sequence_on_real_data(tmp_path):
    indices = [0, 1, 5]
    staging = tmp_path / "head"
    manifest = stage_jpeg_sequence(DATA_ROOT, "head", indices, staging)

    assert manifest == indices
    assert load_staging_manifest(staging) == indices
    staged = sorted(staging.glob("*.jpg"))
    assert [p.name for p in staged] == ["00000.jpg", "00001.jpg", "00002.jpg"]
    # Symlinks resolve to the original per-index head JPEGs.
    assert staged[2].resolve() == (DATA_ROOT / "frame-000005_head_color.jpg").resolve()
    # Restaging replaces stale links instead of failing.
    manifest2 = stage_jpeg_sequence(DATA_ROOT, "head", [2, 3], staging)
    assert manifest2 == [2, 3]
    assert len(list(staging.glob("*.jpg"))) == 2


def test_stage_missing_frame_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        stage_jpeg_sequence(DATA_ROOT, "head", [999999], tmp_path / "head")


@dataclass
class FakeResult:
    frame_index: int
    masks: np.ndarray


class FakeTracker:
    def __init__(self):
        self.prompts = []
        self.opened = None
        self.closed = False

    def open(self, path, **kwargs):
        self.opened = (Path(path), kwargs)

    def add_prompt(self, frame_index, object_id, **kwargs):
        self.prompts.append((frame_index, object_id, kwargs))

    def track(self):
        for position in range(3):
            mask = np.zeros((4, 4), dtype=bool)
            mask[0, position] = True
            yield FakeResult(frame_index=position, masks=mask[None])

    def close(self):
        self.closed = True


def make_segmenter(tracker):
    return SAM2Segmenter(
        checkpoint="/models/fake.pt",
        tracker_factory=lambda *args: tracker,
    )


def test_track_sequence_maps_frames_and_closes(tmp_path):
    staging = tmp_path / "wrist"
    stage_jpeg_sequence(DATA_ROOT, "wrist", [10, 20, 30], staging)

    tracker = FakeTracker()
    segmenter = make_segmenter(tracker)
    prompt = Sam2Prompt(object_id=1, box=(10.0, 20.0, 100.0, 200.0))
    results = list(segmenter.track_sequence(staging, prompt))

    # Staged positions 0..2 map back to original indices via the manifest.
    assert [frame for frame, _ in results] == [10, 20, 30]
    assert all(mask.dtype == bool and mask.shape == (4, 4) for _, mask in results)
    assert tracker.opened[0] == staging
    assert tracker.opened[1] == {"offload_video_to_cpu": True, "offload_state_to_cpu": False}
    assert tracker.prompts == [(0, 1, {
        "box": (10.0, 20.0, 100.0, 200.0), "points": None, "labels": None, "mask": None,
    })]
    assert tracker.closed


def test_prompt_validation():
    with pytest.raises(ValueError):
        Sam2Prompt().validate()
    with pytest.raises(ValueError):
        Sam2Prompt(points=[[1.0, 2.0]]).validate()  # labels missing
    Sam2Prompt(box=(0, 0, 1, 1)).validate()


def test_from_config_reads_sam2_section():
    config = {
        "devices": {"sam2": "cuda"},
        "sam2": {
            "checkpoint": "/models/sam2.1_hiera_tiny.pt",
            "model_type": "tiny",
            "offload_video_to_cpu": True,
            "offload_state_to_cpu": False,
        },
    }
    segmenter = SAM2Segmenter.from_config(config)
    assert segmenter.checkpoint == "/models/sam2.1_hiera_tiny.pt"
    assert segmenter.model_type == "tiny"
    assert segmenter.device == "cuda"
    assert segmenter.offload_video_to_cpu is True
