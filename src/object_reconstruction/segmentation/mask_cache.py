"""Disk cache for precomputed object masks (PLAN section 8).

Layout under output/masks/:
    cam1/000000.png        0/255 uint8 PNG, one per original frame index
    cam2/000000.png
    mask_metadata.jsonl    one record per (frame, cam)

Reconstruction reads this cache by default instead of re-running SAM 2.1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class MaskCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def mask_path(self, cam: int, frame_index: int) -> Path:
        if cam not in (1, 2):
            raise ValueError(f"cam must be 1 or 2, got {cam}")
        return self.root / f"cam{cam}" / f"{frame_index:06d}.png"

    @property
    def metadata_path(self) -> Path:
        return self.root / "mask_metadata.jsonl"

    def save(self, cam: int, frame_index: int, mask: np.ndarray) -> Path:
        mask = np.asarray(mask)
        if mask.ndim != 2:
            raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
        path = self.mask_path(cam, frame_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(bool).astype(np.uint8) * 255).save(path)
        return path

    def get(self, cam: int, frame_index: int) -> np.ndarray | None:
        path = self.mask_path(cam, frame_index)
        if not path.is_file():
            return None
        return np.asarray(Image.open(path)) > 0

    def get_cam1(self, frame_index: int) -> np.ndarray | None:
        return self.get(1, frame_index)

    def get_cam2(self, frame_index: int) -> np.ndarray | None:
        return self.get(2, frame_index)

    def frames(self, cam: int) -> list[int]:
        cam_dir = self.root / f"cam{cam}"
        if not cam_dir.is_dir():
            return []
        return sorted(int(p.stem) for p in cam_dir.glob("*.png"))

    def write_metadata(self, records: list[dict[str, Any]]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    def read_metadata(self) -> list[dict[str, Any]]:
        if not self.metadata_path.is_file():
            return []
        with open(self.metadata_path) as f:
            return [json.loads(line) for line in f if line.strip()]
