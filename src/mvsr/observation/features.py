"""Feature extraction inside the safe foreground (P2 O3).

`FeatureExtractor` is the adapter boundary: ORB is the implemented baseline; a learned
extractor (SuperPoint, DISK, ...) implements the same two calls and declares its own
`descriptor_type` so matchers never compare incompatible descriptors. None is shipped, so
`features.type` other than `orb` is a configuration error, not a silent fallback.
Raw pixel coordinates (distorted image) and descriptors are stored as detected.
"""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np
import numpy.typing as npt

from mvsr.observation.config import FeatureConfig
from mvsr.yaml_config import ConfigError

F32 = npt.NDArray[np.float32]
U8 = npt.NDArray[np.uint8]


class FeatureExtractor(Protocol):
    descriptor_type: str

    def detect(self, gray: U8, mask: npt.NDArray[np.bool_]) -> tuple[F32, np.ndarray]:
        """Keypoints `(M, 2)` inside `mask` and their descriptors `(M, D)`."""
        ...

    def describe(self, gray: U8, uv: F32) -> tuple[npt.NDArray[np.intp], np.ndarray]:
        """Descriptors at given points: (indices of `uv` that got one, descriptors)."""
        ...


class OrbExtractor:
    descriptor_type = "orb"

    def __init__(self, cfg: FeatureConfig) -> None:
        self.cfg = cfg
        self._orb = cv2.ORB.create(
            nfeatures=cfg.n_features,
            scaleFactor=cfg.scale_factor,
            nlevels=cfg.n_levels,
            edgeThreshold=cfg.edge_threshold,
            patchSize=cfg.patch_size,
            fastThreshold=cfg.fast_threshold,
        )

    def detect(self, gray: U8, mask: npt.NDArray[np.bool_]) -> tuple[F32, np.ndarray]:
        if not mask.any():
            return np.zeros((0, 2), np.float32), np.zeros((0, 32), np.uint8)
        kps, desc = self._orb.detectAndCompute(gray, mask.astype(np.uint8) * 255)
        if not kps or desc is None:
            return np.zeros((0, 2), np.float32), np.zeros((0, 32), np.uint8)
        uv = np.array([k.pt for k in kps], dtype=np.float32)
        order = np.lexsort((uv[:, 0], uv[:, 1]))  # deterministic row-major order
        return uv[order], np.asarray(desc, dtype=np.uint8)[order]

    def describe(self, gray: U8, uv: F32) -> tuple[npt.NDArray[np.intp], np.ndarray]:
        if uv.shape[0] == 0:
            return np.zeros(0, np.intp), np.zeros((0, 32), np.uint8)
        kps = [
            cv2.KeyPoint(float(x), float(y), float(self.cfg.patch_size), -1, 0, 0, int(i))
            for i, (x, y) in enumerate(uv)
        ]
        kept, desc = self._orb.compute(gray, kps)
        if not kept or desc is None:
            return np.zeros(0, np.intp), np.zeros((0, 32), np.uint8)
        idx = np.array([k.class_id for k in kept], dtype=np.intp)
        return idx, np.asarray(desc, dtype=np.uint8)


def make_extractor(cfg: FeatureConfig) -> FeatureExtractor:
    if cfg.type == "orb":
        return OrbExtractor(cfg)
    raise ConfigError(f"features.type {cfg.type!r} has no adapter (implemented: orb)")
