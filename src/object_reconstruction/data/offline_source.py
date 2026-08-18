"""OfflineFrameSource for the current flat data/ layout (PLAN section 5.1).

Layout:
    data/
    ├── frame-{idx:06d}_{head,wrist}_color.jpg
    ├── frame-{idx:06d}_{head,wrist}_depth.png
    ├── frame-{idx:06d}_pose.txt      # low-precision copy of a JointStates row
    ├── JointStates.txt               # N x 7, numeric source for raw poses
    ├── intrinsic/{head,wrist}_cam_K.txt
    └── handeye/handeye_tf.txt

Mapping: head -> cam1 (fixed, WORLD), wrist -> cam2 (eye-in-hand).

T_world_cam2 stays None until pose_semantics / quaternion_order /
handeye_convention are confirmed in the config; only data inspection is allowed
before that.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

from ..calibration.intrinsics import intrinsics_from_matrix, load_intrinsics_matrix
from .models import CameraFrame, CameraIntrinsics, FramePacket

_FRAME_RE = re.compile(r"frame-(\d{6})_pose\.txt$")

FILE_SUFFIXES = (
    "{prefix1}_color.jpg",
    "{prefix1}_depth.png",
    "{prefix2}_color.jpg",
    "{prefix2}_depth.png",
    "pose.txt",
)


class OfflineFrameSource:
    def __init__(
        self,
        root: str | Path,
        cam1_prefix: str = "head",
        cam2_prefix: str = "wrist",
        pose_source: str = "JointStates.txt",
        frame_count_expected: int | None = None,
        conventions_confirmed: bool = False,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root not found: {self.root}")
        self.cam1_prefix = cam1_prefix
        self.cam2_prefix = cam2_prefix
        self.conventions_confirmed = conventions_confirmed

        self.indices = self._discover_indices()
        if frame_count_expected is not None and len(self.indices) != frame_count_expected:
            raise ValueError(
                f"expected {frame_count_expected} frames, found {len(self.indices)}"
            )
        self.missing_files = self._check_completeness()

        self.raw_poses = self._load_pose_table(self.root / pose_source)
        if len(self.raw_poses) != len(self.indices):
            raise ValueError(
                f"{pose_source} has {len(self.raw_poses)} rows "
                f"but dataset has {len(self.indices)} frames"
            )

        width, height = self._probe_image_size()
        self.intrinsics_cam1 = self._load_intrinsics(cam1_prefix, width, height)
        self.intrinsics_cam2 = self._load_intrinsics(cam2_prefix, width, height)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OfflineFrameSource":
        ds = config["dataset"]
        confirmed = all(
            ds.get(key) is not None
            for key in ("pose_semantics", "quaternion_order", "handeye_convention")
        )
        return cls(
            root=ds["root"],
            cam1_prefix=ds.get("cam1_prefix", "head"),
            cam2_prefix=ds.get("cam2_prefix", "wrist"),
            pose_source=ds.get("pose_source", "JointStates.txt"),
            frame_count_expected=ds.get("frame_count_expected"),
            conventions_confirmed=confirmed,
        )

    # -- discovery / validation ------------------------------------------------

    def _discover_indices(self) -> list[int]:
        indices = sorted(
            int(m.group(1))
            for p in self.root.iterdir()
            if (m := _FRAME_RE.search(p.name))
        )
        if not indices:
            raise FileNotFoundError(f"no frame-*_pose.txt files in {self.root}")
        return indices

    def _frame_paths(self, index: int) -> dict[str, Path]:
        stem = f"frame-{index:06d}_"
        return {
            "rgb1": self.root / f"{stem}{self.cam1_prefix}_color.jpg",
            "depth1": self.root / f"{stem}{self.cam1_prefix}_depth.png",
            "rgb2": self.root / f"{stem}{self.cam2_prefix}_color.jpg",
            "depth2": self.root / f"{stem}{self.cam2_prefix}_depth.png",
            "pose": self.root / f"{stem}pose.txt",
        }

    def _check_completeness(self) -> dict[int, list[str]]:
        missing: dict[int, list[str]] = {}
        contiguous = set(range(self.indices[0], self.indices[-1] + 1))
        for gap in sorted(contiguous - set(self.indices)):
            missing[gap] = ["pose"]
        for index in self.indices:
            absent = [k for k, p in self._frame_paths(index).items() if not p.is_file()]
            if absent:
                missing[index] = absent
        return missing

    @staticmethod
    def _load_pose_table(path: Path) -> np.ndarray:
        poses = np.loadtxt(path, delimiter=",", dtype=np.float64)
        if poses.ndim == 1:
            poses = poses[None, :]
        if poses.shape[1] != 7:
            raise ValueError(f"{path}: expected 7 columns, got {poses.shape[1]}")
        return poses

    def _probe_image_size(self) -> tuple[int, int]:
        with Image.open(self._frame_paths(self.indices[0])["rgb1"]) as im:
            return im.size  # (width, height)

    def _load_intrinsics(self, prefix: str, width: int, height: int) -> CameraIntrinsics:
        K = load_intrinsics_matrix(self.root / "intrinsic" / f"{prefix}_cam_K.txt")
        return intrinsics_from_matrix(K, width, height)

    # -- per-frame loading -----------------------------------------------------

    @staticmethod
    def _load_rgb(path: Path) -> np.ndarray:
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        return rgb

    @staticmethod
    def _load_depth(path: Path) -> np.ndarray:
        with Image.open(path) as im:
            depth = np.asarray(im)
        if depth.dtype != np.uint16:
            raise ValueError(f"{path}: expected uint16 depth, got {depth.dtype}")
        return depth

    def load_per_frame_pose(self, index: int) -> np.ndarray:
        values = np.loadtxt(self._frame_paths(index)["pose"], dtype=np.float64)
        if values.shape != (7,):
            raise ValueError(f"frame {index}: pose file must contain 7 values")
        return values

    def read_packet(self, position: int) -> FramePacket:
        index = self.indices[position]
        paths = self._frame_paths(index)
        raw_pose = self.raw_poses[position]

        cam1 = CameraFrame(
            rgb=self._load_rgb(paths["rgb1"]),
            depth=self._load_depth(paths["depth1"]),
            intrinsics=self.intrinsics_cam1,
        )
        cam2 = CameraFrame(
            rgb=self._load_rgb(paths["rgb2"]),
            depth=self._load_depth(paths["depth2"]),
            intrinsics=self.intrinsics_cam2,
        )

        T_world_cam2 = None
        if self.conventions_confirmed:
            # Pose composition is implemented in Task 2 once the confirmed
            # conventions define how raw_pose_7d maps to T_world_cam2.
            raise NotImplementedError("T_world_cam2 composition lands in Task 2")

        return FramePacket(
            index=index,
            timestamp=None,
            cam1=cam1,
            cam2=cam2,
            T_world_cam1=np.eye(4, dtype=np.float64),
            T_world_cam2=T_world_cam2,
            raw_pose_7d=raw_pose,
        )

    # -- FrameSource protocol --------------------------------------------------

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[FramePacket]:
        for position in range(len(self.indices)):
            yield self.read_packet(position)
