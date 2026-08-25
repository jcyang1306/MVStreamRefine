"""Single-camera offline source for the current flat data/ layout.

Layout:
    data/
    ├── frame-{idx:06d}_wrist_color.jpg
    ├── frame-{idx:06d}_wrist_depth.png
    ├── frame-{idx:06d}_pose.txt      # low-precision copy of a JointStates row
    ├── JointStates.txt               # N x 7, numeric source for raw poses
    ├── intrinsic/wrist_cam_K.txt
    └── handeye/handeye_tf.txt

The physical ``wrist`` prefix is configurable and maps to the sole logical
camera ``cam``. WORLD is the robot base frame. Legacy head files and the
``base_cam1`` hand-eye block may remain on disk but are deliberately ignored.

With the confirmed conventions (pose = T_base_tcp in xyzw, handeye
wrist_cam2 = T_tcp_cam) each packet carries

    T_world_cam = T_base_tcp @ T_tcp_cam

If any convention key is null in the config, T_world_cam stays None and only
data inspection is allowed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

from ..calibration.handeye import load_handeye_matrices
from ..calibration.intrinsics import intrinsics_from_matrix, load_intrinsics_matrix
from ..calibration.transforms import (
    check_supported_conventions,
    compose_T_world_cam,
    pose7d_to_matrix,
    validate_transform,
)
from .models import CameraFrame, CameraIntrinsics, FramePacket

_FRAME_RE = re.compile(r"frame-(\d{6})_pose\.txt$")

FILE_SUFFIXES = (
    "{prefix}_color.jpg",
    "{prefix}_depth.png",
    "pose.txt",
)


class OfflineFrameSource:
    def __init__(
        self,
        root: str | Path,
        cam_prefix: str = "wrist",
        pose_source: str = "JointStates.txt",
        frame_count_expected: int | None = None,
        pose_semantics: str | None = None,
        quaternion_order: str | None = None,
        handeye_convention: str | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root not found: {self.root}")
        self.cam_prefix = cam_prefix
        self.quaternion_order = quaternion_order
        self.conventions_confirmed = all(
            value is not None
            for value in (pose_semantics, quaternion_order, handeye_convention)
        )

        self.T_tcp_cam: np.ndarray | None = None
        if self.conventions_confirmed:
            check_supported_conventions(pose_semantics, quaternion_order, handeye_convention)
            handeye = load_handeye_matrices(self.root / "handeye" / "handeye_tf.txt")
            try:
                self.T_tcp_cam = handeye["wrist_cam2"]
            except KeyError as exc:
                raise ValueError(
                    f"handeye_tf.txt is missing expected label {exc}; found {sorted(handeye)}"
                ) from exc
            validate_transform(self.T_tcp_cam)

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
        self.intrinsics_cam = self._load_intrinsics(cam_prefix, width, height)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OfflineFrameSource":
        ds = config["dataset"]
        return cls(
            root=ds["root"],
            cam_prefix=ds.get("cam_prefix", "wrist"),
            pose_source=ds.get("pose_source", "JointStates.txt"),
            frame_count_expected=ds.get("frame_count_expected"),
            pose_semantics=ds.get("pose_semantics"),
            quaternion_order=ds.get("quaternion_order"),
            handeye_convention=ds.get("handeye_convention"),
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
            "rgb": self.root / f"{stem}{self.cam_prefix}_color.jpg",
            "depth": self.root / f"{stem}{self.cam_prefix}_depth.png",
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
        with Image.open(self._frame_paths(self.indices[0])["rgb"]) as im:
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

        cam = CameraFrame(
            rgb=self._load_rgb(paths["rgb"]),
            depth=self._load_depth(paths["depth"]),
            intrinsics=self.intrinsics_cam,
        )

        T_world_cam = None
        T_base_tcp = None
        if self.conventions_confirmed:
            T_base_tcp = pose7d_to_matrix(raw_pose, self.quaternion_order)
            T_world_cam = compose_T_world_cam(T_base_tcp, self.T_tcp_cam)

        return FramePacket(
            index=index,
            timestamp=None,
            cam=cam,
            T_world_cam=T_world_cam,
            tcp_pose=T_base_tcp,
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
