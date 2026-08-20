"""TSDF fusion volume wrapping open3d.t.geometry.VoxelBlockGrid (PLAN section 10, Task 5).

Design rules from the PLAN:
- Callers always pass T_world_cam; the world->camera extrinsic that Open3D
  expects is computed here exactly once (no scattered ``inv()`` in the pipeline).
- open3d is imported lazily so that parameter parsing and unit tests run on the
  dev machine, where only the CUDA container ships open3d.
- Requesting a CUDA device on an Open3D build without CUDA support fails fast;
  torch.cuda.is_available() is never used as evidence of Open3D CUDA support.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..calibration.transforms import invert_transform
from ..data.models import CameraIntrinsics

_DEVICE_RE = re.compile(r"^(CPU|CUDA):\d+$")

_POINT_CLOUD_SUFFIXES = (".ply", ".pcd", ".xyzrgb")
_MESH_SUFFIXES = (".obj", ".stl", ".glb", ".gltf")


def _import_open3d() -> Any:
    """Import open3d on first use; TSDF code must run inside the CUDA container."""
    try:
        import open3d  # noqa: PLC0415 (deliberate lazy import)
    except ImportError as exc:
        raise RuntimeError(
            "open3d is not installed in this environment; TSDF fusion must run "
            "inside the CUDA container "
            "(docker compose run --rm reconstruction python3 ...)"
        ) from exc
    return open3d


def check_device_string(device: str) -> str:
    """Validate the tsdf.device string without touching open3d."""
    if not isinstance(device, str) or not _DEVICE_RE.match(device):
        raise ValueError(
            f"unsupported tsdf.device {device!r}; expected 'CPU:<n>' or 'CUDA:<n>'"
        )
    return device


@dataclass(frozen=True)
class TSDFParams:
    """tsdf + depth.scale configuration, importable without open3d."""

    device: str = "CPU:0"
    voxel_size_m: float = 0.002
    block_resolution: int = 16
    block_count: int = 50000
    trunc_voxel_multiplier: float = 4.0
    depth_max_m: float = 1.2
    weight_threshold: float = 3.0
    depth_scale: float = 1000.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TSDFParams":
        tsdf = config["tsdf"]
        depth_scale = config["depth"]["scale"]
        if depth_scale is None:
            raise ValueError("depth.scale is null; fusion is forbidden (PLAN section 9)")
        params = cls(
            device=tsdf.get("device", cls.device),
            voxel_size_m=float(tsdf.get("voxel_size_m", cls.voxel_size_m)),
            block_resolution=int(tsdf.get("block_resolution", cls.block_resolution)),
            block_count=int(tsdf.get("block_count", cls.block_count)),
            trunc_voxel_multiplier=float(
                tsdf.get("trunc_voxel_multiplier", cls.trunc_voxel_multiplier)
            ),
            depth_max_m=float(tsdf.get("depth_max_m", cls.depth_max_m)),
            weight_threshold=float(tsdf.get("weight_threshold", cls.weight_threshold)),
            depth_scale=float(depth_scale),
        )
        params.validate()
        return params

    def validate(self) -> None:
        check_device_string(self.device)
        positive = {
            "voxel_size_m": self.voxel_size_m,
            "block_resolution": self.block_resolution,
            "block_count": self.block_count,
            "trunc_voxel_multiplier": self.trunc_voxel_multiplier,
            "depth_max_m": self.depth_max_m,
            "depth_scale": self.depth_scale,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"tsdf.{name} must be > 0, got {value}")
        if not np.isfinite(self.weight_threshold) or self.weight_threshold < 0:
            raise ValueError(
                f"tsdf.weight_threshold must be >= 0, got {self.weight_threshold}"
            )


class TSDFVolume:
    """Scalable TSDF volume over open3d.t.geometry.VoxelBlockGrid."""

    def __init__(self, params: TSDFParams) -> None:
        params.validate()
        self.params = params
        self._o3d = _import_open3d()
        self._device = self._resolve_device(self._o3d, params.device)
        core = self._o3d.core
        self._grid = self._o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(core.float32, core.float32, core.float32),
            attr_channels=((1,), (1,), (3,)),
            voxel_size=params.voxel_size_m,
            block_resolution=params.block_resolution,
            block_count=params.block_count,
            device=self._device,
        )
        self.integration_count = 0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TSDFVolume":
        return cls(TSDFParams.from_config(config))

    @staticmethod
    def _resolve_device(o3d: Any, device: str) -> Any:
        """Map the config string to an open3d Device, failing fast on CUDA gaps.

        PLAN section 10: torch.cuda.is_available() is not evidence of Open3D
        CUDA support; silently falling back to CPU is forbidden.
        """
        check_device_string(device)
        if device.startswith("CUDA") and not o3d.core.cuda.is_available():
            raise RuntimeError(
                f"tsdf.device={device!r} requested but this Open3D build has no "
                "CUDA support (open3d.core.cuda.is_available() is False); "
                "install a CUDA-enabled Open3D or set tsdf.device to 'CPU:0' "
                "explicitly - silent device fallback is forbidden (PLAN section 10)"
            )
        return o3d.core.Device(device)

    # -- integration -------------------------------------------------------

    def integrate(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
        T_world_cam: np.ndarray,
    ) -> None:
        """Fuse one aligned RGB-D frame given the camera pose in WORLD.

        rgb: uint8 H x W x 3; depth: uint16 H x W in raw units (millimeters
        here, converted via params.depth_scale). The world->camera extrinsic
        required by VoxelBlockGrid is derived internally from T_world_cam.
        """
        rgb = np.asarray(rgb)
        depth = np.asarray(depth)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb must be uint8 H x W x 3, got {rgb.dtype} {rgb.shape}")
        if depth.dtype != np.uint16 or depth.ndim != 2:
            raise ValueError(f"depth must be uint16 H x W, got {depth.dtype} {depth.shape}")
        if depth.shape != rgb.shape[:2]:
            raise ValueError(f"rgb {rgb.shape} and depth {depth.shape} are not aligned")
        intrinsics.validate()
        if depth.shape != (intrinsics.height, intrinsics.width):
            raise ValueError(
                f"depth {depth.shape} does not match intrinsics "
                f"{intrinsics.height} x {intrinsics.width}"
            )

        # Open3D wants extrinsic = T_cam_world; invert exactly once, here.
        extrinsic = invert_transform(np.asarray(T_world_cam, dtype=np.float64))

        core = self._o3d.core
        depth_image = self._o3d.t.geometry.Image(
            core.Tensor(np.ascontiguousarray(depth))
        ).to(self._device)
        color_image = self._o3d.t.geometry.Image(
            core.Tensor(np.ascontiguousarray(rgb))
        ).to(self._device)
        intrinsic_t = core.Tensor(intrinsics.matrix(), core.float64)
        extrinsic_t = core.Tensor(extrinsic, core.float64)

        block_coords = self._grid.compute_unique_block_coordinates(
            depth_image,
            intrinsic_t,
            extrinsic_t,
            self.params.depth_scale,
            self.params.depth_max_m,
            self.params.trunc_voxel_multiplier,
        )
        self._grid.integrate(
            block_coords,
            depth_image,
            color_image,
            intrinsic_t,
            extrinsic_t,
            self.params.depth_scale,
            self.params.depth_max_m,
            self.params.trunc_voxel_multiplier,
        )
        self.integration_count += 1

    # -- extraction / io -----------------------------------------------------

    def _require_integrations(self) -> None:
        if self.integration_count == 0:
            raise RuntimeError("nothing has been integrated into the TSDF volume yet")

    def extract_point_cloud(self) -> Any:
        """Extract the surface point cloud (open3d.t.geometry.PointCloud)."""
        self._require_integrations()
        return self._grid.extract_point_cloud(
            weight_threshold=self.params.weight_threshold
        )

    def extract_mesh(self) -> Any:
        """Extract the surface triangle mesh (open3d.t.geometry.TriangleMesh)."""
        self._require_integrations()
        return self._grid.extract_triangle_mesh(
            weight_threshold=self.params.weight_threshold
        )

    def block_count(self) -> int:
        """Number of allocated voxel blocks (debug metric, PLAN section 15)."""
        if self.integration_count == 0:
            return 0
        return int(self._grid.hashmap().size())

    def save(self, path: str | Path) -> Path:
        """Save by extension: point cloud for .ply/.pcd, mesh for .obj/.stl/.glb."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in _POINT_CLOUD_SUFFIXES:
            return self.save_point_cloud(path)
        if suffix in _MESH_SUFFIXES:
            return self.save_mesh(path)
        raise ValueError(
            f"unsupported save extension {suffix!r}; point cloud: "
            f"{_POINT_CLOUD_SUFFIXES}, mesh: {_MESH_SUFFIXES}"
        )

    def save_point_cloud(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        geometry = self.extract_point_cloud().to_legacy()
        if not self._o3d.io.write_point_cloud(str(path), geometry):
            raise RuntimeError(f"open3d failed to write {path}")
        return path

    def save_mesh(self, path: str | Path) -> Path:
        """Write the triangle mesh regardless of extension (e.g. mesh .ply)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        geometry = self.extract_mesh().to_legacy()
        if not self._o3d.io.write_triangle_mesh(str(path), geometry):
            raise RuntimeError(f"open3d failed to write {path}")
        return path
