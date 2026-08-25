"""ICP pose refinement with safety fallback (PLAN sections 16-17, Task 9).

Point-to-plane ICP refines the robot-provided pose T_world_cam of a wrist-camera
keyframe against the object model fused so far:

    source = current object point cloud, still in the camera frame
    target = object model in WORLD extracted BEFORE integrating this frame
    init   = T_world_cam from robot kinematics (strong prior)

The refined pose is accepted only when all safety gates pass (fitness, rmse,
correction magnitude); otherwise the robot pose is kept. ICP is a local
touch-up and must never pull the camera far from the kinematic prior.

open3d is imported lazily so this module stays importable on the dev machine;
the gate logic (correction_magnitudes / ICPParams.accepts) is pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_MIN_POINTS = 100


def _import_open3d():
    try:
        import open3d  # noqa: PLC0415 (deliberate lazy import)
    except ImportError as exc:
        raise RuntimeError(
            "open3d is required for ICPRefiner; run inside the CUDA container "
            "(see Dockerfile) or disable ICP (icp.enabled: false)"
        ) from exc
    return open3d


def correction_magnitudes(
    T_initial: np.ndarray, T_refined: np.ndarray
) -> tuple[float, float]:
    """(translation_m, rotation_deg) of the correction inv(T_initial) @ T_refined."""
    T_delta = np.linalg.inv(np.asarray(T_initial, dtype=np.float64)) @ np.asarray(
        T_refined, dtype=np.float64
    )
    translation = float(np.linalg.norm(T_delta[:3, 3]))
    cos_angle = np.clip((np.trace(T_delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    rotation = float(np.degrees(np.arccos(cos_angle)))
    return translation, rotation


@dataclass(frozen=True)
class ICPParams:
    enabled: bool = False
    max_correspondence_distance_m: float = 0.01
    min_fitness: float = 0.4
    max_rmse_m: float = 0.008
    max_translation_correction_m: float = 0.02
    max_rotation_correction_deg: float = 5.0
    voxel_downsample_m: float = 0.004
    max_iterations: int = 30

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ICPParams":
        icp = config.get("icp", {})
        params = cls(
            enabled=bool(icp.get("enabled", False)),
            max_correspondence_distance_m=float(
                icp.get("max_correspondence_distance_m", 0.01)
            ),
            min_fitness=float(icp.get("min_fitness", 0.4)),
            max_rmse_m=float(icp.get("max_rmse_m", 0.008)),
            max_translation_correction_m=float(
                icp.get("max_translation_correction_m", 0.02)
            ),
            max_rotation_correction_deg=float(
                icp.get("max_rotation_correction_deg", 5.0)
            ),
            voxel_downsample_m=float(icp.get("voxel_downsample_m", 0.004)),
            max_iterations=int(icp.get("max_iterations", 30)),
        )
        for name in (
            "max_correspondence_distance_m", "min_fitness", "max_rmse_m",
            "max_translation_correction_m", "max_rotation_correction_deg",
            "voxel_downsample_m",
        ):
            if getattr(params, name) <= 0:
                raise ValueError(f"icp.{name} must be > 0")
        if params.max_iterations < 1:
            raise ValueError("icp.max_iterations must be >= 1")
        return params

    def accepts(
        self,
        fitness: float,
        inlier_rmse: float,
        translation_correction_m: float,
        rotation_correction_deg: float,
    ) -> bool:
        """PLAN section 17 gates: reject anything not a small local touch-up."""
        return (
            fitness >= self.min_fitness
            and inlier_rmse <= self.max_rmse_m
            and translation_correction_m <= self.max_translation_correction_m
            and rotation_correction_deg <= self.max_rotation_correction_deg
        )


@dataclass(frozen=True)
class ICPResult:
    success: bool
    T_world_cam_refined: np.ndarray
    fitness: float
    inlier_rmse: float
    translation_correction_m: float
    rotation_correction_deg: float
    reason: str = ""


class ICPRefiner:
    def __init__(self, params: ICPParams) -> None:
        self.params = params
        self._o3d = _import_open3d()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ICPRefiner":
        return cls(ICPParams.from_config(config))

    def refine(
        self,
        current_object_pcd_cam: np.ndarray,
        model_pcd_world: np.ndarray,
        T_world_cam_initial: np.ndarray,
    ) -> ICPResult:
        """Point-to-plane ICP; falls back to the robot pose on any gate failure.

        current_object_pcd_cam: (N, 3) float meters, camera frame.
        model_pcd_world:        (M, 3) float meters, WORLD frame, extracted
                                before integrating the current frame.
        T_world_cam_initial:    4x4 robot-kinematics pose (strong prior).
        """
        T_init = np.asarray(T_world_cam_initial, dtype=np.float64)
        fallback = lambda reason: ICPResult(  # noqa: E731
            success=False, T_world_cam_refined=T_init.copy(),
            fitness=0.0, inlier_rmse=0.0,
            translation_correction_m=0.0, rotation_correction_deg=0.0,
            reason=reason,
        )

        source = self._make_pcd(current_object_pcd_cam)
        target = self._make_pcd(model_pcd_world)
        if len(source.points) < _MIN_POINTS:
            return fallback(f"source has < {_MIN_POINTS} points after downsample")
        if len(target.points) < _MIN_POINTS:
            return fallback(f"target model has < {_MIN_POINTS} points")

        # Point-to-plane needs target normals; the object model is small, so a
        # radius tied to the correspondence distance works at any voxel size.
        radius = 3.0 * self.params.max_correspondence_distance_m
        target.estimate_normals(
            self._o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
        )

        registration = self._o3d.pipelines.registration
        result = registration.registration_icp(
            source,
            target,
            self.params.max_correspondence_distance_m,
            T_init,
            registration.TransformationEstimationPointToPlane(),
            registration.ICPConvergenceCriteria(
                max_iteration=self.params.max_iterations
            ),
        )

        T_refined = np.asarray(result.transformation, dtype=np.float64)
        translation, rotation = correction_magnitudes(T_init, T_refined)
        fitness = float(result.fitness)
        rmse = float(result.inlier_rmse)
        accepted = self.params.accepts(fitness, rmse, translation, rotation)
        return ICPResult(
            success=accepted,
            T_world_cam_refined=T_refined if accepted else T_init.copy(),
            fitness=fitness,
            inlier_rmse=rmse,
            translation_correction_m=translation,
            rotation_correction_deg=rotation,
            reason="" if accepted else "safety gates rejected the correction",
        )

    def _make_pcd(self, points: np.ndarray):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {points.shape}")
        pcd = self._o3d.geometry.PointCloud(
            self._o3d.utility.Vector3dVector(points)
        )
        return pcd.voxel_down_sample(self.params.voxel_downsample_m)
