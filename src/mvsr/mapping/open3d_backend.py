"""Open3D TSDF adapter with the project transform convention.

Open3D's RGBD integration expects world-to-camera extrinsics. The object frame is the TSDF
world, so the only correct extrinsic is `T_cam_obj` (never its inverse).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from mvsr.core.se3 import PoseSE3
from mvsr.core.types import CalibrationBundle, ObjectObservation
from mvsr.io.session_reader import RawFrame
from mvsr.mapping.config import TsdfConfig


@dataclass(frozen=True)
class GeometryResult:
    points_obj_m: npt.NDArray[np.float32]
    normals_obj: npt.NDArray[np.float32]
    colors_u8: npt.NDArray[np.uint8] | None
    vertices_obj_m: npt.NDArray[np.float32]
    triangles: npt.NDArray[np.int32]


def _o3d() -> Any:
    import open3d as o3d

    return o3d


def create_volume(cfg: TsdfConfig) -> Any:
    o3d = _o3d()
    color_type = (
        o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        if cfg.color
        else o3d.pipelines.integration.TSDFVolumeColorType.NoColor
    )
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=cfg.voxel_length_m,
        sdf_trunc=cfg.sdf_trunc_m,
        color_type=color_type,
    )


def integrate(
    volume: Any,
    observation: ObjectObservation,
    raw: RawFrame,
    bundle: CalibrationBundle,
    T_cam_obj: PoseSE3,
    cfg: TsdfConfig,
) -> None:
    T_cam_obj.expect_frames("cam", "obj")
    if raw.depth_m_f32.shape != observation.geometry_mask.shape:
        raise ValueError("raw depth and observation geometry mask shapes differ")
    o3d = _o3d()
    depth = np.where(observation.geometry_mask, raw.depth_m_f32, np.float32(0.0)).astype(np.float32)
    color = np.ascontiguousarray(raw.color_rgb_u8)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color),
        o3d.geometry.Image(np.ascontiguousarray(depth)),
        depth_scale=1.0,
        depth_trunc=cfg.depth_trunc_m,
        convert_rgb_to_intensity=False,
    )
    K = bundle.K
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        bundle.width,
        bundle.height,
        float(K[0, 0]),
        float(K[1, 1]),
        float(K[0, 2]),
        float(K[1, 2]),
    )
    volume.integrate(rgbd, intrinsic, np.asarray(T_cam_obj.matrix))


def extract(volume: Any, *, color: bool) -> GeometryResult:
    pointcloud = volume.extract_point_cloud()
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    points = np.asarray(pointcloud.points, dtype=np.float32)
    normals = np.asarray(pointcloud.normals, dtype=np.float32)
    if normals.shape != points.shape:
        normals = np.zeros_like(points)
    colors: npt.NDArray[np.uint8] | None = None
    if color and len(pointcloud.colors):
        colors = np.clip(np.rint(np.asarray(pointcloud.colors) * 255.0), 0, 255).astype(np.uint8)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    for array in (points, normals, vertices, triangles):
        array.flags.writeable = False
    if colors is not None:
        colors.flags.writeable = False
    return GeometryResult(points, normals, colors, vertices, triangles)


def pointcloud_from_geometry(geometry: GeometryResult) -> Any:
    o3d = _o3d()
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(geometry.points_obj_m.astype(np.float64))
    pointcloud.normals = o3d.utility.Vector3dVector(geometry.normals_obj.astype(np.float64))
    if geometry.colors_u8 is not None:
        pointcloud.colors = o3d.utility.Vector3dVector(
            geometry.colors_u8.astype(np.float64) / 255.0
        )
    return pointcloud


def mesh_from_geometry(geometry: GeometryResult) -> Any:
    o3d = _o3d()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(geometry.vertices_obj_m.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(
        np.array(geometry.triangles, dtype=np.int32, copy=True)
    )
    if len(geometry.vertices_obj_m):
        mesh.compute_vertex_normals()
    return mesh
