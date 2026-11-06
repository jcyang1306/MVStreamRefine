"""P5 object-frame TSDF map and conservative commit protocol."""

from __future__ import annotations

import hashlib
import io
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from mvsr.core import codec
from mvsr.core.errors import MvsrError
from mvsr.core.se3 import PoseSE3
from mvsr.core.types import (
    CalibrationBundle,
    Keyframe,
    KeyframeStatus,
    MapSnapshot,
    MapView,
    ObjectObservation,
    PoseEstimate,
    PoseStatus,
    PoseUpdate,
    TrackState,
)
from mvsr.io.atomic import atomic_write_bytes, atomic_write_json
from mvsr.io.session_reader import SessionReader
from mvsr.mapping.config import MappingConfig
from mvsr.mapping.open3d_backend import (
    GeometryResult,
    create_volume,
    extract,
    integrate,
    mesh_from_geometry,
    pointcloud_from_geometry,
)


class MappingError(MvsrError):
    """A keyframe cannot be fused or a map artifact is inconsistent."""


class ObservationProvider(Protocol):
    def observation(self, keyframe_id: int) -> ObjectObservation: ...


def _same_header(a: Any, b: Any) -> bool:
    return (
        a.schema_version == b.schema_version
        and a.session_id == b.session_id
        and a.camera_id == b.camera_id
        and a.frame_id == b.frame_id
        and a.timestamp_ns == b.timestamp_ns
        and a.clock_domain is b.clock_domain
        and a.calibration_id == b.calibration_id
    )


def _write_o3d_atomic(path: Path, geometry: Any, *, mesh: bool) -> str:
    import open3d as o3d

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".ply", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        ok = (
            o3d.io.write_triangle_mesh(str(temporary), geometry, write_ascii=False)
            if mesh
            else o3d.io.write_point_cloud(str(temporary), geometry, write_ascii=False)
        )
        if not ok:
            raise MappingError(f"Open3D failed to write {path.name}")
        return atomic_write_bytes(path, temporary.read_bytes())
    finally:
        temporary.unlink(missing_ok=True)


class ObjectMap:
    """Implements the P5 `ObjectMap` public ports.

    `commit` updates the published map. `build_provisional` is an explicit diagnostic path
    which never increments map_version and can never be returned by `view()`.
    """

    def __init__(
        self,
        session_dir: str | Path,
        bundle: CalibrationBundle,
        observations: ObservationProvider,
        T_base_obj_init: PoseSE3,
        cfg: MappingConfig,
    ) -> None:
        T_base_obj_init.expect_frames("base", "obj")
        self.session_dir = Path(session_dir)
        self.reader = SessionReader(self.session_dir)
        self.bundle = bundle
        self.observations = observations
        self.T_base_obj_init = T_base_obj_init
        self.cfg = cfg
        self.tsdf_params_hash = hashlib.sha256(
            codec.canonical_dumps(cfg.snapshot()["tsdf"]).encode()
        ).hexdigest()
        self._volume = create_volume(cfg.tsdf)
        self._committed: dict[int, tuple[Keyframe, PoseEstimate]] = {}
        self._map_version = 0
        self._last_snapshot: MapSnapshot | None = None

    @property
    def map_version(self) -> int:
        return self._map_version

    def _measurement(self, keyframe: Keyframe) -> tuple[ObjectObservation, Any]:
        observation = self.observations.observation(keyframe.keyframe_id)
        if not _same_header(keyframe.header, observation.header):
            raise MappingError("keyframe and observation headers differ")
        if keyframe.mask_revision != observation.mask_revision:
            raise MappingError("keyframe and observation mask revisions differ")
        raw = self.reader.read_frame("wrist", keyframe.header.frame_id)
        return observation, raw

    def _validate_commit(self, keyframe: Keyframe, pose: PoseEstimate) -> None:
        if not _same_header(keyframe.header, pose.header):
            raise MappingError("keyframe and pose headers differ")
        if keyframe.status is KeyframeStatus.REJECTED:
            raise MappingError("rejected keyframe cannot be committed")
        if pose.status is not PoseStatus.VALID or pose.T_cam_obj is None:
            raise MappingError("only VALID poses can be committed")
        if (
            self.cfg.commit.require_pose_version_match
            and pose.pose_version != keyframe.pose_version
        ):
            raise MappingError("STALE_VERSION: pose and keyframe versions differ")
        if not pose.T_cam_obj.allclose(keyframe.T_cam_obj_current, atol=1e-8):
            raise MappingError("pose differs from keyframe current pose")
        overlap = float(pose.quality.get("overlap") or 0.0)
        rmse = pose.quality.get("rmse_m")
        if overlap < self.cfg.commit.min_overlap:
            raise MappingError("pose overlap is below mapping threshold")
        if rmse is not None and float(rmse) > self.cfg.commit.max_rmse_m:
            raise MappingError("pose RMSE is above mapping threshold")
        observation, _ = self._measurement(keyframe)
        if len(observation.points_cam_m) < self.cfg.commit.min_geometry_points:
            raise MappingError("keyframe has insufficient geometry")
        existing = self._committed.get(keyframe.keyframe_id)
        if existing is not None and existing[0].pose_version != keyframe.pose_version:
            raise MappingError("STALE_VERSION: keyframe id already committed at another version")

    def commit(self, keyframe: Keyframe, pose: PoseEstimate) -> int:
        self._validate_commit(keyframe, pose)
        if keyframe.keyframe_id in self._committed:
            return self._map_version
        observation, raw = self._measurement(keyframe)
        assert pose.T_cam_obj is not None
        integrate(
            self._volume,
            observation,
            raw,
            self.bundle,
            pose.T_cam_obj,
            self.cfg.tsdf,
        )
        self._committed[keyframe.keyframe_id] = (keyframe, pose)
        self._map_version += 1
        self._last_snapshot = None
        return self._map_version

    def geometry(self) -> GeometryResult:
        return extract(self._volume, color=self.cfg.tsdf.color)

    def view(self) -> MapView:
        geometry = self.geometry()
        return MapView(
            map_version=self._map_version,
            points_obj_m=geometry.points_obj_m,
            normals_obj=geometry.normals_obj,
            colors_u8=geometry.colors_u8,
            source_keyframes={
                keyframe_id: keyframe.pose_version
                for keyframe_id, (keyframe, _) in self._committed.items()
            },
        )

    def build_provisional(self, keyframes: list[Keyframe]) -> GeometryResult:
        """Fuse current keyframe poses into a separate, non-publishable diagnostic volume."""
        volume = create_volume(self.cfg.tsdf)
        for keyframe in keyframes:
            observation, raw = self._measurement(keyframe)
            integrate(
                volume,
                observation,
                raw,
                self.bundle,
                keyframe.T_cam_obj_current,
                self.cfg.tsdf,
            )
        return extract(volume, color=self.cfg.tsdf.color)

    def write_provisional(self, directory: str | Path, geometry: GeometryResult) -> dict[str, str]:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {}
        if self.cfg.output.write_pointcloud:
            rel = "provisional/object_pointcloud.ply"
            files[rel] = _write_o3d_atomic(
                root / rel, pointcloud_from_geometry(geometry), mesh=False
            )
        if self.cfg.output.write_mesh:
            rel = "provisional/object_mesh.ply"
            files[rel] = _write_o3d_atomic(root / rel, mesh_from_geometry(geometry), mesh=True)
        arrays: dict[str, np.ndarray[Any, Any]] = {
            "points_obj_m": geometry.points_obj_m,
            "normals_obj": geometry.normals_obj,
            "vertices_obj_m": geometry.vertices_obj_m,
            "triangles": geometry.triangles,
        }
        if geometry.colors_u8 is not None:
            arrays["colors_u8"] = geometry.colors_u8
        rel = "provisional/geometry.npz"
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)  # type: ignore[arg-type]  # NumPy 2.2 stub treats **arrays as bool
        files[rel] = atomic_write_bytes(root / rel, buffer.getvalue())
        return files

    def publish(self, directory: str | Path, run_id: str) -> MapSnapshot:
        if self._map_version == 0:
            raise MappingError("cannot publish an empty committed map")
        root = Path(directory)
        geometry = self.geometry()
        if len(geometry.points_obj_m) < self.cfg.output.min_point_count:
            raise MappingError("committed map has insufficient extracted geometry")
        files: dict[str, str] = {}
        if self.cfg.output.write_pointcloud:
            rel = "geometry/object_pointcloud.ply"
            files[rel] = _write_o3d_atomic(
                root / rel, pointcloud_from_geometry(geometry), mesh=False
            )
        if self.cfg.output.write_mesh:
            rel = "geometry/object_mesh.ply"
            files[rel] = _write_o3d_atomic(root / rel, mesh_from_geometry(geometry), mesh=True)
        map_view = self.view()
        rel = "geometry/map_view.npz"
        files[rel] = atomic_write_bytes(root / rel, codec.to_npz_bytes(map_view))
        snapshot = MapSnapshot(
            session_id=self.reader.session_id,
            run_id=run_id,
            map_version=self._map_version,
            parent_map_version=self._map_version - 1 if self._map_version > 1 else None,
            T_base_obj_init=self.T_base_obj_init,
            calibration_id=self.bundle.calibration_id,
            committed={
                keyframe_id: keyframe.pose_version
                for keyframe_id, (keyframe, _) in self._committed.items()
            },
            mask_revision=next(iter(self._committed.values()))[0].mask_revision,
            tsdf_params_hash=self.tsdf_params_hash,
            geometry_paths={key: key for key in files},
            created_ns=time.time_ns(),
        )
        files["snapshot.json"] = atomic_write_bytes(
            root / "snapshot.json", codec.to_json(snapshot).encode()
        )
        atomic_write_json(
            root / "manifest.json",
            {
                "schema_version": 1,
                "type": "MapArtifact",
                "snapshot": "snapshot.json",
                "files": files,
            },
        )
        self._last_snapshot = snapshot
        return snapshot

    def snapshot(self) -> MapSnapshot:
        if self._last_snapshot is None:
            raise MappingError("map has not been published")
        return self._last_snapshot

    def apply_pose_updates(self, updates: list[PoseUpdate]) -> int:
        raise MappingError("pose-update reintegration belongs to P7")


def pose_from_keyframe(keyframe: Keyframe) -> PoseEstimate:
    """A validated P4 keyframe pose for the strict commit API."""
    T_base_obj = (
        None
        if keyframe.T_base_cam_prior is None
        else keyframe.T_base_cam_prior @ keyframe.T_cam_obj_current
    )
    return PoseEstimate(
        header=keyframe.header,
        T_cam_obj=keyframe.T_cam_obj_current,
        T_base_obj=T_base_obj,
        T_base_cam_used=keyframe.T_base_cam_prior,
        status=PoseStatus.VALID,
        track_state=TrackState.TRACKING,
        reason_codes=(),
        quality={
            "inliers": len(keyframe.feature_track_ids),
            "overlap": 1.0,
            "rmse_m": 0.0,
            "hessian_min_eig": 0.0,
            "n_hypotheses": 1,
            "correction_from_prediction_m": 0.0,
            "correction_from_prediction_deg": 0.0,
        },
        source="validated_keyframe",
        pose_version=keyframe.pose_version,
    )
