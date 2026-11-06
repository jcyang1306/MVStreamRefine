"""P4 deterministic keyframe selection, retrieval and durable memory."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import numpy.typing as npt

from mvsr.core import codec
from mvsr.core.codec import SCHEMA_VERSION
from mvsr.core.errors import ChecksumMismatchError, MvsrError, SessionFormatError
from mvsr.core.se3 import PoseSE3
from mvsr.core.types import (
    Header,
    Keyframe,
    KeyframeStatus,
    KeyframeView,
    ObjectObservation,
    PoseEstimate,
    PoseStatus,
)
from mvsr.io.atomic import atomic_write_bytes, atomic_write_json, sha256_file
from mvsr.io.session_reader import SessionReader
from mvsr.observation.masks import MaskCache

from .config import KeyframeConfig

MANIFEST_NAME = "manifest.json"


class KeyframeError(MvsrError):
    """Keyframe selection or persistence failed."""


@dataclass(frozen=True)
class _Signature:
    center_obj: npt.NDArray[np.float64]
    optical_axis_obj: npt.NDArray[np.float64]
    appearance: npt.NDArray[np.uint8]
    visibility: frozenset[tuple[int, int, int]]
    spatial_cells: frozenset[int]


@dataclass(frozen=True)
class _Entry:
    keyframe: Keyframe
    observation: ObjectObservation
    signature: _Signature


def _same_header(a: Header, b: Header) -> bool:
    return (
        a.schema_version == b.schema_version
        and a.session_id == b.session_id
        and a.camera_id == b.camera_id
        and a.frame_id == b.frame_id
        and a.timestamp_ns == b.timestamp_ns
        and a.clock_domain is b.clock_domain
        and a.calibration_id == b.calibration_id
    )


def _resolve_under(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    posix = PurePosixPath(relative)
    if (
        not relative
        or posix.is_absolute()
        or "\\" in relative
        or any(part in ("", ".", "..") for part in posix.parts)
    ):
        raise SessionFormatError(f"unsafe relative path {relative!r}")
    base = root.resolve()
    candidate = (base / Path(*posix.parts)).resolve(strict=must_exist)
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise SessionFormatError(f"path escapes root: {relative!r}") from exc
    return candidate


def memory_revision_id(params: dict[str, Any]) -> str:
    digest = hashlib.sha256(codec.canonical_dumps(params).encode()).hexdigest()[:12]
    return f"p4@{digest}"


def _appearance(obs: ObjectObservation) -> npt.NDArray[np.uint8]:
    descriptors = np.asarray(obs.descriptors, dtype=np.uint8)
    if descriptors.shape[0] == 0 or descriptors.shape[1] == 0:
        return np.zeros(32, dtype=np.uint8)
    bits = np.unpackbits(descriptors, axis=1)
    return np.packbits((bits.mean(axis=0) >= 0.5).astype(np.uint8))


def _spatial(obs: ObjectObservation, grid: int) -> frozenset[int]:
    if not len(obs.keypoints_uv):
        return frozenset()
    height, width = obs.mask_binary.shape
    x = np.minimum(grid - 1, (obs.keypoints_uv[:, 0] * grid / width).astype(int))
    y = np.minimum(grid - 1, (obs.keypoints_uv[:, 1] * grid / height).astype(int))
    return frozenset((y * grid + x).tolist())


def _signature(obs: ObjectObservation, pose: PoseSE3, cfg: KeyframeConfig) -> _Signature:
    T_obj_cam = pose.inverse()
    center = np.asarray(T_obj_cam.t, dtype=np.float64)
    axis = T_obj_cam.R @ np.array([0.0, 0.0, 1.0])
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    points_obj = T_obj_cam.transform_points(obs.points_cam_m)
    sample_limit = 4 * cfg.retrieval.max_visibility_voxels
    if len(points_obj) > sample_limit:
        sample_idx = np.linspace(0, len(points_obj) - 1, sample_limit, dtype=np.intp)
        points_obj = points_obj[sample_idx]
    voxels = np.floor(points_obj / cfg.retrieval.visibility_voxel_m).astype(np.int64)
    unique: list[tuple[int, int, int]] = sorted(
        {(int(row[0]), int(row[1]), int(row[2])) for row in voxels}
    )
    if len(unique) > cfg.retrieval.max_visibility_voxels:
        idx = np.linspace(0, len(unique) - 1, cfg.retrieval.max_visibility_voxels, dtype=np.intp)
        unique = [unique[i] for i in idx]
    return _Signature(
        center_obj=center,
        optical_axis_obj=axis,
        appearance=_appearance(obs),
        visibility=frozenset(unique),
        spatial_cells=_spatial(obs, cfg.selection.feature_grid_size),
    )


def _angle_deg(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.degrees(np.arccos(np.clip(float(x @ y), -1.0, 1.0))))


def _jaccard(a: frozenset[Any], b: frozenset[Any]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _hamming(a: npt.NDArray[np.uint8], b: npt.NDArray[np.uint8]) -> int:
    return int(np.unpackbits(np.bitwise_xor(a, b)).sum())


class KeyframeManager:
    """Implements `consider(...)` and deterministic local/global retrieval."""

    def __init__(
        self,
        session_dir: str | Path,
        masks: MaskCache,
        cfg: KeyframeConfig,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.reader = SessionReader(self.session_dir)
        self.masks = masks
        self.cfg = cfg
        if masks.camera_id != "wrist":
            raise KeyframeError("P4 currently supports wrist observations only")
        self._entries: list[_Entry] = []
        self._next_id = 0
        self._T_base_obj_init: PoseSE3 | None = None
        self.last_decision: dict[str, Any] = {}
        self.last_retrieval: dict[str, Any] = {}
        self.revisit_reuses = 0
        self.revisit_opportunities = 0
        self.capacity_rejections = 0

    @property
    def views(self) -> tuple[KeyframeView, ...]:
        return tuple(
            KeyframeView(
                entry.keyframe.keyframe_id,
                entry.observation,
                entry.keyframe.T_cam_obj_current,
                entry.keyframe.pose_version,
            )
            for entry in self._entries
        )

    @property
    def all(self) -> tuple[KeyframeView, ...]:
        """Compatibility with P3's temporary `AnchorWindow`."""
        return self.views

    @property
    def keyframes(self) -> tuple[Keyframe, ...]:
        return tuple(entry.keyframe for entry in self._entries)

    def observation(self, keyframe_id: int) -> ObjectObservation:
        for entry in self._entries:
            if entry.keyframe.keyframe_id == keyframe_id:
                return entry.observation
        raise KeyError(f"unknown keyframe_id {keyframe_id}")

    @property
    def T_base_obj_init(self) -> PoseSE3 | None:
        return self._T_base_obj_init

    def reset(self) -> None:
        self._entries.clear()
        self._next_id = 0
        self._T_base_obj_init = None
        self.last_decision = {}
        self.last_retrieval = {}
        self.revisit_reuses = 0
        self.revisit_opportunities = 0
        self.capacity_rejections = 0

    def _query_signature(
        self, predicted: PoseSE3 | None, observation: ObjectObservation | None
    ) -> _Signature | None:
        if predicted is None or observation is None:
            return None
        return _signature(observation, predicted, self.cfg)

    def _metrics(self, query: _Signature, entry: _Entry) -> dict[str, float | int]:
        sig = entry.signature
        return {
            "translation_m": float(np.linalg.norm(query.center_obj - sig.center_obj)),
            "angle_deg": _angle_deg(query.optical_axis_obj, sig.optical_axis_obj),
            "appearance_hamming": _hamming(query.appearance, sig.appearance),
            "covisibility": _jaccard(query.visibility, sig.visibility),
            "spatial_novelty": 1.0 - _jaccard(query.spatial_cells, sig.spatial_cells),
        }

    def local(
        self,
        predicted: PoseSE3 | None,
        maximum: int,
        observation: ObjectObservation | None = None,
    ) -> tuple[KeyframeView, ...]:
        if not self._entries:
            self.last_retrieval = {"relevant": [], "returned": [], "recall": None}
            return ()
        maximum = min(maximum, self.cfg.retrieval.max_local_keyframes)
        query = self._query_signature(predicted, observation)
        if query is None:
            fallback = self._entries[-maximum:]
            self.last_retrieval = {
                "relevant": [entry.keyframe.keyframe_id for entry in fallback],
                "returned": [entry.keyframe.keyframe_id for entry in fallback],
                "recall": 1.0,
            }
            return tuple(
                KeyframeView(
                    entry.keyframe.keyframe_id,
                    entry.observation,
                    entry.keyframe.T_cam_obj_current,
                    entry.keyframe.pose_version,
                )
                for entry in fallback
            )
        scored: list[tuple[float, _Entry, dict[str, float | int]]] = []
        for entry in self._entries:
            metrics = self._metrics(query, entry)
            score = (
                self.cfg.retrieval.viewpoint_weight
                * (float(metrics["translation_m"]) / 0.10 + float(metrics["angle_deg"]) / 90.0)
                + self.cfg.retrieval.appearance_weight
                * float(metrics["appearance_hamming"])
                / 256.0
                + self.cfg.retrieval.covisibility_weight * (1.0 - float(metrics["covisibility"]))
            )
            scored.append((score, entry, metrics))
        scored.sort(key=lambda item: (item[0], item[1].keyframe.keyframe_id))
        # The best combined viewpoint/appearance/covisibility entry is the retrieval target;
        # the remaining slots are deliberately diversified, so treating every top-K item as
        # mandatory would penalize the diversity policy itself.
        relevant = [scored[0][1].keyframe.keyframe_id]
        selected: list[_Entry] = []
        for _, entry, _ in scored:
            if len(selected) >= maximum:
                break
            if selected and all(
                _angle_deg(
                    entry.signature.optical_axis_obj,
                    other.signature.optical_axis_obj,
                )
                < self.cfg.retrieval.min_viewpoint_separation_deg
                for other in selected
            ):
                continue
            selected.append(entry)
        if self._entries[-1] not in selected:
            if len(selected) >= maximum:
                selected[-1] = self._entries[-1]
            else:
                selected.append(self._entries[-1])
        selected = sorted(
            {entry.keyframe.keyframe_id: entry for entry in selected}.values(),
            key=lambda entry: entry.keyframe.keyframe_id,
        )
        returned = [entry.keyframe.keyframe_id for entry in selected]
        recall = len(set(returned) & set(relevant)) / len(relevant) if relevant else 1.0
        self.last_retrieval = {
            "relevant": relevant,
            "returned": returned,
            "recall": recall,
        }
        return tuple(
            KeyframeView(
                entry.keyframe.keyframe_id,
                entry.observation,
                entry.keyframe.T_cam_obj_current,
                entry.keyframe.pose_version,
            )
            for entry in selected
        )

    def _raw_refs(self, frame_id: int) -> dict[str, tuple[str, str]]:
        frame = self.reader.frame_record("wrist", frame_id)
        if frame.color is None or frame.depth is None:
            raise KeyframeError(f"frame {frame_id} has no recorded RGB-D")
        try:
            mask = self.masks.records[frame_id]
        except KeyError as exc:
            raise KeyframeError(f"frame {frame_id} has no mask") from exc
        mask_path = (self.masks.dir.relative_to(self.session_dir) / mask.path).as_posix()
        return {
            "color": (frame.color.path, frame.color.sha256),
            "depth": (frame.depth.path, frame.depth.sha256),
            "mask": (mask_path, mask.sha256),
        }

    def _insert(
        self, observation: ObjectObservation, pose: PoseEstimate, signature: _Signature
    ) -> Keyframe:
        assert pose.T_cam_obj is not None
        keyframe_id = self._next_id
        self._next_id += 1
        keyframe = Keyframe.create(
            keyframe_id=keyframe_id,
            header=observation.header,
            observation_ref=f"observations/{keyframe_id:06d}.npz",
            mask_revision=observation.mask_revision,
            raw_refs=self._raw_refs(observation.header.frame_id),
            T_base_cam_prior=pose.T_base_cam_used,
            T_cam_obj_provisional=pose.T_cam_obj,
            T_cam_obj_current=pose.T_cam_obj,
            pose_version=pose.pose_version,
            feature_track_ids=observation.track_ids,
            status=KeyframeStatus.PROVISIONAL,
        )
        self._entries.append(_Entry(keyframe, observation, signature))
        if self._T_base_obj_init is None:
            if pose.T_base_obj is None:
                raise KeyframeError("first keyframe requires T_base_obj to persist the gauge")
            self._T_base_obj_init = pose.T_base_obj
        return keyframe

    def consider(self, observation: ObjectObservation, pose: PoseEstimate) -> Keyframe | None:
        if not _same_header(observation.header, pose.header):
            raise KeyframeError("observation and pose headers differ")
        if (
            pose.status is not PoseStatus.VALID
            or pose.T_cam_obj is None
            or len(observation.points_cam_m) < self.cfg.selection.min_geometry_points
        ):
            self.last_decision = {"decision": "REJECT_QUALITY"}
            return None
        signature = _signature(observation, pose.T_cam_obj, self.cfg)
        if not self._entries:
            keyframe = self._insert(observation, pose, signature)
            self.last_decision = {
                "decision": "INSERT",
                "keyframe_id": keyframe.keyframe_id,
                "reason": "FIRST",
            }
            return keyframe
        if len(observation.keypoints_uv) < self.cfg.selection.min_keypoints:
            self.last_decision = {"decision": "REJECT_QUALITY"}
            return None
        last = self._entries[-1]
        gap = observation.header.frame_id - last.keyframe.header.frame_id
        nearest = min(
            self._entries,
            key=lambda entry: (
                float(np.linalg.norm(signature.center_obj - entry.signature.center_obj)),
                _angle_deg(
                    signature.optical_axis_obj,
                    entry.signature.optical_axis_obj,
                ),
                entry.keyframe.keyframe_id,
            ),
        )
        metrics = self._metrics(signature, nearest)
        revisit_opportunity = (
            float(metrics["angle_deg"]) < self.cfg.selection.min_viewpoint_angle_deg
            and int(metrics["appearance_hamming"]) < self.cfg.selection.min_appearance_hamming
            and float(metrics["spatial_novelty"]) < 0.25
        )
        if revisit_opportunity:
            self.revisit_opportunities += 1
        revisit = revisit_opportunity and (
            float(metrics["covisibility"]) >= self.cfg.selection.min_covisibility
            or float(metrics["spatial_novelty"]) < self.cfg.selection.min_spatial_novelty
        )
        if revisit:
            self.revisit_reuses += 1
            self.last_decision = {
                "decision": "REUSE",
                "keyframe_id": nearest.keyframe.keyframe_id,
                "metrics": metrics,
            }
            return None
        if gap < self.cfg.selection.min_frame_spacing:
            self.last_decision = {
                "decision": "REJECT_SPACING",
                "nearest_id": nearest.keyframe.keyframe_id,
                "metrics": metrics,
            }
            return None
        pose_overlap = float(pose.quality.get("overlap") or 0.0)
        insert = (
            gap >= self.cfg.selection.max_frame_gap
            or pose_overlap < self.cfg.selection.min_pose_overlap
            or float(metrics["covisibility"]) < self.cfg.selection.min_covisibility
            or float(metrics["translation_m"]) >= self.cfg.selection.min_viewpoint_translation_m
            or float(metrics["angle_deg"]) >= self.cfg.selection.min_viewpoint_angle_deg
            or int(metrics["appearance_hamming"]) >= self.cfg.selection.min_appearance_hamming
            or float(metrics["spatial_novelty"]) >= self.cfg.selection.min_spatial_novelty
        )
        if not insert:
            self.last_decision = {
                "decision": "REJECT_REDUNDANT",
                "nearest_id": nearest.keyframe.keyframe_id,
                "metrics": metrics,
            }
            return None
        if len(self._entries) >= self.cfg.retrieval.max_global_keyframes:
            self.capacity_rejections += 1
            self.last_decision = {
                "decision": "REJECT_CAPACITY",
                "nearest_id": nearest.keyframe.keyframe_id,
                "metrics": metrics,
            }
            return None
        keyframe = self._insert(observation, pose, signature)
        self.last_decision = {
            "decision": "INSERT",
            "keyframe_id": keyframe.keyframe_id,
            "nearest_id": nearest.keyframe.keyframe_id,
            "metrics": metrics,
        }
        return keyframe

    def digest(self) -> str:
        h = hashlib.sha256()
        for entry in self._entries:
            h.update(entry.keyframe.checksum.encode())
        return h.hexdigest()

    def persist(self, directory: str | Path) -> Path:
        root = Path(directory)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"keyframe memory already exists: {root}")
        (root / "observations").mkdir(parents=True, exist_ok=True)
        (root / "keyframes").mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {}
        entries: list[dict[str, Any]] = []
        for entry in self._entries:
            kid = entry.keyframe.keyframe_id
            obs_rel = entry.keyframe.observation_ref
            key_rel = f"keyframes/{kid:06d}.json"
            files[obs_rel] = atomic_write_bytes(
                root / obs_rel, codec.to_npz_bytes(entry.observation)
            )
            files[key_rel] = atomic_write_bytes(
                root / key_rel, codec.to_json(entry.keyframe).encode()
            )
            entries.append(
                {
                    "keyframe_id": kid,
                    "keyframe_file": key_rel,
                    "observation_file": obs_rel,
                    "keyframe_checksum": entry.keyframe.checksum,
                }
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "type": "KeyframeMemory",
            "session_id": self.reader.session_id,
            "session_manifest_sha256": sha256_file(self.session_dir / MANIFEST_NAME),
            "mask_revision": self.masks.revision,
            "calibration_id": (
                self._entries[0].keyframe.header.calibration_id if self._entries else None
            ),
            "config_sha256": self.cfg.source_sha256,
            "memory_digest": self.digest(),
            "T_base_obj_init": (
                None if self._T_base_obj_init is None else self._T_base_obj_init.matrix.tolist()
            ),
            "next_keyframe_id": self._next_id,
            "entries": entries,
            "files": files,
        }
        atomic_write_json(root / MANIFEST_NAME, manifest)
        return root

    @classmethod
    def load(
        cls,
        directory: str | Path,
        session_dir: str | Path,
        masks: MaskCache,
        cfg: KeyframeConfig,
    ) -> KeyframeManager:
        root = Path(directory)
        path = root / MANIFEST_NAME
        if not path.is_file():
            raise SessionFormatError(f"{root}: incomplete keyframe memory (no manifest)")
        manifest = json.loads(path.read_text())
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("type") != "KeyframeMemory"
        ):
            raise SessionFormatError(f"{path}: unsupported keyframe memory")
        manager = cls(session_dir, masks, cfg)
        if manifest.get("session_id") != manager.reader.session_id:
            raise SessionFormatError("keyframe memory session_id mismatch")
        if manifest.get("mask_revision") != masks.revision:
            raise SessionFormatError("keyframe memory mask_revision mismatch")
        if manifest.get("config_sha256") != cfg.source_sha256:
            raise SessionFormatError("keyframe memory config mismatch")
        expected_session = sha256_file(Path(session_dir) / MANIFEST_NAME)
        if manifest.get("session_manifest_sha256") != expected_session:
            raise ChecksumMismatchError("source session manifest changed")
        files = manifest.get("files") or {}
        for rel, expected in sorted(files.items()):
            actual = sha256_file(_resolve_under(root, str(rel)))
            if actual != expected:
                raise ChecksumMismatchError(f"{rel}: sha256 {actual} != {expected}")
        for item in manifest.get("entries") or []:
            keyframe_path = _resolve_under(root, str(item["keyframe_file"]))
            observation_path = _resolve_under(root, str(item["observation_file"]))
            keyframe = codec.from_json(Keyframe, keyframe_path.read_text())
            observation = codec.from_npz_bytes(ObjectObservation, observation_path.read_bytes())
            if keyframe.checksum != item["keyframe_checksum"]:
                raise ChecksumMismatchError("keyframe contract checksum mismatch")
            if not _same_header(keyframe.header, observation.header):
                raise SessionFormatError("keyframe and observation headers differ")
            if keyframe.mask_revision != observation.mask_revision:
                raise SessionFormatError("keyframe and observation mask revisions differ")
            if not np.array_equal(keyframe.feature_track_ids, observation.track_ids):
                raise SessionFormatError("keyframe feature_track_ids differ from observation")
            if set(keyframe.raw_refs) != {"color", "depth", "mask"}:
                raise SessionFormatError("keyframe raw_refs must contain color/depth/mask")
            if cfg.persistence.verify_raw_refs:
                for rel, expected in keyframe.raw_refs.values():
                    actual = sha256_file(_resolve_under(Path(session_dir), rel))
                    if actual != expected:
                        raise ChecksumMismatchError(f"raw ref {rel}: sha256 {actual} != {expected}")
            manager._entries.append(
                _Entry(
                    keyframe,
                    observation,
                    _signature(observation, keyframe.T_cam_obj_current, cfg),
                )
            )
        ids = [entry.keyframe.keyframe_id for entry in manager._entries]
        if ids != sorted(set(ids)):
            raise SessionFormatError("keyframe ids must be unique and increasing")
        expected_next = max(ids, default=-1) + 1
        if manifest.get("next_keyframe_id") != expected_next:
            raise SessionFormatError("next_keyframe_id does not follow published ids")
        manager._next_id = expected_next
        gauge = manifest.get("T_base_obj_init")
        if gauge is None and manager._entries:
            raise SessionFormatError("keyframe memory has no T_base_obj_init")
        manager._T_base_obj_init = (
            None if gauge is None else PoseSE3(np.asarray(gauge, dtype=np.float64), "base", "obj")
        )
        if manager._entries:
            first = manager._entries[0].keyframe
            if manifest.get("calibration_id") != first.header.calibration_id:
                raise SessionFormatError("keyframe memory calibration_id mismatch")
        if manager.digest() != manifest.get("memory_digest"):
            raise ChecksumMismatchError("keyframe memory digest mismatch")
        return manager
