"""Contract types of contracts/DATA_CONTRACTS.md (schema_version 1).

All classes are frozen dataclasses validated in `__post_init__`; arrays are stored as
non-writeable copies with the exact contract dtype; mappings are read-only views.
Serialize with `mvsr.core.codec` (`to_json_dict` / `to_npz_bytes` and inverses).
"""

from __future__ import annotations

import hashlib
import math
import re
import secrets
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from mvsr.core import codec
from mvsr.core.codec import SCHEMA_VERSION
from mvsr.core.depth import depth_raw_to_m
from mvsr.core.errors import ContractValidationError, FrameMismatchError
from mvsr.core.se3 import PoseSE3

# --------------------------------------------------------------------------- identifiers

SESSION_ID_RE = re.compile(r"^\d{8}T\d{6}_[A-Za-z0-9-]+_[0-9a-f]{4}$")
CALIBRATION_ID_RE = re.compile(r"^[0-9a-f]{16}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MASK_REVISION_RE = re.compile(r"^[A-Za-z0-9._+-]+@[0-9a-f]{4,64}$")

CAMERA_IDS: tuple[str, ...] = ("wrist", "head")
#: Optical frame name of each camera stream (DATA_CONTRACTS.md frame names).
CAMERA_FRAME: Mapping[str, str] = MappingProxyType({"wrist": "cam", "head": "head"})
#: Mount type of each camera stream (ADR-0007).
CAMERA_MOUNT: Mapping[str, str] = MappingProxyType({"wrist": "wrist", "head": "fixed"})


def make_session_id(host: str, when: datetime | None = None, suffix: str | None = None) -> str:
    """`YYYYMMDDTHHMMSS_<host>_<4hex>` from local time; host reduced to `[A-Za-z0-9-]`."""
    stamp = (when or datetime.now()).strftime("%Y%m%dT%H%M%S")
    clean_host = re.sub(r"[^A-Za-z0-9-]", "-", host) or "host"
    tail = suffix if suffix is not None else secrets.token_hex(2)
    sid = f"{stamp}_{clean_host}_{tail}"
    if not SESSION_ID_RE.match(sid):
        raise ContractValidationError(f"invalid session id {sid!r}")
    return sid


# --------------------------------------------------------------------------- enums


class ClockDomain(str, Enum):
    HOST_MONOTONIC = "HOST_MONOTONIC"
    RS_DEVICE = "RS_DEVICE"
    RS_GLOBAL = "RS_GLOBAL"
    ROBOT_CONTROLLER = "ROBOT_CONTROLLER"


class SyncStatus(str, Enum):
    INTERPOLATED = "INTERPOLATED"
    GAP_TOO_LARGE = "GAP_TOO_LARGE"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    NO_ROBOT = "NO_ROBOT"


class PoseStatus(str, Enum):
    VALID = "VALID"
    DEGRADED = "DEGRADED"
    INVALID = "INVALID"


class TrackState(str, Enum):
    INITIALIZING = "INITIALIZING"
    TRACKING = "TRACKING"
    DEGRADED = "DEGRADED"
    LOST = "LOST"
    RELOCALIZING = "RELOCALIZING"


class KeyframeStatus(str, Enum):
    PROVISIONAL = "PROVISIONAL"
    COMMITTED = "COMMITTED"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"


class ReasonCode(str, Enum):
    MASK_TOO_SMALL = "MASK_TOO_SMALL"
    DEPTH_COVERAGE_LOW = "DEPTH_COVERAGE_LOW"
    POSE_SYNC_FAILED = "POSE_SYNC_FAILED"
    FEW_FEATURES = "FEW_FEATURES"
    LOW_INLIERS = "LOW_INLIERS"
    LOW_OVERLAP = "LOW_OVERLAP"
    HIGH_RMSE = "HIGH_RMSE"
    DEGENERATE_HESSIAN = "DEGENERATE_HESSIAN"
    SYMMETRY_AMBIGUOUS = "SYMMETRY_AMBIGUOUS"
    LARGE_CORRECTION = "LARGE_CORRECTION"
    PREDICTION_ONLY = "PREDICTION_ONLY"
    STALE_VERSION = "STALE_VERSION"
    CHECKSUM_MISMATCH = "CHECKSUM_MISMATCH"


# Session-format enums (contracts/SESSION_FORMAT.md).


class SessionStatus(str, Enum):
    RECORDING = "RECORDING"
    COMPLETE = "COMPLETE"
    ABORTED = "ABORTED"


class FrameStatus(str, Enum):
    WRITTEN = "WRITTEN"
    DROPPED_QUEUE_FULL = "DROPPED_QUEUE_FULL"
    DROPPED_INCOMPLETE = "DROPPED_INCOMPLETE"
    DROPPED_DISK = "DROPPED_DISK"
    DROPPED_OPERATOR = "DROPPED_OPERATOR"


class EventType(str, Enum):
    START = "START"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    STOP = "STOP"
    ROI_SELECTED = "ROI_SELECTED"
    ANNOTATION = "ANNOTATION"
    WARNING = "WARNING"
    ERROR = "ERROR"


RobotSourceName = Literal["udp_push", "poll"]
MountName = Literal["wrist", "fixed"]

# --------------------------------------------------------------------------- validators

U8 = npt.NDArray[np.uint8]
U16 = npt.NDArray[np.uint16]
I32 = npt.NDArray[np.int32]
I64 = npt.NDArray[np.int64]
F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
BOOL = npt.NDArray[np.bool_]


def _set(obj: Any, name: str, value: Any) -> None:
    object.__setattr__(obj, name, value)


def _int(value: Any, name: str, lo: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ContractValidationError(f"{name} must be int, got {type(value).__name__}")
    out = int(value)
    if lo is not None and out < lo:
        raise ContractValidationError(f"{name} must be >= {lo}, got {out}")
    return out


def _opt_int(value: Any, name: str, lo: int | None = None) -> int | None:
    return None if value is None else _int(value, name, lo)


def _float(value: Any, name: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ContractValidationError(f"{name} must be a number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out) or (positive and out <= 0.0):
        raise ContractValidationError(
            f"{name} must be finite{' and > 0' if positive else ''}: {out}"
        )
    return out


def _opt_float(value: Any, name: str) -> float | None:
    return None if value is None else _float(value, name)


def _str(value: Any, name: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise ContractValidationError(f"{name} must be a non-empty str")
    if pattern is not None and not pattern.match(value):
        raise ContractValidationError(f"{name}={value!r} does not match {pattern.pattern}")
    return value


def _enum(value: Any, cls: type[Enum], name: str) -> Any:
    try:
        return cls(value)
    except ValueError as exc:
        raise ContractValidationError(f"{name}={value!r} is not a {cls.__name__}") from exc


def _arr(
    value: Any, dtype: type[np.generic], name: str, shape: tuple[int | None, ...]
) -> np.ndarray[Any, Any]:
    """Validate dtype/shape (None = any size) and return a non-writeable array."""
    if isinstance(value, np.ndarray):
        if value.dtype != np.dtype(dtype):
            raise ContractValidationError(f"{name} must be {np.dtype(dtype)}, got {value.dtype}")
        arr = value
    else:
        arr = np.asarray(value, dtype=dtype)
    if arr.ndim != len(shape) or any(
        s is not None and s != a for s, a in zip(shape, arr.shape, strict=True)
    ):
        raise ContractValidationError(f"{name} must have shape {shape}, got {arr.shape}")
    if arr.flags.writeable:
        arr = np.array(arr, copy=True)
        arr.flags.writeable = False
    return arr


def _finite(arr: np.ndarray[Any, Any], name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ContractValidationError(f"{name} contains non-finite values")


def _mapping(value: Any, name: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{name} must be a mapping")
    return MappingProxyType(dict(value))


def _pose(value: Any, dst: str, src: str, name: str, optional: bool = False) -> PoseSE3 | None:
    if value is None:
        if optional:
            return None
        raise ContractValidationError(f"{name} is required")
    if not isinstance(value, PoseSE3):
        raise ContractValidationError(f"{name} must be PoseSE3")
    if value.dst != dst or value.src != src:
        raise FrameMismatchError(f"{name} must be T_{dst}_{src}, got {value.name}")
    return value


def _camera_id(value: Any, name: str = "camera_id") -> str:
    if value not in CAMERA_IDS:
        raise ContractValidationError(f"{name}={value!r} not in {CAMERA_IDS}")
    return str(value)


def _intrinsics(value: Any, name: str = "K") -> F64:
    K = _arr(value, np.float64, name, (3, 3))
    _finite(K, name)
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.array_equal(K[2], [0.0, 0.0, 1.0]) or K[1, 0] != 0:
        raise ContractValidationError(f"{name} is not a pinhole intrinsic matrix: {K.tolist()}")
    return K


def _content_hash(obj: Any, exclude: str) -> str:
    body = codec.to_json_dict(obj)
    body.pop(exclude, None)
    return hashlib.sha256(codec.canonical_dumps(body).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- header + primitives


@dataclass(frozen=True, eq=False)
class Header:
    """Common header; `timestamp_ns` = estimated mid-exposure time in HOST_MONOTONIC."""

    schema_version: int
    session_id: str
    camera_id: str
    frame_id: int
    timestamp_ns: int
    clock_domain: ClockDomain
    calibration_id: str

    def __post_init__(self) -> None:
        if _int(self.schema_version, "schema_version") != SCHEMA_VERSION:
            raise ContractValidationError(f"Header.schema_version must be {SCHEMA_VERSION}")
        _str(self.session_id, "session_id", SESSION_ID_RE)
        _camera_id(self.camera_id)
        _set(self, "frame_id", _int(self.frame_id, "frame_id", 0))
        _set(self, "timestamp_ns", _int(self.timestamp_ns, "timestamp_ns"))
        domain = _enum(self.clock_domain, ClockDomain, "clock_domain")
        if domain is not ClockDomain.HOST_MONOTONIC:
            raise ContractValidationError("Header.timestamp_ns must be in HOST_MONOTONIC")
        _set(self, "clock_domain", domain)
        _str(self.calibration_id, "calibration_id", CALIBRATION_ID_RE)


@dataclass(frozen=True, eq=False)
class CalibrationBundle:
    """Per-camera calibration. `K`/distortion are for the registered color image.

    `calibration_id` = first 16 hex of sha256 over the canonical JSON of this bundle without
    `calibration_id` (use `CalibrationBundle.create`); it is re-verified on construction.
    """

    calibration_id: str
    camera_id: str
    camera_serial: str
    camera_model: str
    mount: MountName
    width: int
    height: int
    K: F64
    distortion_model: str
    distortion: F64
    depth_units_m: float
    T_tcp_cam: PoseSE3 | None
    T_base_head: PoseSE3 | None
    dt_robot_cam_ns: int = 0
    source_files: Mapping[str, str] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        self._normalize()
        _str(self.calibration_id, "calibration_id", CALIBRATION_ID_RE)
        expected = self.compute_id()
        if self.calibration_id != expected:
            raise ContractValidationError(
                f"calibration_id {self.calibration_id} != content hash {expected}"
            )

    def _normalize(self) -> None:
        cam = _camera_id(self.camera_id)
        _str(self.camera_serial, "camera_serial")
        _str(self.camera_model, "camera_model")
        if self.mount != CAMERA_MOUNT[cam]:
            raise ContractValidationError(f"camera {cam} must have mount {CAMERA_MOUNT[cam]!r}")
        _set(self, "width", _int(self.width, "width", 1))
        _set(self, "height", _int(self.height, "height", 1))
        _set(self, "K", _intrinsics(self.K))
        _str(self.distortion_model, "distortion_model")
        dist = _arr(self.distortion, np.float64, "distortion", (5,))
        _finite(dist, "distortion")
        _set(self, "distortion", dist)
        _set(self, "depth_units_m", _float(self.depth_units_m, "depth_units_m", positive=True))
        if self.mount == "wrist":
            _pose(self.T_tcp_cam, "tcp", "cam", "T_tcp_cam")
            if self.T_base_head is not None:
                raise ContractValidationError("wrist bundle must not set T_base_head")
        else:
            _pose(self.T_base_head, "base", "head", "T_base_head")
            if self.T_tcp_cam is not None:
                raise ContractValidationError("fixed bundle must not set T_tcp_cam")
        _set(self, "dt_robot_cam_ns", _int(self.dt_robot_cam_ns, "dt_robot_cam_ns"))
        files = _mapping(self.source_files, "source_files")
        for k, v in files.items():
            _str(k, "source_files key")
            _str(v, f"source_files[{k}]", SHA256_RE)
        _set(self, "source_files", files)
        if not isinstance(self.notes, str):
            raise ContractValidationError("notes must be str")

    def compute_id(self) -> str:
        return _content_hash(self, "calibration_id")[:16]

    @classmethod
    def create(cls, **kwargs: Any) -> CalibrationBundle:
        """Build a bundle and assign its deterministic `calibration_id`."""
        probe = _probe(cls, kwargs, "calibration_id", "0" * 16)
        probe._normalize()
        return cls(calibration_id=probe.compute_id(), **_init_kwargs(probe, "calibration_id"))

    @property
    def T_ext(self) -> PoseSE3:
        """The one extrinsic of this bundle (`T_tcp_cam` or `T_base_head`)."""
        pose = self.T_tcp_cam if self.mount == "wrist" else self.T_base_head
        assert pose is not None
        return pose


def _probe(cls: type[Any], kwargs: dict[str, Any], derived: str, placeholder: str) -> Any:
    """Uninitialized instance holding `kwargs` (+ defaults) so fields can be normalized first."""
    if derived in kwargs:
        raise ContractValidationError(f"{derived} is derived; do not pass it")
    probe = object.__new__(cls)
    for f in fields(cls):
        if f.name == derived:
            value: Any = placeholder
        elif f.name in kwargs:
            value = kwargs[f.name]
        elif f.default is not MISSING:
            value = f.default
        elif f.default_factory is not MISSING:
            value = f.default_factory()
        else:
            raise ContractValidationError(f"{cls.__name__}: missing field {f.name}")
        object.__setattr__(probe, f.name, value)
    unknown = set(kwargs) - {f.name for f in fields(cls)}
    if unknown:
        raise ContractValidationError(f"{cls.__name__}: unknown fields {sorted(unknown)}")
    return probe


def _init_kwargs(obj: Any, derived: str) -> dict[str, Any]:
    return {f.name: getattr(obj, f.name) for f in fields(obj) if f.name != derived}


@dataclass(frozen=True, eq=False)
class RobotSample:
    """One successful robot state read; `t_host_ns` in `clock_domain` (HOST_MONOTONIC)."""

    seq: int
    t_host_ns: int
    clock_domain: ClockDomain
    T_base_tcp: PoseSE3
    joints_rad: F64 | None
    raw_xyzrpy: F64 | None
    source: RobotSourceName
    read_latency_ns: int | None

    def __post_init__(self) -> None:
        _set(self, "seq", _int(self.seq, "seq", 0))
        _set(self, "t_host_ns", _int(self.t_host_ns, "t_host_ns"))
        _set(self, "clock_domain", _enum(self.clock_domain, ClockDomain, "clock_domain"))
        _pose(self.T_base_tcp, "base", "tcp", "T_base_tcp")
        for name in ("joints_rad", "raw_xyzrpy"):
            value = getattr(self, name)
            if value is not None:
                arr = _arr(value, np.float64, name, (6,))
                _finite(arr, name)
                _set(self, name, arr)
        if self.source not in ("udp_push", "poll"):
            raise ContractValidationError(f"source={self.source!r} not in (udp_push, poll)")
        _set(self, "read_latency_ns", _opt_int(self.read_latency_ns, "read_latency_ns", 0))


@dataclass(frozen=True, eq=False)
class FrameTiming:
    """All raw clock values of one camera frame; `method` states how `timestamp_ns` was derived."""

    t_host_arrival_ns: int
    rs_timestamp_ms: float | None
    rs_timestamp_domain: str | None
    rs_sensor_timestamp_us: int | None
    rs_frame_timestamp_us: int | None
    exposure_us: float | None
    rs_frame_number: int | None
    method: str

    def __post_init__(self) -> None:
        _set(self, "t_host_arrival_ns", _int(self.t_host_arrival_ns, "t_host_arrival_ns"))
        _set(self, "rs_timestamp_ms", _opt_float(self.rs_timestamp_ms, "rs_timestamp_ms"))
        if self.rs_timestamp_domain is not None:
            _str(self.rs_timestamp_domain, "rs_timestamp_domain")
        for name in ("rs_sensor_timestamp_us", "rs_frame_timestamp_us", "rs_frame_number"):
            _set(self, name, _opt_int(getattr(self, name), name))
        _set(self, "exposure_us", _opt_float(self.exposure_us, "exposure_us"))
        _str(self.method, "method")


@dataclass(frozen=True, eq=False)
class PoseSyncInfo:
    status: SyncStatus
    method: str
    gap_before_ns: int | None
    gap_after_ns: int | None
    robot_seq_before: int | None
    robot_seq_after: int | None

    def __post_init__(self) -> None:
        _set(self, "status", _enum(self.status, SyncStatus, "status"))
        _str(self.method, "method")
        for name in ("gap_before_ns", "gap_after_ns", "robot_seq_before", "robot_seq_after"):
            _set(self, name, _opt_int(getattr(self, name), name, 0))


# --------------------------------------------------------------------------- messages


@dataclass(frozen=True, eq=False)
class SensorFrame:
    """Registered RGB-D frame. Sync failures are emitted with `T_base_cam_prior=None`."""

    header: Header
    color_rgb_u8: U8
    depth_raw_u16: U16
    depth_units_m: float
    depth_m_f32: F32
    valid_depth_mask: BOOL
    K: F64
    T_base_tcp: PoseSE3 | None
    T_base_cam_prior: PoseSE3 | None
    pose_sync: PoseSyncInfo
    timing: FrameTiming
    source_meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.header, Header):
            raise ContractValidationError("header must be Header")
        color = _arr(self.color_rgb_u8, np.uint8, "color_rgb_u8", (None, None, 3))
        h, w = color.shape[:2]
        raw = _arr(self.depth_raw_u16, np.uint16, "depth_raw_u16", (h, w))
        units = _float(self.depth_units_m, "depth_units_m", positive=True)
        depth_m = _arr(self.depth_m_f32, np.float32, "depth_m_f32", (h, w))
        valid = _arr(self.valid_depth_mask, np.bool_, "valid_depth_mask", (h, w))
        if not np.array_equal(depth_m, depth_raw_to_m(raw, units)):
            raise ContractValidationError("depth_m_f32 != depth_raw_to_m(depth_raw_u16, units)")
        if not np.array_equal(valid, raw != 0):
            raise ContractValidationError("valid_depth_mask must equal depth_raw_u16 != 0")
        for name, value in (("color_rgb_u8", color), ("depth_raw_u16", raw)):
            _set(self, name, value)
        _set(self, "depth_units_m", units)
        _set(self, "depth_m_f32", depth_m)
        _set(self, "valid_depth_mask", valid)
        _set(self, "K", _intrinsics(self.K))
        _pose(self.T_base_tcp, "base", "tcp", "T_base_tcp", optional=True)
        cam_frame = CAMERA_FRAME[self.header.camera_id]
        _pose(self.T_base_cam_prior, "base", cam_frame, "T_base_cam_prior", optional=True)
        if not isinstance(self.pose_sync, PoseSyncInfo) or not isinstance(self.timing, FrameTiming):
            raise ContractValidationError("pose_sync/timing have wrong types")
        if (
            self.pose_sync.status is not SyncStatus.INTERPOLATED
            and self.T_base_cam_prior is not None
        ):
            raise ContractValidationError("T_base_cam_prior must be None when sync failed")
        _set(self, "source_meta", _mapping(self.source_meta, "source_meta"))


OBSERVATION_QUALITY_KEYS = ("mask_area_px", "valid_depth_ratio", "n_keypoints")


@dataclass(frozen=True, eq=False)
class ObjectObservation:
    """Per-frame object evidence. `geometry_mask` (subset of `mask_binary`) may produce 3D points."""

    header: Header
    instance_id: int
    mask_revision: str
    mask_binary: BOOL
    mask_prob: F32 | None
    geometry_mask: BOOL
    pixel_uv: I32
    points_cam_m: F32
    normals_cam: F32 | None
    keypoints_uv: F32
    keypoint_depth_m: F32
    descriptors: np.ndarray[Any, Any]
    descriptor_type: str
    track_ids: I64
    quality: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.header, Header):
            raise ContractValidationError("header must be Header")
        _set(self, "instance_id", _int(self.instance_id, "instance_id", 0))
        _str(self.mask_revision, "mask_revision", MASK_REVISION_RE)
        mask = _arr(self.mask_binary, np.bool_, "mask_binary", (None, None))
        h, w = mask.shape
        _set(self, "mask_binary", mask)
        if self.mask_prob is not None:
            prob = _arr(self.mask_prob, np.float32, "mask_prob", (h, w))
            if not np.all((prob >= 0) & (prob <= 1)):
                raise ContractValidationError("mask_prob must be in [0, 1]")
            _set(self, "mask_prob", prob)
        geom = _arr(self.geometry_mask, np.bool_, "geometry_mask", (h, w))
        if np.any(geom & ~mask):
            raise ContractValidationError("geometry_mask must be a subset of mask_binary")
        _set(self, "geometry_mask", geom)
        uv = _arr(self.pixel_uv, np.int32, "pixel_uv", (None, 2))
        n = uv.shape[0]
        if n and (
            uv[:, 0].min() < 0 or uv[:, 0].max() >= w or uv[:, 1].min() < 0 or uv[:, 1].max() >= h
        ):
            raise ContractValidationError("pixel_uv out of image bounds")
        if n and not np.all(geom[uv[:, 1], uv[:, 0]]):
            raise ContractValidationError("pixel_uv must lie inside geometry_mask")
        _set(self, "pixel_uv", uv)
        pts = _arr(self.points_cam_m, np.float32, "points_cam_m", (n, 3))
        _finite(pts, "points_cam_m")
        _set(self, "points_cam_m", pts)
        if self.normals_cam is not None:
            _set(self, "normals_cam", _arr(self.normals_cam, np.float32, "normals_cam", (n, 3)))
        kp = _arr(self.keypoints_uv, np.float32, "keypoints_uv", (None, 2))
        m = kp.shape[0]
        _set(self, "keypoints_uv", kp)
        _set(
            self,
            "keypoint_depth_m",
            _arr(self.keypoint_depth_m, np.float32, "keypoint_depth_m", (m,)),
        )
        desc = self.descriptors
        if not isinstance(desc, np.ndarray) or desc.ndim != 2 or desc.shape[0] != m:
            raise ContractValidationError(f"descriptors must be ndarray[{m}, D]")
        if desc.flags.writeable:
            desc = np.array(desc, copy=True)
            desc.flags.writeable = False
        _set(self, "descriptors", desc)
        if not isinstance(self.descriptor_type, str):
            raise ContractValidationError("descriptor_type must be str")
        _set(self, "track_ids", _arr(self.track_ids, np.int64, "track_ids", (m,)))
        quality = _mapping(self.quality, "quality")
        missing = [k for k in OBSERVATION_QUALITY_KEYS if k not in quality]
        if missing:
            raise ContractValidationError(f"quality missing keys {missing}")
        _set(self, "quality", quality)


@dataclass(frozen=True, eq=False)
class CorrespondenceSet:
    """2D/3D matches between two frames; rejected matches stay with a `rejection_reason`."""

    src_frame_id: int
    tgt_frame_id: int
    src_uv: F32
    tgt_uv: F32
    src_point_cam_m: F32
    tgt_point_cam_m: F32
    sigma_px: F32
    sigma_m: F32
    track_ids: I64
    method: str
    inlier: BOOL
    rejection_reason: list[str | None]

    def __post_init__(self) -> None:
        _set(self, "src_frame_id", _int(self.src_frame_id, "src_frame_id", 0))
        _set(self, "tgt_frame_id", _int(self.tgt_frame_id, "tgt_frame_id", 0))
        src_uv = _arr(self.src_uv, np.float32, "src_uv", (None, 2))
        k = src_uv.shape[0]
        _set(self, "src_uv", src_uv)
        _set(self, "tgt_uv", _arr(self.tgt_uv, np.float32, "tgt_uv", (k, 2)))
        for name in ("src_point_cam_m", "tgt_point_cam_m"):
            _set(self, name, _arr(getattr(self, name), np.float32, name, (k, 3)))
        sigma_px = _arr(self.sigma_px, np.float32, "sigma_px", (k,))
        if not np.all(np.isfinite(sigma_px) & (sigma_px >= 0)):
            raise ContractValidationError("sigma_px must be finite and >= 0")
        _set(self, "sigma_px", sigma_px)
        sigma_m = _arr(self.sigma_m, np.float32, "sigma_m", (k,))
        if np.any(sigma_m < 0):
            raise ContractValidationError("sigma_m must be >= 0 (NaN = no depth)")
        _set(self, "sigma_m", sigma_m)
        _set(self, "track_ids", _arr(self.track_ids, np.int64, "track_ids", (k,)))
        _str(self.method, "method")
        inlier = _arr(self.inlier, np.bool_, "inlier", (k,))
        _set(self, "inlier", inlier)
        reasons = list(self.rejection_reason)
        if len(reasons) != k:
            raise ContractValidationError(f"rejection_reason must have {k} entries")
        for i, (ok, reason) in enumerate(zip(inlier.tolist(), reasons, strict=True)):
            if ok != (reason is None):
                raise ContractValidationError(f"match {i}: inlier ⇔ rejection_reason is None")
            if reason is not None:
                _str(reason, f"rejection_reason[{i}]")
        _set(self, "rejection_reason", reasons)


POSE_QUALITY_KEYS = (
    "inliers",
    "overlap",
    "rmse_m",
    "hessian_min_eig",
    "n_hypotheses",
    "correction_from_prediction_m",
    "correction_from_prediction_deg",
)
#: Max |T_base_obj - T_base_cam_used @ T_cam_obj| allowed in a PoseEstimate.
POSE_CHAIN_ATOL = 1e-6


@dataclass(frozen=True, eq=False)
class PoseEstimate:
    """`status == INVALID` ⇔ `T_cam_obj is None`; identity is never a failure default."""

    header: Header
    T_cam_obj: PoseSE3 | None
    T_base_obj: PoseSE3 | None
    T_base_cam_used: PoseSE3 | None
    status: PoseStatus
    track_state: TrackState
    reason_codes: tuple[ReasonCode, ...]
    quality: Mapping[str, Any]
    source: str
    pose_version: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.header, Header):
            raise ContractValidationError("header must be Header")
        status = _enum(self.status, PoseStatus, "status")
        _set(self, "status", status)
        _set(self, "track_state", _enum(self.track_state, TrackState, "track_state"))
        cam_obj = _pose(self.T_cam_obj, "cam", "obj", "T_cam_obj", optional=True)
        base_obj = _pose(self.T_base_obj, "base", "obj", "T_base_obj", optional=True)
        base_cam = _pose(self.T_base_cam_used, "base", "cam", "T_base_cam_used", optional=True)
        if (status is PoseStatus.INVALID) != (cam_obj is None):
            raise ContractValidationError("status INVALID ⇔ T_cam_obj is None")
        if cam_obj is None and base_obj is not None:
            raise ContractValidationError("T_base_obj requires T_cam_obj")
        if cam_obj is not None and base_obj is not None and base_cam is not None:
            chained = base_cam @ cam_obj
            if not np.allclose(chained.matrix, base_obj.matrix, atol=POSE_CHAIN_ATOL, rtol=0):
                raise ContractValidationError("T_base_obj != T_base_cam_used @ T_cam_obj")
        _set(
            self,
            "reason_codes",
            tuple(_enum(r, ReasonCode, "reason_codes") for r in self.reason_codes),
        )
        quality = _mapping(self.quality, "quality")
        missing = [k for k in POSE_QUALITY_KEYS if k not in quality]
        if missing:
            raise ContractValidationError(f"quality missing keys {missing}")
        _set(self, "quality", quality)
        _str(self.source, "source")
        _set(self, "pose_version", _int(self.pose_version, "pose_version", 0))


@dataclass(frozen=True, eq=False)
class KeyframeView:
    keyframe_id: int
    observation: ObjectObservation
    T_cam_obj: PoseSE3
    pose_version: int

    def __post_init__(self) -> None:
        _set(self, "keyframe_id", _int(self.keyframe_id, "keyframe_id", 0))
        if not isinstance(self.observation, ObjectObservation):
            raise ContractValidationError("observation must be ObjectObservation")
        _pose(self.T_cam_obj, "cam", "obj", "T_cam_obj")
        _set(self, "pose_version", _int(self.pose_version, "pose_version", 0))


@dataclass(frozen=True, eq=False)
class TrackingReferences:
    prev: PoseEstimate | None
    predicted_T_cam_obj: PoseSE3 | None
    local_keyframes: tuple[KeyframeView, ...]

    def __post_init__(self) -> None:
        if self.prev is not None and not isinstance(self.prev, PoseEstimate):
            raise ContractValidationError("prev must be PoseEstimate | None")
        _pose(self.predicted_T_cam_obj, "cam", "obj", "predicted_T_cam_obj", optional=True)
        kfs = tuple(self.local_keyframes)
        if not all(isinstance(k, KeyframeView) for k in kfs):
            raise ContractValidationError("local_keyframes must be KeyframeView")
        _set(self, "local_keyframes", kfs)


@dataclass(frozen=True, eq=False)
class MapView:
    """Immutable snapshot of tracking geometry in the object frame."""

    map_version: int
    points_obj_m: F32
    normals_obj: F32
    colors_u8: U8 | None
    source_keyframes: Mapping[int, int]

    def __post_init__(self) -> None:
        _set(self, "map_version", _int(self.map_version, "map_version", 0))
        pts = _arr(self.points_obj_m, np.float32, "points_obj_m", (None, 3))
        p = pts.shape[0]
        _set(self, "points_obj_m", pts)
        _set(self, "normals_obj", _arr(self.normals_obj, np.float32, "normals_obj", (p, 3)))
        if self.colors_u8 is not None:
            _set(self, "colors_u8", _arr(self.colors_u8, np.uint8, "colors_u8", (p, 3)))
        src = _mapping(self.source_keyframes, "source_keyframes")
        for k, v in src.items():
            _int(k, "keyframe_id", 0)
            _int(v, "pose_version", 0)
        _set(self, "source_keyframes", src)


@dataclass(frozen=True, eq=False)
class Keyframe:
    """Durable keyframe record; `checksum` = sha256 of its canonical JSON without `checksum`."""

    keyframe_id: int
    header: Header
    observation_ref: str
    mask_revision: str
    raw_refs: Mapping[str, tuple[str, str]]
    T_base_cam_prior: PoseSE3 | None
    T_cam_obj_provisional: PoseSE3
    T_cam_obj_current: PoseSE3
    pose_version: int
    feature_track_ids: I64
    status: KeyframeStatus
    checksum: str

    def __post_init__(self) -> None:
        self._normalize()
        _str(self.checksum, "checksum", SHA256_RE)
        expected = _content_hash(self, "checksum")
        if self.checksum != expected:
            raise ContractValidationError(f"Keyframe checksum {self.checksum} != {expected}")

    def _normalize(self) -> None:
        _set(self, "keyframe_id", _int(self.keyframe_id, "keyframe_id", 0))
        if not isinstance(self.header, Header):
            raise ContractValidationError("header must be Header")
        _str(self.observation_ref, "observation_ref")
        _str(self.mask_revision, "mask_revision", MASK_REVISION_RE)
        refs = {}
        for key, ref in _mapping(self.raw_refs, "raw_refs").items():
            path, digest = tuple(ref)
            refs[_str(key, "raw_refs key")] = (
                _str(path, "path"),
                _str(digest, "sha256", SHA256_RE),
            )
        _set(self, "raw_refs", MappingProxyType(refs))
        _pose(self.T_base_cam_prior, "base", "cam", "T_base_cam_prior", optional=True)
        _pose(self.T_cam_obj_provisional, "cam", "obj", "T_cam_obj_provisional")
        _pose(self.T_cam_obj_current, "cam", "obj", "T_cam_obj_current")
        _set(self, "pose_version", _int(self.pose_version, "pose_version", 0))
        _set(
            self,
            "feature_track_ids",
            _arr(self.feature_track_ids, np.int64, "feature_track_ids", (None,)),
        )
        _set(self, "status", _enum(self.status, KeyframeStatus, "status"))

    @classmethod
    def create(cls, **kwargs: Any) -> Keyframe:
        """Build a keyframe and assign its checksum."""
        probe = _probe(cls, kwargs, "checksum", "0" * 64)
        probe._normalize()
        return cls(checksum=_content_hash(probe, "checksum"), **_init_kwargs(probe, "checksum"))


@dataclass(frozen=True, eq=False)
class PoseUpdate:
    """Accepted only when `old_version` equals the current keyframe version (else STALE_VERSION)."""

    session_id: str
    run_id: str
    keyframe_id: int
    frame_id: int
    old_version: int
    new_version: int
    T_cam_obj: PoseSE3
    covariance: F64 | None
    optimizer_run_id: str
    created_ns: int

    def __post_init__(self) -> None:
        _str(self.session_id, "session_id", SESSION_ID_RE)
        _str(self.run_id, "run_id", SESSION_ID_RE)
        _set(self, "keyframe_id", _int(self.keyframe_id, "keyframe_id", 0))
        _set(self, "frame_id", _int(self.frame_id, "frame_id", 0))
        old = _int(self.old_version, "old_version", 0)
        new = _int(self.new_version, "new_version", 0)
        if new != old + 1:
            raise ContractValidationError("new_version must equal old_version + 1")
        _set(self, "old_version", old)
        _set(self, "new_version", new)
        _pose(self.T_cam_obj, "cam", "obj", "T_cam_obj")
        if self.covariance is not None:
            cov = _arr(self.covariance, np.float64, "covariance", (6, 6))
            _finite(cov, "covariance")
            _set(self, "covariance", cov)
        _str(self.optimizer_run_id, "optimizer_run_id")
        _set(self, "created_ns", _int(self.created_ns, "created_ns"))


@dataclass(frozen=True, eq=False)
class MapSnapshot:
    """Published map state; `T_base_obj_init = [I | c_base]` is fixed per session (ADR-0005)."""

    session_id: str
    run_id: str
    map_version: int
    parent_map_version: int | None
    T_base_obj_init: PoseSE3
    calibration_id: str
    committed: Mapping[int, int]
    mask_revision: str
    tsdf_params_hash: str
    geometry_paths: Mapping[str, str]
    created_ns: int

    def __post_init__(self) -> None:
        _str(self.session_id, "session_id", SESSION_ID_RE)
        _str(self.run_id, "run_id", SESSION_ID_RE)
        version = _int(self.map_version, "map_version", 0)
        _set(self, "map_version", version)
        parent = _opt_int(self.parent_map_version, "parent_map_version", 0)
        if parent is not None and parent >= version:
            raise ContractValidationError("parent_map_version must be < map_version")
        _set(self, "parent_map_version", parent)
        pose = _pose(self.T_base_obj_init, "base", "obj", "T_base_obj_init")
        assert pose is not None
        if not np.array_equal(pose.R, np.eye(3)):
            raise ContractValidationError("T_base_obj_init rotation must be identity (ADR-0005)")
        _str(self.calibration_id, "calibration_id", CALIBRATION_ID_RE)
        committed = _mapping(self.committed, "committed")
        for k, v in committed.items():
            _int(k, "keyframe_id", 0)
            _int(v, "pose_version", 0)
        _set(self, "committed", committed)
        _str(self.mask_revision, "mask_revision", MASK_REVISION_RE)
        _str(self.tsdf_params_hash, "tsdf_params_hash")
        _set(self, "geometry_paths", _mapping(self.geometry_paths, "geometry_paths"))
        _set(self, "created_ns", _int(self.created_ns, "created_ns"))


# --------------------------------------------------------------------------- helpers


def deep_equal(a: Any, b: Any) -> bool:
    """Structural equality for contract objects (arrays compared exactly, NaN == NaN)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return (
            a.dtype == b.dtype
            and a.shape == b.shape
            and bool(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))
        )
    if isinstance(a, PoseSE3):
        return bool(a == b)
    if is_dataclass(a) and not isinstance(a, type):
        return all(deep_equal(getattr(a, f.name), getattr(b, f.name)) for f in fields(a))
    if isinstance(a, Mapping):
        return a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b, strict=True))
    return bool(a == b)
