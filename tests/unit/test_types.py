"""Contract types: validators, immutability, JSON/NPZ round trip, schema versions (F2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mvsr.calib.handeye import load_handeye
from mvsr.core import codec
from mvsr.core.depth import depth_raw_to_m
from mvsr.core.errors import ContractValidationError, FrameMismatchError, SchemaVersionError
from mvsr.core.se3 import PoseSE3, so3_exp
from mvsr.core.types import (
    CalibrationBundle,
    ClockDomain,
    CorrespondenceSet,
    FrameTiming,
    Header,
    Keyframe,
    KeyframeStatus,
    KeyframeView,
    MapSnapshot,
    MapView,
    ObjectObservation,
    PoseEstimate,
    PoseStatus,
    PoseSyncInfo,
    PoseUpdate,
    ReasonCode,
    RobotSample,
    SensorFrame,
    SyncStatus,
    TrackingReferences,
    TrackState,
    deep_equal,
    make_session_id,
)

SID = "20261028T101500_CW20175_a3f1"
H, W = 6, 8
RNG = np.random.default_rng(5)


def pose(dst: str, src: str, t: tuple[float, float, float] = (0.1, -0.2, 0.3)) -> PoseSE3:
    return PoseSE3.from_rt(so3_exp([0.2, -0.1, 0.4]), t, dst, src)


@pytest.fixture(scope="module")
def bundle(reference_handeye_path_module: Path) -> CalibrationBundle:
    he = load_handeye(reference_handeye_path_module)
    return CalibrationBundle.create(
        camera_id="wrist",
        camera_serial="323622271240",
        camera_model="Intel RealSense D405",
        mount="wrist",
        width=W,
        height=H,
        K=[[640.0, 0, 3.5], [0, 640.0, 2.5], [0, 0, 1]],
        distortion_model="inverse_brown_conrady",
        distortion=[0.0] * 5,
        depth_units_m=1e-4,
        T_tcp_cam=he.T_tcp_cam,
        T_base_head=None,
        source_files={"calibration/handeye_tf.txt": he.sha256},
    )


@pytest.fixture(scope="module")
def reference_handeye_path_module() -> Path:
    return Path(__file__).resolve().parents[2] / "configs/handeye/handeye_tf_20261012.txt"


def header(bundle: CalibrationBundle, frame_id: int = 3) -> Header:
    return Header(
        1, SID, "wrist", frame_id, 123_456_789, ClockDomain.HOST_MONOTONIC, bundle.calibration_id
    )


def observation(bundle: CalibrationBundle) -> ObjectObservation:
    mask = np.zeros((H, W), bool)
    mask[1:5, 2:7] = True
    geom = mask.copy()
    geom[1, 2] = False
    vs, us = np.nonzero(geom)
    uv = np.stack([us, vs], 1).astype(np.int32)
    return ObjectObservation(
        header=header(bundle),
        instance_id=0,
        mask_revision="sam2.1-tiny@3f9a",
        mask_binary=mask,
        mask_prob=mask.astype(np.float32) * 0.9,
        geometry_mask=geom,
        pixel_uv=uv,
        points_cam_m=RNG.random((len(uv), 3)).astype(np.float32),
        normals_cam=None,
        keypoints_uv=np.array([[2.5, 1.5], [3.0, 2.0]], np.float32),
        keypoint_depth_m=np.array([0.2, np.nan], np.float32),
        descriptors=np.zeros((2, 32), np.uint8),
        descriptor_type="orb",
        track_ids=np.array([4, -1], np.int64),
        quality={"mask_area_px": int(mask.sum()), "valid_depth_ratio": 0.9, "n_keypoints": 2},
    )


def quality() -> dict[str, Any]:
    return {
        k: None
        for k in (
            "inliers",
            "overlap",
            "rmse_m",
            "hessian_min_eig",
            "n_hypotheses",
            "correction_from_prediction_m",
            "correction_from_prediction_deg",
        )
    }


def estimate(bundle: CalibrationBundle) -> PoseEstimate:
    T_cam_obj = pose("cam", "obj")
    T_base_cam = pose("base", "cam", (0.5, 0.0, 0.4))
    return PoseEstimate(
        header=header(bundle),
        T_cam_obj=T_cam_obj,
        T_base_obj=T_base_cam @ T_cam_obj,
        T_base_cam_used=T_base_cam,
        status=PoseStatus.VALID,
        track_state=TrackState.TRACKING,
        reason_codes=(ReasonCode.PREDICTION_ONLY,),
        quality=quality(),
        source="test",
    )


def all_instances(bundle: CalibrationBundle) -> list[Any]:
    raw = RNG.integers(0, 4000, (H, W)).astype(np.uint16)
    raw[0, 0] = 0
    timing = FrameTiming(10, 1.5, "global_time", 7, 8, 8000.0, 3, "raw_unmapped")
    sync = PoseSyncInfo(SyncStatus.INTERPOLATED, "decoupled", 1000, 2000, 4, 5)
    T_base_tcp = pose("base", "tcp")
    obs = observation(bundle)
    est = estimate(bundle)
    return [
        bundle,
        header(bundle),
        timing,
        sync,
        RobotSample(
            0,
            99,
            ClockDomain.HOST_MONOTONIC,
            T_base_tcp,
            np.zeros(6),
            np.ones(6) * 0.1,
            "poll",
            1200,
        ),
        SensorFrame(
            header=header(bundle),
            color_rgb_u8=RNG.integers(0, 255, (H, W, 3)).astype(np.uint8),
            depth_raw_u16=raw,
            depth_units_m=1e-4,
            depth_m_f32=depth_raw_to_m(raw, 1e-4),
            valid_depth_mask=raw != 0,
            K=bundle.K,
            T_base_tcp=T_base_tcp,
            T_base_cam_prior=T_base_tcp @ bundle.T_ext,
            pose_sync=sync,
            timing=timing,
            source_meta={"session": SID},
        ),
        obs,
        CorrespondenceSet(
            src_frame_id=1,
            tgt_frame_id=2,
            src_uv=np.zeros((2, 2), np.float32),
            tgt_uv=np.ones((2, 2), np.float32),
            src_point_cam_m=np.full((2, 3), np.nan, np.float32),
            tgt_point_cam_m=np.zeros((2, 3), np.float32),
            sigma_px=np.ones(2, np.float32),
            sigma_m=np.array([np.nan, 0.001], np.float32),
            track_ids=np.array([1, 2], np.int64),
            method="klt",
            inlier=np.array([True, False]),
            rejection_reason=[None, "LOW_DEPTH"],
        ),
        est,
        TrackingReferences(
            prev=est,
            predicted_T_cam_obj=pose("cam", "obj"),
            local_keyframes=(KeyframeView(0, obs, pose("cam", "obj"), 1),),
        ),
        MapView(2, np.zeros((3, 3), np.float32), np.ones((3, 3), np.float32), None, {0: 1, 5: 0}),
        Keyframe.create(
            keyframe_id=0,
            header=header(bundle),
            observation_ref="derived/obs/000003.npz",
            mask_revision="sam2.1-tiny@3f9a",
            raw_refs={"color": ("cameras/wrist/color/000003.png", "a" * 64)},
            T_base_cam_prior=pose("base", "cam"),
            T_cam_obj_provisional=pose("cam", "obj"),
            T_cam_obj_current=pose("cam", "obj"),
            pose_version=0,
            feature_track_ids=np.array([1, 2, 3], np.int64),
            status=KeyframeStatus.PROVISIONAL,
        ),
        PoseUpdate(SID, SID, 0, 3, 0, 1, pose("cam", "obj"), np.eye(6) * 1e-6, "opt-1", 5),
        MapSnapshot(
            SID,
            SID,
            1,
            0,
            PoseSE3.from_rt(np.eye(3), [0.4, 0.1, 0.0], "base", "obj"),
            bundle.calibration_id,
            {0: 1},
            "sam2.1-tiny@3f9a",
            "abc123",
            {"mesh": "m.ply"},
            9,
        ),
    ]


def test_npz_roundtrip_all_types(bundle: CalibrationBundle) -> None:
    for obj in all_instances(bundle):
        back = codec.from_npz_bytes(type(obj), codec.to_npz_bytes(obj))
        assert deep_equal(obj, back), type(obj).__name__


def test_json_roundtrip_json_types(bundle: CalibrationBundle) -> None:
    json_types = (
        CalibrationBundle,
        Header,
        FrameTiming,
        PoseSyncInfo,
        RobotSample,
        PoseEstimate,
        Keyframe,
        PoseUpdate,
        MapSnapshot,
        MapView,
    )
    for obj in all_instances(bundle):
        if isinstance(obj, json_types):
            text = codec.to_json(obj)
            d = json.loads(text)
            assert d["schema_version"] == 1 and d["type"] == type(obj).__name__
            assert list(d) == sorted(d)
            assert deep_equal(obj, codec.from_json(type(obj), text)), type(obj).__name__


def test_json_rejects_nan(bundle: CalibrationBundle) -> None:
    obs = observation(bundle)  # keypoint_depth_m holds NaN -> must go to NPZ
    with pytest.raises(ContractValidationError):
        codec.to_json(obs)


@pytest.mark.parametrize("version", [0, 2, None, "1"])
def test_schema_version_rejected(bundle: CalibrationBundle, version: Any) -> None:
    d = codec.to_json_dict(bundle)
    d["schema_version"] = version
    with pytest.raises(SchemaVersionError):
        codec.from_json_dict(CalibrationBundle, d)


def test_type_tag_and_unknown_fields_rejected(bundle: CalibrationBundle) -> None:
    d = codec.to_json_dict(bundle)
    with pytest.raises(SchemaVersionError):
        codec.from_json_dict(Header, d)
    d["extra_field"] = 1
    with pytest.raises(ContractValidationError):
        codec.from_json_dict(CalibrationBundle, d)


def test_npz_schema_version_rejected(bundle: CalibrationBundle) -> None:
    import io

    obj = all_instances(bundle)[5]  # SensorFrame
    blob = codec.to_npz_bytes(obj)
    with np.load(io.BytesIO(blob)) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(arrays["__meta__"].tobytes())
    meta["schema_version"] = 2
    arrays["__meta__"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    with pytest.raises(SchemaVersionError):
        codec.from_npz_bytes(SensorFrame, buf.getvalue())


def test_arrays_non_writeable(bundle: CalibrationBundle) -> None:
    for obj in all_instances(bundle):
        for name, value in vars(obj).items():
            if isinstance(value, np.ndarray):
                assert not value.flags.writeable, f"{type(obj).__name__}.{name}"
    with pytest.raises(TypeError):
        bundle.source_files["x"] = "y"  # type: ignore[index]


def test_calibration_id_deterministic(
    bundle: CalibrationBundle, reference_handeye_path_module: Path
) -> None:
    assert len(bundle.calibration_id) == 16
    kwargs = {k: getattr(bundle, k) for k in vars(bundle) if k != "calibration_id"}
    assert CalibrationBundle.create(**kwargs).calibration_id == bundle.calibration_id
    kwargs["K"] = np.array([[641.0, 0, 3.5], [0, 640.0, 2.5], [0, 0, 1]])
    assert CalibrationBundle.create(**kwargs).calibration_id != bundle.calibration_id
    d = codec.to_json_dict(bundle)
    d["calibration_id"] = "0" * 16
    with pytest.raises(ContractValidationError):
        codec.from_json_dict(CalibrationBundle, d)
    d = codec.to_json_dict(bundle)
    d["depth_units_m"] = 1e-3  # content changed, id not
    with pytest.raises(ContractValidationError):
        codec.from_json_dict(CalibrationBundle, d)


def test_calibration_mount_extrinsic_exclusive(bundle: CalibrationBundle) -> None:
    kwargs = {k: getattr(bundle, k) for k in vars(bundle) if k != "calibration_id"}
    with pytest.raises(ContractValidationError):
        CalibrationBundle.create(**{**kwargs, "T_base_head": pose("base", "head")})
    with pytest.raises(FrameMismatchError):
        CalibrationBundle.create(**{**kwargs, "T_tcp_cam": pose("cam", "tcp")})
    with pytest.raises(ContractValidationError):
        CalibrationBundle.create(**{**kwargs, "mount": "fixed"})


def test_header_rules(bundle: CalibrationBundle) -> None:
    with pytest.raises(ContractValidationError):
        Header(1, SID, "wrist", 0, 1, ClockDomain.RS_GLOBAL, bundle.calibration_id)
    with pytest.raises(ContractValidationError):
        Header(2, SID, "wrist", 0, 1, ClockDomain.HOST_MONOTONIC, bundle.calibration_id)
    with pytest.raises(ContractValidationError):
        Header(1, "bad", "wrist", 0, 1, ClockDomain.HOST_MONOTONIC, bundle.calibration_id)
    with pytest.raises(ContractValidationError):
        Header(1, SID, "left", 0, 1, ClockDomain.HOST_MONOTONIC, bundle.calibration_id)
    with pytest.raises(ContractValidationError):
        Header(1, SID, "wrist", -1, 1, ClockDomain.HOST_MONOTONIC, bundle.calibration_id)


def test_pose_estimate_invariants(bundle: CalibrationBundle) -> None:
    est = estimate(bundle)
    base = {k: getattr(est, k) for k in vars(est)}
    with pytest.raises(ContractValidationError):  # INVALID needs T_cam_obj None
        PoseEstimate(**{**base, "status": PoseStatus.INVALID})
    with pytest.raises(ContractValidationError):  # None needs INVALID
        PoseEstimate(**{**base, "T_cam_obj": None, "T_base_obj": None})
    PoseEstimate(**{**base, "T_cam_obj": None, "T_base_obj": None, "status": PoseStatus.INVALID})
    with pytest.raises(ContractValidationError):  # chain inconsistency
        PoseEstimate(**{**base, "T_base_obj": pose("base", "obj", (9.0, 0.0, 0.0))})
    with pytest.raises(FrameMismatchError):
        PoseEstimate(**{**base, "T_cam_obj": pose("obj", "cam")})
    with pytest.raises(ContractValidationError):
        PoseEstimate(**{**base, "quality": {"inliers": 3}})


def test_sensor_frame_rules(bundle: CalibrationBundle) -> None:
    sf = all_instances(bundle)[5]
    base = {k: getattr(sf, k) for k in vars(sf)}
    with pytest.raises(ContractValidationError):  # depth scaled elsewhere
        SensorFrame(**{**base, "depth_m_f32": sf.depth_m_f32 * np.float32(1.001)})
    with pytest.raises(ContractValidationError):  # sync failed -> no prior
        SensorFrame(
            **{
                **base,
                "pose_sync": PoseSyncInfo(
                    SyncStatus.GAP_TOO_LARGE, "decoupled", None, None, None, None
                ),
            }
        )
    with pytest.raises(ContractValidationError):
        SensorFrame(**{**base, "color_rgb_u8": sf.color_rgb_u8.astype(np.float32)})


def test_correspondence_and_observation_rules(bundle: CalibrationBundle) -> None:
    cs = all_instances(bundle)[7]
    base = {k: getattr(cs, k) for k in vars(cs)}
    with pytest.raises(ContractValidationError):
        CorrespondenceSet(**{**base, "rejection_reason": [None, None]})
    obs = observation(bundle)
    ob = {k: getattr(obs, k) for k in vars(obs)}
    bad = obs.geometry_mask.copy()
    bad[0, 0] = True
    with pytest.raises(ContractValidationError):
        ObjectObservation(**{**ob, "geometry_mask": bad})
    with pytest.raises(ContractValidationError):
        ObjectObservation(**{**ob, "mask_revision": "no-hash"})


def test_keyframe_update_snapshot_rules(bundle: CalibrationBundle) -> None:
    objs = all_instances(bundle)
    kf = objs[11]
    d = codec.to_json_dict(kf)
    d["pose_version"] = 1
    with pytest.raises(ContractValidationError):  # checksum no longer matches
        codec.from_json_dict(Keyframe, d)
    with pytest.raises(ContractValidationError):
        PoseUpdate(SID, SID, 0, 3, 0, 2, pose("cam", "obj"), None, "opt", 5)
    with pytest.raises(ContractValidationError):
        MapSnapshot(
            SID, SID, 1, 0, pose("base", "obj"), bundle.calibration_id, {}, "a@1234", "h", {}, 1
        )
    with pytest.raises(ContractValidationError):
        MapSnapshot(
            SID,
            SID,
            1,
            1,
            PoseSE3.identity("base", "obj"),
            bundle.calibration_id,
            {},
            "a@1234",
            "h",
            {},
            1,
        )


def test_depth_conversion_exact() -> None:
    raw = np.array([[0, 1, 2, 999], [1234, 30000, 65535, 7]], np.uint16)
    m = depth_raw_to_m(raw, 1e-4)
    assert m.dtype == np.float32
    np.testing.assert_array_equal(m, (raw.astype(np.float64) * 1e-4).astype(np.float32))
    assert m[0, 0] == 0.0 and m[1, 2] == np.float32(6.5535)
    np.testing.assert_array_equal(depth_raw_to_m(raw, 1e-3)[1, 0], np.float32(1.234))
    for units in (0.0, -1e-4, float("nan")):
        with pytest.raises(ContractValidationError):
            depth_raw_to_m(raw, units)
    with pytest.raises(ContractValidationError):
        depth_raw_to_m(raw.astype(np.int32), 1e-4)  # type: ignore[arg-type]


def test_session_id_format() -> None:
    sid = make_session_id("CW20175")
    assert sid.split("_")[1] == "CW20175" and len(sid.split("_")[2]) == 4
    assert make_session_id("my_host.lan").split("_")[1] == "my-host-lan"
