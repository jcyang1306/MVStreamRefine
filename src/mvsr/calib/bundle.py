"""Build `CalibrationBundle`s at record time from SDK intrinsics + the hand-eye file."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mvsr.calib.handeye import REFERENCE_HANDEYE_DATE, HandEye
from mvsr.core.errors import ContractValidationError
from mvsr.core.types import CAMERA_MOUNT, CalibrationBundle

#: Session-relative path of the hand-eye copy (SESSION_FORMAT.md).
HANDEYE_SESSION_PATH = "calibration/handeye_tf.txt"


def intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def build_bundle(
    *,
    camera_id: str,
    camera_serial: str,
    camera_model: str,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion_model: str,
    distortion: Sequence[float],
    depth_units_m: float,
    handeye: HandEye,
) -> CalibrationBundle:
    """Bundle for one stream; extrinsic from `wrist_cam2` (wrist) or `base_cam1` (head)."""
    mount = CAMERA_MOUNT.get(camera_id)
    if mount is None:
        raise ContractValidationError(f"unknown camera_id {camera_id!r}")
    if mount == "wrist":
        extr = {"T_tcp_cam": handeye.T_tcp_cam, "T_base_head": None}
        note = f"T_tcp_cam = wrist_cam2 of the {REFERENCE_HANDEYE_DATE} hand-eye calibration"
    else:
        if handeye.T_base_head is None:
            raise ContractValidationError(
                "hand-eye file has no base_cam1 block for the head camera"
            )
        extr = {"T_tcp_cam": None, "T_base_head": handeye.T_base_head}
        note = (
            f"T_base_head = base_cam1 of the {REFERENCE_HANDEYE_DATE} calibration "
            f"({handeye.source_path}); quality exploratory (ADR-0007 amendment)"
        )
    dist = [float(v) for v in distortion]
    if len(dist) != 5:
        raise ContractValidationError(f"expected 5 distortion coefficients, got {len(dist)}")
    return CalibrationBundle.create(
        camera_id=camera_id,
        camera_serial=str(camera_serial),
        camera_model=str(camera_model),
        mount=mount,
        width=int(width),
        height=int(height),
        K=intrinsic_matrix(float(fx), float(fy), float(cx), float(cy)),
        distortion_model=distortion_model,
        distortion=np.array(dist, dtype=np.float64),
        depth_units_m=float(depth_units_m),
        dt_robot_cam_ns=0,
        source_files={HANDEYE_SESSION_PATH: handeye.sha256},
        notes=note + "; K/distortion from SDK color-stream intrinsics at record time; "
        "dt_robot_cam_ns not yet estimated (P1)",
        **extr,
    )


def with_handeye(bundle: CalibrationBundle, handeye: HandEye) -> CalibrationBundle:
    """Same intrinsics, extrinsic replaced. The raw session file is not touched.

    `calibration_id` changes because it covers the extrinsic. Used to evaluate a new
    hand-eye file on an already recorded session.
    """
    from dataclasses import fields

    if bundle.mount == "wrist":
        extrinsic = handeye.T_tcp_cam
        key = "T_tcp_cam"
    else:
        if handeye.T_base_head is None:
            raise ContractValidationError(
                "hand-eye file has no base_cam1 block for the head camera"
            )
        extrinsic = handeye.T_base_head
        key = "T_base_head"
    kwargs = {f.name: getattr(bundle, f.name) for f in fields(bundle) if f.name != "calibration_id"}
    kwargs[key] = extrinsic
    kwargs["source_files"] = {**dict(bundle.source_files), HANDEYE_SESSION_PATH: handeye.sha256}
    note = f"extrinsic replaced by hand-eye {handeye.sha256[:12]} ({handeye.source_path})"
    kwargs["notes"] = f"{bundle.notes}; {note}" if bundle.notes else note
    return CalibrationBundle.create(**kwargs)
