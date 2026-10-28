"""`SensorFrame` assembly shared by the replay and live sources (P1 S1/S5).

After acquisition both sources call `build_sensor_frame`: header at the mapped mid-exposure
time, robot pose at `timestamp_ns + bundle.dt_robot_cam_ns`, and the camera prior
`T_base_cam_prior`:
- wrist: `T_base_tcp @ T_tcp_cam`;
- head (fixed): `T_base_head`. DATA_CONTRACTS.md allows a prior only with an INTERPOLATED
  sync, so a head frame without a robot bracket also has no prior (head is not used by V2
  algorithms, ADR-0007).
Sync failures are still emitted with `T_base_cam_prior=None`; sources never skip silently.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from mvsr.core.codec import SCHEMA_VERSION
from mvsr.core.depth import depth_raw_to_m
from mvsr.core.se3 import PoseSE3
from mvsr.core.types import (
    CalibrationBundle,
    ClockDomain,
    FrameTiming,
    Header,
    PoseSyncInfo,
    SensorFrame,
    SyncStatus,
)
from mvsr.sensors.sync import RobotTrack


def camera_prior(
    bundle: CalibrationBundle, T_base_tcp: PoseSE3 | None, sync: PoseSyncInfo
) -> PoseSE3 | None:
    if sync.status is not SyncStatus.INTERPOLATED:
        return None
    if bundle.mount == "wrist":
        assert bundle.T_tcp_cam is not None and T_base_tcp is not None
        return T_base_tcp @ bundle.T_tcp_cam
    return bundle.T_base_head


def build_sensor_frame(
    *,
    session_id: str,
    bundle: CalibrationBundle,
    frame_id: int,
    timestamp_ns: int,
    timing: FrameTiming,
    color_rgb_u8: npt.NDArray[np.uint8],
    depth_raw_u16: npt.NDArray[np.uint16],
    robot: RobotTrack,
    max_interp_gap_ns: int,
    depth_m_f32: npt.NDArray[np.float32] | None = None,
    source_meta: Mapping[str, Any] | None = None,
) -> SensorFrame:
    """One contract `SensorFrame`; `depth_m_f32` is reused when the reader already converted it."""
    T_base_tcp, sync = robot.pose_at(timestamp_ns, max_interp_gap_ns, bundle.dt_robot_cam_ns)
    depth_m = (
        depth_m_f32
        if depth_m_f32 is not None
        else depth_raw_to_m(depth_raw_u16, bundle.depth_units_m)
    )
    header = Header(
        schema_version=SCHEMA_VERSION,
        session_id=session_id,
        camera_id=bundle.camera_id,
        frame_id=frame_id,
        timestamp_ns=int(timestamp_ns),
        clock_domain=ClockDomain.HOST_MONOTONIC,
        calibration_id=bundle.calibration_id,
    )
    return SensorFrame(
        header=header,
        color_rgb_u8=color_rgb_u8,
        depth_raw_u16=depth_raw_u16,
        depth_units_m=bundle.depth_units_m,
        depth_m_f32=depth_m,
        valid_depth_mask=depth_raw_u16 != 0,
        K=bundle.K,
        T_base_tcp=T_base_tcp,
        T_base_cam_prior=camera_prior(bundle, T_base_tcp, sync),
        pose_sync=sync,
        timing=timing,
        source_meta=dict(source_meta or {}),
    )
