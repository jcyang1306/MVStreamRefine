"""Intel RealSense RGB-D stream (P0 F4): D405 wrist / D435 head.

Ported from V1 `object_reconstruction/data/realsense_source.py` (no import of V1). Changes:
- raw clocks of every frame are captured: host arrival (`CLOCK_MONOTONIC`, taken right after
  `wait_for_frames` returns), `get_timestamp()` + its domain, metadata `SENSOR_TIMESTAMP`,
  `FRAME_TIMESTAMP`, `ACTUAL_EXPOSURE`, frame numbers; unsupported values are `None`;
- `global_time_enabled` is switched on for every sensor that supports it;
- no device post-processing filters are created; device options are dumped to the session;
- the device serial and depth unit are verified against the configuration (`DeviceError`);
- waiting (`wait`) and alignment (`align_and_copy`) are separate so the arrival stamp is not
  delayed by alignment; the recorder runs them on different threads.
"""

from __future__ import annotations

import importlib
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mvsr.core.errors import DeviceError
from mvsr.io.records import RawTiming

log = logging.getLogger(__name__)

#: Relative tolerance between the SDK depth scale (float32) and the configured unit.
DEPTH_UNITS_RTOL = 1e-6


@dataclass
class Arrival:
    """A frameset as delivered by the SDK (kept alive until `align_and_copy`)."""

    t_host_arrival_ns: int
    frameset: Any
    timing: RawTiming
    depth_timing: RawTiming | None
    rs_extra: dict[str, Any]
    complete: bool
    detail: str | None = None


@dataclass
class StreamInfo:
    serial: str
    model: str
    firmware: str | None
    width: int
    height: int
    fps: int
    color_format: str
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str
    distortion: list[float]
    depth_units_m_sdk: float
    device: dict[str, Any] = field(default_factory=dict)
    profiles: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    global_time_enabled: dict[str, bool] = field(default_factory=dict)

    def realsense_json(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "profiles": self.profiles,
            "options": self.options,
            "global_time_enabled": self.global_time_enabled,
            "depth_units_m_sdk": self.depth_units_m_sdk,
            "post_processing_filters": [],
            "aligned_to": "color",
        }


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    return str(name) if name else str(value).split(".")[-1]


def _intr_dict(intr: Any) -> dict[str, Any]:
    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "ppx": float(intr.ppx),
        "ppy": float(intr.ppy),
        "model": _enum_name(intr.model),
        "coeffs": [float(c) for c in intr.coeffs],
    }


class RealSenseStream:
    def __init__(
        self,
        camera_id: str,
        serial: str,
        width: int,
        height: int,
        fps: int,
        color_format: str,
        expected_depth_units_m: float,
        *,
        frame_timeout_ms: int = 1000,
        rs: Any = None,
    ) -> None:
        if color_format not in ("bgr8", "rgb8"):
            raise ValueError(f"color_format must be bgr8 or rgb8, got {color_format!r}")
        self.camera_id = camera_id
        self.serial = str(serial)
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.color_format = color_format
        self.expected_depth_units_m = float(expected_depth_units_m)
        self.frame_timeout_ms = int(frame_timeout_ms)
        self._rs = rs
        self._pipeline: Any = None
        self._align: Any = None
        self.info: StreamInfo | None = None

    # ------------------------------------------------------------------ lifecycle
    def _module(self) -> Any:
        if self._rs is None:
            try:
                self._rs = importlib.import_module("pyrealsense2")
            except ImportError as exc:
                raise DeviceError("pyrealsense2 not installed (use the `rig` image)") from exc
        return self._rs

    def _find_device(self, rs: Any) -> Any:
        ctx = rs.context()
        found = []
        for dev in ctx.query_devices():
            s = dev.get_info(rs.camera_info.serial_number)
            found.append(s)
            if s == self.serial:
                return dev
        raise DeviceError(
            f"{self.camera_id}: RealSense {self.serial} not found (connected: {found})"
        )

    def start(self) -> StreamInfo:
        """Open the device, verify serial + depth unit, start color+depth; returns metadata."""
        rs = self._module()
        dev = self._find_device(rs)
        global_time: dict[str, bool] = {}
        for sensor in dev.query_sensors():
            name = sensor.get_info(rs.camera_info.name)
            if sensor.supports(rs.option.global_time_enabled):
                sensor.set_option(rs.option.global_time_enabled, 1.0)
                global_time[name] = bool(sensor.get_option(rs.option.global_time_enabled))
            else:
                global_time[name] = False
        cfg = rs.config()
        cfg.enable_device(self.serial)
        fmt = rs.format.bgr8 if self.color_format == "bgr8" else rs.format.rgb8
        cfg.enable_stream(rs.stream.color, self.width, self.height, fmt, self.fps)
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        pipeline = rs.pipeline()
        try:
            profile = pipeline.start(cfg)
        except RuntimeError as exc:
            raise DeviceError(f"{self.camera_id}: pipeline start failed: {exc}") from exc
        try:
            info = self._describe(rs, profile, global_time)
        except Exception:
            pipeline.stop()
            raise
        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color)
        self.info = info
        log.info(
            "%s: RealSense %s %s fw %s %dx%d@%d depth_units=%g",
            self.camera_id,
            info.model,
            info.serial,
            info.firmware,
            info.width,
            info.height,
            info.fps,
            info.depth_units_m_sdk,
        )
        return info

    def _describe(self, rs: Any, profile: Any, global_time: dict[str, bool]) -> StreamInfo:
        device = profile.get_device()
        serial = device.get_info(rs.camera_info.serial_number)
        if serial != self.serial:
            raise DeviceError(f"{self.camera_id}: opened serial {serial}, expected {self.serial}")
        dev_info: dict[str, Any] = {}
        for key in (
            "name",
            "serial_number",
            "firmware_version",
            "product_line",
            "usb_type_descriptor",
            "physical_port",
            "product_id",
        ):
            ci = getattr(rs.camera_info, key, None)
            if ci is not None and device.supports(ci):
                dev_info[key] = device.get_info(ci)
        depth_sensor = device.first_depth_sensor()
        scale = float(depth_sensor.get_depth_scale())
        expected = self.expected_depth_units_m
        if not math.isfinite(scale) or abs(scale - expected) > DEPTH_UNITS_RTOL * expected:
            raise DeviceError(
                f"{self.camera_id}: device depth unit {scale!r} m != configured {expected!r} m "
                "(check the depth_units option; recording refused)"
            )
        color_vp = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_vp = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        ci, di = color_vp.get_intrinsics(), depth_vp.get_intrinsics()
        if (int(ci.width), int(ci.height)) != (self.width, self.height):
            raise DeviceError(f"{self.camera_id}: color stream is {ci.width}x{ci.height}")
        extr = depth_vp.get_extrinsics_to(color_vp)
        profiles = {
            "color": {**_intr_dict(ci), "format": self.color_format, "fps": int(color_vp.fps())},
            "depth": {**_intr_dict(di), "format": "z16", "fps": int(depth_vp.fps())},
            "depth_to_color": {
                "rotation_col_major": [float(v) for v in extr.rotation],
                "translation_m": [float(v) for v in extr.translation],
            },
        }
        options: dict[str, Any] = {}
        for sensor in device.query_sensors():
            name = sensor.get_info(rs.camera_info.name)
            values: dict[str, Any] = {}
            for opt in sensor.get_supported_options():
                try:
                    v = float(sensor.get_option(opt))
                    entry: dict[str, Any] = {"value": v}
                    desc = sensor.get_option_value_description(opt, v)
                    if desc:
                        entry["description"] = desc
                    values[_enum_name(opt)] = entry
                except RuntimeError:
                    continue
            options[name] = values
        coeffs = [float(c) for c in ci.coeffs]
        return StreamInfo(
            serial=serial,
            model=str(dev_info.get("name", "unknown")),
            firmware=dev_info.get("firmware_version"),
            width=int(ci.width),
            height=int(ci.height),
            fps=int(color_vp.fps()),
            color_format=self.color_format,
            fx=float(ci.fx),
            fy=float(ci.fy),
            cx=float(ci.ppx),
            cy=float(ci.ppy),
            distortion_model=_enum_name(ci.model),
            distortion=coeffs,
            depth_units_m_sdk=scale,
            device=dev_info,
            profiles=profiles,
            options=options,
            global_time_enabled=global_time,
        )

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except RuntimeError:
                log.warning("%s: pipeline stop failed", self.camera_id, exc_info=True)
            self._pipeline = None

    # ------------------------------------------------------------------ frames
    def wait(self, timeout_ms: int | None = None) -> Arrival:
        """Block for the next frameset. Raises `TimeoutError` on timeout, `DeviceError` on failure."""
        if self._pipeline is None:
            raise DeviceError(f"{self.camera_id}: stream not started")
        rs = self._rs
        try:
            ok, frames = self._pipeline.try_wait_for_frames(timeout_ms or self.frame_timeout_ms)
        except RuntimeError as exc:
            raise DeviceError(f"{self.camera_id}: wait_for_frames failed: {exc}") from exc
        t_arrival = time.monotonic_ns()
        if not ok:
            raise TimeoutError(
                f"{self.camera_id}: no frame within {timeout_ms or self.frame_timeout_ms} ms"
            )
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        complete = bool(color) and bool(depth)
        timing = _raw_timing(rs, color) if color else RawTiming()
        depth_timing = _raw_timing(rs, depth) if depth else None
        extra: dict[str, Any] = {}
        for prefix, fr in (("color", color), ("depth", depth)):
            if fr:
                for key in ("backend_timestamp", "time_of_arrival"):
                    v = _metadata(rs, fr, key)
                    if v is not None:
                        extra[f"{prefix}_{key}_ms"] = v
        if complete:
            frames.keep()
        detail = None if complete else f"color={bool(color)} depth={bool(depth)}"
        return Arrival(
            t_arrival, frames if complete else None, timing, depth_timing, extra, complete, detail
        )

    def align_and_copy(self, arrival: Arrival) -> tuple[np.ndarray, np.ndarray]:
        """Align depth to color and copy both out of SDK memory -> (RGB uint8, raw z16 uint16)."""
        if not arrival.complete or arrival.frameset is None:
            raise DeviceError(f"{self.camera_id}: cannot align an incomplete frameset")
        aligned = self._align.process(arrival.frameset)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        if not color or not depth:
            raise DeviceError(f"{self.camera_id}: alignment produced an incomplete frameset")
        c = np.asanyarray(color.get_data())
        rgb = np.ascontiguousarray(c[:, :, ::-1] if self.color_format == "bgr8" else c[:, :, :3])
        d = np.array(np.asanyarray(depth.get_data()), dtype=np.uint16, copy=True)
        if rgb.shape[:2] != d.shape or rgb.shape[:2] != (self.height, self.width):
            raise DeviceError(f"{self.camera_id}: unexpected frame shapes {rgb.shape} / {d.shape}")
        arrival.frameset = None
        return np.array(rgb, dtype=np.uint8, copy=True), d


def _metadata(rs: Any, frame: Any, key: str) -> int | None:
    md = getattr(rs.frame_metadata_value, key, None)
    if md is None or not frame.supports_frame_metadata(md):
        return None
    return int(frame.get_frame_metadata(md))


def _raw_timing(rs: Any, frame: Any) -> RawTiming:
    ts = float(frame.get_timestamp())
    return RawTiming(
        rs_timestamp_ms=ts if math.isfinite(ts) else None,
        rs_timestamp_domain=_enum_name(frame.get_frame_timestamp_domain()),
        rs_sensor_timestamp_us=_metadata(rs, frame, "sensor_timestamp"),
        rs_frame_timestamp_us=_metadata(rs, frame, "frame_timestamp"),
        exposure_us=(None if (e := _metadata(rs, frame, "actual_exposure")) is None else float(e)),
        rs_frame_number=int(frame.get_frame_number()),
    )
