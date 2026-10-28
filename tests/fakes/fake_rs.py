"""pyrealsense2 test double: just enough API for `mvsr.devices.realsense` control flow.

ADR-0002: used for error paths / control flow only. Pixel content is a seeded pattern and
the timestamps are synthetic counters; nothing here is accuracy or timing evidence.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np


class _Enum:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"<{self.name}>"


def _ns(*names: str) -> SimpleNamespace:
    return SimpleNamespace(**{n: _Enum(n) for n in names})


@dataclass
class DeviceSpec:
    serial: str
    name: str = "Intel RealSense D405"
    depth_scale: float = 1e-4
    period_s: float = 0.005
    disconnect_after: int | None = None
    incomplete_every: int | None = None
    no_frames: bool = False
    starts: list[float] = field(default_factory=list)


class FakeFrame:
    def __init__(self, data: np.ndarray, number: int, ts_ms: float, rs: FakeRS) -> None:
        self._data = data
        self._number = number
        self._ts = ts_ms
        self._rs = rs

    def __bool__(self) -> bool:
        return True

    def get_data(self) -> np.ndarray:
        return self._data

    def get_timestamp(self) -> float:
        return self._ts

    def get_frame_timestamp_domain(self) -> _Enum:
        return self._rs.timestamp_domain.global_time

    def get_frame_number(self) -> int:
        return self._number

    def supports_frame_metadata(self, key: _Enum) -> bool:
        return key.name in (
            "sensor_timestamp",
            "frame_timestamp",
            "actual_exposure",
            "time_of_arrival",
        )

    def get_frame_metadata(self, key: _Enum) -> int:
        counter_us = int(self._ts * 1000) % (1 << 32)  # uint32 device counter
        return {
            "sensor_timestamp": (counter_us - 5000) % (1 << 32),
            "frame_timestamp": counter_us,
            "actual_exposure": 8000,
            "time_of_arrival": int(self._ts) + 3,
        }[key.name]


class FakeFrameset:
    def __init__(self, color: FakeFrame | None, depth: FakeFrame | None) -> None:
        self._color, self._depth = color, depth

    def get_color_frame(self) -> FakeFrame | None:
        return self._color

    def get_depth_frame(self) -> FakeFrame | None:
        return self._depth

    def keep(self) -> None:
        pass


class FakeSensor:
    def __init__(self, name: str, spec: DeviceSpec, rs: FakeRS) -> None:
        self.name, self.spec, self.rs = name, spec, rs
        self.options: dict[str, float] = {"global_time_enabled": 0.0, "enable_auto_exposure": 1.0}
        if "Stereo" in name:
            self.options["visual_preset"] = 0.0

    def get_info(self, key: _Enum) -> str:
        return self.name

    def supports(self, opt: _Enum) -> bool:
        return opt.name in self.options

    def set_option(self, opt: _Enum, value: float) -> None:
        self.options[opt.name] = float(value)

    def get_option(self, opt: _Enum) -> float:
        return self.options[opt.name]

    def get_supported_options(self) -> list[_Enum]:
        return [getattr(self.rs.option, k) for k in self.options]

    def get_option_value_description(self, opt: _Enum, value: float) -> str:
        return "Custom" if opt.name == "visual_preset" else ""

    def get_depth_scale(self) -> float:
        return self.spec.depth_scale


class FakeDevice:
    def __init__(self, spec: DeviceSpec, rs: FakeRS) -> None:
        self.spec = spec
        self.sensors = [FakeSensor("Stereo Module", spec, rs), FakeSensor("RGB Camera", spec, rs)]
        self.rs = rs

    def get_info(self, key: _Enum) -> str:
        return {
            "serial_number": self.spec.serial,
            "name": self.spec.name,
            "firmware_version": "5.16.0.1",
            "product_line": "D400",
        }.get(key.name, "n/a")

    def supports(self, key: _Enum) -> bool:
        return True

    def query_sensors(self) -> list[FakeSensor]:
        return self.sensors

    def first_depth_sensor(self) -> FakeSensor:
        return self.sensors[0]


class _Intr:
    def __init__(self, w: int, h: int) -> None:
        self.width, self.height = w, h
        self.fx, self.fy, self.ppx, self.ppy = 0.9 * w, 0.9 * w, w / 2 - 0.5, h / 2 - 0.5
        self.model = _Enum("inverse_brown_conrady")
        self.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]


class _VideoProfile:
    def __init__(self, w: int, h: int, fps: int) -> None:
        self._w, self._h, self._fps = w, h, fps

    def as_video_stream_profile(self) -> _VideoProfile:
        return self

    def get_intrinsics(self) -> _Intr:
        return _Intr(self._w, self._h)

    def fps(self) -> int:
        return self._fps

    def get_extrinsics_to(self, other: Any) -> SimpleNamespace:
        return SimpleNamespace(rotation=[1, 0, 0, 0, 1, 0, 0, 0, 1], translation=[0, 0, 0])


class _Profile:
    def __init__(self, device: FakeDevice, streams: dict[str, tuple[int, int, int]]) -> None:
        self.device, self.streams = device, streams

    def get_device(self) -> FakeDevice:
        return self.device

    def get_stream(self, stream: _Enum) -> _VideoProfile:
        return _VideoProfile(*self.streams[stream.name])


class FakeConfig:
    def __init__(self) -> None:
        self.serial: str | None = None
        self.streams: dict[str, tuple[int, int, int]] = {}

    def enable_device(self, serial: str) -> None:
        self.serial = serial

    def enable_stream(self, stream: _Enum, w: int, h: int, fmt: _Enum, fps: int) -> None:
        self.streams[stream.name] = (w, h, fps)


class FakePipeline:
    def __init__(self, rs: FakeRS) -> None:
        self.rs = rs
        self.spec: DeviceSpec | None = None
        self.count = 0
        self.size = (0, 0)
        self._t0 = 0.0

    def start(self, cfg: FakeConfig) -> _Profile:
        spec = self.rs.specs[cfg.serial]
        spec.starts.append(time.monotonic())
        self.spec = spec
        w, h, _ = cfg.streams["color"]
        self.size = (w, h)
        self._t0 = time.monotonic()
        with self.rs.lock:
            self.rs.start_order.append(spec.serial)
        return _Profile(FakeDevice(spec, self.rs), cfg.streams)

    def try_wait_for_frames(self, timeout_ms: int) -> tuple[bool, FakeFrameset | None]:
        spec = self.spec
        assert spec is not None
        if spec.disconnect_after is not None and self.count >= spec.disconnect_after:
            raise RuntimeError("No device connected")
        if spec.no_frames:
            time.sleep(min(timeout_ms / 1000.0, 0.05))
            return False, None
        time.sleep(spec.period_s)
        n = self.count
        self.count += 1
        w, h = self.size
        rng = np.random.default_rng(n)
        color = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        depth = rng.integers(0, 2000, size=(h, w), dtype=np.uint16)
        depth[0, :] = 0
        ts = time.time() * 1000.0  # "global_time" = CLOCK_REALTIME ms, like librealsense
        c = FakeFrame(color, n, ts, self.rs)
        d = FakeFrame(depth, n, ts, self.rs)
        if spec.incomplete_every and n % spec.incomplete_every == spec.incomplete_every - 1:
            d = None  # type: ignore[assignment]
        return True, FakeFrameset(c, d)

    def stop(self) -> None:
        pass


class FakeAlign:
    def __init__(self, stream: _Enum) -> None:
        pass

    def process(self, frameset: FakeFrameset) -> FakeFrameset:
        return frameset


class FakeRS:
    """Module-like object; pass as `rs_module=` to the recorder / `rs=` to the stream."""

    def __init__(self, *specs: DeviceSpec) -> None:
        self.specs = {s.serial: s for s in specs}
        self.lock = threading.Lock()
        self.start_order: list[str] = []
        self.camera_info = _ns(
            "serial_number",
            "name",
            "firmware_version",
            "product_line",
            "usb_type_descriptor",
            "physical_port",
            "product_id",
        )
        self.option = _ns(
            "global_time_enabled",
            "enable_auto_exposure",
            "visual_preset",
            "emitter_enabled",
            "laser_power",
        )
        self.stream = _ns("color", "depth")
        self.format = _ns("bgr8", "rgb8", "z16")
        self.frame_metadata_value = _ns(
            "sensor_timestamp",
            "frame_timestamp",
            "actual_exposure",
            "frame_counter",
            "backend_timestamp",
            "time_of_arrival",
        )
        self.timestamp_domain = _ns("global_time", "hardware_clock", "system_time")

    def context(self) -> SimpleNamespace:
        return SimpleNamespace(
            query_devices=lambda: [FakeDevice(s, self) for s in self.specs.values()]
        )

    def config(self) -> FakeConfig:
        return FakeConfig()

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self)

    def align(self, stream: _Enum) -> FakeAlign:
        return FakeAlign(stream)
