"""Online `SensorFrame`s from the rig (P1 S1; consumed by P9). Reads state only, never moves the robot.

Acquisition reuses the P0 adapters (`RealSenseStream`, `RM65StateSource`) and the recorder
configuration (`configs/record.yaml`: serials, resolution, robot endpoint, hand-eye). After
acquisition the frame goes through the same `build_sensor_frame` as replay:
- timestamp: RealSense global time -> HOST_MONOTONIC via a realtime/monotonic pair taken at
  start and one taken per frame (`frame_timestamp_ns`); without global time the declared
  arrival fallback is used;
- robot: successful reads go into a bounded buffer (`live.robot_buffer_s`); before pairing,
  the source waits up to `live.robot_wait_ms` for a sample at or after the frame time so the
  interpolation has a bracket; otherwise the frame is emitted with its failing `SyncStatus`.
Incomplete framesets get a frame_id and are counted in `source_meta.not_written_before`;
framesets whose colour and depth come from different exposures (`OnlinePairCheck`) are
counted in `rgbd_rejected_before` and not emitted when `rgbd.skip_mismatched`.
"""

from __future__ import annotations

import socket
import threading
import time
from collections import deque
from typing import Any

from mvsr.calib.bundle import build_bundle
from mvsr.calib.handeye import load_handeye
from mvsr.core.errors import DeviceError
from mvsr.core.types import (
    CalibrationBundle,
    FrameTiming,
    RobotSample,
    SensorFrame,
    make_session_id,
)
from mvsr.devices.realsense import RealSenseStream
from mvsr.devices.rm65 import RM65StateSource
from mvsr.io.records import RobotRecord
from mvsr.io.session_reader import robot_sample_from_record
from mvsr.record.config import RecordConfig, load_record_config
from mvsr.record.recorder import clock_pair
from mvsr.sensors.clock import RealtimeMap, frame_timestamp_ns
from mvsr.sensors.config import SensorsConfig, load_sensors_config
from mvsr.sensors.frame import build_sensor_frame
from mvsr.sensors.rgbd import REJECTED, OnlinePairCheck
from mvsr.sensors.source import EndOfStream
from mvsr.sensors.sync import RobotTrack


class LiveSource:
    def __init__(
        self,
        record_config: RecordConfig | None = None,
        config: SensorsConfig | None = None,
        *,
        camera_id: str = "wrist",
        session_id: str | None = None,
        rs_module: Any = None,
        rm_sdk: Any = None,
    ) -> None:
        self.record_config = record_config or load_record_config()
        self.config = config or load_sensors_config()
        if camera_id not in self.record_config.cameras:
            raise ValueError(
                f"camera {camera_id!r} not configured in {self.record_config.source_path}"
            )
        self.camera_id = camera_id
        self.session_id = session_id or make_session_id(socket.gethostname())
        self._rs = rs_module
        self._rm = rm_sdk
        rc = self.record_config.robot
        maxlen = max(8, int(self.config.live.robot_buffer_s * rc.poll_hz * 2))
        self._samples: deque[RobotSample] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._stream: RealSenseStream | None = None
        self._robot: RM65StateSource | None = None
        self._clock_start: dict[str, int] | None = None
        self.bundle: CalibrationBundle | None = None
        self._next_id = 0
        self._emitted = 0
        self._not_written = 0
        self._rejected = 0
        self._skipped_since_emit = 0
        self._rejected_since_emit = 0
        self._pairs = OnlinePairCheck(
            self.config.rgbd.limit_ms(camera_id), self.config.clock.device_counter_period_us
        )
        self._closed = False

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> LiveSource:
        rc, lv = self.record_config, self.config.live
        cam = rc.cameras[self.camera_id]
        handeye = load_handeye(rc.handeye_path(), rc.handeye.sha256)
        self._robot = RM65StateSource(
            rc.robot.ip,
            rc.robot.port,
            self._on_robot,
            source=rc.robot.source,
            poll_hz=rc.robot.poll_hz,
            udp_port=rc.robot.udp_port,
            udp_cycle_5ms_units=rc.robot.udp_cycle_5ms_units,
            udp_host_ip=rc.robot.udp_host_ip,
            poll_retries=rc.robot.poll_retries,
            poll_retry_delay_s=rc.robot.poll_retry_delay_s,
            sdk=self._rm,
        )
        try:
            self._robot.connect()
            self._robot.start()
            self._stream = RealSenseStream(
                self.camera_id,
                cam.serial,
                cam.width,
                cam.height,
                cam.fps,
                cam.color_format,
                cam.depth_units_m,
                frame_timeout_ms=lv.frame_timeout_ms,
                rs=self._rs,
            )
            info = self._stream.start()
            self.bundle = build_bundle(
                camera_id=self.camera_id,
                camera_serial=info.serial,
                camera_model=info.model,
                width=info.width,
                height=info.height,
                fx=info.fx,
                fy=info.fy,
                cx=info.cx,
                cy=info.cy,
                distortion_model=info.distortion_model,
                distortion=info.distortion,
                depth_units_m=cam.depth_units_m,
                handeye=handeye,
            )
            self._clock_start = clock_pair()
            for i in range(lv.warmup_frames):
                self._wait(lv.first_frame_timeout_ms if i == 0 else lv.frame_timeout_ms)
        except BaseException:
            self.close()
            raise
        return self

    def __enter__(self) -> LiveSource:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self._closed = True
        if self._stream is not None:
            self._stream.stop()
            self._stream = None
        if self._robot is not None:
            self._robot.close()
            self._robot = None

    # ------------------------------------------------------------------ robot
    def _on_robot(self, rec: RobotRecord) -> None:
        sample = robot_sample_from_record(rec)
        if sample is not None:
            with self._lock:
                self._samples.append(sample)

    def _wait_for_robot(self, t_robot_ns: int) -> None:
        deadline = time.monotonic() + self.config.live.robot_wait_ms / 1e3
        while True:
            with self._lock:
                latest = self._samples[-1].t_host_ns if self._samples else None
            if latest is not None and latest >= t_robot_ns:
                return
            if time.monotonic() >= deadline:
                return
            time.sleep(0.002)

    # ------------------------------------------------------------------ frames
    def _wait(self, timeout_ms: int) -> Any:
        assert self._stream is not None
        timeouts = 0
        while True:
            try:
                return self._stream.wait(timeout_ms)
            except TimeoutError as exc:
                timeouts += 1
                if timeouts >= self.config.live.max_consecutive_timeouts:
                    raise DeviceError(
                        f"{self.camera_id}: {timeouts} consecutive timeouts ({exc})"
                    ) from exc

    def next(self) -> SensorFrame | EndOfStream:
        if self._closed or self._stream is None or self.bundle is None:
            return EndOfStream(
                self.session_id,
                self.camera_id,
                self._emitted,
                self._not_written,
                self._rejected,
                "closed",
            )
        while True:
            arrival = self._wait(self.config.live.frame_timeout_ms)
            fid = self._next_id
            self._next_id += 1
            if not arrival.complete:
                self._not_written += 1
                self._skipped_since_emit += 1
                continue
            pair = self._pairs.check(arrival.timing, arrival.depth_timing)
            if pair.status in REJECTED and self.config.rgbd.skip_mismatched:
                arrival.frameset = None
                self._rejected += 1
                self._rejected_since_emit += 1
                continue
            break
        rgb, depth = self._stream.align_and_copy(arrival)
        assert self._clock_start is not None
        rt_map = RealtimeMap.from_pairs(self._clock_start, clock_pair())
        ts, method = frame_timestamp_ns(
            arrival.timing, arrival.t_host_arrival_ns, rt_map, self.config.clock
        )
        self._wait_for_robot(ts + self.bundle.dt_robot_cam_ns)
        with self._lock:
            track = RobotTrack.from_samples(list(self._samples))
        t = arrival.timing
        timing = FrameTiming(
            t_host_arrival_ns=arrival.t_host_arrival_ns,
            rs_timestamp_ms=t.rs_timestamp_ms,
            rs_timestamp_domain=t.rs_timestamp_domain,
            rs_sensor_timestamp_us=t.rs_sensor_timestamp_us,
            rs_frame_timestamp_us=t.rs_frame_timestamp_us,
            exposure_us=t.exposure_us,
            rs_frame_number=t.rs_frame_number,
            method=method,
        )
        frame = build_sensor_frame(
            session_id=self.session_id,
            bundle=self.bundle,
            frame_id=fid,
            timestamp_ns=ts,
            timing=timing,
            color_rgb_u8=rgb,
            depth_raw_u16=depth,
            robot=track,
            max_interp_gap_ns=self.config.sync.max_interp_gap_ns,
            source_meta={
                "source": "live",
                "session_id": self.session_id,
                "not_written_before": self._skipped_since_emit,
                "rgbd_rejected_before": self._rejected_since_emit,
                "rgbd": pair.status.value,
                "color_depth_dt_ms": None if pair.dt_us is None else pair.dt_us / 1e3,
            },
        )
        self._emitted += 1
        self._skipped_since_emit = 0
        self._rejected_since_emit = 0
        return frame
