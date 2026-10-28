"""Deterministic replay of a recorded session to `SensorFrame`s (P1 S5).

Opening verifies the manifest-listed file checksums; each frame read verifies its image
checksums. Timestamps come from `build_stream_clock` over the whole stream (all records,
written or dropped), robot priors from `RobotTrack` over all successful robot reads, so the
output depends only on the session and `configs/sensors.yaml` and is identical across runs.

Emitted frames: WRITTEN records whose colour/depth pair passed `rgbd.check_pairs` (with
`rgbd.skip_mismatched`). Every other record is accounted for, never skipped silently:
`source_meta.not_written_before` / `rgbd_rejected_before` count the records since the
previous emitted frame, `EndOfStream` carries the totals. Explicit `frame_ids` are emitted
as requested (they must be WRITTEN), with their pairing status in `source_meta.rgbd`.
`speed=None` replays as fast as possible; `speed=s` paces emission at `s` x recorded time.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from mvsr.calib.bundle import with_handeye
from mvsr.calib.handeye import HandEye
from mvsr.core.errors import SessionFormatError
from mvsr.core.types import FrameStatus, SensorFrame, SessionStatus
from mvsr.io.session_reader import SessionReader
from mvsr.sensors.clock import RealtimeMap, StreamClock, build_stream_clock
from mvsr.sensors.config import SensorsConfig, load_sensors_config
from mvsr.sensors.frame import build_sensor_frame
from mvsr.sensors.rgbd import REJECTED, RgbdCheck, check_pairs
from mvsr.sensors.source import EndOfStream
from mvsr.sensors.sync import RobotTrack


class ReplaySource:
    def __init__(
        self,
        session_dir: str | Path,
        camera_id: str = "wrist",
        *,
        config: SensorsConfig | None = None,
        speed: float | None = None,
        frame_ids: Sequence[int] | None = None,
        verify_images: bool = True,
        allow_incomplete: bool = False,
        handeye: HandEye | None = None,
    ) -> None:
        if speed is not None and speed <= 0:
            raise ValueError("speed must be > 0 or None")
        self.config = config or load_sensors_config()
        self.reader = SessionReader(session_dir)
        status = self.reader.status
        if status != SessionStatus.COMPLETE.value and not allow_incomplete:
            raise SessionFormatError(f"session status is {status}, not COMPLETE")
        if camera_id not in self.reader.camera_ids:
            raise SessionFormatError(f"session has no camera {camera_id!r}")
        self.camera_id = camera_id
        self.session_id = self.reader.session_id
        self.handeye = handeye
        self.bundle = self.reader.calibration(camera_id)
        if handeye is not None:
            self.bundle = with_handeye(self.bundle, handeye)
        self.records = self.reader.frame_records(camera_id)
        self.rt_map = RealtimeMap.from_manifest(self.reader.manifest.get("clock"))
        cc = self.config.clock
        self.clock: StreamClock = build_stream_clock(camera_id, self.records, self.rt_map, cc)
        self.rgbd, self.rgbd_report = check_pairs(
            self.records, self.config.rgbd.limit_ms(camera_id), cc.device_counter_period_us
        )
        self.robot = RobotTrack.from_samples(self.reader.robot_samples())
        written = [r.frame_id for r in self.records if r.status is FrameStatus.WRITTEN]
        if frame_ids is not None:
            allowed = set(written)
            bad = [f for f in frame_ids if f not in allowed]
            if bad:
                raise SessionFormatError(f"{camera_id}: frame_ids not WRITTEN: {bad[:10]}")
            selected = sorted(set(frame_ids))
        elif self.config.rgbd.skip_mismatched:
            selected = [f for f in written if self.rgbd[f].status not in REJECTED]
        else:
            selected = written
        self.frame_ids: tuple[int, ...] = tuple(selected)
        self._before = self._skipped_before(self.frame_ids)
        self.speed = speed
        self.verify_images = verify_images
        self._pos = 0
        self._pace0: tuple[float, int] | None = None

    def _rejected(self, fid: int) -> bool:
        return self.rgbd[fid].status in REJECTED

    def _skipped_before(self, ids: Sequence[int]) -> dict[int, tuple[int, int]]:
        """frame_id -> (not written, written but RGB-D rejected) since the previous emitted id."""
        status = {r.frame_id: r.status for r in self.records}
        out: dict[int, tuple[int, int]] = {}
        prev = -1
        for fid in ids:
            gap = range(prev + 1, fid)
            not_written = sum(1 for k in gap if status.get(k) is not FrameStatus.WRITTEN)
            rejected = sum(
                1 for k in gap if status.get(k) is FrameStatus.WRITTEN and self._rejected(k)
            )
            out[fid] = (not_written, rejected)
            prev = fid
        return out

    # ------------------------------------------------------------------ access
    def read(self, frame_id: int) -> SensorFrame:
        """SensorFrame of one WRITTEN frame (thread-safe, independent of the iteration state)."""
        raw = self.reader.read_frame(self.camera_id, frame_id, verify=self.verify_images)
        ts, method = self.clock.timestamp(frame_id)
        check: RgbdCheck = self.rgbd[frame_id]
        not_written, rejected = self._before.get(frame_id, (0, 0))
        meta: dict[str, Any] = {
            "source": "replay",
            "session_id": self.session_id,
            "not_written_before": not_written,
            "rgbd_rejected_before": rejected,
            "rgbd": check.status.value,
            "color_depth_dt_ms": None if check.dt_us is None else check.dt_us / 1e3,
        }
        return build_sensor_frame(
            session_id=self.session_id,
            bundle=self.bundle,
            frame_id=frame_id,
            timestamp_ns=ts,
            timing=replace(raw.timing, method=method),
            color_rgb_u8=raw.color_rgb_u8,
            depth_raw_u16=raw.depth_raw_u16,
            depth_m_f32=raw.depth_m_f32,
            robot=self.robot,
            max_interp_gap_ns=self.config.sync.max_interp_gap_ns,
            source_meta=meta,
        )

    def next(self) -> SensorFrame | EndOfStream:
        if self._pos >= len(self.frame_ids):
            emitted = set(self.frame_ids)
            not_written = sum(1 for r in self.records if r.status is not FrameStatus.WRITTEN)
            rejected = sum(
                1
                for r in self.records
                if r.status is FrameStatus.WRITTEN and r.frame_id not in emitted
            )
            return EndOfStream(
                self.session_id, self.camera_id, self._pos, not_written, rejected, "end_of_session"
            )
        fid = self.frame_ids[self._pos]
        self._pos += 1
        if self.speed is not None:
            self._pace(self.clock.timestamp(fid)[0])
        return self.read(fid)

    def _pace(self, ts_ns: int) -> None:
        now = time.monotonic()
        if self._pace0 is None:
            self._pace0 = (now, ts_ns)
            return
        assert self.speed is not None
        target = self._pace0[0] + (ts_ns - self._pace0[1]) / 1e9 / self.speed
        if target > now:
            time.sleep(target - now)

    def __iter__(self) -> Iterator[SensorFrame]:
        while True:
            item = self.next()
            if isinstance(item, EndOfStream):
                return
            yield item

    def close(self) -> None:
        self._pos = len(self.frame_ids)
