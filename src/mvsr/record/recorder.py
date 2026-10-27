"""Session recorder (subplans/00_FOUNDATION.md F5, ADR-0003/0004/0007).

Per camera:  capture thread  --(process queue)-->  align/copy thread  --(writer queue)-->
             PNG writer pool --> frames.jsonl (ordered by frame_id).
The capture thread only waits for framesets and stamps host arrival time. `frame_id`s are
assigned at arrival once recording is armed, including frames that are later dropped;
every frame_id gets exactly one frames.jsonl record (`WRITTEN` or `DROPPED_<reason>`).
The robot thread writes every state read to robot.jsonl. The recorder never moves the robot.

Stop conditions: operator (Ctrl+C / SIGTERM / `q` in preview), `--duration`, disk reserve
reached (all -> COMPLETE); camera disconnect, robot failure > `abort_after_failure_s`, disk
write error (-> ABORTED). The manifest is finalized in every case once the session exists.
"""

from __future__ import annotations

import logging
import os
import queue
import shutil
import signal
import socket
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mvsr import paths
from mvsr.calib.bundle import HANDEYE_SESSION_PATH, build_bundle
from mvsr.calib.handeye import load_handeye
from mvsr.core.errors import ChecksumMismatchError, DeviceError, DiskBudgetError, MvsrError
from mvsr.core.types import EventType, FrameStatus, SessionStatus, make_session_id
from mvsr.devices.realsense import Arrival, RealSenseStream, StreamInfo
from mvsr.devices.rm65 import RM65StateSource
from mvsr.io.atomic import sha256_bytes
from mvsr.io.records import FrameRecord, RobotRecord
from mvsr.io.session_writer import SessionWriter
from mvsr.record.config import CameraConfig, RecordConfig
from mvsr.software import software_info

log = logging.getLogger(__name__)

#: Scenario tags accepted by `--scenario` (ADR-0002 codes + the P0 SMOKE session).
SCENARIO_TAGS = ("SMOKE", "R-CALIB", "R-STATIC", "R-MARKER", "R-HANDHELD", "R-BOTH", "R-HARD")
#: Head camera is started first: a second RealSense may not deliver frames while an
#: already-open one is left unread (V1 capture/sync.py).
START_ORDER = ("head", "wrist")
_SENTINEL = object()


def clock_pair(samples: int = 5) -> dict[str, int]:
    """Tightest of `samples` (CLOCK_REALTIME, CLOCK_MONOTONIC) pairs; bound in `uncertainty_ns`."""
    best: tuple[int, int, int] | None = None
    for _ in range(samples):
        r0 = time.clock_gettime_ns(time.CLOCK_REALTIME)
        m = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        r1 = time.clock_gettime_ns(time.CLOCK_REALTIME)
        if best is None or r1 - r0 < best[2]:
            best = ((r0 + r1) // 2, m, r1 - r0)
    assert best is not None
    return {"realtime_ns": best[0], "monotonic_ns": best[1], "uncertainty_ns": best[2]}


@dataclass(frozen=True)
class RecordRequest:
    cameras: tuple[str, ...]
    scenario: tuple[str, ...]
    object_name: str
    duration_s: float | None = None
    fps: dict[str, int] = field(default_factory=dict)
    gt_type: str | None = None
    notes: str = ""
    robot_source: str | None = None
    preview: bool = False
    start_paused: bool = False
    data_root: Path | None = None

    def __post_init__(self) -> None:
        cams = tuple(self.cameras)
        if not cams or "wrist" not in cams or any(c not in ("wrist", "head") for c in cams):
            raise ValueError(f"--cameras must include wrist and only wrist/head, got {cams}")
        if len(set(cams)) != len(cams):
            raise ValueError("duplicate camera in --cameras")
        bad = [s for s in self.scenario if s not in SCENARIO_TAGS]
        if not self.scenario or bad:
            raise ValueError(f"unknown scenario tags {bad}; allowed {SCENARIO_TAGS}")
        if not self.object_name.strip():
            raise ValueError("--object must be non-empty")
        if self.duration_s is not None and self.duration_s <= 0:
            raise ValueError("--duration must be > 0")
        if self.start_paused and not self.preview:
            raise ValueError("--start-paused needs --preview (SPACE resumes)")

    @classmethod
    def from_cli(cls, args: Any) -> RecordRequest:
        fps = {}
        if args.wrist_fps is not None:
            fps["wrist"] = int(args.wrist_fps)
        if args.head_fps is not None:
            fps["head"] = int(args.head_fps)
        return cls(
            cameras=tuple(c.strip() for c in args.cameras.split(",") if c.strip()),
            scenario=tuple(s.strip() for s in args.scenario.split(",") if s.strip()),
            object_name=args.object_name,
            duration_s=args.duration,
            fps=fps,
            gt_type=args.gt,
            notes=args.notes,
            robot_source=args.robot_source,
            preview=args.preview,
            start_paused=args.start_paused,
            data_root=args.data_root,
        )


@dataclass
class RecordResult:
    status: str
    session_id: str
    session_dir: str
    stop_reason: str
    errors: list[str]
    counts: dict[str, Any]
    performance: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "session_id": self.session_id,
            "session_dir": self.session_dir,
            "stop_reason": self.stop_reason,
            "errors": self.errors,
            "counts": self.counts,
            "performance": self.performance,
        }


class _CameraPipeline:
    """Threads and bounded queues of one camera stream."""

    def __init__(
        self,
        camera_id: str,
        stream: RealSenseStream,
        cam: CameraConfig,
        recorder: Recorder,
    ) -> None:
        cap = recorder.config.capture
        self.camera_id = camera_id
        self.stream = stream
        self.cam = cam
        self.rec = recorder
        self.writer: SessionWriter | None = None
        self.armed = threading.Event()
        self.stop_capture = threading.Event()
        self.process_q: queue.Queue[Any] = queue.Queue(maxsize=cap.process_queue_size)
        self.write_q: queue.Queue[Any] = queue.Queue(maxsize=cap.writer_queue_size)
        self.lock = threading.Lock()
        self.next_frame_id = 0
        self.discarded = 0
        self.arrivals: deque[int] = deque(maxlen=256)
        self.counts: Counter[str] = Counter()
        self.hwm_process = 0
        self.hwm_write = 0
        self.latest_rgb: np.ndarray | None = None
        self.threads: list[threading.Thread] = []
        self.info: StreamInfo | None = None

    # -- lifecycle -------------------------------------------------------
    def start(self) -> StreamInfo:
        self.info = self.stream.start()
        t = threading.Thread(
            target=self._capture_loop, name=f"{self.camera_id}-capture", daemon=True
        )
        t.start()
        self.threads = [t]
        return self.info

    def arm(self, writer: SessionWriter) -> None:
        self.writer = writer
        workers = [
            threading.Thread(target=self._process_loop, name=f"{self.camera_id}-align", daemon=True)
        ] + [
            threading.Thread(
                target=self._writer_loop, name=f"{self.camera_id}-write{i}", daemon=True
            )
            for i in range(self.cam.writer_threads)
        ]
        for w in workers:
            w.start()
        self.threads.extend(workers)
        self.armed.set()

    def stop_and_drain(self, timeout_s: float = 30.0) -> None:
        self.stop_capture.set()
        self.threads[0].join(timeout=timeout_s)
        if self.writer is None:
            return
        self.process_q.put(_SENTINEL)
        self.threads[1].join(timeout=timeout_s)
        for _ in range(self.cam.writer_threads):
            self.write_q.put(_SENTINEL)
        for t in self.threads[2:]:
            t.join(timeout=timeout_s)
        alive = [t.name for t in self.threads if t.is_alive()]
        if alive:
            self.rec.fatal(f"{self.camera_id}: threads did not stop: {alive}")

    # -- records ---------------------------------------------------------
    def _record(
        self,
        fid: int,
        arr: Arrival,
        status: FrameStatus,
        detail: str | None = None,
        refs: tuple[Any, Any] | None = None,
    ) -> None:
        assert self.writer is not None
        rec = FrameRecord(
            frame_id=fid,
            status=status,
            t_host_arrival_ns=arr.t_host_arrival_ns,
            timing=arr.timing,
            color=None if refs is None else refs[0],
            depth=None if refs is None else refs[1],
            depth_timing=arr.depth_timing,
            rs_extra=arr.rs_extra,
            drop_detail=detail,
        )
        self.writer.record_frame(self.camera_id, rec)
        with self.lock:
            self.counts[status.value] += 1

    # -- threads ---------------------------------------------------------
    def _capture_loop(self) -> None:
        cap = self.rec.config.capture
        timeouts = 0
        first = True
        while not self.stop_capture.is_set():
            try:
                arr = self.stream.wait(
                    cap.first_frame_timeout_ms if first else cap.frame_timeout_ms
                )
            except TimeoutError as exc:
                timeouts += 1
                if timeouts >= cap.max_consecutive_timeouts:
                    self.rec.fatal(
                        f"{self.camera_id}: {timeouts} consecutive timeouts ({exc}); disconnected?"
                    )
                    return
                continue
            except DeviceError as exc:
                self.rec.fatal(str(exc))
                return
            first = False
            timeouts = 0
            if self.stop_capture.is_set():
                return
            if not self.armed.is_set():
                self.discarded += 1
                continue
            fid = self.next_frame_id
            self.next_frame_id += 1
            with self.lock:
                self.arrivals.append(arr.t_host_arrival_ns)
            if self.rec.paused.is_set():
                arr.frameset = None
                self._record(fid, arr, FrameStatus.DROPPED_OPERATOR, "paused")
            elif not arr.complete:
                self._record(fid, arr, FrameStatus.DROPPED_INCOMPLETE, arr.detail)
            else:
                try:
                    self.process_q.put_nowait((fid, arr))
                    self.hwm_process = max(self.hwm_process, self.process_q.qsize())
                except queue.Full:
                    arr.frameset = None
                    self._record(fid, arr, FrameStatus.DROPPED_QUEUE_FULL, "process_queue")

    def _process_loop(self) -> None:
        while True:
            item = self.process_q.get()
            if item is _SENTINEL:
                return
            fid, arr = item
            try:
                rgb, depth = self.stream.align_and_copy(arr)
            except DeviceError as exc:
                self._record(fid, arr, FrameStatus.DROPPED_INCOMPLETE, f"align: {exc}")
                continue
            self.latest_rgb = rgb
            try:
                self.write_q.put_nowait((fid, arr, rgb, depth))
                self.hwm_write = max(self.hwm_write, self.write_q.qsize())
            except queue.Full:
                self._record(fid, arr, FrameStatus.DROPPED_QUEUE_FULL, "writer_queue")

    def _writer_loop(self) -> None:
        assert self.writer is not None
        while True:
            item = self.write_q.get()
            if item is _SENTINEL:
                return
            fid, arr, rgb, depth = item
            try:
                refs = self.writer.write_images(self.camera_id, fid, rgb, depth)
            except OSError as exc:
                self._record(fid, arr, FrameStatus.DROPPED_DISK, f"{type(exc).__name__}: {exc}")
                self.rec.fatal(f"{self.camera_id}: disk write failed: {exc}")
                continue
            self._record(fid, arr, FrameStatus.WRITTEN, refs=refs)

    # -- stats -----------------------------------------------------------
    def recent_fps(self) -> float:
        with self.lock:
            a = list(self.arrivals)
        if len(a) < 2:
            return 0.0
        return (len(a) - 1) / max((a[-1] - a[0]) / 1e9, 1e-9)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            counts = dict(self.counts)
        return {
            "seen": self.next_frame_id,
            "counts": counts,
            "fps": round(self.recent_fps(), 2),
            "process_queue": self.process_q.qsize(),
            "writer_queue": self.write_q.qsize(),
        }


class Recorder:
    """Owns devices, threads and the session writer for one recording."""

    def __init__(
        self,
        config: RecordConfig,
        request: RecordRequest,
        *,
        rs_module: Any = None,
        rm_sdk: Any = None,
        disk_usage: Callable[[str], Any] = shutil.disk_usage,
        install_signal_handlers: bool = True,
        preview_factory: Callable[[int], Any] | None = None,
    ) -> None:
        self.config = config
        self.request = request
        self._rs = rs_module
        self._rm = rm_sdk
        self._disk_usage = disk_usage
        self._signals = install_signal_handlers
        self._preview_factory = preview_factory
        self.paused = threading.Event()
        self._stop = threading.Event()
        self._stop_reason: str | None = None
        self._errors: list[str] = []
        self._err_lock = threading.Lock()
        self.writer: SessionWriter | None = None
        self.pipelines: dict[str, _CameraPipeline] = {}
        self.robot: RM65StateSource | None = None

    # -- control ---------------------------------------------------------
    def request_stop(self, reason: str = "operator") -> None:
        if self._stop_reason is None:
            self._stop_reason = reason
        self._stop.set()

    def fatal(self, message: str) -> None:
        log.error("fatal: %s", message)
        with self._err_lock:
            self._errors.append(message)
        if self.writer is not None and not self.writer.finalized:
            try:
                self.writer.log_event(EventType.ERROR, {"message": message})
            except MvsrError:
                pass
        self.request_stop("error")

    def _toggle_pause(self) -> None:
        assert self.writer is not None
        if self.paused.is_set():
            self.paused.clear()
            self.writer.log_event(EventType.RESUME, {})
        else:
            self.paused.set()
            self.writer.log_event(EventType.PAUSE, {})

    def _on_robot(self, rec: RobotRecord) -> None:
        if self.writer is not None and not self.writer.finalized:
            self.writer.log_robot(rec)

    # -- helpers ---------------------------------------------------------
    def _camera_configs(self) -> dict[str, CameraConfig]:
        out = {}
        for cid in self.request.cameras:
            if cid not in self.config.cameras:
                raise ValueError(f"camera {cid} not configured")
            cam = self.config.cameras[cid]
            if cid in self.request.fps:
                cam = replace(cam, fps=self.request.fps[cid])
            out[cid] = cam
        return out

    def _free_bytes(self, root: Path) -> int:
        return int(self._disk_usage(str(root)).free)

    def _check_disk(self, root: Path, cams: dict[str, CameraConfig]) -> dict[str, Any]:
        d = self.config.disk
        duration = self.request.duration_s or d.estimate_duration_s
        estimate = sum(d.est_bytes_per_frame[c] * cam.fps * duration for c, cam in cams.items())
        reserve = d.reserve_gb * 1e9
        free = self._free_bytes(root)
        info = {
            "free_bytes_start": free,
            "reserve_bytes": int(reserve),
            "estimated_session_bytes": int(estimate),
            "estimate_duration_s": duration,
        }
        if free < reserve + estimate:
            raise DiskBudgetError(
                f"free {free / 1e9:.1f} GB < reserve {reserve / 1e9:.1f} GB + estimate "
                f"{estimate / 1e9:.1f} GB ({duration:.0f} s)"
            )
        return info

    def _manifest_start(self, robot_info: dict[str, Any], disk: dict[str, Any]) -> dict[str, Any]:
        req = self.request
        return {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "host": socket.gethostname(),
            "software": software_info(),
            "robot": robot_info,
            "cameras": {},
            "clock": {"start": clock_pair()},
            "scenario": {
                "tags": list(req.scenario),
                "object": req.object_name,
                "gt_type": req.gt_type,
                "notes": req.notes,
            },
            "disk": disk,
            "recorder": {
                "config": self.config.snapshot(),
                "config_sha256": self.config.source_sha256,
                "cameras_requested": list(req.cameras),
                "duration_s_requested": req.duration_s,
                "warmup_frames_discarded": self.config.capture.warmup_frames,
            },
        }

    @staticmethod
    def _camera_entry(cid: str, cam: CameraConfig, info: StreamInfo) -> dict[str, Any]:
        options = info.options

        def first(key: str) -> Any:
            for values in options.values():
                if key in values:
                    return values[key]
            return None

        return {
            "model": info.model,
            "serial": info.serial,
            "firmware": info.firmware,
            "mount": "wrist" if cid == "wrist" else "fixed",
            "depth_units_m_sdk": info.depth_units_m_sdk,
            "color_codec": "png",
            "png_compression": None,
            "resolution": [info.width, info.height],
            "fps": info.fps,
            "formats": {
                "color_stream": info.color_format,
                "depth_stream": "z16",
                "color_png": "rgb8",
                "depth_png": "uint16_raw",
            },
            "device_options_file": f"calibration/{cid}_realsense.json",
            "preset": first("visual_preset"),
            "emitter": {
                "emitter_enabled": first("emitter_enabled"),
                "laser_power": first("laser_power"),
            },
            "exposure_mode": {"enable_auto_exposure": first("enable_auto_exposure")},
            "post_filters": [],
            "global_time_enabled": info.global_time_enabled,
            "aligned_to": "color",
        }

    def _wait_warmup(self) -> None:
        cap = self.config.capture
        for cid, p in self.pipelines.items():
            budget = cap.first_frame_timeout_ms / 1e3 + 4.0 * cap.warmup_frames / max(p.cam.fps, 1)
            deadline = time.monotonic() + budget
            while p.discarded < cap.warmup_frames:
                if self._stop.is_set():
                    raise DeviceError("; ".join(self._errors) or "stopped during warm-up")
                if time.monotonic() > deadline:
                    raise DeviceError(
                        f"{cid}: only {p.discarded}/{cap.warmup_frames} warm-up frames in {budget:.1f} s"
                    )
                time.sleep(0.01)

    # -- main ------------------------------------------------------------
    def run(self) -> RecordResult:
        cams = self._camera_configs()
        root = (self.request.data_root or paths.data_root()).resolve()
        root.mkdir(parents=True, exist_ok=True)
        disk = self._check_disk(root, cams)
        handeye_path = self.config.handeye_path()
        handeye = load_handeye(handeye_path, self.config.handeye.sha256)
        handeye_bytes = handeye_path.read_bytes()
        if sha256_bytes(handeye_bytes) != handeye.sha256:
            raise ChecksumMismatchError(f"{handeye_path} changed while loading")
        rcfg = self.config.robot
        self.robot = RM65StateSource(
            rcfg.ip,
            rcfg.port,
            self._on_robot,
            source=self.request.robot_source or rcfg.source,
            poll_hz=rcfg.poll_hz,
            udp_port=rcfg.udp_port,
            udp_cycle_5ms_units=rcfg.udp_cycle_5ms_units,
            udp_host_ip=rcfg.udp_host_ip,
            sdk=self._rm,
        )
        old_handlers: dict[int, Any] = {}
        preview = None
        try:
            if self._signals:
                for sig in (signal.SIGINT, signal.SIGTERM):
                    old_handlers[sig] = signal.signal(sig, lambda *_: self.request_stop("operator"))
            robot_info = self.robot.connect()
            for cid in sorted(cams, key=START_ORDER.index):
                cam = cams[cid]
                stream = RealSenseStream(
                    cid,
                    cam.serial,
                    cam.width,
                    cam.height,
                    cam.fps,
                    cam.color_format,
                    cam.depth_units_m,
                    frame_timeout_ms=self.config.capture.frame_timeout_ms,
                    rs=self._rs,
                )
                pipeline = _CameraPipeline(cid, stream, cam, self)
                self.pipelines[cid] = pipeline
                pipeline.start()
            self._wait_warmup()
            if self.request.preview:
                preview = self._make_preview()
            return self._record(cams, root, disk, robot_info, handeye, handeye_bytes, preview)
        finally:
            for p in self.pipelines.values():
                p.stop_capture.set()
            for p in self.pipelines.values():
                if p.threads:
                    p.threads[0].join(timeout=5.0)
                p.stream.stop()
            if self.robot is not None:
                self.robot.close()
            if preview is not None:
                preview.close()
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)

    def _make_preview(self) -> Any:
        if self._preview_factory is not None:
            return self._preview_factory(self.config.monitor.preview_max_width)
        from mvsr.record.preview import Preview

        return Preview(self.config.monitor.preview_max_width)

    def _record(
        self,
        cams: dict[str, CameraConfig],
        root: Path,
        disk: dict[str, Any],
        robot_info: dict[str, Any],
        handeye: Any,
        handeye_bytes: bytes,
        preview: Any,
    ) -> RecordResult:
        cap = self.config.capture
        session_id = make_session_id(socket.gethostname())
        writer = SessionWriter(
            root,
            session_id,
            self._manifest_start(robot_info, disk),
            png_compression=cap.png_compression,
            fsync_images=cap.fsync_images,
        )
        self.writer = writer
        t_start = time.monotonic_ns()
        cpu_start = time.process_time()
        error: BaseException | None = None
        try:
            writer.write_file(HANDEYE_SESSION_PATH, handeye_bytes)
            for cid, p in self.pipelines.items():
                info = p.info
                assert info is not None
                bundle = build_bundle(
                    camera_id=cid,
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
                    depth_units_m=cams[cid].depth_units_m,
                    handeye=handeye,
                )
                entry = self._camera_entry(cid, cams[cid], info)
                entry["png_compression"] = cap.png_compression
                writer.add_camera(cid, bundle, info.realsense_json(), entry)
            writer.log_event(
                EventType.START,
                {
                    "cameras": list(self.pipelines),
                    "scenario": list(self.request.scenario),
                    "object": self.request.object_name,
                },
            )
            assert self.robot is not None
            self.robot.start()
            if self.request.start_paused:
                self.paused.set()
                writer.log_event(EventType.PAUSE, {"reason": "start_paused"})
            for p in self.pipelines.values():
                p.arm(writer)
            self._monitor(root, preview, t_start)
        except BaseException as exc:
            error = exc
            with self._err_lock:
                self._errors.append(f"recorder: {type(exc).__name__}: {exc}")
            self.request_stop("error")
        self._shutdown_threads()
        duration_s = (time.monotonic_ns() - t_start) / 1e9
        writer.log_event(EventType.STOP, {"reason": self._stop_reason or "operator"})
        status = SessionStatus.ABORTED if self._errors else SessionStatus.COMPLETE
        perf = self._performance(duration_s, time.process_time() - cpu_start)
        manifest = writer.finalize(
            status,
            clock={**writer.manifest["clock"], "end": clock_pair()},
            disk={**disk, "free_bytes_end": self._free_bytes(root)},
            performance=perf,
            stop_reason=self._stop_reason or "operator",
            errors=list(self._errors),
            duration_s=duration_s,
        )
        log.info(
            "session %s finalized: %s (%s)", session_id, manifest["status"], manifest["stop_reason"]
        )
        if error is not None:
            raise error
        return RecordResult(
            status=manifest["status"],
            session_id=session_id,
            session_dir=str(writer.dir),
            stop_reason=manifest["stop_reason"],
            errors=manifest["errors"],
            counts=manifest["counts"],
            performance=perf,
        )

    def _shutdown_threads(self) -> None:
        """Stop capture, then the robot, then drain align/writer queues (records stay complete)."""
        for p in self.pipelines.values():
            p.stop_capture.set()
        if self.robot is not None:
            self.robot.stop()
        for p in self.pipelines.values():
            try:
                p.stop_and_drain()
            except Exception as exc:
                with self._err_lock:
                    self._errors.append(f"{p.camera_id}: drain failed: {exc}")

    def _monitor(self, root: Path, preview: Any, t_start: int) -> None:
        d, mon = self.config.disk, self.config.monitor
        assert self.writer is not None and self.robot is not None
        next_disk = next_status = time.monotonic()
        robot_warned = False
        abort_ns = int(self.config.robot.abort_after_failure_s * 1e9)
        lines: list[str] = []
        while not self._stop.is_set():
            now = time.monotonic()
            now_ns = time.monotonic_ns()
            if (
                self.request.duration_s is not None
                and (now_ns - t_start) / 1e9 >= self.request.duration_s
            ):
                self.request_stop("duration")
                break
            if now >= next_disk:
                next_disk = now + d.check_interval_s
                free = self._free_bytes(root)
                if free < d.reserve_gb * 1e9:
                    self.writer.log_event(
                        EventType.WARNING, {"message": "disk reserve reached", "free_bytes": free}
                    )
                    self.request_stop("disk_reserve")
                    break
            st = self.robot.snapshot_stats()
            failing_since = st.first_failure_ns if st.consecutive_failures else None
            if st.last_sample_ns is None and failing_since is None:
                failing_since = t_start
            if failing_since is not None:
                if not robot_warned and st.consecutive_failures:
                    robot_warned = True
                    self.writer.log_event(
                        EventType.WARNING, {"message": "robot read failing", "error": st.last_error}
                    )
                if now_ns - failing_since > abort_ns:
                    self.fatal(
                        f"robot: no valid state for {(now_ns - failing_since) / 1e9:.1f} s ({st.last_error})"
                    )
                    break
            else:
                robot_warned = False
            if preview is not None or now >= next_status:
                lines = self._status_lines(root, st, t_start)
            if now >= next_status:
                next_status = now + mon.status_interval_s
                log.info(" | ".join(lines))
            if preview is not None:
                wrist = self.pipelines["wrist"].latest_rgb
                key = preview.show(wrist, lines, recording=not self.paused.is_set())
                if key in ("q", "esc"):
                    self.request_stop("operator")
                elif key == "space":
                    self._toggle_pause()
                elif key == "a":
                    self.writer.log_event(
                        EventType.ANNOTATION, {"key": "a", "elapsed_s": (now_ns - t_start) / 1e9}
                    )
            else:
                self._stop.wait(0.05)

    def _status_lines(self, root: Path, st: Any, t_start: int) -> list[str]:
        elapsed = (time.monotonic_ns() - t_start) / 1e9
        lines = [f"{'PAUSED' if self.paused.is_set() else 'REC'} {elapsed:6.1f}s"]
        for cid, p in self.pipelines.items():
            s = p.snapshot()
            dropped = s["seen"] - s["counts"].get("WRITTEN", 0)
            lines.append(
                f"{cid} {s['fps']:.1f}fps seen={s['seen']} drop={dropped} q={s['process_queue']}/{s['writer_queue']}"
            )
        recent = st.recent[-101:]
        rate = (
            (len(recent) - 1) / max((recent[-1] - recent[0]) / 1e9, 1e-9)
            if len(recent) > 1
            else 0.0
        )
        lines.append(f"robot {rate:.0f}Hz ok={st.samples} fail={st.failures}")
        lines.append(f"disk free {self._free_bytes(root) / 1e9:.1f}GB")
        return lines

    def _performance(self, duration_s: float, cpu_s: float) -> dict[str, Any]:
        cams = {}
        total_bytes = 0
        for cid, p in self.pipelines.items():
            with p.lock:
                counts = dict(p.counts)
            nbytes = self.writer.camera_bytes(cid) if self.writer is not None else 0
            total_bytes += nbytes
            cams[cid] = {
                "frames_seen": p.next_frame_id,
                "frames_written": counts.get("WRITTEN", 0),
                "capture_fps": p.next_frame_id / duration_s if duration_s > 0 else None,
                "written_fps": counts.get("WRITTEN", 0) / duration_s if duration_s > 0 else None,
                "process_queue_high_water": p.hwm_process,
                "writer_queue_high_water": p.hwm_write,
                "process_queue_size": self.config.capture.process_queue_size,
                "writer_queue_size": self.config.capture.writer_queue_size,
                "png_mb_per_s": nbytes / 1e6 / duration_s if duration_s > 0 else None,
                "warmup_frames_discarded": p.discarded,
            }
        robot: dict[str, Any] = {}
        if self.robot is not None:
            st = self.robot.snapshot_stats()
            robot = {
                "samples": st.samples,
                "failures": st.failures,
                "mean_rate_hz": st.samples / duration_s if duration_s > 0 else None,
            }
        return {
            "duration_s": duration_s,
            "cameras": cams,
            "robot": robot,
            "cpu": {
                "process_cpu_s": cpu_s,
                "mean_cores_busy": cpu_s / duration_s if duration_s > 0 else None,
                "loadavg_end": list(os.getloadavg()),
                "n_cpus": os.cpu_count(),
            },
            "disk_mb_per_s": total_bytes / 1e6 / duration_s if duration_s > 0 else None,
        }


def run_recorder(config: RecordConfig, request: RecordRequest, **kwargs: Any) -> RecordResult:
    return Recorder(config, request, **kwargs).run()
