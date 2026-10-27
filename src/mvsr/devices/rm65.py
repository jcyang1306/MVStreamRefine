"""RealMan RM65 state source (P0 F4). Reads state only; never commands motion (ADR-0007).

Ported from V1 `object_reconstruction/data/robot_pose_source.py` (no import of V1). Changes:
- every read is emitted (successes and failures) with a sequence number instead of a
  nearest-neighbour ring buffer; pairing with frames is a replay-time function (ADR-0004);
- joints are recorded (SDK reports degrees -> converted to radians here);
- poll stamp = midpoint of the request, `read_latency_ns` = request duration;
- optional UDP real-time push (`source="udp"`), which reconfigures the controller's push
  target and restores the previous configuration on close.

Euler angles exist only in this module: controller `[x, y, z, rx, ry, rz]` (m, rad) is
`T_base_tcp` with `R = Rz(rz) @ Ry(ry) @ Rx(rx)` (= scipy `from_euler('xyz')`).
"""

from __future__ import annotations

import ctypes
import importlib
import logging
import math
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from mvsr.core.errors import DeviceError
from mvsr.core.se3 import PoseSE3
from mvsr.io.records import RobotRecord

log = logging.getLogger(__name__)

N_JOINTS = 6


def xyzrpy_to_rotation(rpy: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """`R = Rz(rz) @ Ry(ry) @ Rx(rx)` for controller Euler angles `[rx, ry, rz]` (rad)."""
    rx, ry, rz = (float(v) for v in np.asarray(rpy, dtype=np.float64).reshape(3))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    out: npt.NDArray[np.float64] = Rz @ Ry @ Rx
    return out


def xyzrpy_to_T_base_tcp(xyzrpy: npt.ArrayLike) -> PoseSE3:
    v = np.asarray(xyzrpy, dtype=np.float64)
    if v.shape != (6,) or not np.all(np.isfinite(v)):
        raise ValueError(f"xyzrpy must be 6 finite values, got {v.tolist()}")
    return PoseSE3.from_rt(xyzrpy_to_rotation(v[3:]), v[:3], "base", "tcp")


def _local_ip_towards(remote_ip: str) -> str:
    """Host IP of the interface that routes to `remote_ip` (no packet is sent)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((remote_ip, 9))
        return str(s.getsockname()[0])


@dataclass
class RobotStats:
    samples: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    first_failure_ns: int | None = None
    last_error: str | None = None
    last_sample_ns: int | None = None
    recent: list[int] = field(default_factory=list)


class RM65StateSource:
    """Connects to the RM65 and emits one `RobotRecord` per state read to `sink`.

    `sdk` is the `Robotic_Arm.rm_robot_interface` module (injectable for control-flow tests).
    """

    def __init__(
        self,
        ip: str,
        port: int,
        sink: Callable[[RobotRecord], None],
        *,
        source: str = "poll",
        poll_hz: float = 100.0,
        udp_port: int = 8089,
        udp_cycle_5ms_units: int = 2,
        udp_host_ip: str | None = None,
        udp_first_packet_timeout_s: float = 2.0,
        sdk: Any = None,
    ) -> None:
        if source not in ("poll", "udp"):
            raise ValueError(f"source must be poll or udp, got {source!r}")
        if poll_hz <= 0:
            raise ValueError("poll_hz must be > 0")
        self.ip, self.port = str(ip), int(port)
        self.sink = sink
        self.source = source
        self.poll_period_s = 1.0 / float(poll_hz)
        self.poll_hz = float(poll_hz)
        self.udp_port = int(udp_port)
        self.udp_cycle = int(udp_cycle_5ms_units)
        self.udp_host_ip = udp_host_ip
        self.udp_first_packet_timeout_s = float(udp_first_packet_timeout_s)
        self._sdk = sdk
        self._arm: Any = None
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._udp_cb: Any = None
        self._udp_prev_config: dict[str, Any] | None = None
        self._first_packet = threading.Event()
        self.stats = RobotStats()
        self._stats_lock = threading.Lock()
        self.info: dict[str, Any] = {}

    # ------------------------------------------------------------------ connection
    @property
    def record_source(self) -> str:
        return "udp_push" if self.source == "udp" else "poll"

    def connect(self) -> dict[str, Any]:
        sdk = self._sdk
        if sdk is None:
            try:
                sdk = importlib.import_module("Robotic_Arm.rm_robot_interface")
            except ImportError as exc:
                raise DeviceError("Robotic_Arm SDK not installed (use the `rig` image)") from exc
            self._sdk = sdk
        arm = sdk.RoboticArm(sdk.rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = arm.rm_create_robot_arm(self.ip, self.port)
        if int(getattr(handle, "id", -1)) <= 0:
            raise DeviceError(f"cannot connect to RM65 at {self.ip}:{self.port}")
        self._arm = arm
        info: dict[str, Any] = {
            "model": "RM65",
            "endpoint": f"{self.ip}:{self.port}",
            "state_source": self.record_source,
        }
        try:
            from importlib.metadata import version

            info["sdk_version"] = version("Robotic_Arm")
        except Exception:
            info["sdk_version"] = None
        api_version = getattr(sdk, "rm_api_version", None)
        info["sdk_c_api_version"] = _safe_call(api_version)
        ret, robot_info = _safe_pair(getattr(arm, "rm_get_robot_info", None))
        info["controller_info"] = _jsonable(robot_info) if ret == 0 else {"error": f"ret={ret}"}
        ret, sw = _safe_pair(getattr(arm, "rm_get_arm_software_info", None))
        info["software_info"] = _jsonable(sw) if ret == 0 else {"error": f"ret={ret}"}
        if self.source == "poll":
            info["nominal_rate_hz"] = self.poll_hz
        else:
            info["nominal_rate_hz"] = 1000.0 / (5.0 * self.udp_cycle)
            info["udp"] = {"port": self.udp_port, "cycle_5ms_units": self.udp_cycle}
        self.info = info
        return info

    def close(self) -> None:
        self.stop()
        if self._arm is None:
            return
        if self._udp_prev_config is not None:
            self._restore_udp_config()
        try:
            code = self._arm.rm_delete_robot_arm()
            if int(code) != 0:
                log.warning("rm_delete_robot_arm returned %s", code)
        finally:
            self._arm = None

    # ------------------------------------------------------------------ streaming
    def start(self) -> None:
        if self._arm is None:
            raise DeviceError("connect() before start()")
        if self._thread is not None:
            raise DeviceError("robot source already started")
        self._stop.clear()
        if self.source == "udp":
            self._start_udp()
            return
        self._thread = threading.Thread(target=self._poll_loop, name="rm65-poll", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                log.warning("RM65 poll thread did not stop within 2 s")
            self._thread = None

    def _next_seq(self) -> int:
        with self._seq_lock:
            seq = self._seq
            self._seq += 1
            return seq

    def _emit(
        self,
        t_ns: int,
        joints_deg: Any,
        pose: Any,
        latency_ns: int | None,
        error: str | None,
    ) -> None:
        joints_rad: list[float] | None = None
        xyzrpy: list[float] | None = None
        if error is None:
            try:
                xyzrpy = [float(v) for v in list(pose)[:6]]
                if len(xyzrpy) != 6 or not all(math.isfinite(v) for v in xyzrpy):
                    raise ValueError(f"pose {pose!r}")
                if joints_deg is not None:
                    jd = [float(v) for v in list(joints_deg)[:N_JOINTS]]
                    if len(jd) != N_JOINTS or not all(math.isfinite(v) for v in jd):
                        raise ValueError(f"joints {joints_deg!r}")
                    joints_rad = [math.radians(v) for v in jd]
            except (TypeError, ValueError) as exc:
                error, xyzrpy, joints_rad = f"malformed state: {exc}", None, None
        rec = RobotRecord(
            seq=self._next_seq(),
            t_host_ns=int(t_ns),
            source=self.record_source,
            joints_rad=joints_rad,
            tcp_xyzrpy=xyzrpy,
            read_latency_ns=latency_ns,
            error=error,
        )
        with self._stats_lock:
            st = self.stats
            if error is None:
                st.samples += 1
                st.consecutive_failures = 0
                st.first_failure_ns = None
                st.last_sample_ns = rec.t_host_ns
                st.recent.append(rec.t_host_ns)
                if len(st.recent) > 512:
                    del st.recent[:256]
            else:
                st.failures += 1
                st.consecutive_failures += 1
                if st.first_failure_ns is None:
                    st.first_failure_ns = rec.t_host_ns
                st.last_error = error
        self.sink(rec)

    def _poll_loop(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set():
            t0 = time.monotonic_ns()
            try:
                ret, state = self._arm.rm_get_current_arm_state()
                t1 = time.monotonic_ns()
                if int(ret) != 0:
                    self._emit(
                        (t0 + t1) // 2, None, None, t1 - t0, f"rm_get_current_arm_state ret={ret}"
                    )
                else:
                    self._emit((t0 + t1) // 2, state.get("joint"), state.get("pose"), t1 - t0, None)
            except Exception as exc:  # SDK/network failure: logged as a failed sample
                t1 = time.monotonic_ns()
                self._emit((t0 + t1) // 2, None, None, t1 - t0, f"{type(exc).__name__}: {exc}")
            next_t += self.poll_period_s
            now = time.monotonic()
            if next_t < now:  # overrun: do not burst to catch up
                next_t = now
            self._stop.wait(next_t - now)

    # ------------------------------------------------------------------ UDP push
    def _start_udp(self) -> None:
        sdk, arm = self._sdk, self._arm
        ret, prev = _safe_pair(getattr(arm, "rm_get_realtime_push", None))
        if ret != 0:
            raise DeviceError(f"rm_get_realtime_push failed (ret={ret}); UDP push unsupported?")
        self._udp_prev_config = dict(prev)
        host_ip = self.udp_host_ip or _local_ip_towards(self.ip)
        config = sdk.rm_realtime_push_config_t(
            cycle=self.udp_cycle, enable=True, port=self.udp_port, force_coordinate=-1, ip=host_ip
        )
        code = arm.rm_set_realtime_push(config)
        if int(code) != 0:
            raise DeviceError(f"rm_set_realtime_push failed (ret={code})")

        def on_state(data: Any) -> None:
            t = time.monotonic_ns()
            self._first_packet.set()
            if self._stop.is_set():
                return
            try:
                if int(data.errCode) != 0:
                    self._emit(t, None, None, None, f"udp errCode={data.errCode}")
                    return
                wp = data.waypoint
                pose = [
                    wp.position.x,
                    wp.position.y,
                    wp.position.z,
                    wp.euler.rx,
                    wp.euler.ry,
                    wp.euler.rz,
                ]
                self._emit(t, list(data.joint_status.joint_position), pose, None, None)
            except Exception as exc:
                self._emit(t, None, None, None, f"udp callback: {type(exc).__name__}: {exc}")

        self._udp_cb = sdk.rm_realtime_arm_state_callback_ptr(on_state)
        arm.rm_realtime_arm_state_call_back(self._udp_cb)
        if not self._first_packet.wait(self.udp_first_packet_timeout_s):
            raise DeviceError(
                f"no UDP push packet within {self.udp_first_packet_timeout_s} s "
                f"(host {host_ip}:{self.udp_port}); use --robot-source poll"
            )
        self.info["udp"]["host_ip"] = host_ip
        self.info["udp"]["previous_config"] = _jsonable(self._udp_prev_config)

    def _restore_udp_config(self) -> None:
        prev = self._udp_prev_config or {}
        try:
            ip = prev.get("ip", b"")
            ip = ip.decode() if isinstance(ip, bytes) else str(ip)
            config = self._sdk.rm_realtime_push_config_t(
                cycle=int(prev.get("cycle", 1)),
                enable=bool(prev.get("enable", False)),
                port=int(prev.get("port", self.udp_port)),
                force_coordinate=int(prev.get("force_coordinate", -1)),
                ip=ip,
            )
            code = self._arm.rm_set_realtime_push(config)
            if int(code) != 0:
                log.warning("restoring UDP push config failed (ret=%s)", code)
        except Exception:
            log.warning("restoring UDP push config failed", exc_info=True)
        self._udp_prev_config = None

    def snapshot_stats(self) -> RobotStats:
        with self._stats_lock:
            st = self.stats
            return RobotStats(
                st.samples,
                st.failures,
                st.consecutive_failures,
                st.first_failure_ns,
                st.last_error,
                st.last_sample_ns,
                list(st.recent),
            )


def _safe_call(fn: Any) -> Any:
    if fn is None:
        return None
    try:
        return _jsonable(fn())
    except Exception:
        return None


def _safe_pair(fn: Any) -> tuple[int, Any]:
    if fn is None:
        return -99, None
    try:
        ret, value = fn()
        return int(ret), value
    except Exception as exc:
        log.debug("SDK call failed: %s", exc)
        return -98, None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace").rstrip("\x00")
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, ctypes.Array):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return str(value)
