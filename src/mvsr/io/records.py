"""Line records of the session JSONL files (contracts/SESSION_FORMAT.md).

Fields beyond SESSION_FORMAT.md (all optional, raw-data only, see README "P0 format notes"):
- `frames.jsonl`: `color.raw_sha256` / `depth.raw_sha256` (sha256 of the pixel array before PNG
  encoding; proves the lossless round trip), `depth_timing` (the depth stream's own raw clock
  values; on the D405 color and depth share one imager, on the D435 they do not),
  `rs_extra` (librealsense `backend_timestamp` / `time_of_arrival`, ms), `drop_detail`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from mvsr.core.errors import SessionFormatError
from mvsr.core.types import SHA256_RE, EventType, FrameStatus

FRAME_TIMING_KEYS = (
    "rs_timestamp_ms",
    "rs_timestamp_domain",
    "rs_sensor_timestamp_us",
    "rs_frame_timestamp_us",
    "exposure_us",
    "rs_frame_number",
)


def dumps_line(obj: dict[str, Any]) -> str:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _req(d: dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise SessionFormatError(f"{where}: missing key '{key}'")
    return d[key]


def _opt_num(v: Any, where: str, integer: bool = False) -> Any:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise SessionFormatError(f"{where}: expected number or null, got {v!r}")
    if integer and not isinstance(v, int):
        raise SessionFormatError(f"{where}: expected int or null, got {v!r}")
    if isinstance(v, float) and not math.isfinite(v):
        raise SessionFormatError(f"{where}: non-finite value")
    return v


@dataclass(frozen=True)
class FileRef:
    path: str
    sha256: str
    raw_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"path": self.path, "sha256": self.sha256}
        if self.raw_sha256 is not None:
            out["raw_sha256"] = self.raw_sha256
        return out

    @classmethod
    def from_dict(cls, d: Any, where: str) -> FileRef:
        if not isinstance(d, dict):
            raise SessionFormatError(f"{where}: file ref must be an object")
        path, sha = _req(d, "path", where), _req(d, "sha256", where)
        raw = d.get("raw_sha256")
        if not isinstance(path, str) or not isinstance(sha, str) or not SHA256_RE.match(sha):
            raise SessionFormatError(f"{where}: bad file ref {d!r}")
        if raw is not None and (not isinstance(raw, str) or not SHA256_RE.match(raw)):
            raise SessionFormatError(f"{where}: bad raw_sha256")
        return cls(path, sha, raw)


@dataclass(frozen=True)
class RawTiming:
    """Raw clock values of one RealSense frame (each None when the device does not report it)."""

    rs_timestamp_ms: float | None = None
    rs_timestamp_domain: str | None = None
    rs_sensor_timestamp_us: int | None = None
    rs_frame_timestamp_us: int | None = None
    exposure_us: float | None = None
    rs_frame_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in FRAME_TIMING_KEYS}

    @classmethod
    def from_dict(cls, d: dict[str, Any], where: str) -> RawTiming:
        domain = _req(d, "rs_timestamp_domain", where)
        if domain is not None and not isinstance(domain, str):
            raise SessionFormatError(f"{where}: rs_timestamp_domain must be str or null")
        return cls(
            rs_timestamp_ms=_opt_num(_req(d, "rs_timestamp_ms", where), where),
            rs_timestamp_domain=domain,
            rs_sensor_timestamp_us=_opt_num(_req(d, "rs_sensor_timestamp_us", where), where, True),
            rs_frame_timestamp_us=_opt_num(_req(d, "rs_frame_timestamp_us", where), where, True),
            exposure_us=_opt_num(_req(d, "exposure_us", where), where),
            rs_frame_number=_opt_num(_req(d, "rs_frame_number", where), where, True),
        )


@dataclass(frozen=True)
class FrameRecord:
    """One `cameras/<camera_id>/frames.jsonl` line; raw timing fields come from the color frame."""

    frame_id: int
    status: FrameStatus
    t_host_arrival_ns: int
    timing: RawTiming
    color: FileRef | None = None
    depth: FileRef | None = None
    depth_timing: RawTiming | None = None
    rs_extra: dict[str, Any] = field(default_factory=dict)
    drop_detail: str | None = None

    def __post_init__(self) -> None:
        written = self.status is FrameStatus.WRITTEN
        if written != (self.color is not None and self.depth is not None):
            raise SessionFormatError(
                f"frame {self.frame_id}: WRITTEN ⇔ color and depth refs present ({self.status})"
            )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "frame_id": self.frame_id,
            "status": self.status.value,
            "t_host_arrival_ns": self.t_host_arrival_ns,
            **self.timing.to_dict(),
        }
        if self.color is not None and self.depth is not None:
            out["color"] = self.color.to_dict()
            out["depth"] = self.depth.to_dict()
        if self.depth_timing is not None:
            out["depth_timing"] = self.depth_timing.to_dict()
        if self.rs_extra:
            out["rs_extra"] = dict(self.rs_extra)
        if self.drop_detail is not None:
            out["drop_detail"] = self.drop_detail
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any], where: str = "frames.jsonl") -> FrameRecord:
        fid = _req(d, "frame_id", where)
        where = f"{where} frame_id={fid}"
        if isinstance(fid, bool) or not isinstance(fid, int) or fid < 0:
            raise SessionFormatError(f"{where}: bad frame_id")
        try:
            status = FrameStatus(_req(d, "status", where))
        except ValueError as exc:
            raise SessionFormatError(f"{where}: bad status {d.get('status')!r}") from exc
        arrival = _req(d, "t_host_arrival_ns", where)
        if isinstance(arrival, bool) or not isinstance(arrival, int):
            raise SessionFormatError(f"{where}: t_host_arrival_ns must be int")
        color = FileRef.from_dict(d["color"], where) if d.get("color") is not None else None
        depth = FileRef.from_dict(d["depth"], where) if d.get("depth") is not None else None
        dt = d.get("depth_timing")
        return cls(
            frame_id=fid,
            status=status,
            t_host_arrival_ns=arrival,
            timing=RawTiming.from_dict(d, where),
            color=color,
            depth=depth,
            depth_timing=None if dt is None else RawTiming.from_dict(dt, where + " depth_timing"),
            rs_extra=dict(d.get("rs_extra") or {}),
            drop_detail=d.get("drop_detail"),
        )


@dataclass(frozen=True)
class RobotRecord:
    """One `robot.jsonl` line. A failed read has `error` set and null state fields."""

    seq: int
    t_host_ns: int
    source: str
    joints_rad: list[float] | None
    tcp_xyzrpy: list[float] | None
    read_latency_ns: int | None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.source not in ("udp_push", "poll"):
            raise SessionFormatError(f"robot seq {self.seq}: bad source {self.source!r}")
        if self.error is None and self.tcp_xyzrpy is None:
            raise SessionFormatError(f"robot seq {self.seq}: successful read without tcp_xyzrpy")
        for name in ("joints_rad", "tcp_xyzrpy"):
            v = getattr(self, name)
            if v is not None and (len(v) != 6 or not all(math.isfinite(float(x)) for x in v)):
                raise SessionFormatError(f"robot seq {self.seq}: {name} must be 6 finite values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "t_host_ns": self.t_host_ns,
            "source": self.source,
            "joints_rad": self.joints_rad,
            "tcp_xyzrpy": self.tcp_xyzrpy,
            "read_latency_ns": self.read_latency_ns,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], where: str = "robot.jsonl") -> RobotRecord:
        where = f"{where} seq={d.get('seq')}"
        seq, t = _req(d, "seq", where), _req(d, "t_host_ns", where)
        if not isinstance(seq, int) or not isinstance(t, int):
            raise SessionFormatError(f"{where}: seq/t_host_ns must be int")
        return cls(
            seq=seq,
            t_host_ns=t,
            source=_req(d, "source", where),
            joints_rad=_req(d, "joints_rad", where),
            tcp_xyzrpy=_req(d, "tcp_xyzrpy", where),
            read_latency_ns=_opt_num(_req(d, "read_latency_ns", where), where, True),
            error=_req(d, "error", where),
        )


@dataclass(frozen=True)
class EventRecord:
    t_host_ns: int
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"t_host_ns": self.t_host_ns, "type": self.type.value, "payload": self.payload}

    @classmethod
    def from_dict(cls, d: dict[str, Any], where: str = "events.jsonl") -> EventRecord:
        t = _req(d, "t_host_ns", where)
        if not isinstance(t, int):
            raise SessionFormatError(f"{where}: t_host_ns must be int")
        try:
            etype = EventType(_req(d, "type", where))
        except ValueError as exc:
            raise SessionFormatError(f"{where}: bad event type {d.get('type')!r}") from exc
        payload = d.get("payload") or {}
        if not isinstance(payload, dict):
            raise SessionFormatError(f"{where}: payload must be an object")
        return cls(t, etype, payload)
