"""P3 session orchestration and the temporary in-memory keyframe window.

P4 will own durable keyframe selection. This bounded window exists only to make P3
keyframe-anchored from its first runnable slice; it never writes or mutates observations.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from mvsr.core.se3 import PoseSE3, so3_log
from mvsr.core.types import (
    Keyframe,
    KeyframeView,
    MapView,
    ObjectObservation,
    PoseEstimate,
    PoseStatus,
    SensorFrame,
    TrackingReferences,
)
from mvsr.tracking.config import KeyframeConfig
from mvsr.tracking.tracker import ObjectTracker


def _distance(a: PoseSE3, b: PoseSE3) -> tuple[float, float]:
    d = a.matrix @ np.linalg.inv(b.matrix)
    return float(np.linalg.norm(d[:3, 3])), float(np.degrees(np.linalg.norm(so3_log(d[:3, :3]))))


@dataclass(frozen=True)
class TrackingStep:
    pose: PoseEstimate
    keyframe_inserted: int | None
    local_keyframe_ids: tuple[int, ...]


class AnchorMemory(Protocol):
    @property
    def all(self) -> tuple[KeyframeView, ...]: ...

    def local(
        self,
        predicted: PoseSE3 | None,
        maximum: int,
        observation: ObjectObservation | None = None,
    ) -> tuple[KeyframeView, ...]: ...

    def consider(
        self, observation: ObjectObservation, pose: PoseEstimate
    ) -> Keyframe | int | None: ...

    def reset(self) -> None: ...


class AnchorWindow:
    def __init__(self, cfg: KeyframeConfig) -> None:
        self.cfg = cfg
        self._keyframes: list[KeyframeView] = []
        self._next_id = 0

    @property
    def all(self) -> tuple[KeyframeView, ...]:
        return tuple(self._keyframes)

    def reset(self) -> None:
        self._keyframes.clear()
        self._next_id = 0

    def local(
        self,
        predicted: PoseSE3 | None,
        maximum: int,
        observation: ObjectObservation | None = None,
    ) -> tuple[KeyframeView, ...]:
        if predicted is None:
            return tuple(self._keyframes[-maximum:])
        ranked = sorted(
            self._keyframes,
            key=lambda k: (*_distance(predicted, k.T_cam_obj), -k.keyframe_id),
        )
        chosen = ranked[:maximum]
        if self._keyframes and self._keyframes[-1] not in chosen:
            chosen[-1:] = [self._keyframes[-1]]
        return tuple(sorted(chosen, key=lambda k: k.keyframe_id))

    def should_insert(self, obs: ObjectObservation, pose: PoseEstimate) -> bool:
        if pose.status is not PoseStatus.VALID or pose.T_cam_obj is None:
            return False
        if not self._keyframes:
            return True
        last = self._keyframes[-1]
        gap = obs.header.frame_id - last.observation.header.frame_id
        overlap = float(pose.quality.get("overlap") or 0.0)
        dt, dr = min(
            (_distance(pose.T_cam_obj, k.T_cam_obj) for k in self._keyframes),
            key=lambda x: (x[0], x[1]),
        )
        return (
            gap >= self.cfg.max_frame_gap
            or overlap < self.cfg.min_anchor_overlap
            or dt >= self.cfg.max_anchor_translation_m
            or dr >= self.cfg.max_anchor_rotation_deg
        )

    def insert(self, obs: ObjectObservation, pose: PoseEstimate) -> int:
        if pose.T_cam_obj is None:
            raise ValueError("cannot insert an invalid pose")
        kid = self._next_id
        self._next_id += 1
        self._keyframes.append(KeyframeView(kid, obs, pose.T_cam_obj, pose.pose_version))
        if len(self._keyframes) > self.cfg.max_keyframes:
            # Keep the gauge keyframe and the most recent anchors. P4 replaces this policy.
            self._keyframes.pop(1)
        return kid

    def consider(self, observation: ObjectObservation, pose: PoseEstimate) -> int | None:
        if self.should_insert(observation, pose):
            return self.insert(observation, pose)
        return None


class TrackingSession:
    """Build `TrackingReferences`, call the tracker, and maintain bounded anchors."""

    def __init__(
        self,
        tracker: ObjectTracker,
        map_view: MapView | None = None,
        keyframe_memory: AnchorMemory | None = None,
    ) -> None:
        self.tracker = tracker
        self.window: AnchorMemory = keyframe_memory or AnchorWindow(tracker.cfg.keyframes)
        self.map_view = map_view
        self.previous: PoseEstimate | None = None

    def reset(self) -> None:
        self.tracker.reset()
        self.window.reset()
        self.previous = None

    def process(self, frame: SensorFrame, observation: ObjectObservation) -> TrackingStep:
        predicted = self.tracker.predict(frame, self.previous)
        selection_pose = predicted or (None if self.previous is None else self.previous.T_cam_obj)
        local = self.window.local(
            selection_pose,
            self.tracker.cfg.hypothesis.max_local_keyframes,
            observation,
        )
        refs = TrackingReferences(self.previous, predicted, local)
        pose = self.tracker.track(frame, observation, refs, self.map_view)
        considered = self.window.consider(observation, pose)
        inserted = considered.keyframe_id if isinstance(considered, Keyframe) else considered
        self.previous = pose
        return TrackingStep(pose, inserted, tuple(k.keyframe_id for k in local))


def pose_digest(poses: list[PoseEstimate]) -> str:
    h = hashlib.sha256()
    for p in poses:
        h.update(f"{p.header.frame_id}:{p.status.value}:{p.track_state.value}:{p.source}".encode())
        if p.T_cam_obj is not None:
            h.update(p.T_cam_obj.matrix.tobytes())
        for reason in p.reason_codes:
            h.update(reason.value.encode())
    return h.hexdigest()
