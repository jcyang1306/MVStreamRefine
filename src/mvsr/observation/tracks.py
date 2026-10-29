"""Long-lived feature tracks (P2 O4).

Per frame, in order:
1. Every live track is propagated from the previous frame with pyramidal KLT and checked
   forward-backward (`fb_max_px`). A propagated point inside the safe foreground is
   `TRACKED` (confidence `1 - fb / fb_max_px`); otherwise it is `OCCLUDED` and keeps its last
   position. Invisible for more than `max_occluded_frames` frames -> `RETIRED`.
2. New ORB keypoints spawn `NEW` tracks until `target_tracks` are visible, at least
   `min_spacing_px` away from visible tracks.
3. Visible points get a descriptor at their current position; a point that cannot be
   described (too close to the border) is retired.
Track ids are never reused. Occluded points are not observations (no pixel this frame) and
are reported only in the per-frame `FrameTracks`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from mvsr.observation.config import TrackConfig
from mvsr.observation.features import FeatureExtractor

F32 = npt.NDArray[np.float32]
I64 = npt.NDArray[np.int64]


class TrackStatus(str, Enum):
    NEW = "NEW"
    TRACKED = "TRACKED"
    OCCLUDED = "OCCLUDED"
    RETIRED = "RETIRED"


@dataclass
class _Track:
    uv: npt.NDArray[np.float32]
    first_frame: int
    last_seen: int
    occluded: int = 0
    birth_descriptor: np.ndarray | None = None


def hamming_bits(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.unpackbits(np.bitwise_xor(a, b)).sum())


@dataclass(frozen=True, eq=False)
class FrameTracks:
    """Visible points of one frame (aligned arrays) plus occlusion/retirement events."""

    frame_id: int
    uv: F32
    track_ids: I64
    status: tuple[TrackStatus, ...]
    confidence: F32
    descriptors: np.ndarray
    occluded_ids: tuple[int, ...] = field(default_factory=tuple)
    retired_ids: tuple[int, ...] = field(default_factory=tuple)

    def counts(self) -> dict[str, int]:
        return {
            "NEW": sum(s is TrackStatus.NEW for s in self.status),
            "TRACKED": sum(s is TrackStatus.TRACKED for s in self.status),
            "OCCLUDED": len(self.occluded_ids),
            "RETIRED": len(self.retired_ids),
        }


class TrackManager:
    def __init__(self, cfg: TrackConfig, extractor: FeatureExtractor) -> None:
        self.cfg = cfg
        self.extractor = extractor
        self._prev: npt.NDArray[np.uint8] | None = None
        self._tracks: dict[int, _Track] = {}
        self._next_id = 0
        self._win = (cfg.klt_window_px, cfg.klt_window_px)
        self._crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

    def _lk(
        self, a: npt.NDArray[np.uint8], b: npt.NDArray[np.uint8], p: npt.NDArray[np.float32]
    ) -> tuple[Any, Any]:
        cv: Any = cv2  # the OpenCV stubs do not accept nextPts=None
        nxt, st, _ = cv.calcOpticalFlowPyrLK(
            a, b, p, None, winSize=self._win, maxLevel=self.cfg.klt_levels, criteria=self._crit
        )
        return nxt, st

    def _propagate(self, gray: npt.NDArray[np.uint8]) -> tuple[list[int], F32, F32]:
        ids = sorted(self._tracks)
        if self._prev is None or not ids:
            return [], np.zeros((0, 2), np.float32), np.zeros(0, np.float32)
        p0 = np.array([self._tracks[i].uv for i in ids], dtype=np.float32).reshape(-1, 1, 2)
        p1, st = self._lk(self._prev, gray, p0)
        pb, stb = self._lk(gray, self._prev, np.asarray(p1, dtype=np.float32))
        fb = np.linalg.norm(p0.reshape(-1, 2) - pb.reshape(-1, 2), axis=1)
        ok = (st.reshape(-1) == 1) & (stb.reshape(-1) == 1) & (fb <= self.cfg.fb_max_px)
        fb = np.where(ok, fb, np.inf).astype(np.float32)
        return ids, p1.reshape(-1, 2).astype(np.float32), fb

    def update(
        self, frame_id: int, gray: npt.NDArray[np.uint8], safe: npt.NDArray[np.bool_]
    ) -> FrameTracks:
        h, w = gray.shape
        ids, p1, fb = self._propagate(gray)
        vis_ids: list[int] = []
        vis_uv: list[npt.NDArray[np.float32]] = []
        vis_status: list[TrackStatus] = []
        vis_conf: list[float] = []
        occluded: list[int] = []
        retired: list[int] = []
        for k, tid in enumerate(ids):
            tr = self._tracks[tid]
            x, y = float(p1[k, 0]), float(p1[k, 1])
            xi, yi = round(x), round(y)
            inside = np.isfinite(fb[k]) and 0 <= xi < w and 0 <= yi < h and bool(safe[yi, xi])
            if inside:
                tr.uv = p1[k].copy()
                tr.occluded = 0
                tr.last_seen = frame_id
                vis_ids.append(tid)
                vis_uv.append(tr.uv)
                vis_status.append(TrackStatus.TRACKED)
                vis_conf.append(max(0.0, 1.0 - float(fb[k]) / self.cfg.fb_max_px))
            else:
                tr.occluded += 1
                if tr.occluded > self.cfg.max_occluded_frames:
                    retired.append(tid)
                else:
                    occluded.append(tid)
        for tid in retired:
            del self._tracks[tid]
        need = self.cfg.target_tracks - len(vis_ids)
        if need > 0 and safe.any():
            cand, _ = self.extractor.detect(gray, safe)
            taken = np.array(vis_uv, dtype=np.float32).reshape(-1, 2)
            sp2 = float(self.cfg.min_spacing_px) ** 2
            for uv in cand:
                if need <= 0:
                    break
                if taken.size and float(np.min(np.sum((taken - uv) ** 2, axis=1))) < sp2:
                    continue
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = _Track(uv=uv.copy(), first_frame=frame_id, last_seen=frame_id)
                vis_ids.append(tid)
                vis_uv.append(uv.copy())
                vis_status.append(TrackStatus.NEW)
                vis_conf.append(1.0)
                taken = np.vstack([taken, uv[None, :]])
                need -= 1
        uv_arr = np.array(vis_uv, dtype=np.float32).reshape(-1, 2)
        keep, desc = self.extractor.describe(gray, uv_arr)
        limit = self.cfg.max_descriptor_drift_bits
        ok_rows = []
        for row, k in enumerate(keep.tolist()):
            tr = self._tracks[vis_ids[k]]
            if tr.birth_descriptor is None:
                tr.birth_descriptor = desc[row].copy()
            elif limit is not None and hamming_bits(desc[row], tr.birth_descriptor) > limit:
                continue
            ok_rows.append(row)
        rows = np.asarray(ok_rows, dtype=np.intp)
        keep, desc = keep[rows], desc[rows]
        dropped = sorted(set(range(len(vis_ids))) - set(keep.tolist()))
        for k in dropped:
            tid = vis_ids[k]
            self._tracks.pop(tid, None)
            retired.append(tid)
        self._prev = gray.copy()
        ids_arr = np.array([vis_ids[k] for k in keep], dtype=np.int64)
        return FrameTracks(
            frame_id=frame_id,
            uv=uv_arr[keep] if keep.size else np.zeros((0, 2), np.float32),
            track_ids=ids_arr,
            status=tuple(vis_status[k] for k in keep),
            confidence=np.array([vis_conf[k] for k in keep], dtype=np.float32),
            descriptors=desc,
            occluded_ids=tuple(sorted(occluded)),
            retired_ids=tuple(sorted(retired)),
        )

    def reset(self) -> None:
        self._prev = None
        self._tracks.clear()
