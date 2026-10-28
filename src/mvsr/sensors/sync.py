"""Robot pose at camera time (ADR-0004 §6, COORDINATES_AND_TIMING.md, P1 S4).

`T_base_tcp(t)` = rotation slerp + translation lerp between the two successful robot samples
bracketing `t_robot = t_cam + dt_robot_cam_ns`; valid only when the bracket is at most
`max_interp_gap_ns` wide. No extrapolation: outside the samples the status is `OUT_OF_RANGE`.
Failed robot reads are not samples, so they widen the bracket instead of being bridged silently.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from mvsr.core.errors import ContractValidationError
from mvsr.core.se3 import PoseSE3, interpolate_decoupled
from mvsr.core.types import PoseSyncInfo, RobotSample, SyncStatus


def interp_method(max_gap_ns: int, dt_robot_cam_ns: int) -> str:
    return f"decoupled_slerp_lerp(max_gap_ns={max_gap_ns},dt_robot_cam_ns={dt_robot_cam_ns})"


def nearest_method(dt_robot_cam_ns: int) -> str:
    return f"nearest(dt_robot_cam_ns={dt_robot_cam_ns})"


@dataclass(frozen=True, eq=False)
class RobotTrack:
    """Successful robot samples sorted by host time (stable on `seq` for equal times)."""

    samples: tuple[RobotSample, ...]
    times_ns: npt.NDArray[np.int64]

    @classmethod
    def from_samples(cls, samples: Iterable[RobotSample]) -> RobotTrack:
        s = tuple(sorted(samples, key=lambda x: (x.t_host_ns, x.seq)))
        times = np.array([x.t_host_ns for x in s], dtype=np.int64)
        times.flags.writeable = False
        return cls(s, times)

    def __len__(self) -> int:
        return len(self.samples)

    def pose_at(
        self, t_cam_ns: int, max_gap_ns: int, dt_robot_cam_ns: int = 0
    ) -> tuple[PoseSE3 | None, PoseSyncInfo]:
        if max_gap_ns <= 0:
            raise ContractValidationError("max_gap_ns must be > 0")
        method = interp_method(max_gap_ns, dt_robot_cam_ns)
        n = len(self.samples)
        if n == 0:
            return None, PoseSyncInfo(SyncStatus.NO_ROBOT, method, None, None, None, None)
        t = int(t_cam_ns) + int(dt_robot_cam_ns)
        times = self.times_ns
        i = int(np.searchsorted(times, t, side="right"))
        if i > 0 and int(times[i - 1]) == t:
            s = self.samples[i - 1]
            return s.T_base_tcp, PoseSyncInfo(SyncStatus.INTERPOLATED, method, 0, 0, s.seq, s.seq)
        if i == 0:
            after = self.samples[0]
            return None, PoseSyncInfo(
                SyncStatus.OUT_OF_RANGE, method, None, int(times[0]) - t, None, after.seq
            )
        if i == n:
            before = self.samples[-1]
            return None, PoseSyncInfo(
                SyncStatus.OUT_OF_RANGE, method, t - int(times[-1]), None, before.seq, None
            )
        a, b = self.samples[i - 1], self.samples[i]
        ta, tb = int(times[i - 1]), int(times[i])
        info_args = (method, t - ta, tb - t, a.seq, b.seq)
        if tb - ta > max_gap_ns:
            return None, PoseSyncInfo(SyncStatus.GAP_TOO_LARGE, *info_args)
        pose = interpolate_decoupled(a.T_base_tcp, b.T_base_tcp, (t - ta) / (tb - ta))
        return pose, PoseSyncInfo(SyncStatus.INTERPOLATED, *info_args)

    def nearest(
        self, t_cam_ns: int, max_dist_ns: int, dt_robot_cam_ns: int = 0
    ) -> tuple[PoseSE3 | None, PoseSyncInfo]:
        """Nearest sample within `max_dist_ns` (comparison baseline only; not used for priors)."""
        method = nearest_method(dt_robot_cam_ns)
        if not self.samples:
            return None, PoseSyncInfo(SyncStatus.NO_ROBOT, method, None, None, None, None)
        t = int(t_cam_ns) + int(dt_robot_cam_ns)
        k = int(np.argmin(np.abs(self.times_ns - t)))
        s = self.samples[k]
        dist = abs(int(self.times_ns[k]) - t)
        before, after = (dist, None) if int(self.times_ns[k]) <= t else (None, dist)
        status = SyncStatus.INTERPOLATED if dist <= max_dist_ns else SyncStatus.GAP_TOO_LARGE
        pose = s.T_base_tcp if status is SyncStatus.INTERPOLATED else None
        return pose, PoseSyncInfo(status, method, before, after, s.seq, s.seq)
