"""`SensorSource` port (DATA_CONTRACTS.md §Public ports) and its end-of-stream marker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mvsr.core.types import SensorFrame


@dataclass(frozen=True)
class EndOfStream:
    """Returned by `SensorSource.next()` once no frame follows; `reason` says why.

    `not_written`: frame_ids without pixels (dropped / incomplete); `not_emitted`: frames with
    pixels that were not emitted (colour/depth from different exposures, or not selected).
    """

    session_id: str
    camera_id: str
    frames_emitted: int
    not_written: int
    not_emitted: int
    reason: str


class SensorSource(Protocol):
    def next(self) -> SensorFrame | EndOfStream: ...

    def close(self) -> None: ...
