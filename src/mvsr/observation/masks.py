"""Offline mask cache (P2 O1; SESSION_FORMAT.md `derived/masks/<camera_id>/<mask_revision>/`).

Layout of one revision directory:
- `<frame_id:06d>.png`: 0/255 mask of the object (one instance);
- `masks.jsonl`: one `MaskRecord` per segmented frame (status, area, score, bbox, png sha256);
- `revision.json`: producer, parameters, prompt, checkpoint and input-session checksums, the
  sha256 of `masks.jsonl`. It is written last, so a directory without it is an interrupted
  run and is never read.

`mask_revision = <producer>@<first 12 hex of sha256(canonical parameters)>`: the same producer,
checkpoint, prompt and session give the same revision id; a revision is immutable once
complete (a second writer raises `FileExistsError`). Raw session files are never touched.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from mvsr.core import codec
from mvsr.core.codec import SCHEMA_VERSION
from mvsr.core.errors import ChecksumMismatchError, SessionFormatError
from mvsr.core.types import MASK_REVISION_RE
from mvsr.io import images
from mvsr.io.atomic import atomic_write_bytes, atomic_write_json, sha256_bytes, sha256_file
from mvsr.io.records import dumps_line
from mvsr.io.session_reader import read_jsonl

REVISION_FILE = "revision.json"
RECORDS_FILE = "masks.jsonl"


class MaskStatus(str, Enum):
    OK = "OK"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    EMPTY = "EMPTY"


@dataclass(frozen=True)
class MaskRecord:
    frame_id: int
    status: MaskStatus
    area_px: int
    score: float | None
    bbox_xyxy: tuple[int, int, int, int] | None
    path: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "status": self.status.value,
            "area_px": self.area_px,
            "score": self.score,
            "bbox_xyxy": None if self.bbox_xyxy is None else list(self.bbox_xyxy),
            "path": self.path,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MaskRecord:
        try:
            bbox = d["bbox_xyxy"]
            return cls(
                frame_id=int(d["frame_id"]),
                status=MaskStatus(d["status"]),
                area_px=int(d["area_px"]),
                score=None if d["score"] is None else float(d["score"]),
                bbox_xyxy=None
                if bbox is None
                else (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                path=str(d["path"]),
                sha256=str(d["sha256"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SessionFormatError(f"{RECORDS_FILE}: bad record {d!r}: {exc}") from exc


def mask_revision_id(producer: str, params: dict[str, Any]) -> str:
    digest = hashlib.sha256(codec.canonical_dumps(params).encode("utf-8")).hexdigest()[:12]
    rev = f"{producer}@{digest}"
    if not MASK_REVISION_RE.match(rev):
        raise ValueError(f"invalid mask revision {rev!r}")
    return rev


def revisions_dir(session_dir: str | Path, camera_id: str) -> Path:
    return Path(session_dir) / "derived" / "masks" / camera_id


def list_revisions(session_dir: str | Path, camera_id: str = "wrist") -> list[str]:
    """Complete revisions (with `revision.json`), sorted by name."""
    base = revisions_dir(session_dir, camera_id)
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if (p / REVISION_FILE).is_file())


def mask_bbox(mask: npt.NDArray[np.bool_]) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


class MaskCacheWriter:
    """Writes one revision; `finalize()` makes it visible to readers."""

    def __init__(
        self, session_dir: str | Path, camera_id: str, revision: str, meta: dict[str, Any]
    ) -> None:
        if not MASK_REVISION_RE.match(revision):
            raise ValueError(f"invalid mask revision {revision!r}")
        self.dir = revisions_dir(session_dir, camera_id) / revision
        if (self.dir / REVISION_FILE).exists():
            raise FileExistsError(f"mask revision {revision} already exists: {self.dir}")
        if self.dir.exists():
            shutil.rmtree(self.dir)  # interrupted earlier run: never readable, safe to replace
        self.dir.mkdir(parents=True)
        self.camera_id = camera_id
        self.revision = revision
        self.meta = dict(meta)
        self.records: dict[int, MaskRecord] = {}

    def add(
        self, frame_id: int, mask: npt.NDArray[np.bool_], status: MaskStatus, score: float | None
    ) -> MaskRecord:
        if frame_id in self.records:
            raise SessionFormatError(f"mask for frame {frame_id} written twice")
        rel = f"{frame_id:06d}.png"
        digest = atomic_write_bytes(self.dir / rel, images.encode_png_mask(mask), sync_dir=False)
        rec = MaskRecord(
            frame_id=int(frame_id),
            status=status,
            area_px=int(mask.sum()),
            score=None if score is None else round(float(score), 6),
            bbox_xyxy=mask_bbox(mask),
            path=rel,
            sha256=digest,
        )
        self.records[frame_id] = rec
        return rec

    def finalize(self, **extra: Any) -> dict[str, Any]:
        lines = "".join(dumps_line(self.records[k].to_dict()) + "\n" for k in sorted(self.records))
        jsonl_sha = atomic_write_bytes(self.dir / RECORDS_FILE, lines.encode("utf-8"))
        counts: dict[str, int] = {}
        for r in self.records.values():
            counts[r.status.value] = counts.get(r.status.value, 0) + 1
        revision = {
            "schema_version": SCHEMA_VERSION,
            "type": "MaskRevision",
            "mask_revision": self.revision,
            "camera_id": self.camera_id,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "frames": len(self.records),
            "by_status": dict(sorted(counts.items())),
            "records_sha256": jsonl_sha,
            **self.meta,
            **extra,
        }
        atomic_write_json(self.dir / REVISION_FILE, revision)
        return revision


class MaskCache:
    """Read-only access to one complete revision (checksums verified)."""

    def __init__(self, session_dir: str | Path, revision: str, camera_id: str = "wrist") -> None:
        self.dir = revisions_dir(session_dir, camera_id) / revision
        rev_path = self.dir / REVISION_FILE
        if not rev_path.is_file():
            raise SessionFormatError(
                f"mask revision {revision} not found or incomplete under {self.dir.parent}"
            )
        self.revision_meta: dict[str, Any] = _load_json(rev_path)
        if self.revision_meta.get("schema_version") != SCHEMA_VERSION:
            raise SessionFormatError(f"{rev_path}: unsupported schema_version")
        if self.revision_meta.get("mask_revision") != revision:
            raise SessionFormatError(f"{rev_path}: mask_revision field != directory name")
        actual = sha256_file(self.dir / RECORDS_FILE)
        if actual != self.revision_meta.get("records_sha256"):
            raise ChecksumMismatchError(f"{RECORDS_FILE}: sha256 {actual} != revision.json")
        self.revision = revision
        self.camera_id = camera_id
        self.records = {
            r.frame_id: r
            for r in (MaskRecord.from_dict(d) for d in read_jsonl(self.dir / RECORDS_FILE))
        }

    def __contains__(self, frame_id: int) -> bool:
        return frame_id in self.records

    def frame_ids(self) -> list[int]:
        return sorted(self.records)

    def mask(self, frame_id: int, *, verify: bool = True) -> npt.NDArray[np.bool_]:
        try:
            rec = self.records[frame_id]
        except KeyError as exc:
            raise SessionFormatError(
                f"mask revision {self.revision} has no frame {frame_id}"
            ) from exc
        data = (self.dir / rec.path).read_bytes()
        if verify and sha256_bytes(data) != rec.sha256:
            raise ChecksumMismatchError(f"{rec.path}: sha256 differs from {RECORDS_FILE}")
        m = images.decode_png_mask(data)
        m.flags.writeable = False
        return m


def _load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SessionFormatError(f"{path}: expected a JSON object")
    return obj
