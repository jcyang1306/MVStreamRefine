"""Atomic file writes and sha256 utilities (DATA_CONTRACTS.md §Serialization).

Atomic write = write `<name>.<random>.tmp` in the target directory -> flush -> fsync ->
`os.replace` onto the target -> (optionally) fsync the directory. A crash leaves either the
old file or the new file, never a partial one; stray `*.tmp` files are removed on failure.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from pathlib import Path
from typing import Any

from mvsr.core.codec import pretty_dumps

_CHUNK = 1 << 20


def sha256_bytes(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def fsync_dir(path: str | os.PathLike[str]) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(
    path: str | os.PathLike[str], data: bytes | memoryview, *, sync_dir: bool = True
) -> str:
    """Atomically replace `path` with `data`; returns the sha256 of `data`."""
    target = Path(path)
    tmp = target.with_name(f"{target.name}.{secrets.token_hex(4)}.tmp")
    digest = sha256_bytes(data)
    try:
        with open(tmp, "xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if sync_dir:
        fsync_dir(target.parent)
    return digest


def atomic_write_text(path: str | os.PathLike[str], text: str, *, sync_dir: bool = True) -> str:
    return atomic_write_bytes(path, text.encode("utf-8"), sync_dir=sync_dir)


def atomic_write_json(path: str | os.PathLike[str], obj: Any, *, sync_dir: bool = True) -> str:
    """Sorted-key, indented UTF-8 JSON (NaN rejected)."""
    return atomic_write_text(path, pretty_dumps(obj), sync_dir=sync_dir)
