"""Data-root and session path resolution (ADR-0006, SESSION_FORMAT.md)."""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT_ENV = "MVSR_DATA_ROOT"
DEFAULT_DATA_ROOT = "./data"


def data_root() -> Path:
    """`$MVSR_DATA_ROOT`, default `./data` relative to the working directory."""
    return Path(os.environ.get(DATA_ROOT_ENV) or DEFAULT_DATA_ROOT).expanduser().resolve()


def sessions_dir(root: Path | None = None) -> Path:
    return (root if root is not None else data_root()) / "sessions"


def resolve_session_dir(session: str | os.PathLike[str], root: Path | None = None) -> Path:
    """Accept a session directory path or a bare `session_id` under `<data_root>/sessions/`."""
    candidate = Path(session)
    if candidate.is_dir():
        return candidate.resolve()
    by_id = sessions_dir(root) / str(session)
    if by_id.is_dir():
        return by_id
    raise FileNotFoundError(f"session not found: {session} (looked in {sessions_dir(root)})")


def repo_root() -> Path:
    """Repository root for an editable/src-layout install (src/mvsr/paths.py -> repo)."""
    return Path(__file__).resolve().parents[2]
