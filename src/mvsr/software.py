"""Software provenance recorded in manifests and reports (package version + git state)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from mvsr import __version__
from mvsr.paths import repo_root


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def software_info() -> dict[str, Any]:
    """`{"package", "version", "git_hash", "git_dirty"}`; git fields are None outside a checkout."""
    root = repo_root()
    git_hash = _git(["rev-parse", "HEAD"], root)
    dirty: bool | None = None
    if git_hash is not None:
        status = _git(["status", "--porcelain", "--untracked-files=no"], root)
        dirty = None if status is None else bool(status)
    return {"package": "mvsr", "version": __version__, "git_hash": git_hash, "git_dirty": dirty}


def version_string() -> str:
    info = software_info()
    if info["git_hash"] is None:
        return f"mvsr {__version__} (git: unavailable)"
    suffix = "-dirty" if info["git_dirty"] else ""
    return f"mvsr {__version__} (git {info['git_hash'][:12]}{suffix})"
