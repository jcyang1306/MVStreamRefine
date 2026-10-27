"""Fixture registry `fixtures/registry.yaml` (SESSION_FORMAT.md, ADR-0002)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mvsr.core.errors import ChecksumMismatchError, SessionFormatError
from mvsr.io.atomic import atomic_write_text, sha256_file
from mvsr.io.session_reader import load_manifest
from mvsr.io.session_writer import MANIFEST_NAME
from mvsr.io.verify import verify_session
from mvsr.paths import repo_root, sessions_dir

DEFAULT_REGISTRY = "fixtures/registry.yaml"
ENTRY_KEYS = ("session_id", "manifest_sha256", "cameras", "scenario", "gt", "object", "used_by")


def registry_path(path: str | Path | None = None) -> Path:
    return Path(path) if path is not None else repo_root() / DEFAULT_REGISTRY


def load_registry(path: str | Path | None = None) -> list[dict[str, Any]]:
    data = yaml.safe_load(registry_path(path).read_text(encoding="utf-8"))
    entries = data or []
    if not isinstance(entries, list):
        raise SessionFormatError("registry must be a YAML list")
    for e in entries:
        missing = [k for k in ENTRY_KEYS if k not in e]
        if missing:
            raise SessionFormatError(f"registry entry {e.get('session_id')} missing {missing}")
    return entries


def resolve_fixture(entry: dict[str, Any], root: Path | None = None) -> Path:
    """Session dir of a registry entry; raises FileNotFoundError / ChecksumMismatchError."""
    d = sessions_dir(root) / entry["session_id"]
    manifest = d / MANIFEST_NAME
    if not manifest.is_file():
        raise FileNotFoundError(f"fixture {entry['session_id']} missing under {d.parent}")
    actual = sha256_file(manifest)
    if actual != entry["manifest_sha256"]:
        raise ChecksumMismatchError(
            f"fixture {entry['session_id']}: manifest sha256 {actual} != registry"
        )
    return d


def register_session(
    session_dir: Path, *, used_by: list[str], gt: str | None = None, registry: Path | None = None
) -> dict[str, Any]:
    """Full `verify_session`, then add/replace the entry. Refuses sessions that do not verify."""
    report = verify_session(session_dir, check_images=True)
    if not report.ok:
        raise SessionFormatError(f"session does not verify; not registered: {report.errors[:5]}")
    m = load_manifest(session_dir)
    scenario = m.get("scenario") or {}
    entry = {
        "session_id": m["session_id"],
        "manifest_sha256": sha256_file(session_dir / MANIFEST_NAME),
        "cameras": sorted(m["cameras"]),
        "scenario": list(scenario.get("tags") or []),
        "gt": gt or scenario.get("gt_type"),
        "object": scenario.get("object"),
        "used_by": sorted(set(used_by)),
    }
    path = registry_path(registry)
    text = path.read_text(encoding="utf-8")
    header = "".join(line for line in text.splitlines(keepends=True) if line.startswith("#"))
    entries = [e for e in load_registry(path) if e["session_id"] != entry["session_id"]]
    entries.append(entry)
    entries.sort(key=lambda e: e["session_id"])
    atomic_write_text(path, header + yaml.safe_dump(entries, sort_keys=False, allow_unicode=True))
    return entry
