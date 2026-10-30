"""Provisional parameter registry `configs/provisional_params.yaml` (owner decision 2026-10-30).

Each entry: `key` (`<config>.<section>.<field>[.<sub>]`, config = sensors | observation),
`value` (must equal the config file), `derived_on`, `basis` (session ids), `rule`, `evidence`
(run directory), `recheck_when` {`scenarios`: tags, `min_new`: n}. An entry is DUE when at
least `min_new` registered sessions (fixtures/registry.yaml) with one of the tags are not in
its basis. `mvsr params-status`, `mvsr register-session` and the test suite report DUE
entries; a value differing from its config file is an error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mvsr.io.registry import load_registry
from mvsr.paths import repo_root

REGISTRY = "configs/provisional_params.yaml"
CONFIG_FILES = {"sensors": "configs/sensors.yaml", "observation": "configs/observation.yaml"}


def load_params(path: str | Path | None = None) -> list[dict[str, Any]]:
    p = Path(path) if path is not None else repo_root() / REGISTRY
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or []
    if not isinstance(data, list):
        raise ValueError(f"{p}: expected a list")
    for e in data:
        for k in ("key", "value", "derived_on", "basis", "rule", "recheck_when"):
            if k not in e:
                raise ValueError(f"{p}: entry {e.get('key')} missing {k}")
    return data


def config_value(key: str) -> Any:
    head, *rest = key.split(".")
    raw = yaml.safe_load((repo_root() / CONFIG_FILES[head]).read_text(encoding="utf-8"))
    node: Any = raw
    for part in rest:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"{key}: '{part}' not found in {CONFIG_FILES[head]}")
        node = node[part]
    return node


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(a)))
    return bool(a == b)


def status(
    params: list[dict[str, Any]] | None = None, registry: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    params = load_params() if params is None else params
    sessions = load_registry() if registry is None else registry
    out = []
    for e in params:
        cond = e["recheck_when"] or {}
        tags = set(cond.get("scenarios") or [])
        basis = set(e["basis"] or [])
        new = sorted(
            s["session_id"]
            for s in sessions
            if s["session_id"] not in basis and tags & set(s["scenario"])
        )
        try:
            current = config_value(e["key"])
            mismatch = not _same(current, e["value"])
        except KeyError:
            current, mismatch = None, True
        out.append(
            {
                "key": e["key"],
                "value": e["value"],
                "config_value": current,
                "config_mismatch": mismatch,
                "new_sessions": new,
                "due": len(new) >= int(cond.get("min_new", 1)),
            }
        )
    return out


def reminder_lines(rows: list[dict[str, Any]]) -> list[str]:
    due = [r for r in rows if r["due"]]
    if not due:
        return []
    keys = ", ".join(r["key"] for r in due)
    sessions = sorted({s for r in due for s in r["new_sessions"]})
    return [
        f"RECHECK provisional parameters: {keys}",
        f"  new data: {', '.join(sessions)}; recompute with `mvsr derive-params` and update {REGISTRY}",
    ]
