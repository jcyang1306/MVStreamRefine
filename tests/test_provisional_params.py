"""Provisional parameter registry: config agreement (hard) and recompute reminders (warning)."""

from __future__ import annotations

import warnings

from mvsr.tuning.registry import REGISTRY, load_params, reminder_lines, status


def test_registry_values_match_configs() -> None:
    rows = status()
    assert rows, f"{REGISTRY} is empty"
    bad = [(r["key"], r["value"], r["config_value"]) for r in rows if r["config_mismatch"]]
    assert not bad, f"{REGISTRY} disagrees with the config files: {bad}"


def test_registry_keys_unique() -> None:
    keys = [e["key"] for e in load_params()]
    assert len(keys) == len(set(keys))


def test_recheck_reminder() -> None:
    for line in reminder_lines(status()):
        warnings.warn(line, UserWarning, stacklevel=1)
