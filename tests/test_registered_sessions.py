"""Deterministic checks on registered real sessions (fixtures/registry.yaml, ADR-0002).

A missing or checksum-mismatched fixture is an explicit SKIP here; the I-P0 gate itself is
BLOCKED while any required session is missing (subplans/00_FOUNDATION.md).
"""

from __future__ import annotations

from typing import Any

import pytest

from mvsr.core.errors import ChecksumMismatchError
from mvsr.io.registry import load_registry, resolve_fixture
from mvsr.io.verify import verify_session

ENTRIES = [e for e in load_registry() if "P0-gate" in e["used_by"]]


def _fixture(entry: dict[str, Any]) -> Any:
    try:
        return resolve_fixture(entry)
    except FileNotFoundError:
        pytest.skip(f"fixture {entry['session_id']} missing")
    except ChecksumMismatchError as exc:
        pytest.skip(f"fixture {entry['session_id']} checksum mismatch: {exc}")


@pytest.mark.skipif(
    not ENTRIES, reason="no P0-gate sessions registered yet (owner recordings pending)"
)
@pytest.mark.parametrize("entry", ENTRIES, ids=[e["session_id"] for e in ENTRIES])
def test_registered_session_verifies(entry: dict[str, Any]) -> None:
    rep = verify_session(_fixture(entry), p0_gate=True)
    assert rep.ok, rep.errors
