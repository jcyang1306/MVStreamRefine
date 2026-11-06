"""Short P4 checks on registered real sessions (`used_by: P4`)."""

from __future__ import annotations

from typing import Any

import pytest

from mvsr.core.errors import ChecksumMismatchError
from mvsr.io.registry import load_registry, resolve_fixture
from mvsr.keyframes.benchmark import run_benchmark
from mvsr.keyframes.config import load_keyframe_config
from mvsr.observation.config import load_observation_config
from mvsr.observation.masks import list_revisions
from mvsr.tracking.config import load_tracking_config

ENTRIES = [entry for entry in load_registry() if "P4" in entry["used_by"]]


def _fixture_and_revision(entry: dict[str, Any]) -> tuple[Any, str]:
    try:
        directory = resolve_fixture(entry)
    except FileNotFoundError:
        pytest.skip(f"fixture {entry['session_id']} missing")
    except ChecksumMismatchError as exc:
        pytest.skip(f"fixture {entry['session_id']} checksum mismatch: {exc}")
    revisions = list_revisions(directory)
    if not revisions:
        pytest.skip(f"{entry['session_id']}: no mask revision")
    return directory, revisions[-1]


@pytest.mark.skipif(not ENTRIES, reason="no P4 sessions registered")
@pytest.mark.parametrize("entry", ENTRIES, ids=[entry["session_id"] for entry in ENTRIES])
def test_short_keyframe_memory_is_reproducible(entry: dict[str, Any]) -> None:
    directory, revision = _fixture_and_revision(entry)
    report, manager = run_benchmark(
        directory,
        revision,
        load_keyframe_config(),
        load_tracking_config(),
        load_observation_config(),
        limit=20,
    )
    assert report["reproducibility"]["identical"]
    assert 0 < len(manager.keyframes) <= load_keyframe_config().gate.max_keyframes
    assert report["immutable_observations"]
