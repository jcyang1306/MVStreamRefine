"""I1 checks on registered real sessions (`used_by: I1`) that have a complete mask revision.

Missing fixtures or masks are explicit SKIPs here; `mvsr i1-gate` reports them as BLOCKED.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from mvsr.core.errors import ChecksumMismatchError
from mvsr.io.registry import load_registry, resolve_fixture
from mvsr.observation.benchmark import iter_observations, observation_digest, run_benchmark
from mvsr.observation.config import load_observation_config
from mvsr.observation.masks import list_revisions

ENTRIES = [e for e in load_registry() if "I1" in e["used_by"]]


def _fixture_and_revision(entry: dict[str, Any]) -> tuple[Any, str]:
    try:
        d = resolve_fixture(entry)
    except FileNotFoundError:
        pytest.skip(f"fixture {entry['session_id']} missing")
    except ChecksumMismatchError as exc:
        pytest.skip(f"fixture {entry['session_id']} checksum mismatch: {exc}")
    revs = list_revisions(d)
    if not revs:
        pytest.skip(f"{entry['session_id']}: no mask revision (run mvsr segment)")
    return d, revs[-1]


@pytest.mark.skipif(not ENTRIES, reason="no I1 sessions registered")
@pytest.mark.parametrize("entry", ENTRIES, ids=[e["session_id"] for e in ENTRIES])
def test_observations_are_reproducible(entry: dict[str, Any]) -> None:
    d, rev = _fixture_and_revision(entry)
    cfg = load_observation_config()
    a = [observation_digest(o) for _, o, _ in iter_observations(d, rev, cfg, limit=15)]
    b = [observation_digest(o) for _, o, _ in iter_observations(d, rev, cfg, limit=15)]
    assert a == b and len(a) == 15


@pytest.mark.skipif(not ENTRIES, reason="no I1 sessions registered")
@pytest.mark.parametrize("entry", ENTRIES, ids=[e["session_id"] for e in ENTRIES])
def test_short_benchmark_has_tracks_and_consistent_masks(entry: dict[str, Any]) -> None:
    d, rev = _fixture_and_revision(entry)
    cfg = load_observation_config()
    cfg = replace(
        cfg,
        benchmark=replace(
            cfg.benchmark, frame_gaps=[1, 5], pair_stride=5, reproducibility_frames=5
        ),
    )
    rep = run_benchmark(d, rev, cfg, limit=60)
    assert rep["reproducibility"]["identical"]
    assert rep["tracks"]["count"] > 0
    if rep["reference_available"]:
        agreement = rep["pairs"]["gap_1"]["mask_agreement"]["median"]
        assert agreement > 0.5, agreement
