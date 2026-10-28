from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[1]
REFERENCE_HANDEYE = REPO / "configs" / "handeye" / "handeye_tf_20261028.txt"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-hardware", action="store_true", help="run @pytest.mark.hardware tests on the rig"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-hardware"):
        return
    skip = pytest.mark.skip(reason="hardware test: needs the rig and --run-hardware (rig service)")
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def reference_handeye_path() -> Path:
    return REFERENCE_HANDEYE
