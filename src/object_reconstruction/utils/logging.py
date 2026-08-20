"""Run logging: console plus a timestamped file under output/logs/ (Task 8)."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(output_root: str | Path, level: int = logging.INFO) -> Path:
    """Configure the root logger once per run; returns the log file path."""
    log_dir = Path(output_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    formatter = logging.Formatter(_FORMAT)
    for handler in (logging.StreamHandler(), logging.FileHandler(log_path)):
        handler.setFormatter(formatter)
        root.addHandler(handler)
    return log_path
