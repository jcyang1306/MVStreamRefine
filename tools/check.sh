#!/usr/bin/env bash
# Format / lint / type / test gate for the dev container:
#   docker compose run --rm dev tools/check.sh [extra pytest args]
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== ruff format --check"
ruff format --check src tests
echo "== ruff check"
ruff check src tests
echo "== mypy --strict mvsr.core"
mypy --strict src/mvsr/core
echo "== mypy mvsr"
mypy src/mvsr
echo "== pytest"
pytest -q "$@"
