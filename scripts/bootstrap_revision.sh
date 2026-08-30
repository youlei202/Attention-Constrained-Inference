#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

mkdir -p environment logs state result/table result/figure result/artifact

if [[ ! -f environment/base_commit.txt ]]; then
  git rev-parse HEAD > environment/base_commit.txt
fi
if [[ ! -f environment/base_status.txt ]]; then
  git status --short > environment/base_status.txt
fi
python3 --version > environment/python_before.txt 2>&1
nproc > environment/nproc.txt
free -h > environment/memory.txt
uname -a > environment/system.txt

if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install -e .
.venv/bin/python --version > environment/python_after.txt 2>&1
.venv/bin/python -m pip freeze > environment/pip_freeze.txt

echo "Revision environment ready at $REPO_ROOT/.venv"
