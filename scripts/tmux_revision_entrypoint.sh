#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg

NCPU=$(nproc)
WORKERS=$((NCPU - 8))
if [[ "$WORKERS" -gt 112 ]]; then WORKERS=112; fi
if [[ "$WORKERS" -lt 1 ]]; then WORKERS=1; fi
export ACI_WORKERS="$WORKERS"

mkdir -p environment logs state result/artifact
PHASE=preflight
on_exit() {
  local exit_code=$?
  if [[ "$exit_code" -ne 0 ]]; then
    python scripts/update_revision_status.py \
      --phase "$PHASE" --pipeline-status failed --exit-code "$exit_code" || true
  fi
  exit "$exit_code"
}
trap on_exit EXIT
python scripts/update_revision_status.py --phase "$PHASE" --pipeline-status running

DIRTY_SOURCE=$(git status --porcelain --untracked-files=all)
if [[ -n "$DIRTY_SOURCE" ]]; then
  echo "Refusing formal execution from a dirty source tree:" >&2
  echo "$DIRTY_SOURCE" >&2
  exit 1
fi
BRANCH=$(git branch --show-current)
if [[ "$BRANCH" != "major-revision-reproducibility" ]]; then
  echo "Refusing formal execution on branch '$BRANCH'" >&2
  exit 1
fi
git rev-parse HEAD > environment/run_commit.txt
git rev-parse HEAD > environment/final_commit.txt
python --version > environment/python_formal.txt 2>&1
python -m pip freeze > environment/pip_freeze.txt

PHASE=pytest
python scripts/update_revision_status.py --phase "$PHASE" --pipeline-status running
pytest -q --junitxml=result/artifact/pytest.xml
PHASE=legacy_regressions
python scripts/update_revision_status.py --phase "$PHASE" --pipeline-status running
bash scripts/run_revision_regressions.sh
PHASE=paper_experiments
python scripts/update_revision_status.py --phase "$PHASE" --pipeline-status running
python experiment/10_run_major_revision_suite.py --profile paper --workers "$WORKERS" --resume
PHASE=notebook
python scripts/update_revision_status.py --phase "$PHASE" --pipeline-status running
bash scripts/execute_revision_notebook.sh
PHASE=validation
python scripts/update_revision_status.py --phase "$PHASE" --pipeline-status running
python scripts/validate_revision_outputs.py
PHASE=packaging
python scripts/update_revision_status.py --phase "$PHASE" --pipeline-status running
PHASE=complete
python scripts/update_revision_status.py --phase "$PHASE" --pipeline-status completed
python scripts/package_revision_bundle.py
trap - EXIT
