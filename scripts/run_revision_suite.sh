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

PROFILE=${1:-smoke}
shift || true
python experiment/10_run_major_revision_suite.py --profile "$PROFILE" --workers "$WORKERS" "$@"
