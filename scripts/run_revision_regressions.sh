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

python experiment/00_benchmark_theory_vs_sim.py
python experiment/01_theorem6_upper_bound_hits.py
python experiment/02_lemma4_enrichment_bound.py
python experiment/03_theorem10_achievability_weak_screening.py --smoke
python experiment/04_tail_leverage_gaussian_vs_pareto.py --smoke
python experiment/05_finite_length_simulation_validation.py --profile smoke --workers 1

python - <<'PY'
from pathlib import Path
from acli.revision.suite import atomic_write_json, git_commit

root = Path.cwd()
atomic_write_json(
    root / "result/artifact/legacy_regressions.json",
    {
        "status": "passed",
        "git_commit": git_commit(root),
        "commands": [
            "experiment/00_benchmark_theory_vs_sim.py",
            "experiment/01_theorem6_upper_bound_hits.py",
            "experiment/02_lemma4_enrichment_bound.py",
            "experiment/03_theorem10_achievability_weak_screening.py --smoke",
            "experiment/04_tail_leverage_gaussian_vs_pareto.py --smoke",
            "experiment/05_finite_length_simulation_validation.py --profile smoke --workers 1",
        ],
    },
)
PY
