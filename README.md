# Attention-Constrained Inference (ACI)

Reproducible numerical experiments for **“Epistemic Throughput: Fundamental Limits of Attention-Constrained Inference.”**

The repository keeps a small core library, standalone experiment entrypoints, and plotting notebooks that read cached result tables.  The original experiments `00`–`05` remain available; the major-revision pipeline adds exact calibration, sharp binary-information frontiers, shared-target accumulation, finite-pool convergence, stress audits, and complete paper-figure reproduction.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

Run the original core experiments (each writes a CSV under `result/table/`):

```bash
python experiment/00_benchmark_theory_vs_sim.py
python experiment/01_theorem6_upper_bound_hits.py
python experiment/02_lemma4_enrichment_bound.py
```

The historical filenames are retained for compatibility.  New documentation and outputs use semantic names rather than theorem numbers where numbering may change.

## Major-revision numerical modules

The `acli.revision` package provides:

- entropy, binary KL, mutual information, and stable task-level PCG64DXSM randomness;
- exactly calibrated continuous and discrete posterior-score channels, including tie-correct AUC;
- exact two-branch screening frontiers and feasibility-clipped Pinsker comparisons;
- shared-target, component-target, and decoupled accumulation profiles;
- exact Poisson-binomial gain evaluation and transcript-identity stress audits;
- deterministic finite-`K` top-`B` precision and large-`K` top-tail limits;
- checkpointed experiment orchestration, plotting helpers, and output validation.

Standalone major-revision experiments are `experiment/06_*.py` through `experiment/09_*.py`.  The orchestrator is `experiment/10_run_major_revision_suite.py`.

## Reproduce all paper figures

A small end-to-end check is:

```bash
bash scripts/run_revision_suite.sh smoke
bash scripts/execute_revision_notebook.sh
```

The formal CPU reproduction uses the paper profile and restart-safe checkpoints:

```bash
tmux new-session -d -s aci_major_revision_integration \
  "cd /work/Users/leiyo/GitHub/Attention-Constrained-Inference && bash scripts/tmux_revision_entrypoint.sh 2>&1 | tee -a logs/aci_major_revision_integration.log"
```

Monitor it with:

```bash
bash scripts/status_revision.sh
```

The paper notebook is `notebook/06_major_revision_paper_figures.ipynb`.  It only reads `result/table/*.csv` and performs lightweight deterministic formulas; the 50,000-channel stress test and large simulations run in the experiment suite, not in the notebook.  Every paper figure is saved as vector PDF and at least 300-dpi PNG under `result/figure/paper/`.

## Result layout

```text
result/
├── table/       # experiment CSVs with seeds, full configuration, commit, and method
├── figure/      # notebook-produced PDF and PNG figures
└── artifact/    # executed notebook, HTML, validation manifests, and test report
```

Generated results, logs, checkpoints, environments, and delivery archives are ignored by Git.  `scripts/package_revision_bundle.py` packages the validated outputs plus a source snapshot into `ACI_MAJOR_REVISION_REPRODUCIBILITY_BUNDLE.zip` and writes its SHA-256 checksum beside it.

## Numerical conventions

Entropy, KL divergence, mutual information, and log-loss gain are measured in bits.  Finite-epsilon logistic posterior scores solve their intercept so `abs(E[eta] - p) <= 1e-10`.  Exact global-target gain uses the selected probabilities’ Poisson-binomial hit distribution; a homogeneous binomial proxy appears only when explicitly labeled as an approximation.
