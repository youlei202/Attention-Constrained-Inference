# Attention-Constrained Inference (ACI)

Reproducible numerical experiments for **“Epistemic Throughput: Fundamental Limits of Attention-Constrained Inference.”**

The repository contains a compact numerical library, standalone experiment entrypoints, and plotting notebooks that read cached result tables. Experiments `00`–`05` cover the core benchmarks, while `06`–`09` provide exact calibration, sharp binary-information frontiers, shared-target accumulation, finite-pool convergence, stress audits, and complete paper-figure reproduction.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

Run the core experiments (each writes a CSV under `result/table/`):

```bash
python experiment/00_benchmark_theory_vs_sim.py
python experiment/01_theorem6_upper_bound_hits.py
python experiment/02_lemma4_enrichment_bound.py
```

The historical filenames are retained for compatibility.  New documentation and outputs use semantic names rather than theorem numbers where numbering may change.

## Numerical modules

The `acli.revision` package provides:

- entropy, binary KL, mutual information, and stable task-level PCG64DXSM randomness;
- exactly calibrated continuous and discrete posterior-score channels, including tie-correct AUC;
- exact two-branch screening frontiers and feasibility-clipped Pinsker comparisons;
- shared-target, component-target, and decoupled accumulation profiles;
- exact Poisson-binomial gain evaluation and transcript-identity stress audits;
- deterministic finite-`K` top-`B` precision and large-`K` top-tail limits;
- checkpointed experiment orchestration, plotting helpers, and output validation.

Standalone analysis experiments are numbered `06` through `09`. The suite runner discovers and executes them with deterministic task seeds and restart-safe checkpoints.

## Reproduce all paper figures

A small end-to-end check is:

```bash
bash scripts/run_revision_suite.sh smoke
bash scripts/execute_revision_notebook.sh
```

Run the complete paper configuration with restart-safe checkpoints:

```bash
bash scripts/run_revision_suite.sh paper
bash scripts/execute_revision_notebook.sh
```

The paper-figure notebook under `notebook/` only reads `result/table/*.csv` and performs lightweight deterministic formulas; the 50,000-channel stress test and large simulations run in the experiment suite, not in the notebook. Every paper figure is saved as vector PDF and at least 300-dpi PNG under `result/figure/paper/`.

## Result layout

```text
result/
├── table/       # experiment CSVs with seeds, full configuration, commit, and method
├── figure/      # notebook-produced PDF and PNG figures
└── artifact/    # executed notebook, HTML, validation manifests, and test report
```

Generated results, logs, checkpoints, environments, and delivery archives are ignored by Git. After a validated run, `scripts/package_revision_bundle.py` packages the outputs together with a source snapshot and writes a SHA-256 checksum beside the archive.

## Numerical conventions

Entropy, KL divergence, mutual information, and log-loss gain are measured in bits.  Finite-epsilon logistic posterior scores solve their intercept so `abs(E[eta] - p) <= 1e-10`.  Exact global-target gain uses the selected probabilities’ Poisson-binomial hit distribution; a homogeneous binomial proxy appears only when explicitly labeled as an approximation.
