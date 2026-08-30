# Major-Revision Integration Report

## Repository and Git provenance

- Repository: `/work/Users/leiyo/GitHub/Attention-Constrained-Inference`
- Branch: `major-revision-reproducibility`
- Base commit: `20582350a85faf752f06b1fe69dafd8735e775f7`
- Final commit: recorded after the final source commit in `environment/final_commit.txt`
- Upstream push status: recorded after the authenticated push attempt in `environment/push_status.txt`
- History policy: `main` was not modified, history was not rewritten, and no pull request was opened.

The optional directory `/work/Users/leiyo/GitHub/_inputs/` contained no major-revision result bundle or paper source package at integration start.  Consequently, this implementation was derived from the runbook’s mathematical specification and no external reference CSV comparison was performed.  The validator records the reference-comparison status as `not_performed_no_reference_input`, not as a numerical failure.

## Integrated source

The original `acli` package name, experiments `00`–`05`, and notebooks `00`–`05` are retained.  Version `0.2.0` adds:

- `acli/revision/information.py`: canonical bit-valued entropy, KL, and mutual information;
- `acli/revision/channels.py`: calibrated continuous/discrete posterior-score channels and exact AUC;
- `acli/revision/frontier.py`: exact two-branch selection frontier and Pinsker comparisons;
- `acli/revision/accumulation.py`: shared/component/decoupled information profiles and exact Poisson-binomial gain;
- `acli/revision/finite_k.py`: deterministic finite-pool top-`B` precision and tail limits;
- `acli/revision/stress.py`: finite-channel concavity and transcript-identity audits;
- `acli/revision/reproducibility.py`: SHA-256-derived PCG64DXSM task seeds;
- `acli/revision/suite.py`, `plotting.py`, and `validation.py`: checkpoints, figure style, and formal validation;
- experiments `06`–`10`, smoke/paper YAML profiles, formal scripts, tests, and the paper-figure notebook.

The compatibility wrapper `acli.utils.h2` now delegates to the canonical boundary-correct implementation.  Existing imports remain valid.

## Numerical corrections covered

- Every finite-epsilon posterior uses a solved intercept with `E[eta]=p`; `logit(p)+epsilon*G` is not presented as an exact calibrated posterior.
- Discrete AUC awards half credit to tied positive/negative score pairs.
- Exact global-target gain uses a Poisson-binomial PMF for heterogeneous selected posterior probabilities.  The binomial proxy is explicitly labeled as an approximation.
- The sharp frontier enforces `p <= q <= min(1,p/alpha)` and audits inversion residuals away from the feasibility ceiling.
- Large arrays use partial selection (`argpartition`) instead of full-pool sorting.

## Experiments and figures

Formal data products are written under `result/table/`.  `notebook/06_major_revision_paper_figures.ipynb` reads those tables and produces Figures 1–8 as PDF and PNG under `result/figure/paper/`; the executed notebook and HTML are stored under `result/artifact/`.

The authoritative figure-to-source mapping is `paper_figures/manifest.yaml`.  Numerical metrics—including calibration error, q-star residual, accumulation stress maxima, and finite-`K` convergence summaries—are recorded in `result/artifact/output_validation.json`.  This avoids copying rounded values into this report and makes the delivered claims machine-checkable.

The notebook outputs are, concretely:

1. `paper_fig01_jakob_discovery_law.{pdf,png}`;
2. `paper_fig02_aci_pipeline.{pdf,png}`;
3. `paper_fig03_accumulation_profiles.{pdf,png}`;
4. `paper_fig04_equal_J_tail_yield.{pdf,png}`;
5. `paper_fig05_tail_leverage.{pdf,png}`;
6. `paper_fig06_finite_length_validation.{pdf,png}`;
7. `paper_fig07_finite_K_convergence.{pdf,png}`;
8. `paper_fig08_weak_screening_validity.{pdf,png}`.

## Changed-file inventory

The tracked change set comprises `.gitignore`, `README.md`, `INTEGRATION_REPORT.md`, packaging metadata and requirements; compatibility updates in `acli/{__init__,utils,screening,benchmark}.py`, experiments `00`–`05`, and notebook `05`; the complete `acli/revision/` package; YAML profiles; experiments `06`–`10`; notebook `06`; the figure manifest; formal scripts; and six `tests/test_revision_*.py` files.  The delivery bundle’s `changed_files.txt` is the exact Git-generated name/status inventory, and `source_diff.patch` is the exact binary-capable `base..final` patch.

## Validation and tests

The formal tmux entrypoint runs, in order:

1. the complete pytest suite with JUnit output;
2. compatibility regressions for experiments `00`–`05`;
3. the checkpointed paper-profile experiments;
4. execution and HTML conversion of the paper notebook;
5. schema, finiteness, tolerance, figure, provenance, Git, and commit validation;
6. reproducibility-bundle creation and ZIP integrity testing.

Formal test results are in `result/artifact/pytest.xml`; validation results are in `result/artifact/output_validation.json`.  The run commit is in `environment/run_commit.txt`, the software environment in `environment/` and `result/artifact/environment_manifest.txt`, and the complete tmux transcript in `logs/aci_major_revision_integration.log`.

Before the source commit, the complete unit/schema suite reported **145 passed**, experiments `00`–`05` completed under the regression runner, the smoke profile completed all five checkpointed stages, and notebook `06` executed end to end.  The final smoke validator recorded maximum calibration error `8.500145032286355e-17`, maximum interior q-star residual `1.3877787807814454e-16`, zero q-star ceiling error, and at `K=10000` finite-pool relative-boost errors `(median, p90, maximum) = (0.001140555607864, 0.0110513979578199, 0.0161597081178427)` with median log-log slope `-0.9924233044732712`.  These are preflight values; the authoritative paper-profile values and 12,000-repetition finite-`K` MC audit are the formal JSON/CSV artifacts named above.

No deterministic values were compared with an external reference archive because none was present.  The finite-`K` table reports the actual differences from the runbook’s rounded `K=10000` reference targets for the full paper grid; non-applicable smoke or non-`K=10000` comparisons are explicitly labeled `not_applicable`.  Integer budgets use `B=max(1, round(alpha*K))`, and the asymptotic comparator is evaluated at the disclosed realized fraction `B/K`.

## Delivery bundle

`ACI_MAJOR_REVISION_REPRODUCIBILITY_BUNDLE.zip` contains this report, environment records, tables, figures, artifacts, formal log and state, source diff and changed-file list, and a source snapshot.  `.venv`, `.git`, caches, and external input archives are excluded.  ZIP integrity and the adjacent SHA-256 file are checked before delivery.
