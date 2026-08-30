#!/usr/bin/env python3
"""Build the source notebook for deterministic paper-figure rendering."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebook/06_major_revision_paper_figures.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    markdown(
        r"""
# Major-revision paper figures

This notebook reproduces all eight quantitative and conceptual figures for
**“Epistemic Throughput: Fundamental Limits of Attention-Constrained Inference.”**
It reads cached CSV tables produced by experiments 05–09 and performs only
lightweight deterministic formula evaluation.  Large Monte Carlo and stress
audits are deliberately kept out of the notebook.
"""
    ),
    code(
        r"""
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from scipy.stats import norm

cwd = Path.cwd().resolve()
ROOT = next(path for path in (cwd, cwd.parent) if (path / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT))

from acli.revision.plotting import PAPER_COLORS, apply_paper_style, draw_aci_pipeline, save_paper_figure

TABLE_DIR = ROOT / "result" / "table"
FIGURE_DIR = ROOT / "result" / "figure" / "paper"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
apply_paper_style()

def read_table(name):
    path = TABLE_DIR / name
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Run the paper experiment suite first: {path}")
    frame = pd.read_csv(path)
    print(f"data source: {path.relative_to(ROOT)} ({len(frame):,} rows)")
    return frame

print(f"repository: {ROOT}")
print(f"figure output: {FIGURE_DIR.relative_to(ROOT)}")
"""
    ),
    markdown(
        r"""
## Figure 1 — JaKoB discovery law

Parameters: prevalence $p=0.05$, coefficient $c=0.45$,
$JK\in\{10,100,200,400\}$, and $B=0,\ldots,200$.
The vertical quantity is informative discoveries, not universal throughput.
"""
    ),
    code(
        r"""
p, c = 0.05, 0.45
B = np.arange(0, 201)
JK_values = (10, 100, 200, 400)
print("data source: deterministic JaKoB discovery formula")
print({"p": p, "c": c, "JK": JK_values, "B_range": [int(B.min()), int(B.max())]})

fig, ax = plt.subplots(figsize=(5.3, 3.4))
for JK in JK_values:
    discoveries = np.minimum(B, B * p + c * np.sqrt(JK * B))
    ax.plot(B, discoveries, label=fr"$JK={JK}$")
ax.plot(B, B * p, color="0.35", linestyle="--", label=r"random baseline $Bp$")
ax.set(title="JaKoB Discovery Law", xlabel="Verification budget B", ylabel=r"Informative discoveries $\mathbb{E}[N_B]$")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
save_paper_figure(fig, "paper_fig01_jakob_discovery_law", FIGURE_DIR)
plt.show()
"""
    ),
    markdown(
        r"""
## Figure 2 — ACI pipeline

This conceptual diagram has no experimental input.  It is drawn with
Matplotlib patches so the source notebook is self-contained on systems without
a LaTeX installation.
"""
    ),
    code(
        r"""
print("data source: deterministic Matplotlib concept diagram")
print({"parameters": "not applicable", "role": "conceptual ACI pipeline"})
fig, ax = plt.subplots(figsize=(8.0, 2.5))
draw_aci_pipeline(ax)
save_paper_figure(fig, "paper_fig02_aci_pipeline", FIGURE_DIR)
plt.show()
"""
    ),
    markdown(
        r"""
## Figure 3 — Shared-target accumulation

Panel A uses a balanced shared target ($\pi=0.5$), BSC noise
$\beta\in\{0.05,0.1,0.2,0.3,0.4\}$, and $n\le128$.  Panel B shows the
diminishing margins through $n=2048$ on log–log axes.  The decoupled linear
curve is intentionally omitted from the shared-target scale.
"""
    ),
    code(
        r"""
accumulation = read_table("06_accumulation_profiles.csv")
shared = accumulation.query("curve_type == 'shared_bsc' and pi == 0.5").copy()
betas = (0.05, 0.10, 0.20, 0.30, 0.40)
print({"pi": 0.5, "beta": betas, "panel_A_n": [0, 128], "panel_B_n": [1, 2048]})

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.25))
for color, beta in zip(PAPER_COLORS, betas):
    subset = shared.loc[np.isclose(shared["beta"], beta)].sort_values("n")
    panel_a = subset.loc[subset["n"] <= 128]
    panel_b = subset.loc[(subset["n"] >= 1) & (subset["n"] <= 2048) & (subset["delta_bits"] > 0)]
    axes[0].plot(panel_a["n"], panel_a["phi_bits"], color=color, label=fr"$\beta={beta:g}$")
    axes[1].loglog(panel_b["n"], panel_b["delta_bits"], color=color)
axes[0].axhline(1.0, color="0.25", linestyle="--", linewidth=1.1, label=r"$H(\Theta)=1$")
axes[0].set(xlabel="Informative verifications n", ylabel=r"$\Phi(n)$ (bits)", ylim=(0, 1.05), title="A. Accumulation")
axes[1].set(xlabel="n", ylabel=r"$\Delta_n=\Phi(n)-\Phi(n-1)$", title="B. Diminishing margins")
for ax in axes: ax.grid(alpha=0.22, which="both")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout(rect=(0, 0, 1, 0.91))
save_paper_figure(fig, "paper_fig03_accumulation_profiles", FIGURE_DIR)
plt.show()
"""
    ),
    markdown(
        r"""
## Figure 4 — Equal-information tail yield

All score families are matched at $p=0.01$ and $J/H_2(p)=0.01$.  The figure
compares their exact large-pool top-tail precision across the prescribed
selection fractions, including discrete channels with fractional cutoff atoms.
"""
    ),
    code(
        r"""
same_j = read_table("07_same_J_tail_comparison.csv")
same_j = same_j.loc[np.isclose(same_j["p"], 0.01)].copy()
family_order = ("gaussian", "student_t5", "pareto4", "pareto5", "rare_spike", "three_point", "uniform")
print({"p": 0.01, "J_over_H2": 0.01, "families": family_order, "alpha": sorted(same_j.alpha.unique())})

fig, ax = plt.subplots(figsize=(6.4, 3.8))
markers = ("o", "s", "^", "v", "D", "P", "X")
for color, marker, family in zip(PAPER_COLORS, markers, family_order):
    subset = same_j.loc[same_j["family"] == family].sort_values("alpha")
    ax.semilogx(subset["alpha"], subset["q_alpha"], marker=marker, color=color, label=family.replace("_", " "))
ax.set(xlabel=r"Selected fraction $\alpha=B/K$", ylabel=r"Top-tail precision $q_\alpha$", title="Equal-J tail yield")
ax.grid(alpha=0.24, which="both")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=4, frameon=False)
fig.subplots_adjust(bottom=0.30)
save_paper_figure(fig, "paper_fig04_equal_J_tail_yield", FIGURE_DIR)
plt.show()
"""
    ),
    markdown(
        r"""
## Figure 5 — Gaussian and Pareto tail leverage

This deterministic comparison evaluates upper-tail score means for
$K/B=10^1,\ldots,10^5$.  Pareto base variables are centered and normalized to
unit variance before computing $m_G(\alpha)$.
"""
    ),
    code(
        r"""
ratio = np.logspace(1, 5, 240)
alpha = 1.0 / ratio
gaussian_m = norm.pdf(norm.isf(alpha)) / alpha
print("data source: deterministic Gaussian/Pareto tail formulas")

def standardized_pareto_tail_mean(alpha, nu):
    mean = nu / (nu - 1.0)
    sd = np.sqrt(nu / ((nu - 1.0) ** 2 * (nu - 2.0)))
    conditional_raw_mean = (nu / (nu - 1.0)) * alpha ** (-1.0 / nu)
    return (conditional_raw_mean - mean) / sd

print({"K_over_B": [float(ratio.min()), float(ratio.max())], "pareto_nu": [3, 4, 5]})
fig, ax = plt.subplots(figsize=(5.7, 3.6))
ax.loglog(ratio, gaussian_m, color=PAPER_COLORS[0], label="Gaussian")
for color, nu in zip(PAPER_COLORS[1:], (3, 4, 5)):
    ax.loglog(ratio, standardized_pareto_tail_mean(alpha, nu), color=color, label=fr"Pareto $\nu={nu}$")
ax.set(xlabel=r"Oversampling ratio $K/B$", ylabel=r"Tail leverage $m_G(\alpha)$", title="Gaussian and Pareto tail leverage")
ax.grid(alpha=0.24, which="both")
ax.legend(frameon=False)
save_paper_figure(fig, "paper_fig05_tail_leverage", FIGURE_DIR)
plt.show()
"""
    ),
    markdown(
        r"""
## Figure 6 — Finite-length validation

Parameters: $p=0.01$, $K=10{,}000$, BSC noise $\delta=0.1$, and target AUC
$0.55,0.70,0.79,0.90$.  Every panel uses the calibrated AUC–$\epsilon$–$J$
mapping and compares Monte Carlo gain with the exact finite-$K$ benchmark,
large-pool tail benchmark, approximations, information bounds, and finite-pool
oracle.
"""
    ),
    code(
        r"""
finite_length = read_table("05_finite_length_simulation_validation.csv")
auc_values = tuple(sorted(finite_length["target_auc"].unique()))
print({"p": 0.01, "K": 10000, "delta": 0.1, "target_auc": auc_values, "B": sorted(finite_length.B.unique())})

fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.3), sharex=True)
for ax, target_auc in zip(axes.ravel(), auc_values):
    subset = finite_length.loc[np.isclose(finite_length["target_auc"], target_auc)].sort_values("B")
    ax.errorbar(subset["B"], subset["mc_gain_bits"], yerr=2*subset["mc_gain_se_bits"], fmt="o", color="black", label="MC gain ±2 SE", zorder=6)
    ax.plot(subset["B"], subset["exact_finite_K_gain_bits"], label="exact finite-K")
    ax.plot(subset["B"], subset["large_K_top_tail_gain_bits"], linestyle="--", label="large-K tail")
    ax.plot(subset["B"], subset["weak_epsilon_gain_bits"], linestyle=":", label=r"weak-$\epsilon$")
    ax.plot(subset["B"], subset["weak_J_gain_bits"], linestyle="-.", label="weak-J")
    ax.plot(subset["B"], subset["sharp_binary_KL_gain_bits"], linewidth=1.1, label="sharp binary-KL")
    ax.plot(subset["B"], subset["original_pinsker_jakob_clipped_gain_bits"], linewidth=1.1, label="original Pinsker (feasibility-clipped)")
    ax.plot(subset["B"], subset["finite_pool_oracle_gain_bits"], color="0.5", linestyle="--", linewidth=1.1, label="finite-pool oracle")
    epsilon = subset["epsilon"].iloc[0]
    J = subset["J_bits"].iloc[0]
    ax.set_title(fr"AUC={target_auc:.2f}, $\epsilon$={epsilon:.3g}, J={J:.2e}")
    ax.grid(alpha=0.22)
for ax in axes[-1]: ax.set_xlabel("Verification budget B")
for ax in axes[:, 0]: ax.set_ylabel("Information gain (bits)")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.90))
save_paper_figure(fig, "paper_fig06_finite_length_validation", FIGURE_DIR)
plt.show()
"""
    ),
    markdown(
        r"""
## Figure 7 — Finite-$K$ convergence

The paper configuration fixes $p=0.01$ and
$J=0.03H_2(p)\approx2.423794\times10^{-3}$ bits.  Each panel shows the relative
boost error for one score family and five selection fractions.  A single
shared legend is placed above the panels.
"""
    ),
    code(
        r"""
finite_k = read_table("08_finite_K_convergence.csv")
paper_k = finite_k.loc[finite_k["paper_figure_configuration"].astype(bool)].copy()
families = ("gaussian", "pareto4", "student_t5")
alphas = tuple(sorted(paper_k["alpha"].unique()))
print({"p": 0.01, "J_bits": float(paper_k.target_J_bits.iloc[0]), "families": families, "alpha": alphas, "K": sorted(paper_k.K.unique())})

fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.05), sharey=True)
for ax, family in zip(axes, families):
    family_data = paper_k.loc[paper_k["family"] == family]
    for color, alpha_value in zip(PAPER_COLORS, alphas):
        subset = family_data.loc[np.isclose(family_data["alpha"], alpha_value)].sort_values("K")
        ax.loglog(subset["K"], subset["relative_boost_error"], marker="o", color=color, label=fr"$\alpha={alpha_value:g}$")
    ax.set(title=family.replace("_", " "), xlabel="Pool size K")
    ax.grid(alpha=0.22, which="both")
axes[0].set_ylabel("Relative boost error")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=5, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.89))
save_paper_figure(fig, "paper_fig07_finite_K_convergence", FIGURE_DIR)
plt.show()
"""
    ),
    markdown(
        r"""
## Figure 8 — Weak-screening validity

For $p=0.01$, each family panel shows the relative boost error of the weak-$J$
approximation over the joint $(\alpha,J/H_2(p))$ grid.  All panels share one
logarithmic color scale.
"""
    ),
    code(
        r"""
weak = read_table("09_weak_screening_validity.csv")
families = ("gaussian", "pareto4", "student_t5", "uniform")
print({"p": 0.01, "families": families, "alpha_points": weak.alpha.nunique(), "J_points": weak.J_fraction_h2.nunique()})

positive_errors = np.maximum(weak["relative_boost_error_J"].to_numpy(dtype=float), 1e-7)
vmin = max(1e-7, float(np.nanquantile(positive_errors, 0.02)))
vmax = max(vmin * 10, float(np.nanmax(positive_errors)))
normalizer = LogNorm(vmin=vmin, vmax=vmax)
fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.9), sharex=True, sharey=True)
mesh = None
for ax, family in zip(axes, families):
    subset = weak.loc[weak["family"] == family]
    pivot = subset.pivot(index="J_fraction_h2", columns="alpha", values="relative_boost_error_J").sort_index().sort_index(axis=1)
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    z = np.maximum(pivot.to_numpy(dtype=float), vmin)
    mesh = ax.pcolormesh(x, y, z, norm=normalizer, cmap="viridis", shading="auto")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set(title=family.replace("_", " "), xlabel=r"$\alpha$")
axes[0].set_ylabel(r"$J/H_2(p)$")
colorbar = fig.colorbar(mesh, ax=axes, location="right", fraction=0.03, pad=0.025)
colorbar.set_label("Relative boost error (weak-J)")
fig.subplots_adjust(left=0.07, right=0.90, bottom=0.18, top=0.86, wspace=0.10)
save_paper_figure(fig, "paper_fig08_weak_screening_validity", FIGURE_DIR)
plt.show()
"""
    ),
    markdown(
        r"""
## Output summary

All eight figures above are saved both as vector PDF and 320-dpi PNG.  Their
stable paper mapping is recorded in `paper_figures/manifest.yaml`.
"""
    ),
    code(
        r"""
for path in sorted(FIGURE_DIR.glob("paper_fig*")):
    if path.suffix in {".pdf", ".png"}:
        print(f"{path.relative_to(ROOT)}  {path.stat().st_size:,} bytes")
"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT}")
