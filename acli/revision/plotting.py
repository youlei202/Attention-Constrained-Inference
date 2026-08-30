"""Shared Matplotlib style and deterministic figure-saving helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt


PAPER_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
)


def apply_paper_style() -> None:
    """Apply the single style used by every revision figure."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.7,
            "lines.markersize": 4.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
            "figure.dpi": 120,
        }
    )
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=PAPER_COLORS)


def save_paper_figure(fig: Any, stem: str, output_dir: str | Path) -> tuple[Path, Path]:
    """Save vector PDF and a 320-dpi PNG, returning both paths."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    pdf = target / f"{stem}.pdf"
    png = target / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=320)
    return pdf, png


def draw_aci_pipeline(ax: Any) -> None:
    """Draw the data-independent ACI pipeline used for paper Figure 2."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.03, "Inspect K records", "cheap observations"),
        (0.28, "Posterior scores", r"$\eta_i=P(T_i=1\mid Z_i)$"),
        (0.53, "Select top B", "attention constraint"),
        (0.78, "Verify & infer", "log-loss reduction"),
    ]
    for x, title, subtitle in boxes:
        patch = FancyBboxPatch(
            (x, 0.34), 0.19, 0.31, boxstyle="round,pad=0.018", facecolor="#EAF2F8",
            edgecolor="#0072B2", linewidth=1.4,
        )
        ax.add_patch(patch)
        ax.text(x + 0.095, 0.535, title, ha="center", va="center", weight="bold", fontsize=9)
        ax.text(x + 0.095, 0.435, subtitle, ha="center", va="center", fontsize=8)
    for x in (0.225, 0.475, 0.725):
        ax.add_patch(FancyArrowPatch((x, 0.495), (x + 0.05, 0.495), arrowstyle="-|>", mutation_scale=12))
    ax.text(0.5, 0.82, "Attention-Constrained Inference (ACI)", ha="center", weight="bold", fontsize=12)


__all__ = ["PAPER_COLORS", "apply_paper_style", "draw_aci_pipeline", "save_paper_figure"]
