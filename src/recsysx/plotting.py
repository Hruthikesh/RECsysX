"""Shared matplotlib setup so every figure in results/figures looks the same."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import project_path  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 11,
    "legend.frameon": False,
})

# one fixed colour per model so the same model looks the same in every plot
MODEL_COLORS = {
    "Popularity": "#7f7f7f",
    "Popularity (recent)": "#a6a6a6",
    "Popularity (demographic)": "#c7c7c7",
    "ItemKNN": "#8c564b",
    "MF-BPR": "#1f77b4",
    "Two-Tower": "#ff7f0e",
    "Node2Vec": "#9467bd",
    "GraphSAGE": "#2ca02c",
    "GAT": "#d62728",
    "LightGCN": "#bcbd22",
    "Hybrid (id+content+graph)": "#17becf",
    "Final pipeline": "#000000",
}


def color(label: str, default: str | None = "#333333") -> str | None:
    if label in MODEL_COLORS:
        return MODEL_COLORS[label]
    if label.startswith("Final pipeline"):
        return "#000000" if "MMR" not in label else "#555555"
    if label.startswith("Ranker"):
        return "#555555" if "MMR" in label else "#000000"
    if label.startswith("Actual"):
        return "#e377c2"
    return default


def figure_path(name: str, results_dir: str = "results") -> Path:
    out = project_path(results_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out / name


def save(fig, name: str, results_dir: str = "results") -> Path:
    path = figure_path(name, results_dir)
    fig.savefig(path)
    plt.close(fig)
    return path
