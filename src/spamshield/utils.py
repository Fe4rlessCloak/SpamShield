"""Small shared helpers for plotting and saving figures.

Matplotlib is forced to the non-interactive ``Agg`` backend so figures can be generated
from scripts, tests and CI without a display.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402

from .constants import FIGURES_DIR  # noqa: E402

BAR_COLOR = "#4C72B0"
BEST_COLOR = "#C44E52"


def ensure_directory(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_figure(fig: plt.Figure, filename: str, *, directory: str | Path = FIGURES_DIR) -> Path:
    """Tighten, save and close a figure, returning the written path."""
    target = ensure_directory(directory) / filename
    fig.tight_layout()
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target


def bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    ylabel: str,
    filename: str,
    value_format: str = "%.3f",
    highlight: str | None = None,
    directory: str | Path = FIGURES_DIR,
) -> Path:
    """Save a labelled vertical bar chart, optionally highlighting one bar."""
    values = list(values)
    colors = [BEST_COLOR if label == highlight else BAR_COLOR for label in labels]
    upper = max(values) * 1.15 if values else 1.0

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars = ax.bar(list(labels), values, color=colors)
    ax.bar_label(bars, fmt=value_format, padding=4, fontsize=9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, upper if upper > 0 else 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")

    if highlight:
        ax.legend(
            handles=[
                plt.Rectangle((0, 0), 1, 1, color=BAR_COLOR),
                plt.Rectangle((0, 0), 1, 1, color=BEST_COLOR),
            ],
            labels=["Model", f"Selected model ({highlight})"],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=2,
            fontsize=8,
            frameon=False,
        )

    return save_figure(fig, filename, directory=directory)


def confusion_matrix_figure(
    matrix: np.ndarray,
    class_names: Sequence[str],
    *,
    title: str,
    filename: str,
    directory: str | Path = FIGURES_DIR,
) -> Path:
    """Save a single labelled confusion-matrix heatmap."""
    matrix = np.asarray(matrix)

    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(matrix, cmap="Blues", interpolation="nearest")

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            ax.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=11,
                color="white" if value > threshold else "black",
            )

    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    return save_figure(fig, filename, directory=directory)
