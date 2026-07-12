from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Color-blind-friendly palette adapted from Okabe-Ito.
PAPER_COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
    "dark": "#1F2937",
}

TUG_PHASE_COLORS = [
    "#4C78A8",  # Sitting
    "#F58518",  # SitToStand
    "#54A24B",  # Walk
    "#E45756",  # Turn
    "#B279A2",  # TurnToSit
]


def apply_publication_style(font_size: float = 8.5) -> None:
    """Apply a compact journal-friendly Matplotlib style."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": font_size,
            "axes.titlesize": font_size + 0.5,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 0.5,
            "ytick.labelsize": font_size - 0.5,
            "legend.fontsize": font_size - 0.5,
            "figure.titlesize": font_size + 1.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure_bundle(
    figure: mpl.figure.Figure,
    path: str | Path,
    *,
    formats: Iterable[str] = ("png", "pdf", "svg"),
    dpi: int = 600,
) -> list[Path]:
    """Save a figure in raster and vector publication formats."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix("")
    saved: list[Path] = []
    for extension in formats:
        target = stem.with_suffix(f".{extension}")
        kwargs = {"dpi": dpi} if extension.lower() == "png" else {}
        figure.savefig(target, **kwargs)
        saved.append(target)
    return saved


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    denominator = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        denominator,
        out=np.zeros_like(matrix, dtype=float),
        where=denominator != 0,
    )


def draw_confusion_matrix(
    axis: mpl.axes.Axes,
    matrix: pd.DataFrame | np.ndarray,
    class_names: list[str] | None = None,
    *,
    title: str | None = None,
    normalize: bool = True,
    show_counts: bool = True,
    cmap: str = "Blues",
    vmin: float = 0.0,
    vmax: float | None = None,
) -> mpl.image.AxesImage:
    if isinstance(matrix, pd.DataFrame):
        counts = matrix.to_numpy(dtype=float)
        names = list(matrix.index)
    else:
        counts = np.asarray(matrix, dtype=float)
        names = class_names or [str(index) for index in range(counts.shape[0])]
    values = _row_normalize(counts) if normalize else counts
    if vmax is None:
        vmax = 1.0 if normalize else float(max(counts.max(), 1.0))
    image = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    axis.set_xticks(range(len(names)))
    axis.set_yticks(range(len(names)))
    axis.set_xticklabels(names, rotation=35, ha="right", rotation_mode="anchor")
    axis.set_yticklabels(names)
    axis.set_xlabel("Predicted phase")
    axis.set_ylabel("True phase")
    if title:
        axis.set_title(title, pad=6)

    threshold = (vmax or 1.0) * 0.55
    for row in range(counts.shape[0]):
        for column in range(counts.shape[1]):
            if normalize:
                text = f"{100 * values[row, column]:.1f}%"
                if show_counts:
                    text += f"\n({int(counts[row, column])})"
            else:
                text = str(int(counts[row, column]))
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=7.0,
                color="white" if values[row, column] > threshold else PAPER_COLORS["dark"],
            )
    return image


def save_confusion_matrix_plot(
    matrix: pd.DataFrame,
    title: str,
    path: str | Path,
) -> None:
    apply_publication_style()
    figure, axis = plt.subplots(figsize=(4.3, 3.8), constrained_layout=True)
    image = draw_confusion_matrix(axis, matrix, title=title, normalize=True)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Recall within true class")
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.set_ticklabels(["0%", "50%", "100%"])
    save_figure_bundle(figure, path)
    plt.close(figure)


def save_comparison_plot(
    frame: pd.DataFrame,
    label_column: str,
    metric_columns: list[str],
    title: str,
    path: str | Path,
) -> None:
    apply_publication_style()
    figure, axis = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
    x = np.arange(len(frame))
    width = 0.76 / max(len(metric_columns), 1)
    colors = [PAPER_COLORS["blue"], PAPER_COLORS["orange"], PAPER_COLORS["green"]]
    for index, metric in enumerate(metric_columns):
        offset = (index - (len(metric_columns) - 1) / 2) * width
        values = frame[metric].to_numpy(dtype=float) * 100
        bars = axis.bar(
            x + offset,
            values,
            width,
            label=metric.replace("_", " ").title(),
            color=colors[index % len(colors)],
            edgecolor="white",
            linewidth=0.6,
        )
        axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
    axis.set_xticks(x)
    axis.set_xticklabels(frame[label_column], rotation=22, ha="right")
    axis.set_ylabel("Score (%)")
    axis.set_ylim(0, min(105, max(100, float(frame[metric_columns].max().max() * 100 + 8))))
    axis.set_title(title)
    axis.legend(ncol=max(1, len(metric_columns)), loc="upper center")
    axis.spines["left"].set_bounds(0, 100)
    save_figure_bundle(figure, path)
    plt.close(figure)


def save_timeline_plot(
    frame: pd.DataFrame,
    class_names: list[str],
    title: str,
    path: str | Path,
) -> None:
    apply_publication_style()
    ordered = frame.sort_values("time_step")
    time = (
        ordered["time_sec"].to_numpy(dtype=float)
        if "time_sec" in ordered
        else ordered["time_step"].to_numpy(dtype=float)
    )
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 2.4),
        sharex=True,
        constrained_layout=True,
    )
    for axis, column, label in zip(
        axes,
        ("true_id", "pred_id"),
        ("Ground truth", "Prediction"),
    ):
        axis.imshow(
            ordered[column].to_numpy(dtype=int)[None, :],
            aspect="auto",
            interpolation="nearest",
            extent=(time.min(), time.max(), 0, 1),
            cmap=mpl.colors.ListedColormap(TUG_PHASE_COLORS[: len(class_names)]),
            vmin=-0.5,
            vmax=len(class_names) - 0.5,
        )
        axis.set_yticks([0.5])
        axis.set_yticklabels([label])
        axis.tick_params(axis="y", length=0)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Time (s)" if "time_sec" in ordered else "Timestep")
    save_figure_bundle(figure, path)
    plt.close(figure)
