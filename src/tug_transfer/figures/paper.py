from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from sklearn.metrics import f1_score

from tug_transfer.constants import TUG_PHASES
from tug_transfer.training.plots import (
    PAPER_COLORS,
    TUG_PHASE_COLORS,
    apply_publication_style,
    draw_confusion_matrix,
    save_figure_bundle,
)
from tug_transfer.utils import ensure_dir


CONDITION_ORDER = [
    "zero_shot_tug_head_projection",
    "frozen_pretrained_encoder_mlp_head",
    "finetune_pretrained_cnn_bilstm",
    "from_scratch_cnn_bilstm",
]
CONDITION_LABELS = {
    "zero_shot_tug_head_projection": "Zero-shot\nprojection",
    "frozen_pretrained_encoder_mlp_head": "Frozen encoder\n+ new head",
    "finetune_pretrained_cnn_bilstm": "Fine-tuned\npretrained",
    "from_scratch_cnn_bilstm": "From scratch",
}
CONDITION_COLORS = {
    "zero_shot_tug_head_projection": PAPER_COLORS["gray"],
    "frozen_pretrained_encoder_mlp_head": PAPER_COLORS["sky"],
    "finetune_pretrained_cnn_bilstm": PAPER_COLORS["blue"],
    "from_scratch_cnn_bilstm": PAPER_COLORS["orange"],
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _read_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, index_col=0)


def _save(
    figure: mpl.figure.Figure,
    output_dir: Path,
    stem: str,
    config: dict[str, Any],
) -> list[Path]:
    style = config.get("style", {})
    return save_figure_bundle(
        figure,
        output_dir / f"{stem}.png",
        formats=tuple(style.get("formats", ["png", "pdf", "svg"])),
        dpi=int(style.get("dpi", 600)),
    )


def _phase_strip(
    axis: mpl.axes.Axes,
    time: np.ndarray,
    values: np.ndarray,
    label: str,
    class_names: list[str],
) -> None:
    cmap = mpl.colors.ListedColormap(TUG_PHASE_COLORS[: len(class_names)])
    edges = np.r_[time, time[-1] + np.median(np.diff(time))] if len(time) > 1 else [0, 1]
    axis.pcolormesh(
        edges,
        [0, 1],
        values[None, :],
        cmap=cmap,
        vmin=-0.5,
        vmax=len(class_names) - 0.5,
        shading="flat",
        rasterized=False,
    )
    axis.set_ylim(0, 1)
    axis.set_yticks([0.5])
    axis.set_yticklabels([label])
    axis.tick_params(axis="y", length=0, pad=4)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_visible(False)
    axis.tick_params(axis="x", bottom=False, labelbottom=False)


def _record_f1(frame: pd.DataFrame, class_count: int) -> pd.Series:
    values = {}
    for record_id, group in frame.groupby("record_id"):
        values[record_id] = f1_score(
            group["true_id"],
            group["pred_id"],
            labels=list(range(class_count)),
            average="macro",
            zero_division=0,
        )
    return pd.Series(values, name="macro_f1")


def _select_timeline_records(
    dense: pd.DataFrame,
    window: pd.DataFrame,
    config: dict[str, Any],
) -> list[Any]:
    timeline_config = config.get("timeline", {})
    explicit = timeline_config.get("record_ids")
    common = sorted(set(dense["record_id"]) & set(window["record_id"]))
    if explicit:
        selected = [record for record in explicit if record in common]
        if len(selected) != len(explicit):
            missing = sorted(set(explicit) - set(selected))
            raise ValueError(f"Configured timeline record IDs are unavailable: {missing}")
        return selected

    scores = _record_f1(dense[dense["record_id"].isin(common)], len(TUG_PHASES))
    quantiles = timeline_config.get("quantiles", [0.50, 0.25])
    selected: list[Any] = []
    for quantile in quantiles:
        target = float(scores.quantile(float(quantile)))
        candidates = scores.drop(index=selected, errors="ignore")
        selected.append((candidates - target).abs().sort_values().index[0])
    return selected


def figure_tug_timelines(paths: dict[str, Path], output_dir: Path, config: dict[str, Any]):
    dense = _read_csv(paths["tug"] / "dense_sequence_timestep_predictions.csv")
    window = _read_csv(paths["tug"] / "window_reconstructed_timestep_predictions.csv")
    dense = dense[dense["split"] == "test"].copy()
    window = window[window["split"] == "test"].copy()
    selected = _select_timeline_records(dense, window, config)

    apply_publication_style()
    n_records = len(selected)
    figure, axes = plt.subplots(
        3,
        n_records,
        figsize=(7.2, 3.05 if n_records > 1 else 2.75),
        squeeze=False,
        gridspec_kw={"hspace": 0.12, "wspace": 0.12},
    )
    selection_rows = []

    for record_index, record_id in enumerate(selected):
        d = dense[dense["record_id"] == record_id].sort_values("time_step")
        w = window[window["record_id"] == record_id].sort_values("time_step")
        merged = d[["time_step", "true_id", "pred_id"]].merge(
            w[["time_step", "true_id", "pred_id"]],
            on="time_step",
            suffixes=("_dense", "_window"),
            validate="one_to_one",
        )
        if not np.array_equal(merged["true_id_dense"], merged["true_id_window"]):
            raise ValueError(f"Ground-truth mismatch for record {record_id}")
        hz = 1.0 / np.median(np.diff(d["time_sec"])) if "time_sec" in d and len(d) > 1 else 20.0
        time = merged["time_step"].to_numpy(dtype=float) / hz
        column_axes = axes[:, record_index]
        _phase_strip(column_axes[0], time, merged["true_id_dense"].to_numpy(), "Ground truth", TUG_PHASES)
        _phase_strip(column_axes[1], time, merged["pred_id_dense"].to_numpy(), "Dense", TUG_PHASES)
        _phase_strip(column_axes[2], time, merged["pred_id_window"].to_numpy(), "Window", TUG_PHASES)

        boundaries = np.flatnonzero(np.diff(merged["true_id_dense"].to_numpy()) != 0) + 1
        for axis in column_axes:
            for boundary in boundaries:
                axis.axvline(time[boundary], color="white", linewidth=0.55, alpha=0.9)
            axis.set_xlim(time.min(), time.max())

        column_axes[2].spines["bottom"].set_visible(True)
        column_axes[2].tick_params(axis="x", bottom=True, labelbottom=True)
        column_axes[2].set_xlabel("Time (s)")
        if record_index > 0:
            for axis in column_axes:
                axis.set_yticklabels([])
        subject = d["subject_id"].iloc[0] if "subject_id" in d else record_id
        dense_f1 = f1_score(
            merged["true_id_dense"], merged["pred_id_dense"],
            labels=list(range(len(TUG_PHASES))), average="macro", zero_division=0,
        )
        window_f1 = f1_score(
            merged["true_id_dense"], merged["pred_id_window"],
            labels=list(range(len(TUG_PHASES))), average="macro", zero_division=0,
        )
        panel_label = chr(ord("a") + record_index)
        descriptor = "Typical held-out trial" if record_index == 0 else "Lower-quartile held-out trial"
        column_axes[0].set_title(
            f"({panel_label}) {descriptor} — subject {subject}\n"
            f"Dense F1 {100*dense_f1:.1f}% | Window F1 {100*window_f1:.1f}%",
            fontweight="bold",
            pad=5,
        )
        selection_rows.append(
            {
                "record_id": record_id,
                "subject_id": subject,
                "descriptor": descriptor,
                "dense_macro_f1": dense_f1,
                "window_macro_f1": window_f1,
            }
        )

    legend_handles = [
        Patch(facecolor=color, label=phase)
        for phase, color in zip(TUG_PHASES, TUG_PHASE_COLORS)
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(TUG_PHASES),
        bbox_to_anchor=(0.55, -0.01),
        columnspacing=1.3,
        handlelength=1.4,
    )
    figure.subplots_adjust(left=0.12, right=0.995, top=0.83, bottom=0.22, hspace=0.12, wspace=0.08)
    saved = _save(figure, output_dir, "figure_4_tug_phase_timelines", config)
    plt.close(figure)
    pd.DataFrame(selection_rows).to_csv(output_dir / "figure_4_selected_records.csv", index=False)
    return saved, {"selected_records": selection_rows}


def figure_tug_confusions(paths: dict[str, Path], output_dir: Path, config: dict[str, Any]):
    dense = _read_matrix(paths["tug"] / "dense_sequence_test_confusion_matrix.csv")
    window = _read_matrix(paths["tug"] / "window_reconstructed_test_confusion_matrix.csv")
    apply_publication_style()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), constrained_layout=True)
    image = draw_confusion_matrix(
        axes[0], dense, title="(a) Dense timestep labeling", normalize=True, show_counts=True
    )
    draw_confusion_matrix(
        axes[1], window, title="(b) Window reconstruction", normalize=True, show_counts=True
    )
    axes[1].set_ylabel("")
    colorbar = figure.colorbar(image, ax=axes, fraction=0.028, pad=0.025)
    colorbar.set_label("Recall within true phase")
    colorbar.set_ticks([0, 0.5, 1])
    colorbar.set_ticklabels(["0%", "50%", "100%"])
    saved = _save(figure, output_dir, "figure_5_tug_confusion_matrices", config)
    plt.close(figure)
    return saved, {}


def _load_transfer_summary(
    output_dir: Path,
    experiment: str,
    reported_results: Path | None,
) -> tuple[pd.DataFrame, str]:
    path = output_dir / f"{experiment}_pretrained_vs_scratch_final_comparison.csv"
    if path.exists():
        frame = pd.read_csv(path)
        return frame, str(path)
    if reported_results is None or not reported_results.exists():
        raise FileNotFoundError(path)
    reported = pd.read_csv(reported_results)
    dataset_name = "SPmT" if experiment == "spmt" else "FoG-STAR"
    subset = reported[reported["experiment"] == dataset_name].copy()
    reverse = {
        "Zero-shot TUG-head projection": "zero_shot_tug_head_projection",
        "Frozen TUG encoder + target head": "frozen_pretrained_encoder_mlp_head",
        "Fine-tuned pretrained CNN-BiLSTM": "finetune_pretrained_cnn_bilstm",
        "From-scratch CNN-BiLSTM": "from_scratch_cnn_bilstm",
    }
    subset["condition"] = subset["configuration"].map(reverse)
    subset["test_macro_f1"] = subset["macro_f1_percent"] / 100.0
    subset["test_balanced_accuracy"] = subset["balanced_accuracy_percent"] / 100.0
    subset["test_accuracy"] = subset["accuracy_percent"] / 100.0
    return subset, str(reported_results) + " (reported fallback)"


def figure_transfer_performance(paths: dict[str, Path], output_dir: Path, config: dict[str, Any]):
    reported = paths.get("reported_results")
    spmt, spmt_source = _load_transfer_summary(paths["spmt"], "spmt", reported)
    fog, fog_source = _load_transfer_summary(paths["fogstar"], "fogstar", reported)
    apply_publication_style()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    panel_specs = [
        (axes[0], spmt, "(a) WearGait-PD SPmT", (88, 97)),
        (axes[1], fog, "(b) FoG-STAR", (0, 82)),
    ]
    for axis, frame, title, limits in panel_specs:
        ordered = frame.set_index("condition").reindex(CONDITION_ORDER).dropna(how="all")
        x = np.arange(len(ordered))
        values = ordered["test_macro_f1"].to_numpy(dtype=float) * 100
        colors = [CONDITION_COLORS[condition] for condition in ordered.index]
        bars = axis.bar(
            x,
            values,
            width=0.68,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
        )
        axis.set_xticks(x)
        axis.set_xticklabels([CONDITION_LABELS[c] for c in ordered.index])
        axis.set_ylabel("Macro F1 (%)")
        axis.set_ylim(*limits)
        axis.set_title(title, fontweight="bold")
        axis.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=2, fontsize=7.5)
        axis.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
        axis.axhline(limits[0], color=PAPER_COLORS["dark"], linewidth=0.8)
    saved = _save(figure, output_dir, "figure_6_transfer_macro_f1", config)
    plt.close(figure)
    return saved, {"spmt_source": spmt_source, "fogstar_source": fog_source}


def figure_window_ambiguity(paths: dict[str, Path], output_dir: Path, config: dict[str, Any]):
    source = paths["tug"] / "window_test_purity_analysis.csv"
    if source.exists():
        frame = pd.read_csv(source)
        provenance = str(source)
    else:
        fallback = paths.get("reported_window_purity")
        if fallback is None or not fallback.exists():
            raise FileNotFoundError(source)
        frame = pd.read_csv(fallback)
        provenance = str(fallback) + " (reported fallback)"
    frame = frame.set_index("window_type").reindex(["pure", "mixed_boundary"])

    apply_publication_style()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.95),
        gridspec_kw={"width_ratios": [1.1, 1.45]},
        constrained_layout=True,
    )
    # Conceptual panel: a 2-s window that spans Walk -> Turn.
    axis = axes[0]
    axis.set_xlim(0, 2)
    axis.set_ylim(0, 2.2)
    axis.add_patch(Rectangle((0, 1.15), 1.25, 0.42, facecolor=TUG_PHASE_COLORS[2], edgecolor="none"))
    axis.add_patch(Rectangle((1.25, 1.15), 0.75, 0.42, facecolor=TUG_PHASE_COLORS[3], edgecolor="none"))
    axis.text(0.62, 1.36, "Walk", ha="center", va="center", color="white", fontweight="bold")
    axis.text(1.62, 1.36, "Turn", ha="center", va="center", color="white", fontweight="bold")
    axis.annotate(
        "Majority label: Walk",
        xy=(1.0, 1.05),
        xytext=(1.0, 0.45),
        ha="center",
        arrowprops={"arrowstyle": "-|>", "linewidth": 0.9, "color": PAPER_COLORS["dark"]},
        fontweight="bold",
    )
    axis.plot([0, 2], [1.72, 1.72], color=PAPER_COLORS["dark"], linewidth=1.0)
    axis.plot([0, 0], [1.65, 1.79], color=PAPER_COLORS["dark"], linewidth=1.0)
    axis.plot([2, 2], [1.65, 1.79], color=PAPER_COLORS["dark"], linewidth=1.0)
    axis.text(1, 1.84, "2-s fixed window", ha="center")
    axis.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    axis.set_xlabel("Time (s)")
    axis.set_yticks([])
    axis.set_title("(a) Boundary-crossing window", fontweight="bold")
    axis.spines["left"].set_visible(False)

    axis = axes[1]
    labels = ["Pure", "Mixed / boundary"]
    x = np.arange(2)
    width = 0.36
    percentages = frame["percentage"].to_numpy(dtype=float)
    accuracies = frame["accuracy"].to_numpy(dtype=float) * 100
    bars_a = axis.bar(
        x - width / 2,
        percentages,
        width,
        color=PAPER_COLORS["sky"],
        label="Share of test windows",
        edgecolor="white",
    )
    bars_b = axis.bar(
        x + width / 2,
        accuracies,
        width,
        color=PAPER_COLORS["orange"],
        label="Classification accuracy",
        edgecolor="white",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 112)
    axis.set_ylabel("Percentage (%)")
    axis.set_title("(b) Frequency and performance", fontweight="bold", pad=18)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    axis.bar_label(bars_a, fmt="%.1f", padding=2, fontsize=7.5)
    axis.bar_label(bars_b, fmt="%.1f", padding=2, fontsize=7.5)
    for index, count in enumerate(frame["n_windows"].to_numpy(dtype=int)):
        axis.text(index - width / 2, percentages[index] / 2, f"n={count}", ha="center", va="center", fontsize=7)
    saved = _save(figure, output_dir, "figure_s1_window_ambiguity", config)
    plt.close(figure)
    return saved, {"source": provenance}


def figure_boundary_tolerance(paths: dict[str, Path], output_dir: Path, config: dict[str, Any]):
    candidates = {
        "TUG dense": paths["tug"] / "dense_sequence_boundary_tolerant_test.csv",
        "TUG window": paths["tug"] / "window_reconstructed_boundary_tolerant_test.csv",
        "SPmT": paths["spmt"] / "boundary_tolerant_test_metrics_paper_best.csv",
        "FoG-STAR": paths["fogstar"] / "boundary_tolerant_test_metrics_paper_best.csv",
    }
    available = {name: pd.read_csv(path) for name, path in candidates.items() if path.exists()}
    if not available:
        raise FileNotFoundError("No boundary-tolerance CSVs were found")
    apply_publication_style()
    figure, axis = plt.subplots(figsize=(5.2, 3.25), constrained_layout=True)
    colors = [PAPER_COLORS["blue"], PAPER_COLORS["orange"], PAPER_COLORS["green"], PAPER_COLORS["purple"]]
    markers = ["o", "s", "^", "D"]
    for (name, frame), color, marker in zip(available.items(), colors, markers):
        axis.plot(
            frame["tolerance_sec"],
            frame["weighted_tolerant_accuracy"] * 100,
            marker=marker,
            color=color,
            label=name,
        )
    axis.set_xlabel("Boundary tolerance (s)")
    axis.set_ylabel("Tolerant accuracy (%)")
    axis.set_xticks([0, 0.25, 0.5, 1.0])
    axis.set_ylim(0, 101)
    axis.legend(ncol=2, loc="lower right")
    saved = _save(figure, output_dir, "figure_s2_boundary_tolerance", config)
    plt.close(figure)
    return saved, {"sources": {name: str(path) for name, path in candidates.items() if path.exists()}}


def figure_subject_distributions(paths: dict[str, Path], output_dir: Path, config: dict[str, Any]):
    dense = _read_csv(paths["tug"] / "dense_sequence_subjectwise_metrics.csv")
    dense = dense[dense["split"] == "test"].copy()
    window = _read_csv(paths["tug"] / "window_reconstructed_subjectwise_metrics.csv")
    frames = [
        ("Dense", dense["macro_f1"].to_numpy(dtype=float) * 100, PAPER_COLORS["blue"]),
        ("Window", window["macro_f1"].to_numpy(dtype=float) * 100, PAPER_COLORS["orange"]),
    ]
    apply_publication_style()
    figure, axis = plt.subplots(figsize=(3.8, 3.25), constrained_layout=True)
    rng = np.random.default_rng(42)
    box = axis.boxplot(
        [values for _, values, _ in frames],
        positions=[1, 2],
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": PAPER_COLORS["dark"], "linewidth": 1.3},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch, (_, _, color) in zip(box["boxes"], frames):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
    for position, (_, values, color) in zip([1, 2], frames):
        jitter = rng.normal(0, 0.045, size=len(values))
        axis.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=18,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.82,
            zorder=3,
        )
    axis.set_xticks([1, 2])
    axis.set_xticklabels(["Dense", "Window reconstruction"])
    axis.set_ylabel("Record-level macro F1 (%)")
    axis.set_ylim(0, 101)
    saved = _save(figure, output_dir, "figure_s3_tug_record_distributions", config)
    plt.close(figure)
    return saved, {}


def figure_fogstar_condition_confusions(paths: dict[str, Path], output_dir: Path, config: dict[str, Any]):
    base = paths["fogstar"] / "condition_predictions"
    fine = _read_matrix(base / "finetune_pretrained_cnn_bilstm_test_confusion_matrix.csv")
    scratch = _read_matrix(base / "from_scratch_cnn_bilstm_test_confusion_matrix.csv")
    apply_publication_style()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), constrained_layout=True)
    image = draw_confusion_matrix(
        axes[0], fine, title="(a) Fine-tuned TUG initialization", normalize=True, show_counts=True
    )
    draw_confusion_matrix(
        axes[1], scratch, title="(b) From-scratch baseline", normalize=True, show_counts=True
    )
    axes[1].set_ylabel("")
    colorbar = figure.colorbar(image, ax=axes, fraction=0.028, pad=0.025)
    colorbar.set_label("Recall within true activity")
    colorbar.set_ticks([0, 0.5, 1])
    colorbar.set_ticklabels(["0%", "50%", "100%"])
    saved = _save(figure, output_dir, "figure_s4_fogstar_confusion_matrices", config)
    plt.close(figure)
    return saved, {}


FIGURE_BUILDERS: list[tuple[str, Callable]] = [
    ("figure_4_tug_phase_timelines", figure_tug_timelines),
    ("figure_5_tug_confusion_matrices", figure_tug_confusions),
    ("figure_6_transfer_macro_f1", figure_transfer_performance),
    ("figure_s1_window_ambiguity", figure_window_ambiguity),
    ("figure_s2_boundary_tolerance", figure_boundary_tolerance),
    ("figure_s3_tug_record_distributions", figure_subject_distributions),
    ("figure_s4_fogstar_confusion_matrices", figure_fogstar_condition_confusions),
]


def run_paper_figures(config: dict[str, Any]) -> pd.DataFrame:
    """Build all available main-text and supplementary paper figures.

    Missing experiment outputs do not abort the entire build unless
    ``strict=true``. A manifest records every generated or skipped figure.
    """
    path_config = config["paths"]
    output_dir = ensure_dir(path_config["output_dir"])
    paths = {
        "tug": Path(path_config["tug_output_dir"]),
        "spmt": Path(path_config["spmt_output_dir"]),
        "fogstar": Path(path_config["fogstar_output_dir"]),
        "reported_results": Path(path_config["reported_results_csv"])
        if path_config.get("reported_results_csv")
        else None,
        "reported_window_purity": Path(path_config["reported_window_purity_csv"])
        if path_config.get("reported_window_purity_csv")
        else None,
    }
    apply_publication_style(float(config.get("style", {}).get("font_size", 8.5)))
    enabled = config.get("figures", {})
    strict = bool(config.get("strict", False))
    manifest_rows: list[dict[str, Any]] = []

    for figure_name, builder in FIGURE_BUILDERS:
        if enabled and not bool(enabled.get(figure_name, True)):
            manifest_rows.append({"figure": figure_name, "status": "disabled"})
            continue
        try:
            saved, metadata = builder(paths, output_dir, config)
            manifest_rows.append(
                {
                    "figure": figure_name,
                    "status": "generated",
                    "files": ";".join(str(path) for path in saved),
                    "metadata": json.dumps(metadata, default=str),
                }
            )
        except (FileNotFoundError, ValueError, KeyError) as error:
            manifest_rows.append(
                {
                    "figure": figure_name,
                    "status": "skipped_missing_input",
                    "reason": str(error),
                }
            )
            if strict:
                raise

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "paper_figures_manifest.csv", index=False)
    return manifest
