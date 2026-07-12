from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                labels=labels,
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                labels=labels,
                zero_division=0,
            )
        ),
    }


def classification_report_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).T


def confusion_matrix_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )
    return pd.DataFrame(matrix, index=class_names, columns=class_names)


def inverse_frequency_weights(
    labels: np.ndarray,
    num_classes: int,
    normalize_mean: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    if normalize_mean:
        weights = weights / weights.mean()
    return weights.astype(np.float32), counts


def tolerant_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    radius_steps: int,
) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if radius_steps <= 0:
        return float(np.mean(y_true == y_pred))
    correct = np.zeros(len(y_true), dtype=bool)
    for index in range(len(y_true)):
        start = max(0, index - radius_steps)
        end = min(len(y_true), index + radius_steps + 1)
        correct[index] = y_pred[index] in set(y_true[start:end].tolist())
    return float(correct.mean())


def boundary_tolerant_table(
    timestep_predictions: pd.DataFrame,
    effective_hz: float,
    record_column: str = "record_id",
    time_column: str = "time_step",
    true_column: str = "true_id",
    pred_column: str = "pred_id",
    tolerances_seconds: Iterable[float] = (0.0, 0.25, 0.50, 1.00),
) -> pd.DataFrame:
    rows = []
    for tolerance in tolerances_seconds:
        radius = int(round(tolerance * effective_hz))
        scores: list[float] = []
        weights: list[int] = []
        for _, group in timestep_predictions.groupby(record_column):
            ordered = group.sort_values(time_column)
            y_true = ordered[true_column].to_numpy()
            y_pred = ordered[pred_column].to_numpy()
            scores.append(tolerant_accuracy(y_true, y_pred, radius))
            weights.append(len(group))
        rows.append(
            {
                "tolerance_sec": float(tolerance),
                "radius_steps": radius,
                "mean_sequence_tolerant_accuracy": float(np.mean(scores)),
                "weighted_tolerant_accuracy": float(
                    np.average(scores, weights=weights)
                ),
            }
        )
    return pd.DataFrame(rows)


def grouped_metrics(
    frame: pd.DataFrame,
    group_columns: list[str],
    num_classes: int,
    record_column: str = "record_id",
    subject_column: str = "subject_id",
    true_column: str = "true_id",
    pred_column: str = "pred_id",
) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = frame.groupby(group_columns, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        y_true = group[true_column].to_numpy()
        y_pred = group[pred_column].to_numpy()
        row = {column: value for column, value in zip(group_columns, keys)}
        row.update(
            {
                "n_timesteps": int(len(group)),
                "n_records": int(group[record_column].nunique()),
                "n_subjects": int(group[subject_column].nunique()),
                **classification_metrics(y_true, y_pred, num_classes),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
