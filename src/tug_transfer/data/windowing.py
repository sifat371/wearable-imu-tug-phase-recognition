from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .records import SequenceRecord


def window_starts(length: int, window_length: int, stride: int) -> list[int]:
    if length <= window_length:
        return [0]
    starts = list(range(0, length - window_length + 1, stride))
    last = length - window_length
    if starts[-1] != last:
        starts.append(last)
    return starts


@dataclass
class WindowCollection:
    x: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame


def build_windows(
    records: list[SequenceRecord],
    window_length: int,
    stride: int,
    num_classes: int,
) -> WindowCollection:
    windows: list[np.ndarray] = []
    labels: list[int] = []
    rows: list[dict[str, Any]] = []

    for record in records:
        for start in window_starts(record.length, window_length, stride):
            end = min(start + window_length, record.length)
            x_window = record.x[start:end]
            y_window = record.y[start:end]
            actual_length = len(x_window)
            if actual_length < window_length:
                x_window = np.pad(
                    x_window,
                    ((0, window_length - actual_length), (0, 0)),
                    mode="constant",
                )
            counts = np.bincount(y_window, minlength=num_classes)
            label = int(np.argmax(counts))
            windows.append(x_window.astype(np.float32))
            labels.append(label)
            rows.append(
                {
                    "record_id": record.record_id,
                    "subject_id": record.subject_id,
                    "split": record.split,
                    "start": start,
                    "end": end,
                    "sequence_length": record.length,
                    "true_window_label": label,
                    "purity": float(counts.max() / max(len(y_window), 1)),
                    "is_pure": bool((counts > 0).sum() == 1),
                    **record.metadata,
                }
            )

    if not windows:
        raise ValueError("No windows were generated")
    return WindowCollection(
        x=np.stack(windows).astype(np.float32),
        y=np.asarray(labels, dtype=np.int64),
        metadata=pd.DataFrame(rows),
    )


class WindowDataset(Dataset):
    def __init__(self, collection: WindowCollection) -> None:
        self.collection = collection
        self.metadata = collection.metadata.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.collection.y)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "x": torch.tensor(self.collection.x[index], dtype=torch.float32),
            "y": torch.tensor(self.collection.y[index], dtype=torch.long),
            "metadata": self.metadata.iloc[index].to_dict(),
        }


def collate_windows(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "y": torch.stack([item["y"] for item in batch]),
        "metadata": [item["metadata"] for item in batch],
    }


def reconstruct_timestep_probabilities(
    window_predictions: pd.DataFrame,
    records: list[SequenceRecord],
    phase_names: list[str],
    effective_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    probability_columns = [f"prob_{phase}" for phase in phase_names]
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    for record in records:
        sequence_length = record.length
        probability_sum = np.zeros((sequence_length, len(phase_names)), dtype=np.float64)
        coverage = np.zeros(sequence_length, dtype=np.float64)
        subset = window_predictions[window_predictions["record_id"] == record.record_id]

        for _, window in subset.iterrows():
            start = int(window["start"])
            end = int(window["end"])
            probabilities = window[probability_columns].to_numpy(dtype=np.float64)
            probability_sum[start:end] += probabilities[None, :]
            coverage[start:end] += 1.0

        uncovered = coverage == 0
        probability_sum[uncovered] = 1.0 / len(phase_names)
        coverage[uncovered] = 1.0
        averaged_probabilities = probability_sum / coverage[:, None]
        predictions = averaged_probabilities.argmax(axis=1)

        all_true.append(record.y)
        all_pred.append(predictions)
        for timestep in range(sequence_length):
            true_id = int(record.y[timestep])
            pred_id = int(predictions[timestep])
            row: dict[str, Any] = {
                "record_id": record.record_id,
                "subject_id": record.subject_id,
                "split": record.split,
                "time_step": timestep,
                "true_id": true_id,
                "pred_id": pred_id,
                "true_phase": phase_names[true_id],
                "pred_phase": phase_names[pred_id],
                "n_covering_windows": int(coverage[timestep]),
                **record.metadata,
            }
            if effective_hz is not None:
                row["time_sec"] = timestep / effective_hz
            for class_index, phase_name in enumerate(phase_names):
                row[f"prob_{phase_name}"] = float(
                    averaged_probabilities[timestep, class_index]
                )
            rows.append(row)

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        pd.DataFrame(rows),
    )
