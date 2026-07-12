from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .records import SequenceRecord


def normalize_column_name(value: object) -> str:
    return "".join(ch.lower() for ch in str(value).strip() if ch.isalnum())


def resolve_numeric_columns(
    frame: pd.DataFrame,
    desired_columns: list[str],
    allow_missing_fill: bool,
) -> tuple[pd.DataFrame, list[str]]:
    lookup = {normalize_column_name(column): column for column in frame.columns}
    output = pd.DataFrame(index=frame.index)
    missing: list[str] = []
    for desired in desired_columns:
        key = normalize_column_name(desired)
        if key in lookup:
            output[desired] = pd.to_numeric(frame[lookup[key]], errors="coerce")
        elif allow_missing_fill:
            output[desired] = 0.0
            missing.append(desired)
        else:
            raise KeyError(f"Missing required input channel: {desired}")
    output = output.interpolate(method="linear", limit_direction="both").fillna(0.0)
    return output, missing


@dataclass
class ChannelNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, records: Iterable[SequenceRecord]) -> "ChannelNormalizer":
        arrays = [record.x for record in records]
        if not arrays:
            raise ValueError("Cannot fit a normalizer without records")
        stacked = np.concatenate(arrays, axis=0).astype(np.float32)
        mean = stacked.mean(axis=0).astype(np.float32)
        std = stacked.std(axis=0).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return cls(mean=mean, std=std)

    def transform_array(self, x: np.ndarray) -> np.ndarray:
        if x.shape[-1] != len(self.mean):
            raise ValueError(
                f"Channel mismatch: input has {x.shape[-1]}, normalizer has {len(self.mean)}"
            )
        return ((x.astype(np.float32) - self.mean) / self.std).astype(np.float32)

    def transform_record(self, record: SequenceRecord) -> SequenceRecord:
        return record.clone_with_x(self.transform_array(record.x))

    def save(self, mean_path: str | Path, std_path: str | Path) -> None:
        np.save(mean_path, self.mean)
        np.save(std_path, self.std)

    @classmethod
    def load(cls, mean_path: str | Path, std_path: str | Path) -> "ChannelNormalizer":
        mean = np.load(mean_path).astype(np.float32).reshape(-1)
        std = np.load(std_path).astype(np.float32).reshape(-1)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return cls(mean=mean, std=std)
