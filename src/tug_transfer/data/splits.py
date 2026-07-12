from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .preprocessing import normalize_column_name
from .records import SequenceRecord


VALID_SPLITS = {"train", "val", "test"}


def load_subject_split(path: str | Path) -> dict[str, str]:
    frame = pd.read_csv(path)
    subject_column = None
    split_column = None
    for column in frame.columns:
        token = normalize_column_name(column)
        if token in {"subjectid", "subject", "subjid"}:
            subject_column = column
        if token in {"split", "set", "partition"}:
            split_column = column
    if subject_column is None or split_column is None:
        raise ValueError(
            "Split CSV must contain a subject column and a split column. "
            f"Found: {list(frame.columns)}"
        )

    subset = frame[[subject_column, split_column]].dropna().copy()
    subset[subject_column] = subset[subject_column].astype(str)
    subset[split_column] = subset[split_column].astype(str).str.lower()
    subset = subset[subset[split_column].isin(VALID_SPLITS)]
    duplicated = subset.groupby(subject_column)[split_column].nunique()
    if (duplicated > 1).any():
        conflicting = duplicated[duplicated > 1].index.tolist()
        raise ValueError(f"Subjects have conflicting split assignments: {conflicting[:10]}")
    return (
        subset.drop_duplicates(subject_column)
        .set_index(subject_column)[split_column]
        .to_dict()
    )


def assign_split_map(
    records: list[SequenceRecord],
    split_map: dict[str, str],
    drop_missing: bool = True,
) -> list[SequenceRecord]:
    assigned: list[SequenceRecord] = []
    for record in records:
        split = split_map.get(str(record.subject_id))
        if split in VALID_SPLITS:
            record.split = split
            assigned.append(record)
        elif not drop_missing:
            assigned.append(record)
    return assigned


def _subject_table(records: list[SequenceRecord]) -> pd.DataFrame:
    rows = []
    seen: set[str] = set()
    for record in records:
        if record.subject_id in seen:
            continue
        seen.add(record.subject_id)
        rows.append(
            {
                "subject_id": record.subject_id,
                "site": str(record.metadata.get("site", "UNKNOWN")),
                "group": str(record.metadata.get("group", "UNKNOWN")),
            }
        )
    return pd.DataFrame(rows)


def make_stratified_subject_split(
    records: list[SequenceRecord],
    seed: int = 42,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    stratify: bool = True,
) -> dict[str, str]:
    total = train_fraction + val_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError("Split fractions must sum to 1.0")

    subjects = _subject_table(records)
    if len(subjects) < 3:
        raise ValueError("At least three subjects are required")

    ids = subjects["subject_id"].to_numpy()
    strata = (subjects["site"] + "_" + subjects["group"]).to_numpy()
    temp_fraction = val_fraction + test_fraction
    stratify_values = strata if stratify else None

    try:
        train_ids, temp_ids, _, temp_strata = train_test_split(
            ids,
            strata,
            test_size=temp_fraction,
            random_state=seed,
            stratify=stratify_values,
        )
        test_share_of_temp = test_fraction / temp_fraction
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=test_share_of_temp,
            random_state=seed,
            stratify=temp_strata if stratify else None,
        )
    except ValueError:
        train_ids, temp_ids = train_test_split(
            ids, test_size=temp_fraction, random_state=seed
        )
        test_share_of_temp = test_fraction / temp_fraction
        val_ids, test_ids = train_test_split(
            temp_ids, test_size=test_share_of_temp, random_state=seed
        )

    split_map = {str(subject): "train" for subject in train_ids}
    split_map.update({str(subject): "val" for subject in val_ids})
    split_map.update({str(subject): "test" for subject in test_ids})
    return split_map


def make_fixed_count_subject_split(
    subjects: np.ndarray,
    stratify_values: np.ndarray | None,
    seed: int,
    train_count: int,
    val_count: int,
    test_count: int,
) -> dict[str, str]:
    if train_count + val_count + test_count != len(subjects):
        raise ValueError(
            "Configured train/val/test subject counts do not match the available subjects: "
            f"{train_count}+{val_count}+{test_count}!={len(subjects)}"
        )
    try:
        train_val, test = train_test_split(
            subjects,
            test_size=test_count,
            random_state=seed,
            stratify=stratify_values,
        )
        if stratify_values is None:
            train_val_strata = None
        else:
            lookup = dict(zip(subjects.tolist(), stratify_values.tolist()))
            train_val_strata = np.array([lookup[value] for value in train_val])
        train, val = train_test_split(
            train_val,
            test_size=val_count,
            random_state=seed,
            stratify=train_val_strata,
        )
    except ValueError:
        rng = np.random.default_rng(seed)
        shuffled = subjects.copy()
        rng.shuffle(shuffled)
        test = shuffled[:test_count]
        val = shuffled[test_count : test_count + val_count]
        train = shuffled[test_count + val_count :]

    split_map = {str(subject): "train" for subject in train}
    split_map.update({str(subject): "val" for subject in val})
    split_map.update({str(subject): "test" for subject in test})
    return split_map


def split_dataframe(records: list[SequenceRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "record_id": record.record_id,
                "subject_id": record.subject_id,
                "split": record.split,
                "length": record.length,
                **record.metadata,
            }
        )
    return pd.DataFrame(rows)
