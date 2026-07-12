from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from tug_transfer.constants import IMU_CHANNELS, map_spmt_event, map_tug_event

from .preprocessing import normalize_column_name, resolve_numeric_columns
from .records import SequenceRecord


def infer_subject_id(path: Path) -> str:
    """Infer a participant ID from a WearGait-PD task filename."""
    name = path.stem
    patterns = [
        r"_SelfPace_matTURN$",
        r"_SelfPace_mat$",
        r"_HurriedPace_mat$",
        r"_SelfPace$",
        r"_HurriedPace$",
        r"_TUG$",
    ]
    for pattern in patterns:
        name = re.sub(pattern, "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+\(control\)", "", name, flags=re.IGNORECASE)
    return name.split("_")[0] if "_TUG" in path.stem.upper() else name


def infer_group(path: Path, subject_id: str) -> str:
    path_text = str(path).lower()
    subject = subject_id.upper()
    if "control" in path_text or subject.startswith(("HC", "WHC")):
        return "HC"
    return "PD"


def infer_site(subject_id: str, path: Path | None = None) -> str:
    subject = subject_id.upper()
    if subject.startswith(("NLS", "HC")):
        return "JHU_NLS_HC"
    if subject.startswith(("WPD", "WHC")):
        return "VA_WPD_WHC"
    if path is not None:
        text = str(path).lower()
        if any(token in text for token in ("va", "whc", "wpd")):
            return "VA_WPD_WHC"
    return "JHU_NLS_HC"


def discover_tug_files(control_dir: Path, pd_dir: Path) -> list[Path]:
    files: list[Path] = []
    for directory in (control_dir, pd_dir):
        if not directory.exists():
            raise FileNotFoundError(f"WearGait-PD directory not found: {directory}")
        for path in directory.glob("*.csv"):
            name = path.name.lower()
            if any(
                token in name
                for token in ("tug", "timedupgo", "timed_up_go", "timed-up-go")
            ):
                files.append(path)
    return sorted(files)


def discover_spmt_files(
    control_dir: Path,
    pd_dir: Path,
    pattern: str = "*_SelfPace_matTURN.csv",
) -> list[Path]:
    files: list[Path] = []
    for directory in (control_dir, pd_dir):
        if not directory.exists():
            raise FileNotFoundError(f"WearGait-PD directory not found: {directory}")
        files.extend(sorted(directory.glob(pattern)))
    return sorted(files)


def _find_label_column(frame: pd.DataFrame) -> str:
    for column in frame.columns:
        if normalize_column_name(column) == "generalevent":
            return column
    raise KeyError("GeneralEvent column not found")


def _load_weargait_record(
    path: Path,
    label_mapper: Callable[[object], int | None],
    sample_step: int,
    max_seq_len: int,
    min_seq_len: int,
    allow_missing_channels: bool,
    task: str,
) -> tuple[SequenceRecord | None, dict | None]:
    try:
        frame = pd.read_csv(path)
        label_column = _find_label_column(frame)
        mapped = frame[label_column].map(label_mapper)
        valid = mapped.notna()
        if int(valid.sum()) < min_seq_len:
            raise ValueError(f"Too few valid labeled rows: {int(valid.sum())}")

        valid_frame = frame.loc[valid].copy()
        y = mapped.loc[valid].astype(int).to_numpy(dtype=np.int64)
        x_frame, missing = resolve_numeric_columns(
            valid_frame,
            IMU_CHANNELS,
            allow_missing_fill=allow_missing_channels,
        )
        x = x_frame.to_numpy(dtype=np.float32)

        x = x[::sample_step]
        y = y[::sample_step]
        if len(y) > max_seq_len:
            x = x[:max_seq_len]
            y = y[:max_seq_len]
        if len(y) < min_seq_len:
            raise ValueError(f"Too short after subsampling: {len(y)}")
        if task == "TUG" and len(np.unique(y)) < 2:
            raise ValueError("TUG record has fewer than two phases")

        subject_id = infer_subject_id(path)
        metadata = {
            "path": str(path),
            "site": infer_site(subject_id, path),
            "group": infer_group(path, subject_id),
            "task": task,
            "missing_channels": missing,
            "raw_rows": int(len(frame)),
            "mapped_rows": int(valid.sum()),
        }
        return (
            SequenceRecord(
                record_id=-1,
                subject_id=subject_id,
                x=x,
                y=y,
                metadata=metadata,
            ),
            None,
        )
    except Exception as error:
        return None, {"path": str(path), "error": str(error), "task": task}


def _load_collection(
    paths: list[Path],
    label_mapper: Callable[[object], int | None],
    sample_step: int,
    max_seq_len: int,
    min_seq_len: int,
    allow_missing_channels: bool,
    task: str,
) -> tuple[list[SequenceRecord], list[dict]]:
    records: list[SequenceRecord] = []
    errors: list[dict] = []
    for path in paths:
        record, error = _load_weargait_record(
            path=path,
            label_mapper=label_mapper,
            sample_step=sample_step,
            max_seq_len=max_seq_len,
            min_seq_len=min_seq_len,
            allow_missing_channels=allow_missing_channels,
            task=task,
        )
        if record is not None:
            record.record_id = len(records)
            records.append(record)
        elif error is not None:
            errors.append(error)
    return records, errors


def load_tug_records(
    control_dir: Path,
    pd_dir: Path,
    sample_step: int = 5,
    max_seq_len: int = 1600,
    min_seq_len: int = 20,
    allow_missing_channels: bool = True,
) -> tuple[list[SequenceRecord], list[dict]]:
    return _load_collection(
        discover_tug_files(control_dir, pd_dir),
        map_tug_event,
        sample_step,
        max_seq_len,
        min_seq_len,
        allow_missing_channels,
        "TUG",
    )


def load_spmt_records(
    control_dir: Path,
    pd_dir: Path,
    file_pattern: str = "*_SelfPace_matTURN.csv",
    sample_step: int = 5,
    max_seq_len: int = 1600,
    min_seq_len: int = 20,
) -> tuple[list[SequenceRecord], list[dict]]:
    return _load_collection(
        discover_spmt_files(control_dir, pd_dir, file_pattern),
        map_spmt_event,
        sample_step,
        max_seq_len,
        min_seq_len,
        False,
        "SPmT",
    )
