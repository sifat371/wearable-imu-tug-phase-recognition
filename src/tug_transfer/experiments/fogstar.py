from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tug_transfer.data.fogstar import inspect_fogstar_zip, load_fogstar_records
from tug_transfer.data.preprocessing import ChannelNormalizer
from tug_transfer.data.splits import (
    assign_split_map,
    load_subject_split,
    make_fixed_count_subject_split,
    split_dataframe,
)
from tug_transfer.training.transfer import run_transfer_experiment
from tug_transfer.utils import ensure_dir, save_json, seed_everything


def _numeric_subject_key(subject_id: str | int | float) -> int:
    """Return a numeric key for FoG-STAR subject identifiers.

    FoG-STAR subject IDs are numeric. Converting before sorting prevents
    lexicographic ordering such as 1, 10, 11, ..., 2, 20, ... .
    """
    return int(float(str(subject_id).strip()))


def _normalise_split_map(split_map: dict[str, str]) -> dict[str, str]:
    """Normalise numeric subject keys to canonical integer strings."""
    normalised: dict[str, str] = {}
    for subject_id, split in split_map.items():
        key = str(_numeric_subject_key(subject_id))
        if key in normalised and normalised[key] != split:
            raise ValueError(
                f"Subject {key} has conflicting split assignments: "
                f"{normalised[key]!r} and {split!r}."
            )
        normalised[key] = str(split).lower()
    return normalised


def _load_or_create_subject_split(
    *,
    records: list,
    paths: dict[str, Any],
    split_config: dict[str, Any],
    seed: int,
    output_dir: Path,
) -> dict[str, str]:
    """Load the fixed paper split, or create a numerically sorted fallback.

    For exact paper reproduction, set:

        paths.subject_split_csv: artifacts/paper/fogstar_subject_split.csv

    The fallback remains available for new experiments, but it sorts subject
    IDs numerically before applying the seeded split.
    """
    configured_split = paths.get("subject_split_csv")

    available_subjects = sorted(
        {_numeric_subject_key(record.subject_id) for record in records}
    )
    available_keys = {str(subject_id) for subject_id in available_subjects}

    if configured_split:
        split_path = Path(configured_split)
        if not split_path.exists():
            raise FileNotFoundError(
                f"Configured FoG-STAR split file was not found: {split_path}"
            )

        split_map = _normalise_split_map(load_subject_split(split_path))
        split_keys = set(split_map)

        missing = sorted(available_keys - split_keys, key=int)
        extra = sorted(split_keys - available_keys, key=int)
        if missing:
            raise ValueError(
                "The fixed FoG-STAR split is missing available subjects: "
                f"{missing}"
            )
        if extra:
            raise ValueError(
                "The fixed FoG-STAR split contains subjects not present in "
                f"the loaded data: {extra}"
            )

        counts = pd.Series(split_map).value_counts().to_dict()
        expected_counts = {
            "train": int(split_config["train_subjects"]),
            "val": int(split_config["val_subjects"]),
            "test": int(split_config["test_subjects"]),
        }
        for split_name, expected_count in expected_counts.items():
            actual_count = int(counts.get(split_name, 0))
            if actual_count != expected_count:
                raise ValueError(
                    f"Fixed split has {actual_count} {split_name} subjects; "
                    f"expected {expected_count}."
                )

        split_source = {
            "mode": "fixed_csv",
            "path": str(split_path.resolve()),
        }
    else:
        subjects = np.asarray(available_subjects, dtype=int)

        has_fog_lookup: dict[int, int] = {}
        for subject in subjects:
            has_fog_lookup[int(subject)] = max(
                int(record.metadata.get("has_fog", 0))
                for record in records
                if _numeric_subject_key(record.subject_id) == int(subject)
            )

        stratify_values = (
            np.asarray(
                [has_fog_lookup[int(subject)] for subject in subjects],
                dtype=int,
            )
            if bool(split_config.get("stratify_by_has_fog", True))
            else None
        )

        split_map = make_fixed_count_subject_split(
            subjects,
            stratify_values,
            seed=seed,
            train_count=int(split_config["train_subjects"]),
            val_count=int(split_config["val_subjects"]),
            test_count=int(split_config["test_subjects"]),
        )
        split_map = _normalise_split_map(split_map)
        split_source = {
            "mode": "generated_numeric_sort",
            "seed": seed,
            "warning": (
                "A split was generated because paths.subject_split_csv was "
                "not configured. Use a fixed CSV for exact reproduction."
            ),
        }

    canonical_rows = [
        {"subject_id": int(subject_id), "split": split_map[subject_id]}
        for subject_id in sorted(split_map, key=int)
    ]
    pd.DataFrame(canonical_rows).to_csv(
        output_dir / "canonical_subject_split.csv", index=False
    )
    save_json(split_source, output_dir / "subject_split_source.json")

    print("FoG-STAR subject split:")
    for split_name in ("train", "val", "test"):
        ids = [
            int(subject_id)
            for subject_id in sorted(split_map, key=int)
            if split_map[subject_id] == split_name
        ]
        print(f"  {split_name}: {ids}")

    return split_map


def run_fogstar_experiment(config: dict[str, Any]) -> pd.DataFrame:
    seed = int(config["seed"])
    seed_everything(seed)

    paths = config["paths"]
    data_config = config["data"]
    split_config = config["split"]

    output_dir = ensure_dir(paths["output_dir"])
    archive_path = Path(paths["fogstar_zip"])
    tug_output_dir = Path(paths["tug_output_dir"])

    source_checkpoint = (
        tug_output_dir / "checkpoints" / "dense_sequence_cnn_bilstm_best.pt"
    )
    mean_path = tug_output_dir / "imu_train_mean.npy"
    std_path = tug_output_dir / "imu_train_std.npy"

    for required in (source_checkpoint, mean_path, std_path):
        if not required.exists():
            raise FileNotFoundError(
                f"Required TUG artifact not found: {required}. "
                "Run the TUG experiment first."
            )

    inspection = inspect_fogstar_zip(archive_path)
    save_json(inspection, output_dir / "fogstar_archive_inspection.json")

    records, errors, clinical, target_phases, tug_to_target = load_fogstar_records(
        archive_path,
        target_mode=str(data_config.get("target_mode", "activity6")),
        drop_activity_zero=bool(data_config.get("drop_activity_zero", True)),
        raw_hz=int(data_config["raw_hz"]),
        target_hz=int(data_config["target_hz"]),
        max_seq_len=int(data_config["max_seq_len"]),
        min_seq_len=int(data_config.get("min_seq_len", 20)),
        convert_acc_g_to_ms2=bool(
            data_config.get("convert_acc_g_to_ms2", True)
        ),
        convert_gyro_deg_to_rad=bool(
            data_config.get("convert_gyro_deg_to_rad", True)
        ),
        freeacc_method=str(
            data_config.get("freeacc_method", "rolling_center")
        ),
        freeacc_rolling_sec=float(
            data_config.get("freeacc_rolling_sec", 1.0)
        ),
    )

    if not records:
        raise RuntimeError("No usable FoG-STAR records were found")

    pd.DataFrame(errors).to_csv(output_dir / "excluded_records.csv", index=False)
    clinical.to_csv(output_dir / "clinical_data_copy.csv", index=False)

    # Canonicalise record subject IDs before split assignment so that keys from
    # the fixed split CSV and the loaded records always match.
    for record in records:
        record.subject_id = str(_numeric_subject_key(record.subject_id))

    split_map = _load_or_create_subject_split(
        records=records,
        paths=paths,
        split_config=split_config,
        seed=seed,
        output_dir=output_dir,
    )

    records = assign_split_map(records, split_map, drop_missing=True)
    if not records:
        raise RuntimeError("No FoG-STAR records remained after split assignment")

    for index, record in enumerate(records):
        record.record_id = index

    metadata = split_dataframe(records)
    metadata.to_csv(output_dir / "subject_split.csv", index=False)
    metadata.to_csv(output_dir / "fogstar_record_metadata.csv", index=False)

    normalizer = ChannelNormalizer.load(mean_path, std_path)
    effective_hz = float(data_config["raw_hz"]) / int(
        round(
            float(data_config["raw_hz"])
            / float(data_config["target_hz"])
        )
    )

    save_json(
        {
            "normalizer_mean": str(mean_path),
            "normalizer_std": str(std_path),
            "source_checkpoint": str(source_checkpoint),
            "target_phases": target_phases,
            "tug_to_target": tug_to_target,
            "note": "FoG-STAR uses the TUG training-set normalizer.",
        },
        output_dir / "used_tug_artifacts.json",
    )

    return run_transfer_experiment(
        records=records,
        normalizer=normalizer,
        target_phases=target_phases,
        tug_to_target=tug_to_target,
        source_checkpoint=source_checkpoint,
        output_dir=output_dir,
        config=config,
        experiment_name="fogstar",
        effective_hz=effective_hz,
        subgroup_analyses={
            "subjectwise_test_metrics_paper_best.csv": ["subject_id"],
            "taskwise_test_metrics_paper_best.csv": ["task_id"],
        },
        fog_analysis=True,
    )