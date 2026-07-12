from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tug_transfer.constants import SPMT_PHASES, TUG_TO_SPMT
from tug_transfer.data.preprocessing import ChannelNormalizer
from tug_transfer.data.splits import assign_split_map, load_subject_split, split_dataframe
from tug_transfer.data.weargait import load_spmt_records
from tug_transfer.training.transfer import run_transfer_experiment
from tug_transfer.utils import ensure_dir, save_json, seed_everything


def run_spmt_experiment(config: dict[str, Any]) -> pd.DataFrame:
    seed = int(config["seed"])
    seed_everything(seed)
    paths = config["paths"]
    data_config = config["data"]

    output_dir = ensure_dir(paths["output_dir"])
    tug_output_dir = Path(paths["tug_output_dir"])
    source_checkpoint = (
        tug_output_dir / "checkpoints" / "dense_sequence_cnn_bilstm_best.pt"
    )
    mean_path = tug_output_dir / "imu_train_mean.npy"
    std_path = tug_output_dir / "imu_train_std.npy"
    split_path = tug_output_dir / "subject_level_split.csv"
    for required in (source_checkpoint, mean_path, std_path, split_path):
        if not required.exists():
            raise FileNotFoundError(
                f"Required TUG artifact not found: {required}. Run the TUG experiment first."
            )

    records, errors = load_spmt_records(
        Path(paths["control_dir"]),
        Path(paths["pd_dir"]),
        file_pattern=str(data_config.get("file_pattern", "*_SelfPace_matTURN.csv")),
        sample_step=int(data_config["sample_step"]),
        max_seq_len=int(data_config["max_seq_len"]),
        min_seq_len=int(data_config.get("min_seq_len", 20)),
    )
    if not records:
        raise RuntimeError("No usable SPmT records were found")
    pd.DataFrame(errors).to_csv(output_dir / "excluded_spmt_records.csv", index=False)

    split_map = load_subject_split(split_path)
    records = assign_split_map(records, split_map, drop_missing=True)
    if not records:
        raise RuntimeError("No SPmT records matched the TUG subject split")
    for index, record in enumerate(records):
        record.record_id = index
    split_dataframe(records).to_csv(output_dir / "spmt_subject_split.csv", index=False)
    split_dataframe(records).to_csv(output_dir / "spmt_record_metadata.csv", index=False)

    normalizer = ChannelNormalizer.load(mean_path, std_path)
    effective_hz = float(data_config["assumed_raw_hz"]) / int(data_config["sample_step"])

    save_json(
        {
            "normalizer_mean": str(mean_path),
            "normalizer_std": str(std_path),
            "source_checkpoint": str(source_checkpoint),
            "source_split": str(split_path),
            "note": "SPmT reuses the TUG training-set normalizer and subject split.",
        },
        output_dir / "used_tug_artifacts.json",
    )

    return run_transfer_experiment(
        records=records,
        normalizer=normalizer,
        target_phases=SPMT_PHASES,
        tug_to_target=TUG_TO_SPMT,
        source_checkpoint=source_checkpoint,
        output_dir=output_dir,
        config=config,
        experiment_name="spmt",
        effective_hz=effective_hz,
        subgroup_analyses={
            "sitewise_test_metrics_paper_best.csv": ["site"],
            "groupwise_test_metrics_paper_best.csv": ["group"],
            "site_group_test_metrics_paper_best.csv": ["site", "group"],
        },
        fog_analysis=False,
    )
