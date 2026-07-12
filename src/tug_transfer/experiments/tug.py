from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tug_transfer.constants import IMU_CHANNELS, TUG_PHASES
from tug_transfer.data.datasets import IGNORE_INDEX, SequenceDataset, collate_sequences
from tug_transfer.data.preprocessing import ChannelNormalizer
from tug_transfer.data.splits import (
    assign_split_map,
    load_subject_split,
    make_stratified_subject_split,
    split_dataframe,
)
from tug_transfer.data.weargait import load_tug_records
from tug_transfer.data.windowing import (
    WindowDataset,
    build_windows,
    collate_windows,
    reconstruct_timestep_probabilities,
)
from tug_transfer.models import DenseCNNBiLSTM, WindowCNNBiLSTM
from tug_transfer.training.engine import (
    evaluate_sequence_model,
    evaluate_window_model,
    fit_sequence_model,
    fit_window_model,
)
from tug_transfer.training.metrics import (
    boundary_tolerant_table,
    classification_metrics,
    classification_report_frame,
    confusion_matrix_frame,
    grouped_metrics,
    inverse_frequency_weights,
)
from tug_transfer.training.plots import (
    save_comparison_plot,
    save_confusion_matrix_plot,
)
from tug_transfer.utils import ensure_dir, save_json, seed_everything, select_device


def _make_sequence_loaders(
    records,
    normalizer,
    batch_size,
    num_workers,
    pin_memory,
):
    loaders = {}
    for split in ("train", "val", "test"):
        subset = [record for record in records if record.split == split]
        if not subset:
            raise ValueError(f"No TUG records in split: {split}")
        loaders[split] = DataLoader(
            SequenceDataset(subset, normalizer=normalizer),
            batch_size=batch_size,
            shuffle=split == "train",
            collate_fn=collate_sequences,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return loaders


def run_tug_experiment(config: dict[str, Any]) -> pd.DataFrame:
    seed = int(config["seed"])
    seed_everything(seed)
    paths = config["paths"]
    data_config = config["data"]
    split_config = config["split"]
    window_config = config["window"]
    model_config = config["model"]
    training_config = config["training"]
    run_config = config.get("run", {"dense": True, "window": True})
    if training_config.get("torch_num_threads") is not None:
        torch.set_num_threads(int(training_config["torch_num_threads"]))

    output_dir = ensure_dir(paths["output_dir"])
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    plots_dir = ensure_dir(output_dir / "plots")
    device = select_device(training_config.get("device", "auto"))
    pin_memory = bool(training_config.get("pin_memory", True)) and device.type == "cuda"
    use_amp = bool(training_config.get("use_amp", False)) and device.type == "cuda"

    control_dir = Path(paths["control_dir"])
    pd_dir = Path(paths["pd_dir"])
    records, errors = load_tug_records(
        control_dir,
        pd_dir,
        sample_step=int(data_config["sample_step"]),
        max_seq_len=int(data_config["max_seq_len"]),
        min_seq_len=int(data_config.get("min_seq_len", 20)),
        allow_missing_channels=bool(data_config.get("allow_missing_channels", True)),
    )
    if not records:
        raise RuntimeError("No usable TUG records were found")
    pd.DataFrame(errors).to_csv(output_dir / "excluded_records.csv", index=False)

    split_csv = paths.get("split_csv")
    if split_csv:
        split_map = load_subject_split(split_csv)
        split_source = str(split_csv)
    else:
        split_map = make_stratified_subject_split(
            records,
            seed=seed,
            train_fraction=float(split_config["train_fraction"]),
            val_fraction=float(split_config["val_fraction"]),
            test_fraction=float(split_config["test_fraction"]),
            stratify=bool(split_config.get("stratify_by_site_group", True)),
        )
        split_source = "generated_seeded_subject_split"
    records = assign_split_map(records, split_map, drop_missing=True)
    if not records:
        raise RuntimeError("No records remained after applying the split")
    for index, record in enumerate(records):
        record.record_id = index

    metadata = split_dataframe(records)
    metadata.to_csv(output_dir / "subject_level_split.csv", index=False)
    metadata.to_csv(output_dir / "record_metadata.csv", index=False)

    train_records = [record for record in records if record.split == "train"]
    normalization_config = config.get("normalization", {})
    reference_mean = paths.get("normalization_mean")
    reference_std = paths.get("normalization_std")
    use_reference = bool(normalization_config.get("use_reference", False))
    if use_reference:
        if not reference_mean or not reference_std:
            raise ValueError(
                "normalization.use_reference=true requires paths.normalization_mean "
                "and paths.normalization_std"
            )
        normalizer = ChannelNormalizer.load(reference_mean, reference_std)
        if len(normalizer.mean) != len(IMU_CHANNELS):
            raise ValueError(
                "Reference normalization vectors must contain one value per IMU channel"
            )
        fitted = ChannelNormalizer.fit(train_records)
        comparison = pd.DataFrame(
            {
                "channel": IMU_CHANNELS,
                "reference_mean": normalizer.mean,
                "recomputed_mean": fitted.mean,
                "mean_abs_difference": np.abs(normalizer.mean - fitted.mean),
                "reference_std": normalizer.std,
                "recomputed_std": fitted.std,
                "std_abs_difference": np.abs(normalizer.std - fitted.std),
            }
        )
        comparison.to_csv(output_dir / "normalization_reference_check.csv", index=False)
        tolerance = float(normalization_config.get("verification_tolerance", 1e-4))
        if bool(normalization_config.get("strict_verification", False)) and (
            comparison["mean_abs_difference"].max() > tolerance
            or comparison["std_abs_difference"].max() > tolerance
        ):
            raise ValueError(
                "Recomputed training normalization does not match the supplied paper vectors. "
                "Inspect normalization_reference_check.csv."
            )
        normalization_source = "reference_artifacts"
    else:
        normalizer = ChannelNormalizer.fit(train_records)
        normalization_source = "computed_from_training_split"
    normalizer.save(
        output_dir / "imu_train_mean.npy",
        output_dir / "imu_train_std.npy",
    )

    effective_hz = float(data_config["assumed_raw_hz"]) / int(data_config["sample_step"])
    sequence_loaders = _make_sequence_loaders(
        records,
        normalizer,
        batch_size=int(training_config["batch_size_dense"]),
        num_workers=int(training_config.get("num_workers", 0)),
        pin_memory=pin_memory,
    )

    summary_rows: list[dict[str, Any]] = []

    if bool(run_config.get("dense", True)):
        train_labels = np.concatenate([record.y for record in train_records])
        weights_numpy, counts = inverse_frequency_weights(
            train_labels,
            len(TUG_PHASES),
            normalize_mean=False,
        )
        pd.DataFrame(
            {
                "phase": TUG_PHASES,
                "train_count": counts.astype(int),
                "class_weight": weights_numpy,
            }
        ).to_csv(output_dir / "dense_sequence_class_weights.csv", index=False)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights_numpy, dtype=torch.float32, device=device),
            ignore_index=IGNORE_INDEX,
        )
        dense_model = DenseCNNBiLSTM(
            in_channels=len(IMU_CHANNELS),
            conv_channels=int(model_config["conv_channels"]),
            lstm_hidden=int(model_config["lstm_hidden"]),
            num_classes=len(TUG_PHASES),
            dropout=float(model_config["dropout"]),
        ).to(device)
        optimizer = torch.optim.AdamW(
            dense_model.parameters(),
            lr=float(training_config["learning_rate"]),
            weight_decay=float(training_config["weight_decay"]),
        )
        dense_model, history, best = fit_sequence_model(
            dense_model,
            sequence_loaders["train"],
            sequence_loaders["val"],
            optimizer,
            criterion,
            device,
            len(TUG_PHASES),
            int(training_config["max_epochs"]),
            int(training_config["patience"]),
            float(training_config["grad_clip"]),
            checkpoints_dir / "dense_sequence_cnn_bilstm_best.pt",
            use_amp=use_amp,
            selection_order=tuple(training_config.get("dense_selection_order", ["macro_f1"])),
        )
        history.to_csv(output_dir / "dense_sequence_training_history.csv", index=False)

        sequence_parts = []
        timestep_parts = []
        evaluations = {}
        for split, loader in sequence_loaders.items():
            result = evaluate_sequence_model(
                dense_model,
                loader,
                device,
                TUG_PHASES,
                split_name=split,
                criterion=criterion,
                use_amp=use_amp,
                effective_hz=effective_hz,
            )
            evaluations[split] = result
            sequence_parts.append(result.sequence_predictions)
            timestep_parts.append(result.timestep_predictions)
        pd.concat(sequence_parts, ignore_index=True).to_csv(
            output_dir / "dense_sequence_subjectwise_metrics.csv",
            index=False,
        )
        pd.concat(timestep_parts, ignore_index=True).to_csv(
            output_dir / "dense_sequence_timestep_predictions.csv",
            index=False,
        )
        test_result = evaluations["test"]
        classification_report_frame(
            test_result.y_true,
            test_result.y_pred,
            TUG_PHASES,
        ).to_csv(output_dir / "dense_sequence_test_classification_report.csv")
        dense_confusion = confusion_matrix_frame(
            test_result.y_true,
            test_result.y_pred,
            TUG_PHASES,
        )
        dense_confusion.to_csv(output_dir / "dense_sequence_test_confusion_matrix.csv")
        save_confusion_matrix_plot(
            dense_confusion,
            "Dense sequence CNN-BiLSTM - test",
            plots_dir / "dense_sequence_test_confusion_matrix.png",
        )
        boundary_tolerant_table(
            test_result.timestep_predictions,
            effective_hz=effective_hz,
        ).to_csv(
            output_dir / "dense_sequence_boundary_tolerant_test.csv",
            index=False,
        )
        summary_rows.append(
            {
                "approach": "Dense timestep classification",
                "model": "CNNBiLSTM",
                "input": "full/long sequence [B,L,27]",
                "output": "label per timestep",
                "best_epoch": int(best["epoch"]),
                **test_result.metrics,
            }
        )

    if bool(run_config.get("window", True)):
        normalized_records = [normalizer.transform_record(record) for record in records]
        collections = {
            split: build_windows(
                [record for record in normalized_records if record.split == split],
                window_length=int(window_config["length"]),
                stride=int(window_config["stride"]),
                num_classes=len(TUG_PHASES),
            )
            for split in ("train", "val", "test")
        }
        pd.concat(
            [collection.metadata for collection in collections.values()],
            ignore_index=True,
        ).to_csv(output_dir / "window_metadata.csv", index=False)

        window_loaders = {
            split: DataLoader(
                WindowDataset(collection),
                batch_size=int(training_config["batch_size_window"]),
                shuffle=split == "train",
                collate_fn=collate_windows,
                num_workers=int(training_config.get("num_workers", 0)),
                pin_memory=pin_memory,
            )
            for split, collection in collections.items()
        }
        weights_numpy, counts = inverse_frequency_weights(
            collections["train"].y,
            len(TUG_PHASES),
            normalize_mean=False,
        )
        pd.DataFrame(
            {
                "phase": TUG_PHASES,
                "train_count": counts.astype(int),
                "class_weight": weights_numpy,
            }
        ).to_csv(output_dir / "window_class_weights.csv", index=False)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights_numpy, dtype=torch.float32, device=device)
        )
        window_model = WindowCNNBiLSTM(
            in_channels=len(IMU_CHANNELS),
            conv_channels=int(model_config["conv_channels"]),
            lstm_hidden=int(model_config["lstm_hidden"]),
            num_classes=len(TUG_PHASES),
            dropout=float(model_config["dropout"]),
        ).to(device)
        optimizer = torch.optim.AdamW(
            window_model.parameters(),
            lr=float(training_config["learning_rate"]),
            weight_decay=float(training_config["weight_decay"]),
        )
        window_model, history, best = fit_window_model(
            window_model,
            window_loaders["train"],
            window_loaders["val"],
            optimizer,
            criterion,
            device,
            TUG_PHASES,
            int(training_config["max_epochs"]),
            int(training_config["patience"]),
            float(training_config["grad_clip"]),
            checkpoints_dir / "window_cnn_bilstm_best.pt",
            selection_order=tuple(training_config.get("window_selection_order", ["macro_f1"])),
        )
        history.to_csv(output_dir / "window_training_history.csv", index=False)

        window_results = {
            split: evaluate_window_model(
                window_model,
                loader,
                device,
                TUG_PHASES,
                split_name=split,
                criterion=criterion,
            )
            for split, loader in window_loaders.items()
        }
        pd.concat(
            [result.predictions for result in window_results.values()],
            ignore_index=True,
        ).to_csv(output_dir / "window_native_predictions.csv", index=False)

        native_test = window_results["test"]
        classification_report_frame(
            native_test.y_true,
            native_test.y_pred,
            TUG_PHASES,
        ).to_csv(output_dir / "window_native_test_classification_report.csv")
        native_confusion = confusion_matrix_frame(
            native_test.y_true,
            native_test.y_pred,
            TUG_PHASES,
        )
        native_confusion.to_csv(output_dir / "window_native_test_confusion_matrix.csv")
        save_confusion_matrix_plot(
            native_confusion,
            "Window CNN-BiLSTM native labels - test",
            plots_dir / "window_native_test_confusion_matrix.png",
        )

        test_normalized_records = [
            record for record in normalized_records if record.split == "test"
        ]
        recon_true, recon_pred, recon_frame = reconstruct_timestep_probabilities(
            native_test.predictions,
            test_normalized_records,
            TUG_PHASES,
            effective_hz=effective_hz,
        )
        recon_frame.to_csv(
            output_dir / "window_reconstructed_timestep_predictions.csv",
            index=False,
        )
        recon_metrics = classification_metrics(
            recon_true,
            recon_pred,
            len(TUG_PHASES),
        )
        classification_report_frame(
            recon_true,
            recon_pred,
            TUG_PHASES,
        ).to_csv(output_dir / "window_reconstructed_test_classification_report.csv")
        recon_confusion = confusion_matrix_frame(
            recon_true,
            recon_pred,
            TUG_PHASES,
        )
        recon_confusion.to_csv(
            output_dir / "window_reconstructed_test_confusion_matrix.csv"
        )
        save_confusion_matrix_plot(
            recon_confusion,
            "Window predictions reconstructed to timesteps - test",
            plots_dir / "window_reconstructed_test_confusion_matrix.png",
        )

        purity_rows = []
        total_test_windows = len(native_test.predictions)
        for flag, name in ((True, "pure"), (False, "mixed_boundary")):
            subset = native_test.predictions[
                native_test.predictions["is_pure"] == flag
            ]
            if not subset.empty:
                purity_rows.append(
                    {
                        "window_type": name,
                        "n_windows": len(subset),
                        "percentage": 100.0 * len(subset) / total_test_windows,
                        **classification_metrics(
                            subset["y_true"].to_numpy(),
                            subset["y_pred"].to_numpy(),
                            len(TUG_PHASES),
                        ),
                    }
                )
        pd.DataFrame(purity_rows).to_csv(
            output_dir / "window_test_purity_analysis.csv",
            index=False,
        )

        grouped_metrics(
            recon_frame,
            ["record_id"],
            len(TUG_PHASES),
        ).to_csv(
            output_dir / "window_reconstructed_subjectwise_metrics.csv",
            index=False,
        )
        boundary_tolerant_table(
            recon_frame,
            effective_hz=effective_hz,
        ).to_csv(
            output_dir / "window_reconstructed_boundary_tolerant_test.csv",
            index=False,
        )

        summary_rows.extend(
            [
                {
                    "approach": "Window classification",
                    "model": "WindowCNNBiLSTM",
                    "input": (
                        f"{int(window_config['length'])}-sample window "
                        f"[B,{int(window_config['length'])},27]"
                    ),
                    "output": "one majority label per window",
                    "best_epoch": int(best["epoch"]),
                    **native_test.metrics,
                },
                {
                    "approach": "Window to timestep reconstruction",
                    "model": "WindowCNNBiLSTM",
                    "input": (
                        f"overlapping {int(window_config['length'])}-sample windows"
                    ),
                    "output": "reconstructed label per timestep",
                    "best_epoch": int(best["epoch"]),
                    "loss": float("nan"),
                    **recon_metrics,
                },
            ]
        )

    comparison = pd.DataFrame(summary_rows)
    comparison.to_csv(
        output_dir / "imu_sequence_vs_window_cnn_bilstm_comparison.csv",
        index=False,
    )
    if not comparison.empty:
        save_comparison_plot(
            comparison,
            "approach",
            ["balanced_accuracy", "macro_f1"],
            "IMU-only CNN-BiLSTM: dense versus window formulation",
            plots_dir / "imu_sequence_vs_window_comparison.png",
        )

    save_json(
        {
            "seed": seed,
            "device": str(device),
            "split_source": split_source,
            "normalization_source": normalization_source,
            "input_channels": len(IMU_CHANNELS),
            "phase_names": TUG_PHASES,
            "sample_step": int(data_config["sample_step"]),
            "assumed_raw_hz": float(data_config["assumed_raw_hz"]),
            "effective_hz": effective_hz,
            "max_seq_len": int(data_config["max_seq_len"]),
            "window_length": int(window_config["length"]),
            "window_stride": int(window_config["stride"]),
            "n_records": len(records),
            "n_train_records": sum(record.split == "train" for record in records),
            "n_val_records": sum(record.split == "val" for record in records),
            "n_test_records": sum(record.split == "test" for record in records),
        },
        output_dir / "experiment_config.json",
    )
    print(f"TUG outputs saved to: {output_dir.resolve()}")
    return comparison
