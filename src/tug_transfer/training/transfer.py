from __future__ import annotations

import json
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
from tug_transfer.data.records import SequenceRecord
from tug_transfer.models import (
    DenseCNNBiLSTM,
    FrozenEncoderWithHead,
    TimeStepMLPHead,
    copy_encoder_weights,
    load_dense_checkpoint,
)
from tug_transfer.utils import ensure_dir, save_json, seed_everything, torch_load

from .engine import (
    evaluate_sequence_metrics,
    evaluate_sequence_model,
    fit_sequence_model,
)
from .metrics import (
    boundary_tolerant_table,
    classification_report_frame,
    confusion_matrix_frame,
    grouped_metrics,
    inverse_frequency_weights,
)
from .plots import save_comparison_plot, save_confusion_matrix_plot, save_timeline_plot


def _loaders(
    records: list[SequenceRecord],
    normalizer: ChannelNormalizer,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> dict[str, DataLoader]:
    output: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        split_records = [record for record in records if record.split == split]
        if not split_records:
            raise ValueError(f"No records are assigned to the {split} split")
        output[split] = DataLoader(
            SequenceDataset(split_records, normalizer=normalizer),
            batch_size=batch_size,
            shuffle=split == "train",
            collate_fn=collate_sequences,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return output


@torch.no_grad()
def _evaluate_zero_shot(
    source_model: DenseCNNBiLSTM,
    loader: DataLoader,
    mapping: dict[int, int],
    target_phases: list[str],
    device: torch.device,
    split: str,
) -> dict[str, Any]:
    from .metrics import classification_metrics

    source_model.eval()
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    mapping_array = np.array(
        [mapping[index] for index in range(len(TUG_PHASES))],
        dtype=np.int64,
    )
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        source_predictions = source_model(
            x, mask=mask, lengths=batch["lengths"]
        ).argmax(dim=-1)
        target_predictions = mapping_array[source_predictions.detach().cpu().numpy()]
        y_numpy = y.detach().cpu().numpy()
        mask_numpy = mask.detach().cpu().numpy()
        all_true.append(y_numpy[mask_numpy])
        all_pred.append(target_predictions[mask_numpy])

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return {
        "condition": "zero_shot_tug_head_projection",
        "split": split,
        **classification_metrics(y_true, y_pred, len(target_phases)),
    }


def _candidate_rank(candidate: dict[str, Any]) -> tuple[float, float, float]:
    validation = candidate["val"]
    return (
        validation["balanced_accuracy"],
        validation["macro_f1"],
        -validation["loss"],
    )


def _evaluate_all_splits(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    target_phases: list[str],
    criterion: nn.Module,
    use_amp: bool,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for split, loader in loaders.items():
        metrics[split] = evaluate_sequence_metrics(
            model,
            loader,
            device,
            len(target_phases),
            criterion=criterion,
            use_amp=use_amp,
        )
    return metrics


def _build_target_model(
    model_config: dict[str, Any],
    num_classes: int,
    dropout: float,
    device: torch.device,
) -> DenseCNNBiLSTM:
    return DenseCNNBiLSTM(
        in_channels=len(IMU_CHANNELS),
        conv_channels=int(model_config["conv_channels"]),
        lstm_hidden=int(model_config["lstm_hidden"]),
        num_classes=num_classes,
        dropout=float(dropout),
    ).to(device)


def run_transfer_experiment(
    *,
    records: list[SequenceRecord],
    normalizer: ChannelNormalizer,
    target_phases: list[str],
    tug_to_target: dict[int, int],
    source_checkpoint: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    experiment_name: str,
    effective_hz: float,
    subgroup_analyses: dict[str, list[str]] | None = None,
    fog_analysis: bool = False,
) -> pd.DataFrame:
    """Run zero-shot, frozen-head, fine-tuned, and from-scratch conditions."""
    output_dir = ensure_dir(output_dir)
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    plots_dir = ensure_dir(output_dir / "plots")
    seed = int(config["seed"])
    seed_everything(seed)

    training_config = config["training"]
    model_config = config["model"]
    if training_config.get("torch_num_threads") is not None:
        torch.set_num_threads(int(training_config["torch_num_threads"]))
    device_name = training_config.get("device", "auto")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    use_amp = bool(training_config.get("use_amp", False)) and device.type == "cuda"
    pin_memory = bool(training_config.get("pin_memory", True)) and device.type == "cuda"

    loaders = _loaders(
        records,
        normalizer,
        batch_size=int(training_config["batch_size"]),
        num_workers=int(training_config.get("num_workers", 0)),
        pin_memory=pin_memory,
    )

    source_model, source_info = load_dense_checkpoint(
        source_checkpoint,
        in_channels=len(IMU_CHANNELS),
        conv_channels=int(model_config["conv_channels"]),
        lstm_hidden=int(model_config["lstm_hidden"]),
        num_classes=len(TUG_PHASES),
        dropout=float(model_config.get("source_dropout", 0.25)),
        device=device,
        freeze=True,
    )
    save_json(
        {
            "checkpoint": str(source_checkpoint),
            "checkpoint_info": source_info,
            "source_phases": TUG_PHASES,
            "target_phases": target_phases,
            "tug_to_target": tug_to_target,
        },
        output_dir / "source_checkpoint_info.json",
    )

    train_labels = np.concatenate(
        [record.y for record in records if record.split == "train"]
    )
    weights_numpy, counts = inverse_frequency_weights(
        train_labels,
        len(target_phases),
        normalize_mean=True,
    )
    pd.DataFrame(
        {
            "phase": target_phases,
            "train_count": counts.astype(int),
            "class_weight": weights_numpy,
        }
    ).to_csv(output_dir / "class_weights.csv", index=False)
    class_weights = torch.tensor(weights_numpy, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=IGNORE_INDEX,
    )

    zero_rows = [
        _evaluate_zero_shot(
            source_model,
            loaders[split],
            tug_to_target,
            target_phases,
            device,
            split,
        )
        for split in ("train", "val", "test")
    ]
    zero_frame = pd.DataFrame(zero_rows)
    zero_frame.to_csv(output_dir / "zero_shot_tug_projection_summary.csv", index=False)

    candidates: list[dict[str, Any]] = []

    for config_id, head_config in enumerate(config["frozen_head_configs"]):
        print(f"\nFrozen encoder configuration {config_id}: {head_config}")
        seed_everything(seed)
        encoder, _ = load_dense_checkpoint(
            source_checkpoint,
            in_channels=len(IMU_CHANNELS),
            conv_channels=int(model_config["conv_channels"]),
            lstm_hidden=int(model_config["lstm_hidden"]),
            num_classes=len(TUG_PHASES),
            dropout=float(model_config.get("source_dropout", 0.25)),
            device=device,
            freeze=True,
        )
        head = TimeStepMLPHead(
            input_dim=encoder.feature_dim,
            hidden_dims=tuple(head_config["hidden_dims"]),
            dropout=float(head_config["dropout"]),
            num_classes=len(target_phases),
        ).to(device)
        model = FrozenEncoderWithHead(encoder, head).to(device)
        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=float(head_config["learning_rate"]),
            weight_decay=float(head_config["weight_decay"]),
        )
        checkpoint_path = checkpoints_dir / f"frozen_head_cfg{config_id}.pt"
        model, history, best = fit_sequence_model(
            model,
            loaders["train"],
            loaders["val"],
            optimizer,
            criterion,
            device,
            len(target_phases),
            int(training_config["max_epochs"]),
            int(training_config["patience"]),
            float(training_config["grad_clip"]),
            checkpoint_path,
            use_amp=use_amp,
        )
        history.to_csv(
            output_dir / f"history_frozen_encoder_head_cfg{config_id}.csv",
            index=False,
        )
        split_metrics = _evaluate_all_splits(
            model,
            loaders,
            device,
            target_phases,
            criterion,
            use_amp,
        )
        candidate = {
            "condition": "frozen_pretrained_encoder_mlp_head",
            "cfg_id": config_id,
            "cfg": dict(head_config),
            "best_epoch": int(best["epoch"]),
            "checkpoint": str(checkpoint_path),
            **split_metrics,
        }
        candidates.append(candidate)

    for condition in (
        "finetune_pretrained_cnn_bilstm",
        "from_scratch_cnn_bilstm",
    ):
        for config_id, full_config in enumerate(config["full_model_configs"]):
            print(f"\n{condition} configuration {config_id}: {full_config}")
            seed_everything(seed)
            model = _build_target_model(
                model_config,
                len(target_phases),
                float(full_config["dropout"]),
                device,
            )
            if condition == "finetune_pretrained_cnn_bilstm":
                model = copy_encoder_weights(model, source_model)
                encoder_scale = float(full_config.get("encoder_lr_scale", 0.2))
            else:
                encoder_scale = 1.0

            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": list(model.conv.parameters())
                        + list(model.lstm.parameters()),
                        "lr": float(full_config["learning_rate"]) * encoder_scale,
                    },
                    {
                        "params": model.head.parameters(),
                        "lr": float(full_config["learning_rate"]),
                    },
                ],
                weight_decay=float(full_config["weight_decay"]),
            )
            checkpoint_path = checkpoints_dir / f"{condition}_cfg{config_id}.pt"
            model, history, best = fit_sequence_model(
                model,
                loaders["train"],
                loaders["val"],
                optimizer,
                criterion,
                device,
                len(target_phases),
                int(training_config["max_epochs"]),
                int(training_config["patience"]),
                float(training_config["grad_clip"]),
                checkpoint_path,
                use_amp=use_amp,
            )
            history.to_csv(
                output_dir / f"history_{condition}_cfg{config_id}.csv",
                index=False,
            )
            split_metrics = _evaluate_all_splits(
                model,
                loaders,
                device,
                target_phases,
                criterion,
                use_amp,
            )
            candidates.append(
                {
                    "condition": condition,
                    "cfg_id": config_id,
                    "cfg": dict(full_config),
                    "best_epoch": int(best["epoch"]),
                    "checkpoint": str(checkpoint_path),
                    **split_metrics,
                }
            )

    best_by_condition: dict[str, dict[str, Any]] = {}
    for condition in (
        "frozen_pretrained_encoder_mlp_head",
        "finetune_pretrained_cnn_bilstm",
        "from_scratch_cnn_bilstm",
    ):
        matches = [candidate for candidate in candidates if candidate["condition"] == condition]
        best_by_condition[condition] = sorted(
            matches,
            key=_candidate_rank,
            reverse=True,
        )[0]

    candidate_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_rows.append(
            {
                "condition": candidate["condition"],
                "cfg_id": candidate["cfg_id"],
                "best_epoch": candidate["best_epoch"],
                "cfg": json.dumps(candidate["cfg"]),
                "checkpoint": candidate["checkpoint"],
                **{
                    f"{split}_{metric}": value
                    for split in ("train", "val", "test")
                    for metric, value in candidate[split].items()
                },
            }
        )
    candidate_frame = pd.DataFrame(candidate_rows)
    candidate_frame.to_csv(output_dir / "all_candidate_configs_summary.csv", index=False)

    # Save test-set predictions for the validation-selected configuration of
    # every trained condition. This supports fair condition-level confusion
    # matrices and publication figures without rerunning inference later.
    condition_predictions_dir = ensure_dir(output_dir / "condition_predictions")
    for condition, candidate in best_by_condition.items():
        if condition == "frozen_pretrained_encoder_mlp_head":
            encoder, _ = load_dense_checkpoint(
                source_checkpoint,
                in_channels=len(IMU_CHANNELS),
                conv_channels=int(model_config["conv_channels"]),
                lstm_hidden=int(model_config["lstm_hidden"]),
                num_classes=len(TUG_PHASES),
                dropout=float(model_config.get("source_dropout", 0.25)),
                device=device,
                freeze=True,
            )
            head = TimeStepMLPHead(
                input_dim=encoder.feature_dim,
                hidden_dims=tuple(candidate["cfg"]["hidden_dims"]),
                dropout=float(candidate["cfg"]["dropout"]),
                num_classes=len(target_phases),
            ).to(device)
            condition_model: nn.Module = FrozenEncoderWithHead(encoder, head).to(device)
        else:
            condition_model = _build_target_model(
                model_config,
                len(target_phases),
                float(candidate["cfg"]["dropout"]),
                device,
            )
        condition_model.load_state_dict(
            torch_load(candidate["checkpoint"], map_location=device)
        )
        condition_result = evaluate_sequence_model(
            condition_model,
            loaders["test"],
            device,
            target_phases,
            split_name="test",
            criterion=criterion,
            use_amp=use_amp,
            effective_hz=effective_hz,
        )
        condition_result.sequence_predictions.to_csv(
            condition_predictions_dir / f"{condition}_test_sequence_metrics.csv",
            index=False,
        )
        condition_result.timestep_predictions.to_csv(
            condition_predictions_dir / f"{condition}_test_timestep_predictions.csv",
            index=False,
        )
        condition_confusion = confusion_matrix_frame(
            condition_result.y_true,
            condition_result.y_pred,
            target_phases,
        )
        condition_confusion.to_csv(
            condition_predictions_dir / f"{condition}_test_confusion_matrix.csv"
        )
        classification_report_frame(
            condition_result.y_true,
            condition_result.y_pred,
            target_phases,
        ).to_csv(
            condition_predictions_dir / f"{condition}_test_classification_report.csv"
        )

    zero_val = zero_frame[zero_frame["split"] == "val"].iloc[0]
    zero_test = zero_frame[zero_frame["split"] == "test"].iloc[0]
    summary_rows = [
        {
            "condition": "zero_shot_tug_head_projection",
            "cfg_id": -1,
            "best_epoch": 0,
            "trained_now": "nothing",
            "val_balanced_accuracy": zero_val["balanced_accuracy"],
            "val_macro_f1": zero_val["macro_f1"],
            "test_accuracy": zero_test["accuracy"],
            "test_balanced_accuracy": zero_test["balanced_accuracy"],
            "test_macro_f1": zero_test["macro_f1"],
        }
    ]
    for condition, candidate in best_by_condition.items():
        summary_rows.append(
            {
                "condition": condition,
                "cfg_id": candidate["cfg_id"],
                "best_epoch": candidate["best_epoch"],
                "trained_now": {
                    "frozen_pretrained_encoder_mlp_head": "new target head only",
                    "finetune_pretrained_cnn_bilstm": "pretrained encoder and new head",
                    "from_scratch_cnn_bilstm": "full model",
                }[condition],
                "val_balanced_accuracy": candidate["val"]["balanced_accuracy"],
                "val_macro_f1": candidate["val"]["macro_f1"],
                "test_accuracy": candidate["test"]["accuracy"],
                "test_balanced_accuracy": candidate["test"]["balanced_accuracy"],
                "test_macro_f1": candidate["test"]["macro_f1"],
                "cfg": json.dumps(candidate["cfg"]),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["test_balanced_accuracy", "test_macro_f1"],
        ascending=False,
    )
    summary.to_csv(
        output_dir / f"{experiment_name}_pretrained_vs_scratch_final_comparison.csv",
        index=False,
    )
    save_comparison_plot(
        summary,
        "condition",
        ["test_balanced_accuracy", "test_macro_f1"],
        f"{experiment_name}: pretrained versus scratch",
        plots_dir / "final_comparison.png",
    )

    # Select the reported condition using validation data only. Test metrics are
    # never used for model/configuration selection, including as a tie-breaker.
    paper_best = sorted(
        best_by_condition.values(),
        key=lambda candidate: (
            candidate["val"]["balanced_accuracy"],
            candidate["val"]["macro_f1"],
            -candidate["val"]["loss"],
            -int(candidate["cfg_id"]),
        ),
        reverse=True,
    )[0]

    if paper_best["condition"] == "frozen_pretrained_encoder_mlp_head":
        encoder, _ = load_dense_checkpoint(
            source_checkpoint,
            in_channels=len(IMU_CHANNELS),
            conv_channels=int(model_config["conv_channels"]),
            lstm_hidden=int(model_config["lstm_hidden"]),
            num_classes=len(TUG_PHASES),
            dropout=float(model_config.get("source_dropout", 0.25)),
            device=device,
            freeze=True,
        )
        head = TimeStepMLPHead(
            input_dim=encoder.feature_dim,
            hidden_dims=tuple(paper_best["cfg"]["hidden_dims"]),
            dropout=float(paper_best["cfg"]["dropout"]),
            num_classes=len(target_phases),
        ).to(device)
        paper_model: nn.Module = FrozenEncoderWithHead(encoder, head).to(device)
    else:
        paper_model = _build_target_model(
            model_config,
            len(target_phases),
            float(paper_best["cfg"]["dropout"]),
            device,
        )
    paper_model.load_state_dict(torch_load(paper_best["checkpoint"], map_location=device))

    all_sequence_predictions: list[pd.DataFrame] = []
    all_timestep_predictions: list[pd.DataFrame] = []
    test_result = None
    for split, loader in loaders.items():
        result = evaluate_sequence_model(
            paper_model,
            loader,
            device,
            target_phases,
            split_name=split,
            criterion=criterion,
            use_amp=use_amp,
            effective_hz=effective_hz,
        )
        all_sequence_predictions.append(result.sequence_predictions)
        all_timestep_predictions.append(result.timestep_predictions)
        if split == "test":
            test_result = result

    if test_result is None:
        raise RuntimeError("Test result was not generated")

    sequence_predictions = pd.concat(all_sequence_predictions, ignore_index=True)
    timestep_predictions = pd.concat(all_timestep_predictions, ignore_index=True)
    sequence_filename = (
        "spmt_sequence_metrics_predictions_paper_best.csv"
        if experiment_name == "spmt"
        else "sequence_metrics_predictions_paper_best.csv"
    )
    timestep_filename = (
        "spmt_timestep_predictions_paper_best.csv"
        if experiment_name == "spmt"
        else "timestep_predictions_paper_best.csv"
    )
    sequence_predictions.to_csv(output_dir / sequence_filename, index=False)
    timestep_predictions.to_csv(output_dir / timestep_filename, index=False)

    classification_report_frame(
        test_result.y_true,
        test_result.y_pred,
        target_phases,
    ).to_csv(output_dir / "classification_report_test_paper_best.csv")
    confusion = confusion_matrix_frame(
        test_result.y_true,
        test_result.y_pred,
        target_phases,
    )
    confusion.to_csv(output_dir / "confusion_matrix_test_paper_best.csv")
    save_confusion_matrix_plot(
        confusion,
        f"{experiment_name} test confusion matrix",
        plots_dir / "confusion_matrix_test_paper_best.png",
    )

    test_steps = timestep_predictions[timestep_predictions["split"] == "test"].copy()
    boundary_tolerant_table(
        test_steps,
        effective_hz=effective_hz,
    ).to_csv(output_dir / "boundary_tolerant_test_metrics_paper_best.csv", index=False)

    if subgroup_analyses:
        for filename, columns in subgroup_analyses.items():
            available = [column for column in columns if column in test_steps.columns]
            if available:
                grouped_metrics(
                    test_steps,
                    available,
                    len(target_phases),
                ).to_csv(output_dir / filename, index=False)

    if fog_analysis and "fog" in test_steps.columns:
        fog_rows = []
        from .metrics import classification_metrics

        for fog_value, name in ((0, "non_fog"), (1, "fog")):
            subset = test_steps[test_steps["fog"] == fog_value]
            if subset.empty:
                continue
            fog_rows.append(
                {
                    "segment_type": name,
                    "n_timesteps": len(subset),
                    "n_records": subset["record_id"].nunique(),
                    "n_subjects": subset["subject_id"].nunique(),
                    **classification_metrics(
                        subset["true_id"].to_numpy(),
                        subset["pred_id"].to_numpy(),
                        len(target_phases),
                    ),
                }
            )
        pd.DataFrame(fog_rows).to_csv(
            output_dir / "fog_vs_nonfog_test_metrics_paper_best.csv",
            index=False,
        )

    for record_id in (
        sequence_predictions[sequence_predictions["split"] == "test"]["record_id"]
        .head(5)
        .tolist()
    ):
        subset = test_steps[test_steps["record_id"] == record_id]
        if not subset.empty:
            save_timeline_plot(
                subset,
                target_phases,
                f"{experiment_name} record {record_id}",
                plots_dir / f"timeline_test_record_{record_id}.png",
            )

    best_bundle = {
        "best_by_condition": {
            condition: {
                "cfg_id": candidate["cfg_id"],
                "cfg": candidate["cfg"],
                "best_epoch": candidate["best_epoch"],
                "model_state_dict": torch_load(
                    candidate["checkpoint"], map_location="cpu"
                ),
            }
            for condition, candidate in best_by_condition.items()
        },
        "paper_best_condition": paper_best["condition"],
        "target_phases": target_phases,
        "tug_to_target": tug_to_target,
        "source_checkpoint": str(source_checkpoint),
    }
    torch.save(best_bundle, output_dir / f"best_{experiment_name}_transfer_models.pt")
    save_json(
        {
            "experiment_name": experiment_name,
            "device": str(device),
            "target_phases": target_phases,
            "effective_hz": effective_hz,
            "paper_best_condition": paper_best["condition"],
            "paper_best_config": paper_best["cfg"],
            "source_checkpoint": str(source_checkpoint),
            "n_records": len(records),
            "n_train_records": sum(record.split == "train" for record in records),
            "n_val_records": sum(record.split == "val" for record in records),
            "n_test_records": sum(record.split == "test" for record in records),
        },
        output_dir / "experiment_summary.json",
    )
    return summary
