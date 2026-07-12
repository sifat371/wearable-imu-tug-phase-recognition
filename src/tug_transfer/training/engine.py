from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .metrics import classification_metrics


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    sequence_predictions: pd.DataFrame
    timestep_predictions: pd.DataFrame


@dataclass
class WindowEvaluationResult:
    metrics: dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray
    predictions: pd.DataFrame


def _autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        enabled=enabled and device.type == "cuda",
    )


def _make_grad_scaler(device: torch.device, enabled: bool):
    active = enabled and device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=active)
    except (AttributeError, TypeError):  # PyTorch < 2.3 compatibility
        return torch.cuda.amp.GradScaler(enabled=active)


@torch.no_grad()
def evaluate_sequence_metrics(
    model: nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    criterion: nn.Module | None = None,
    use_amp: bool = False,
) -> dict[str, float]:
    model.eval()
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    loss_sum = 0.0
    loss_weight = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        lengths = batch["lengths"]
        with _autocast_context(device, use_amp):
            logits = model(x, mask=mask, lengths=lengths)
            if criterion is not None:
                loss = criterion(logits.reshape(-1, num_classes), y.reshape(-1))
                valid_count = int(mask.sum().item())
                loss_sum += float(loss.item()) * valid_count
                loss_weight += valid_count
        predictions = logits.float().argmax(dim=-1)
        y_numpy = y.detach().cpu().numpy()
        pred_numpy = predictions.detach().cpu().numpy()
        mask_numpy = mask.detach().cpu().numpy()
        all_true.append(y_numpy[mask_numpy])
        all_pred.append(pred_numpy[mask_numpy])

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return {
        "loss": loss_sum / loss_weight if loss_weight else float("nan"),
        **classification_metrics(y_true, y_pred, num_classes),
    }


@torch.no_grad()
def evaluate_sequence_model(
    model: nn.Module,
    loader,
    device: torch.device,
    class_names: list[str],
    split_name: str,
    criterion: nn.Module | None = None,
    use_amp: bool = False,
    effective_hz: float | None = None,
) -> EvaluationResult:
    model.eval()
    num_classes = len(class_names)
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    loss_sum = 0.0
    loss_weight = 0
    sequence_rows: list[dict[str, Any]] = []
    timestep_rows: list[dict[str, Any]] = []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        lengths = batch["lengths"]
        with _autocast_context(device, use_amp):
            logits = model(x, mask=mask, lengths=lengths)
            if criterion is not None:
                loss = criterion(logits.reshape(-1, num_classes), y.reshape(-1))
                valid_count = int(mask.sum().item())
                loss_sum += float(loss.item()) * valid_count
                loss_weight += valid_count
        probabilities = torch.softmax(logits.float(), dim=-1)
        predictions = probabilities.argmax(dim=-1)

        y_numpy = y.detach().cpu().numpy()
        pred_numpy = predictions.detach().cpu().numpy()
        prob_numpy = probabilities.detach().cpu().numpy()
        auxiliary_numpy = {
            key: values.detach().cpu().numpy()
            for key, values in batch.get("auxiliary", {}).items()
        }

        for batch_index, metadata in enumerate(batch["metadata"]):
            length = int(batch["lengths"][batch_index].item())
            y_true_record = y_numpy[batch_index, :length]
            y_pred_record = pred_numpy[batch_index, :length]
            all_true.append(y_true_record)
            all_pred.append(y_pred_record)

            record_metrics = classification_metrics(
                y_true_record,
                y_pred_record,
                num_classes,
            )
            base_metadata = dict(metadata)
            sequence_rows.append(
                {
                    "split": split_name,
                    **base_metadata,
                    "length": length,
                    **record_metrics,
                }
            )

            for timestep in range(length):
                row: dict[str, Any] = {
                    "split": split_name,
                    **base_metadata,
                    "time_step": timestep,
                    "true_id": int(y_true_record[timestep]),
                    "pred_id": int(y_pred_record[timestep]),
                    "true_phase": class_names[int(y_true_record[timestep])],
                    "pred_phase": class_names[int(y_pred_record[timestep])],
                }
                if effective_hz is not None:
                    row["time_sec"] = timestep / effective_hz
                for class_index, class_name in enumerate(class_names):
                    row[f"prob_{class_name}"] = float(
                        prob_numpy[batch_index, timestep, class_index]
                    )
                for key, values in auxiliary_numpy.items():
                    row[key] = int(values[batch_index, timestep])
                timestep_rows.append(row)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    metrics = {
        "loss": loss_sum / loss_weight if loss_weight else float("nan"),
        **classification_metrics(y_true, y_pred, num_classes),
    }
    return EvaluationResult(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sequence_predictions=pd.DataFrame(sequence_rows),
        timestep_predictions=pd.DataFrame(timestep_rows),
    )


def train_sequence_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    grad_clip: float,
    use_amp: bool,
    scaler=None,
) -> float:
    model.train()
    loss_sum = 0.0
    token_count = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, use_amp):
            logits = model(
                x,
                mask=batch["mask"].to(device, non_blocking=True),
                lengths=batch["lengths"],
            )
            loss = criterion(logits.reshape(-1, num_classes), y.reshape(-1))
        valid_count = int(batch["mask"].sum().item())
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        loss_sum += float(loss.item()) * valid_count
        token_count += valid_count
    return loss_sum / max(token_count, 1)


def fit_sequence_model(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    max_epochs: int,
    patience: int,
    grad_clip: float,
    checkpoint_path: str | Path,
    use_amp: bool = False,
    selection_order: tuple[str, ...] = ("balanced_accuracy", "macro_f1"),
) -> tuple[nn.Module, pd.DataFrame, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best: dict[str, Any] | None = None
    bad_epochs = 0
    history: list[dict[str, float]] = []
    scaler = _make_grad_scaler(device, use_amp)

    for epoch in range(1, max_epochs + 1):
        train_loss = train_sequence_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            num_classes,
            grad_clip,
            use_amp,
            scaler=scaler,
        )
        validation = evaluate_sequence_metrics(
            model,
            val_loader,
            device,
            num_classes,
            criterion=criterion,
            use_amp=use_amp,
        )
        row = {"epoch": epoch, "train_loss": train_loss, **validation}
        history.append(row)
        score = tuple(validation[name] for name in selection_order) + (-validation["loss"],)
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "epoch": epoch,
                "state_dict": copy.deepcopy(model.state_dict()),
                "validation": dict(validation),
            }
            torch.save(best["state_dict"], checkpoint_path)
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} train_loss={train_loss:.4f} "
                f"val_bal={validation['balanced_accuracy']:.4f} "
                f"val_f1={validation['macro_f1']:.4f}"
            )
        if bad_epochs >= patience:
            break

    if best is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best["state_dict"])
    return model, pd.DataFrame(history), best


@torch.no_grad()
def evaluate_window_model(
    model: nn.Module,
    loader,
    device: torch.device,
    class_names: list[str],
    split_name: str,
    criterion: nn.Module | None = None,
) -> WindowEvaluationResult:
    model.eval()
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    loss_sum = 0.0
    sample_count = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        logits = model(x).float()
        if criterion is not None:
            batch_loss = float(criterion(logits, y).item())
            loss_sum += batch_loss * len(y)
            sample_count += len(y)
        probabilities = torch.softmax(logits, dim=-1)
        predictions = probabilities.argmax(dim=-1)

        y_numpy = y.detach().cpu().numpy()
        pred_numpy = predictions.detach().cpu().numpy()
        prob_numpy = probabilities.detach().cpu().numpy()
        all_true.append(y_numpy)
        all_pred.append(pred_numpy)
        all_probabilities.append(prob_numpy)

        for index, metadata in enumerate(batch["metadata"]):
            row = {
                "split": split_name,
                **metadata,
                "y_true": int(y_numpy[index]),
                "y_pred": int(pred_numpy[index]),
            }
            for class_index, class_name in enumerate(class_names):
                row[f"prob_{class_name}"] = float(prob_numpy[index, class_index])
            rows.append(row)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    probabilities = np.concatenate(all_probabilities)
    metrics = {
        "loss": loss_sum / sample_count if sample_count else float("nan"),
        **classification_metrics(y_true, y_pred, len(class_names)),
    }
    return WindowEvaluationResult(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        predictions=pd.DataFrame(rows),
    )


def train_window_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        loss_sum += float(loss.item()) * len(y)
        sample_count += len(y)
    return loss_sum / max(sample_count, 1)


def fit_window_model(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
    max_epochs: int,
    patience: int,
    grad_clip: float,
    checkpoint_path: str | Path,
    selection_order: tuple[str, ...] = ("balanced_accuracy", "macro_f1"),
) -> tuple[nn.Module, pd.DataFrame, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best: dict[str, Any] | None = None
    bad_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_window_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            grad_clip,
        )
        validation = evaluate_window_model(
            model,
            val_loader,
            device,
            class_names,
            split_name="val",
            criterion=criterion,
        ).metrics
        row = {"epoch": epoch, "train_loss": train_loss, **validation}
        history.append(row)
        score = tuple(validation[name] for name in selection_order) + (-validation["loss"],)
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "epoch": epoch,
                "state_dict": copy.deepcopy(model.state_dict()),
                "validation": dict(validation),
            }
            torch.save(best["state_dict"], checkpoint_path)
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} train_loss={train_loss:.4f} "
                f"val_bal={validation['balanced_accuracy']:.4f} "
                f"val_f1={validation['macro_f1']:.4f}"
            )
        if bad_epochs >= patience:
            break

    if best is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best["state_dict"])
    return model, pd.DataFrame(history), best
