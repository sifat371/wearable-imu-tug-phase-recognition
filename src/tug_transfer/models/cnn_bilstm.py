from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tug_transfer.utils import torch_load


class MaskedBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that excludes padded timesteps from training statistics.

    The module keeps the same parameters and buffer names as ``nn.BatchNorm1d``.
    Consequently, checkpoints produced by the original notebook architecture can
    be loaded without renaming BatchNorm keys.
    """

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is None:
            return super().forward(x)
        if x.ndim != 3:
            raise ValueError(f"MaskedBatchNorm1d expects [B,C,L], received {tuple(x.shape)}")
        if mask.shape != (x.shape[0], x.shape[2]):
            raise ValueError(
                "Mask shape must be [B,L] and match the input; "
                f"received mask={tuple(mask.shape)}, input={tuple(x.shape)}"
            )

        # Match PyTorch BatchNorm's numerically stable behavior under AMP by
        # accumulating statistics in float32 for fp16/bf16 activations.
        stats_x = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        valid = mask[:, None, :].to(device=x.device, dtype=stats_x.dtype)
        valid_count = valid.sum()
        if valid_count.item() <= 0:
            raise ValueError("A sequence batch cannot contain zero valid timesteps")

        if self.training:
            mean = (stats_x * valid).sum(dim=(0, 2)) / valid_count
            centered = stats_x - mean[None, :, None]
            variance = (centered.square() * valid).sum(dim=(0, 2)) / valid_count

            if self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked.add_(1)
                    if self.momentum is None:
                        factor = 1.0 / float(self.num_batches_tracked.item())
                    else:
                        factor = float(self.momentum)
                    self.running_mean.lerp_(mean.detach(), factor)
                    if valid_count.item() > 1:
                        unbiased = variance.detach() * valid_count / (valid_count - 1.0)
                    else:
                        unbiased = variance.detach()
                    self.running_var.lerp_(unbiased, factor)
        else:
            if self.track_running_stats:
                mean = self.running_mean
                variance = self.running_var
            else:
                mean = (stats_x * valid).sum(dim=(0, 2)) / valid_count
                centered = stats_x - mean[None, :, None]
                variance = (centered.square() * valid).sum(dim=(0, 2)) / valid_count

        output = (stats_x - mean[None, :, None]) / torch.sqrt(
            variance[None, :, None] + self.eps
        )
        if self.affine:
            output = output * self.weight[None, :, None] + self.bias[None, :, None]
        return output * valid


def _resolve_lengths(
    x: torch.Tensor,
    mask: torch.Tensor | None,
    lengths: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, max_length, _ = x.shape
    if lengths is None:
        if mask is None:
            lengths = torch.full(
                (batch_size,),
                max_length,
                dtype=torch.long,
                device=x.device,
            )
        else:
            lengths = mask.to(dtype=torch.long).sum(dim=1)
    lengths = lengths.to(device=x.device, dtype=torch.long)
    if lengths.shape != (batch_size,):
        raise ValueError(
            f"lengths must have shape [B]; received {tuple(lengths.shape)} for B={batch_size}"
        )
    if torch.any(lengths <= 0) or torch.any(lengths > max_length):
        raise ValueError("Every sequence length must be in the interval [1, padded_length]")

    if mask is None:
        positions = torch.arange(max_length, device=x.device)[None, :]
        mask = positions < lengths[:, None]
    else:
        mask = mask.to(device=x.device, dtype=torch.bool)
        expected = torch.arange(max_length, device=x.device)[None, :] < lengths[:, None]
        if not torch.equal(mask, expected):
            raise ValueError("Sequence masks must be contiguous prefixes consistent with lengths")
    return lengths, mask


class DenseCNNBiLSTM(nn.Module):
    """CNN-BiLSTM with one class prediction per valid timestep.

    Variable-length sequences are handled in two places:

    * padded timesteps are excluded from convolutional BatchNorm statistics;
    * ``pack_padded_sequence`` prevents the bidirectional LSTM from reading the
      padded tail.
    """

    def __init__(
        self,
        in_channels: int = 27,
        conv_channels: int = 64,
        lstm_hidden: int = 64,
        num_classes: int = 5,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.lstm_hidden = lstm_hidden
        self.num_classes = num_classes
        self.feature_dim = lstm_hidden * 2

        # The sequential indices intentionally match the original notebook so
        # old state dictionaries remain loadable.
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=5, padding=2),
            MaskedBatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            MaskedBatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def _masked_conv(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_channels = mask[:, None, :].to(dtype=x.dtype)
        z = x.transpose(1, 2) * mask_channels
        z = self.conv[0](z)
        z = self.conv[1](z, mask)
        z = self.conv[2](z)
        z = self.conv[3](z)
        z = z * mask_channels
        z = self.conv[4](z)
        z = self.conv[5](z, mask)
        z = self.conv[6](z)
        z = self.conv[7](z)
        return z * mask_channels

    def encode(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        lengths, mask = _resolve_lengths(x, mask, lengths)
        features = self._masked_conv(x, mask).transpose(1, 2)
        packed = pack_padded_sequence(
            features,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_features, _ = self.lstm(packed)
        features, _ = pad_packed_sequence(
            packed_features,
            batch_first=True,
            total_length=x.shape[1],
        )
        return features * mask[:, :, None].to(dtype=features.dtype)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.head(self.encode(x, mask=mask, lengths=lengths))


class WindowCNNBiLSTM(nn.Module):
    """Encoder-matched CNN-BiLSTM with one class prediction per window."""

    def __init__(
        self,
        in_channels: int = 27,
        conv_channels: int = 64,
        lstm_hidden: int = 64,
        num_classes: int = 5,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x.transpose(1, 2)).transpose(1, 2)
        features, _ = self.lstm(features)
        pooled = torch.cat(
            [features.mean(dim=1), features.max(dim=1).values],
            dim=1,
        )
        return self.head(pooled)


class TimeStepMLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: tuple[int, ...] = (64,),
        dropout: float = 0.30,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.LayerNorm(previous),
                    nn.Linear(previous, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = hidden
        layers.extend(
            [
                nn.LayerNorm(previous),
                nn.Dropout(dropout),
                nn.Linear(previous, num_classes),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class FrozenEncoderWithHead(nn.Module):
    def __init__(self, encoder: DenseCNNBiLSTM, head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The encoder remains in eval mode and is not part of the gradient graph.
        self.encoder.eval()
        with torch.no_grad():
            features = self.encoder.encode(x, mask=mask, lengths=lengths)
        return self.head(features.float())


def clean_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def extract_state_dict(payload: Any) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    if isinstance(payload, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in payload:
                info = {name: str(value) for name, value in payload.items() if name != key}
                return clean_state_dict_keys(payload[key]), info
    return clean_state_dict_keys(payload), {"checkpoint_format": "raw_state_dict"}


def load_dense_checkpoint(
    checkpoint_path: str | Path,
    *,
    in_channels: int,
    conv_channels: int,
    lstm_hidden: int,
    num_classes: int,
    dropout: float,
    device: torch.device | str = "cpu",
    freeze: bool = False,
) -> tuple[DenseCNNBiLSTM, dict[str, str]]:
    model = DenseCNNBiLSTM(
        in_channels=in_channels,
        conv_channels=conv_channels,
        lstm_hidden=lstm_hidden,
        num_classes=num_classes,
        dropout=dropout,
    )
    payload = torch_load(checkpoint_path, map_location="cpu")
    state_dict, info = extract_state_dict(payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    critical_missing = [
        key for key in missing if key.startswith("conv.") or key.startswith("lstm.")
    ]
    if critical_missing:
        raise RuntimeError(f"Critical encoder parameters are missing: {critical_missing}")
    if unexpected:
        info["unexpected_keys"] = str(unexpected)
    if missing:
        info["missing_keys"] = str(missing)
    model.to(device)
    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.eval()
    return model, info


def copy_encoder_weights(
    target: DenseCNNBiLSTM,
    source: DenseCNNBiLSTM,
) -> DenseCNNBiLSTM:
    target.conv.load_state_dict(copy.deepcopy(source.conv.state_dict()))
    target.lstm.load_state_dict(copy.deepcopy(source.lstm.state_dict()))
    return target
