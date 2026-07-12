from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from .preprocessing import ChannelNormalizer
from .records import SequenceRecord

IGNORE_INDEX = -100


class SequenceDataset(Dataset):
    def __init__(
        self,
        records: list[SequenceRecord],
        normalizer: ChannelNormalizer | None = None,
    ) -> None:
        self.records = records
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        x = (
            self.normalizer.transform_array(record.x)
            if self.normalizer is not None
            else record.x
        )
        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(record.y, dtype=torch.long),
            "length": record.length,
            "record_id": record.record_id,
            "subject_id": record.subject_id,
            "split": record.split,
            "metadata": dict(record.metadata),
            "auxiliary": {
                key: torch.tensor(values)
                for key, values in record.auxiliary.items()
            },
        }


def collate_sequences(batch: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    max_length = int(lengths.max().item())
    channels = int(batch[0]["x"].shape[1])
    batch_size = len(batch)

    x = torch.zeros(batch_size, max_length, channels, dtype=torch.float32)
    y = torch.full((batch_size, max_length), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)

    auxiliary_keys = sorted(
        {key for item in batch for key in item["auxiliary"].keys()}
    )
    auxiliary: dict[str, torch.Tensor] = {
        key: torch.zeros(batch_size, max_length, dtype=torch.long)
        for key in auxiliary_keys
    }

    metadata: list[dict[str, Any]] = []
    for i, item in enumerate(batch):
        length = item["length"]
        x[i, :length] = item["x"]
        y[i, :length] = item["y"]
        mask[i, :length] = True
        for key in auxiliary_keys:
            if key in item["auxiliary"]:
                values = item["auxiliary"][key]
                auxiliary[key][i, :length] = values.to(dtype=torch.long)
        metadata.append(
            {
                "record_id": item["record_id"],
                "subject_id": item["subject_id"],
                "split": item["split"],
                **item["metadata"],
            }
        )

    return {
        "x": x,
        "y": y,
        "mask": mask,
        "lengths": lengths,
        "metadata": metadata,
        "auxiliary": auxiliary,
    }
