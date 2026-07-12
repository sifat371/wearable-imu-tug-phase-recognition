from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SequenceRecord:
    """A single variable-length multivariate sequence."""

    record_id: int
    subject_id: str
    x: np.ndarray
    y: np.ndarray
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    auxiliary: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.int64)
        if self.x.ndim != 2:
            raise ValueError(f"x must have shape [L, C], got {self.x.shape}")
        if self.y.ndim != 1:
            raise ValueError(f"y must have shape [L], got {self.y.shape}")
        if len(self.x) != len(self.y):
            raise ValueError(f"x/y length mismatch: {len(self.x)} vs {len(self.y)}")
        for name, values in self.auxiliary.items():
            array = np.asarray(values)
            if len(array) != len(self.y):
                raise ValueError(f"Auxiliary sequence {name!r} has incorrect length")
            self.auxiliary[name] = array

    @property
    def length(self) -> int:
        return len(self.y)

    def clone_with_x(self, x: np.ndarray) -> "SequenceRecord":
        return SequenceRecord(
            record_id=self.record_id,
            subject_id=self.subject_id,
            x=x,
            y=self.y.copy(),
            split=self.split,
            metadata=dict(self.metadata),
            auxiliary={k: v.copy() for k, v in self.auxiliary.items()},
        )
