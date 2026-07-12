import numpy as np

from tug_transfer.data.records import SequenceRecord
from tug_transfer.data.windowing import build_windows, window_starts


def test_window_starts_includes_last_window():
    assert window_starts(95, 40, 10)[-1] == 55


def test_pure_and_mixed_windows():
    x = np.zeros((8, 27), dtype=np.float32)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    record = SequenceRecord(0, "S1", x, y, split="test")
    collection = build_windows([record], window_length=4, stride=2, num_classes=2)
    assert collection.x.shape[1:] == (4, 27)
    assert collection.metadata["is_pure"].any()
    assert (~collection.metadata["is_pure"]).any()
