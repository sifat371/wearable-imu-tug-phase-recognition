import numpy as np

from tug_transfer.data.preprocessing import ChannelNormalizer
from tug_transfer.data.records import SequenceRecord


def test_train_normalizer():
    records = [
        SequenceRecord(0, "S1", np.ones((5, 3)), np.zeros(5)),
        SequenceRecord(1, "S2", np.full((5, 3), 3.0), np.zeros(5)),
    ]
    normalizer = ChannelNormalizer.fit(records)
    transformed = np.concatenate(
        [normalizer.transform_array(record.x) for record in records],
        axis=0,
    )
    assert np.allclose(transformed.mean(axis=0), 0.0)
    assert np.allclose(transformed.std(axis=0), 1.0)
