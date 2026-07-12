import numpy as np

from tug_transfer.data.records import SequenceRecord
from tug_transfer.data.splits import assign_split_map, make_stratified_subject_split


def test_subject_split_has_no_overlap():
    records = []
    for index in range(20):
        records.append(
            SequenceRecord(
                index,
                f"S{index:02d}",
                np.zeros((10, 2)),
                np.zeros(10),
                metadata={
                    "site": "A" if index < 10 else "B",
                    "group": "HC" if index % 2 == 0 else "PD",
                },
            )
        )
    split_map = make_stratified_subject_split(records, seed=42)
    assigned = assign_split_map(records, split_map)
    subject_splits = {}
    for record in assigned:
        subject_splits.setdefault(record.subject_id, set()).add(record.split)
    assert all(len(values) == 1 for values in subject_splits.values())
    assert {record.split for record in assigned} == {"train", "val", "test"}
