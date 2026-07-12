import numpy as np
import pandas as pd

from tug_transfer.constants import fogstar_target_definition
from tug_transfer.data.fogstar import (
    FOGSTAR_BASE_SENSOR_COLUMNS,
    fogstar_to_weargait_imu27,
)


def test_fogstar_transform_has_27_channels():
    frame = pd.DataFrame(
        {
            column: np.linspace(0, 1, 15)
            for column in FOGSTAR_BASE_SENSOR_COLUMNS
        }
    )
    frame["activity"] = 1
    x = fogstar_to_weargait_imu27(frame, raw_hz=60)
    assert x.shape == (15, 27)


def test_activity6_mapping():
    phases, raw_to_target, tug_to_target = fogstar_target_definition("activity6")
    assert len(phases) == 6
    assert raw_to_target[6] == raw_to_target[7]
    assert 2 not in set(tug_to_target.values())  # no direct TUG Stand class
