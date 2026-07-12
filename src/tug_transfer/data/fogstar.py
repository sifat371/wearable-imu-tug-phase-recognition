from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from tug_transfer.constants import FOGSTAR_RAW_ACTIVITY_NAMES, fogstar_target_definition

from .records import SequenceRecord


FOGSTAR_BASE_SENSOR_COLUMNS = [
    "ankleL_acc_x",
    "ankleL_acc_y",
    "ankleL_acc_z",
    "ankleL_gyro_x",
    "ankleL_gyro_y",
    "ankleL_gyro_z",
    "ankleR_acc_x",
    "ankleR_acc_y",
    "ankleR_acc_z",
    "ankleR_gyro_x",
    "ankleR_gyro_y",
    "ankleR_gyro_z",
    "back_acc_x",
    "back_acc_y",
    "back_acc_z",
    "back_gyro_x",
    "back_gyro_y",
    "back_gyro_z",
]


def _numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame[columns].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    return output.interpolate(limit_direction="both").fillna(0.0)


def _rolling_free_acceleration(
    acceleration: pd.DataFrame,
    raw_hz: float,
    window_seconds: float,
) -> pd.DataFrame:
    window = max(3, int(round(raw_hz * window_seconds)))
    if window % 2 == 0:
        window += 1
    trend = acceleration.rolling(window=window, center=True, min_periods=1).mean()
    return acceleration - trend


def fogstar_to_weargait_imu27(
    frame: pd.DataFrame,
    raw_hz: float,
    convert_acc_g_to_ms2: bool = True,
    convert_gyro_deg_to_rad: bool = True,
    freeacc_method: str = "rolling_center",
    freeacc_rolling_sec: float = 1.0,
) -> np.ndarray:
    required = FOGSTAR_BASE_SENSOR_COLUMNS + ["activity"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing FoG-STAR columns: {missing}")

    acc_back = _numeric_frame(frame, ["back_acc_x", "back_acc_y", "back_acc_z"])
    gyr_back = _numeric_frame(frame, ["back_gyro_x", "back_gyro_y", "back_gyro_z"])
    acc_left = _numeric_frame(frame, ["ankleL_acc_x", "ankleL_acc_y", "ankleL_acc_z"])
    gyr_left = _numeric_frame(frame, ["ankleL_gyro_x", "ankleL_gyro_y", "ankleL_gyro_z"])
    acc_right = _numeric_frame(frame, ["ankleR_acc_x", "ankleR_acc_y", "ankleR_acc_z"])
    gyr_right = _numeric_frame(frame, ["ankleR_gyro_x", "ankleR_gyro_y", "ankleR_gyro_z"])

    if convert_acc_g_to_ms2:
        for acceleration in (acc_back, acc_left, acc_right):
            acceleration *= 9.80665
    if convert_gyro_deg_to_rad:
        for gyroscope in (gyr_back, gyr_left, gyr_right):
            gyroscope *= np.pi / 180.0

    if freeacc_method == "rolling_center":
        free_back = _rolling_free_acceleration(acc_back, raw_hz, freeacc_rolling_sec)
        free_left = _rolling_free_acceleration(acc_left, raw_hz, freeacc_rolling_sec)
        free_right = _rolling_free_acceleration(acc_right, raw_hz, freeacc_rolling_sec)
    elif freeacc_method == "zero":
        free_back = acc_back * 0.0
        free_left = acc_left * 0.0
        free_right = acc_right * 0.0
    else:
        raise ValueError(f"Unsupported free-acceleration method: {freeacc_method}")

    x = np.concatenate(
        [
            acc_back.to_numpy(),
            free_back.to_numpy(),
            gyr_back.to_numpy(),
            acc_left.to_numpy(),
            free_left.to_numpy(),
            gyr_left.to_numpy(),
            acc_right.to_numpy(),
            free_right.to_numpy(),
            gyr_right.to_numpy(),
        ],
        axis=1,
    ).astype(np.float32)
    if x.shape[1] != 27:
        raise RuntimeError(f"Unexpected transformed channel count: {x.shape[1]}")
    return x


def inspect_fogstar_zip(path: str | Path) -> dict:
    archive_path = Path(path)
    if not archive_path.exists():
        raise FileNotFoundError(f"FoG-STAR archive not found: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        required = {"sensor_data.csv", "clinical_data.csv", "README.txt"}
        missing = sorted(required - names)
        if missing:
            raise KeyError(f"FoG-STAR archive is missing: {missing}")
        clinical = pd.read_csv(archive.open("clinical_data.csv"))
        sensor_sample = pd.read_csv(archive.open("sensor_data.csv"), nrows=5)
    return {
        "files": sorted(names),
        "clinical_shape": list(clinical.shape),
        "sensor_columns": sensor_sample.columns.tolist(),
    }


def load_fogstar_records(
    archive_path: str | Path,
    target_mode: str = "activity6",
    drop_activity_zero: bool = True,
    raw_hz: int = 60,
    target_hz: int = 20,
    max_seq_len: int = 4000,
    min_seq_len: int = 20,
    convert_acc_g_to_ms2: bool = True,
    convert_gyro_deg_to_rad: bool = True,
    freeacc_method: str = "rolling_center",
    freeacc_rolling_sec: float = 1.0,
) -> tuple[list[SequenceRecord], list[dict], pd.DataFrame, list[str], dict[int, int]]:
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"FoG-STAR archive not found: {archive_path}")

    target_phases, raw_to_target, tug_to_target = fogstar_target_definition(target_mode)
    sample_step = int(round(raw_hz / target_hz))
    if sample_step < 1:
        raise ValueError("target_hz must not exceed raw_hz")

    with zipfile.ZipFile(archive_path) as archive:
        sensor = pd.read_csv(archive.open("sensor_data.csv"))
        clinical = pd.read_csv(archive.open("clinical_data.csv"))

    records: list[SequenceRecord] = []
    errors: list[dict] = []

    group_columns = ["subjectID", "sessionID", "taskID"]
    for (subject, session, task), group in sensor.groupby(group_columns, sort=True):
        try:
            ordered = group.sort_values("timestamp").reset_index(drop=True)
            mapped = ordered["activity"].map(raw_to_target)
            valid = mapped.notna().to_numpy()
            if drop_activity_zero:
                valid &= ordered["activity"].to_numpy() != 0
            if int(valid.sum()) < min_seq_len:
                raise ValueError(f"Too few valid rows: {int(valid.sum())}")

            valid_frame = ordered.loc[valid].reset_index(drop=True)
            y = mapped.loc[valid].astype(int).to_numpy(dtype=np.int64)
            x = fogstar_to_weargait_imu27(
                valid_frame,
                raw_hz=raw_hz,
                convert_acc_g_to_ms2=convert_acc_g_to_ms2,
                convert_gyro_deg_to_rad=convert_gyro_deg_to_rad,
                freeacc_method=freeacc_method,
                freeacc_rolling_sec=freeacc_rolling_sec,
            )
            fog = (
                pd.to_numeric(valid_frame["fog"], errors="coerce")
                .fillna(0)
                .astype(int)
                .to_numpy()
            )

            x = x[::sample_step]
            y = y[::sample_step]
            fog = fog[::sample_step]
            if len(y) > max_seq_len:
                x = x[:max_seq_len]
                y = y[:max_seq_len]
                fog = fog[:max_seq_len]
            if len(y) < min_seq_len:
                raise ValueError(f"Too short after subsampling: {len(y)}")

            records.append(
                SequenceRecord(
                    record_id=len(records),
                    subject_id=str(int(subject)),
                    x=x,
                    y=y,
                    metadata={
                        "session_id": int(session),
                        "task_id": int(task),
                        "has_fog": int(fog.max() > 0),
                        "n_fog_steps": int(fog.sum()),
                        "task": "FoG-STAR",
                    },
                    auxiliary={"fog": fog.astype(np.int64)},
                )
            )
        except Exception as error:
            errors.append(
                {
                    "subject_id": subject,
                    "session_id": session,
                    "task_id": task,
                    "error": str(error),
                }
            )

    return records, errors, clinical, target_phases, tug_to_target


def raw_activity_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"activity_id": activity_id, "activity_name": name}
            for activity_id, name in FOGSTAR_RAW_ACTIVITY_NAMES.items()
        ]
    )
