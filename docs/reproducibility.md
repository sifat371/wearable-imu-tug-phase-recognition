# Reproducibility notes

## Fixed artifacts

The repository includes the exact participant assignments and TUG training statistics used by the final experiments:

- `artifacts/paper/subject_level_split.csv`: 117/25/25 TUG train/validation/test participants;
- `artifacts/paper/fogstar_subject_split.csv`: 11/5/6 FoG-STAR train/validation/test participants;
- `artifacts/paper/imu_train_mean.npy` and `imu_train_std.npy`: 27-channel TUG training normalization;
- `artifacts/paper/architecture_sensitivity.csv`: reported single-seed 1D-CNN, BiLSTM, and CNN-BiLSTM comparison.

## Main settings

- random seed: 42;
- WearGait-PD retained rate: approximately 20 Hz by every-fifth-sample decimation;
- FoG-STAR retained rate: approximately 20 Hz by every-third-sample decimation;
- TUG maximum retained length: 1600 timesteps;
- FoG-STAR maximum retained length: 4000 timesteps;
- TUG windows: 40 samples with stride 10;
- optimizer: AdamW;
- checkpoint selection: validation metrics only;
- padded timesteps excluded from loss and evaluation;
- packed BiLSTM processing prevents padded tails from affecting recurrent outputs.

## Paper-to-code map

| Paper component | Main code |
|---|---|
| WearGait-PD loading and preprocessing | `src/tug_transfer/data/weargait.py` |
| FoG-STAR mapping | `src/tug_transfer/data/fogstar.py` |
| fixed windows and reconstruction | `src/tug_transfer/data/windowing.py` |
| CNN-BiLSTM models | `src/tug_transfer/models/cnn_bilstm.py` |
| TUG experiment | `src/tug_transfer/experiments/tug.py` |
| SPmT transfer | `src/tug_transfer/experiments/spmt.py` |
| FoG-STAR transfer | `src/tug_transfer/experiments/fogstar.py` |
| validation-based transfer selection | `src/tug_transfer/training/transfer.py` |
| paper result figures | `src/tug_transfer/figures/paper.py` |

## Numerical variation

The repository fixes splits and seeds but does not guarantee bitwise-identical GPU results. Small variations can occur between software and hardware environments.
