# Wearable IMU-Based TUG Phase Recognition

Code for the paper:

> **Wearable IMU-Based TUG Phase Recognition: Dense Temporal Labeling and Cross-Task Transfer**

Authors: Md Sifat, Sania Akter, Akif Islam, and Md. Ekramul Hamid.

## What is included

This compact repository contains only the code needed for the final paper:

- dense timestep-level TUG phase recognition;
- encoder-matched fixed-window TUG classification and timestep reconstruction;
- pure-versus-mixed window analysis;
- transfer to WearGait-PD SPmT;
- exploratory transfer to FoG-STAR;
- generation of the result figures used in the paper;
- fixed subject splits and TUG training normalization statistics;
- unit tests for preprocessing, splitting, masking, models, and window reconstruction.

Raw datasets, trained checkpoints, outputs, logs, notebooks, and generated figures are intentionally excluded by `.gitignore`.

## Repository structure

```text
artifacts/paper/   Fixed splits, normalization vectors, and the reported architecture table
configs/           Experiment configuration files
scripts/           Direct entry points
src/tug_transfer/  Data loading, models, training, transfer, and figure code
tests/             Unit tests
docs/              Data layout and reproducibility notes
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
```

Install the PyTorch build appropriate for your CUDA version before the editable installation when required.

## Dataset setup

The datasets are not redistributed. Arrange them as described in [`docs/data_layout.md`](docs/data_layout.md), or edit the paths in the YAML files.

Default relative paths:

```text
data/weargait_pd/controls/
data/weargait_pd/pd/
data/fogstar/17838806.zip
```

## Run the experiments

Run TUG first because SPmT and FoG-STAR reuse the source checkpoint and normalization.

```bash
tug-transfer tug --config configs/tug.yaml
tug-transfer spmt --config configs/spmt.yaml
tug-transfer fogstar --config configs/fogstar.yaml
tug-transfer figures --config configs/figures.yaml
```

Equivalent scripts:

```bash
python scripts/run_tug.py --config configs/tug.yaml
python scripts/run_spmt.py --config configs/spmt.yaml
python scripts/run_fogstar.py --config configs/fogstar.yaml
python scripts/make_paper_figures.py --config configs/figures.yaml
```

Generated files are written under `outputs/`, which is excluded from Git.

## Expected paper results

| Experiment | Accuracy (%) | Balanced accuracy (%) | Macro F1 (%) |
|---|---:|---:|---:|
| Dense TUG | 94.77 | 94.67 | 93.46 |
| Fixed-window reconstruction | 91.21 | 89.87 | 88.89 |

Transfer macro F1:

| Condition | SPmT (%) | FoG-STAR (%) |
|---|---:|---:|
| Fine-tuned pretrained | 95.7 | 74.14 |
| Frozen encoder | 95.2 | 46.88 |
| From scratch | 94.9 | 69.19 |
| Heuristic zero-shot | 90.8 | 1.75 |

Small numerical differences may occur across PyTorch, CUDA, cuDNN, GPU, and driver versions. The paper reports one fixed participant split and seed 42.

## Reproducibility artifacts

```text
artifacts/paper/subject_level_split.csv
artifacts/paper/fogstar_subject_split.csv
artifacts/paper/imu_train_mean.npy
artifacts/paper/imu_train_std.npy
artifacts/paper/architecture_sensitivity.csv
```

The split files contain portable subject identifiers only; local filesystem paths were removed.

## Data and code scope

- WearGait-PD TUG: five labels — Sitting, SitToStand, Walk, Turn, TurnToSit.
- WearGait-PD SPmT: three labels — Standing, Walk, Turn.
- FoG-STAR: six-class activity recognition stress test.
- The models use 27 IMU channels from lower-back and bilateral-foot locations or their mapped proxies.
- The code performs offline, non-causal recognition and is not a clinical diagnostic system.

## License

The original source code in this repository is released under the MIT License.
See [LICENSE](LICENSE) for details.

The WearGait-PD and FoG-STAR datasets are not distributed with this repository
and remain subject to their respective licenses and terms of use. Third-party
figures and assets remain subject to their original licenses.