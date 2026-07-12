# Expected data layout

## WearGait-PD

The loader scans two directories for CSV files.

```text
data/weargait_pd/
├── controls/
│   ├── HC001_TUG.csv
│   ├── HC001_SelfPace_matTURN.csv
│   └── ...
└── pd/
    ├── NLS001_TUG.csv
    ├── NLS001_SelfPace_matTURN.csv
    └── ...
```

The exact participant prefixes can differ. Subject ID, site, and group are inferred using the same rules used in the original notebooks.

### TUG file discovery

A CSV is considered a TUG candidate when its filename contains one of:

```text
tug
timedupgo
timed_up_go
timed-up-go
```

### SPmT file discovery

The default pattern is:

```text
*_SelfPace_matTURN.csv
```

This can be changed in `configs/spmt.yaml`.

### Required WearGait-PD columns

The final experiments use 27 IMU channels:

```text
LowerBack_Acc_X
LowerBack_Acc_Y
LowerBack_Acc_Z
LowerBack_FreeAcc_E
LowerBack_FreeAcc_N
LowerBack_FreeAcc_U
LowerBack_Gyr_X
LowerBack_Gyr_Y
LowerBack_Gyr_Z
L_DorsalFoot_Acc_X
L_DorsalFoot_Acc_Y
L_DorsalFoot_Acc_Z
L_DorsalFoot_FreeAcc_E
L_DorsalFoot_FreeAcc_N
L_DorsalFoot_FreeAcc_U
L_DorsalFoot_Gyr_X
L_DorsalFoot_Gyr_Y
L_DorsalFoot_Gyr_Z
R_DorsalFoot_Acc_X
R_DorsalFoot_Acc_Y
R_DorsalFoot_Acc_Z
R_DorsalFoot_FreeAcc_E
R_DorsalFoot_FreeAcc_N
R_DorsalFoot_FreeAcc_U
R_DorsalFoot_Gyr_X
R_DorsalFoot_Gyr_Y
R_DorsalFoot_Gyr_Z
```

The TUG and SPmT label column is `GeneralEvent`.

The `Time` column is deliberately excluded.

## FoG-STAR

Place the original archive at the path configured in `configs/fogstar.yaml`.

```text
data/fogstar/17838806.zip
```

The ZIP must contain:

```text
sensor_data.csv
clinical_data.csv
README.txt
```

The external-transfer loader uses lower-back and bilateral ankle accelerometer and gyroscope channels. Ankle sensors are treated as proxies for the WearGait-PD dorsal-foot sensors. Wrist channels are excluded.

## Split CSV format

An explicit TUG split file should contain at least:

```csv
subject_id,split
HC001,train
HC002,val
NLS001,test
```

The subject column may also be named `subject` or `subj_id`. Split values must be `train`, `val`, or `test`.
