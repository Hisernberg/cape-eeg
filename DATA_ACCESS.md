# Data access

This repository does **not** redistribute any HMS data. It contains code, configuration, synthetic tests, aggregate results and aggregate figures only.

## Source
The study uses the original *HMS – Harmful Brain Activity Classification* competition files (Kaggle, 2024): `train.csv`, `train_eegs/<eeg_id>.parquet` (200 Hz, 19 electrodes + EKG, float32) and `train_spectrograms/<spectrogram_id>.parquet` (2 s rows, four regions × 100 frequencies 0.59–19.92 Hz). Obtain them from Kaggle under the competition's rules; the download is about 19.8 GB compressed and 25.9 GB extracted (measured locally). The 2024 competition has ended; this project is a local research experiment, not a submission.

## Rights
The data are used for research on the local machine only. Patient-derived artefacts (the derived spectral cache, split membership with identifiers, per-row predictions, model checkpoints trained on the data, executed notebooks with per-row outputs) stay under `private/` and are never uploaded. Public tables suppress cells with fewer than ten independent patients.

## Reconstruction
1. Place the files in a read-only directory and export `HMS_DATA_ROOT=/path/to/hms` and `CAPE_ROOT=/path/to/workspace` (the workspace holds `private/`).
2. `python scripts/audit_source.py` — schema gate, inventory, checksums (source manifest hash is reported in `results/aggregate/`).
3. `python scripts/make_splits.py` — deterministic 60/10/5/5/20 patient-component split, seed 20260907 (split hash `4455edaac7cb7e5d` for the 106,800-row source).
4. `python scripts/build_cache.py --smoke 64` then `python scripts/build_cache.py` — 5.6 GB paired float16 cache with packed validity masks (preprocess hash `2406ce595fbd7457`).
5. Notebooks 02–05 or `scripts/run_dev_pipeline.sh` and `scripts/run_final_pipeline.sh` reproduce training and the locked evaluation.

Hashes let an authorised researcher verify that they rebuilt exactly the same cohort, split and cache without any identifier leaving this repository.
