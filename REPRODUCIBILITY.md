# Reproducibility

* **Environment.** DGX Spark (NVIDIA GB10, compute capability 12.1, 121 GiB unified memory, 20 CPU cores), Ubuntu kernel 6.17, Python 3.12, PyTorch 2.9.0+cu130, CUDA 13.0, cuDNN 9.13, timm 1.0.29. The environment snapshot without secrets is written by `scripts/audit_source.py` to `private/provenance/preflight.json`.
* **Determinism.** Split seed 20260907; final training seeds 101/202/303; bootstrap seed 20260907 with 2,000 replicates; robustness subset seed 20260907. Every run id embeds the split hash, preprocess hash and a configuration hash; a changed parameter produces a new run rather than overwriting a metric.
* **Hashes.** Source manifest, split, cache, normalizer and protocol hashes are recorded in `results/aggregate/final_evaluation_summary.json`.
* **Ledger.** All GPU wall time (including failed or smoke jobs) is appended to `private/runs/gpu_ledger.jsonl`; the study ceiling is 12 GPU-wall-hours and the totals are exported in `results/aggregate/table5_resources.csv`.
* **Tests.** `python -m pytest -q` runs 170 CPU-only synthetic tests (metrics, bootstrap, calibration, contracts, alignment, spectral encoders, splits, cache round trip, model, losses, engine, figures, revision extensions). They certify code behaviour, not clinical validity.
* **Notebooks.** `notebooks/` holds the seven thin notebooks; their executed copies with aggregate-only outputs are in `notebooks/executed/`. Notebooks call tested package code through `scripts/`; they never re-implement training.
* **What is not reproducible from this repository alone.** The raw data (see `DATA_ACCESS.md`) and the trained weights (withheld pending a rights review). Everything else is rebuildable from source with the commands in `DATA_ACCESS.md`.
