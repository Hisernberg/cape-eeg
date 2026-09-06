# Codex execution brief

## Paste this instruction, not credentials

> Implement and run the CAPE-EEG experiment from this package on my local DGX Spark. Treat AGENTS.md and documents 01–08 as the specification. My dataset is at the local path configured in HMS_DATA_ROOT. First inspect existing code and run synthetic tests, then produce the CSV/Parquet audit, time-alignment tests, component-isolated split, exact cache-byte estimate and bounded-memory one-shard smoke. Do not start full training until those gates pass and the measured 12-GPU-hour plan is feasible. Implement missing SPEC modules with tests and turn the seven notebook blueprints into thin runnable interfaces. Keep all patient-derived artifacts in private/. Preserve every failed or unrun status. Do not read locked test outcomes during development, change the source dataset, print tokens, upload data or push to GitHub without the separate destination/visibility/export-hash approval described in Document 07.

## First runnable commands

```bash
python scripts/init_workspace.py --root "$HOME/research/cape-eeg"
python -m pytest -q
python scripts/budget.py --rows 106800
python scripts/preflight.py --out "$HOME/research/cape-eeg/private/provenance/preflight.json"
python scripts/audit_metadata.py --train-csv "$HMS_DATA_ROOT/train.csv" \
  --out "$HOME/research/cape-eeg/private/provenance/metadata_pre_audit.json"
```

The row count in the first byte estimate is illustrative. Rerun with the actual eligible count after source audit. The current preflight probes information; it does not certify GPU training compatibility.

## Implementation phases and bounded acceptance

| Phase | Implement | Acceptance artifact | GPU permission |
|---|---|---|---|
| I0 | Local paths, status ledger, environment probe | environment/safety report | None |
| I1 | Source inventory, montage, offsets, components | data and leakage audit | None |
| I2 | Paired spectral cache and masks | verified one-shard smoke then cache manifest | None |
| I3 | Loader, losses, resource supervisor, trainer | synthetic finite-gradient test and budget stop test | Tiny approved smoke |
| I4 | B0/B1/B2/B3 and A1/A2/P pilots | tune results and resource projection | Development allocation |
| I5 | Freeze protocol and final fits | six completed-fit manifests and calibration objects | Final allocation |
| I6 | Metrics/bootstrap/stress/figures | test report and 20-entry figure manifest | Inference allocation |
| I7 | Sanitization/scans/release orchestration | approved exact export | No GPU |

Only phases I0 and parts of I1/I3/I7 have starter utilities in this package. Do not advertise an end-to-end run before implementing and validating the remaining phases.

## Configuration resolution

Read configs/study.yaml as the default specification. Resolve local paths into a private config copy. Pin the working environment and HF model revision after verification; null entries are blockers for the affected stage, not placeholders to guess. Store credentials only in the local credential manager or process-scoped environment.

## Required implementation report

After each phase report what changed, tests run, measured memory/time, source and output hashes, remaining blockers and the next permitted phase. Do not claim work is running in the background or that a later result is guaranteed. Stop when a gate fails or the budget is exhausted.

## Publication automation implementation target

Implement the state machine in Document 07 using an authenticated documented GitHub connector or local CLI only after the user records exact owner/repository/branch/visibility. Start with dry-run mode. Content and history scans must fail closed; approval binds to the exact exported artifact hash. No repository is created or modified by this package itself.
