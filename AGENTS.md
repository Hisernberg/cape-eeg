# CAPE-EEG Codex operating contract

Read START_HERE.md and docs/01–08 before changing code. The authoritative task is six-class HMS expert-vote EEG-pattern prediction, NOT Alzheimer diagnosis. User provides source data locally. No credentials belong in this repository or in prompts.

## Immutable constraints

- Original data read-only; patient/recording components never cross partitions.
- No zero-vote target, no majority-only replacement, no hidden difficult-case filtering.
- Source offsets and raw/spectrogram clocks follow Document 02 exactly.
- Active cache at most 6,000,000,000 bytes, with masks/metadata counted.
- One GPU process, 12 GPU-wall-hour study ceiling; GPU allocation stop 12 GiB; process-tree RSS stop 32 GiB; keep 32 GiB system available.
- No host-driver changes, large teacher, unrestricted HF snapshot, paid compute, raw-data upload or clinical deployment as automatic actions.
- No test data/labels/figures used for tuning. Network weights train on train+tune only after method selection; calibration-T and calibration-P remain separate.
- All metrics from real measured evidence or null with NOT_RUN/NOT_ESTIMABLE. Synthetic fixture outputs are never medical results.
- GitHub upload is a separate explicitly authorized phase with exact destination/visibility and immutable export hash.

## Work order

1. Inventory existing implemented files and tests; do not assume SPEC modules exist.
2. Run CPU synthetic tests and local hardware preflight. Record failures verbatim without secrets.
3. Implement source inventory, montage, alignment and split allocator with unit tests.
4. Implement bounded cache conversion, normalizers and dataset loader; validate one shard before full conversion.
5. Implement resource supervisor and training engine; run tiny synthetic GPU smoke only after permission gates.
6. Run bounded development baselines/ablations; freeze protocol and final epoch schedule before any test access.
7. Complete two methods × seeds101/202/303 sequentially, calibrate in the prescribed partitions.
8. Evaluate once, generate measured tables and V01–V20, preserve all failure statuses.
9. Prepare clean public export; perform required scans and await exact locally recorded publication approval.

## Completion evidence per task

Every task changes tests, implementation_status.json and a private status record. Include configuration/source/split/cache hashes, measured resource use and exact generated paths. A notebook blueprint is not a completed training notebook. Do not create placeholder metrics to satisfy report generation.

## Automatic stop conditions

Stop on unknown source units needed by a transform; invalid temporal alignment; cross-partition linked records; missing rights for a requested release; invalid probabilities; nonfinite loss; resource ceiling; failed scanner; stale approval; or an attempted test-label read from a development phase. Return a concrete BLOCKED reason rather than bypassing the gate.

## Local secrets

HF/GitHub authentication is configured outside the repo. Never request the token string in a chat or notebook; never log the full environment. No public push, visibility change or model-weight publication is implied by the request to build this local experiment.
