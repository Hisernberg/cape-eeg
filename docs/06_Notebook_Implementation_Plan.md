# Seven-notebook implementation plan

CAPE-EEG | Thin notebooks, tested shared modules | 7 September 2026

## 1. Implementation boundary

This package supplies notebook **blueprints**, not completed medical-data training notebooks. Their executable starter cells verify paths/status only and their markdown cells specify required sections. The reusable compact model, scoring functions, byte calculator, CSV audit, preflight and export precheck are starter implementations tested only on synthetic data here.

Codex must implement the source-specific Parquet conversion, final split allocator, trainer, resource supervisor, calibration/evaluation pipeline, plotting suite and gated GitHub orchestration before those notebooks can execute the experiment end-to-end. Stubs must raise or report `NOT_IMPLEMENTED`; they must never emit plausible synthetic medical scores.

## 2. Shared package structure

```text
src/cape_eeg/
  contracts.py                # implemented basic labels, offsets and byte rules
  metrics.py                  # implemented core distributional metrics
  model.py                    # implemented synthetic-test compact architecture
  data/
    inventory.py              # SPEC: source/schema/checksum audit
    montage.py                # SPEC: validated electrode mapping
    alignment.py              # SPEC: Parquet time coordinates
    cache.py                  # SPEC: paired spectral shards + masks
    splits.py                 # SPEC: connected components + frozen allocation
    dataset.py                # SPEC: bounded mmap mini-batches
  training/
    losses.py                 # SPEC: model loss assembly and masked auxiliary loss
    engine.py                 # SPEC: bounded fit/resume + complete schedule gate
    supervisor.py             # SPEC: process-tree/system memory and GPU-time guard
  evaluation/
    calibrate.py              # SPEC: calibration-T and calibration-P separation
    bootstrap.py              # SPEC: paired patient/component uncertainty
    robustness.py             # SPEC: fixed perturbation suite
    report.py                 # SPEC: sanitized tables and immutable evidence
  visualization/
    registry.py               # SPEC: V01–V20 source/artifact contracts
```

Modules marked SPEC are not silently supplied as working code. Add them with corresponding tests, then update the implementation status ledger. Avoid code duplication between notebooks and the command-line interface.

## 3. Notebook 00 — environment and data intake

**File:** `00_environment_and_intake.ipynb`. **GPU:** off except a separately approved tiny smoke. **Inputs:** local paths, source files, environment config. **Outputs:** private preflight, source inventory and readiness report.

Required cells: project root resolution; configuration display with secret redaction; ARM64/GPU/driver probe; disk/shared-memory accounting; train.csv schema audit; file counts/checksums; usage-term checklist; tiny fixed-window alignment inspection; cache-byte estimate; go/no-go summary.

Acceptance tests: no environment/token dump; all output paths under the private root; no source mutation; invalid votes/offsets fail; no raw-file copy; unknown source rights explicitly recorded. The provided CSV pre-audit is only the start of this notebook.

## 4. Notebook 01 — cohort, splits and compact cache

**File:** `01_cohort_splits_and_cache.ipynb`. **GPU:** off. **Inputs:** validated source manifest. **Outputs:** frozen split manifest, exclusions, normalization specifications, cache shards and checksums.

Required cells: component construction from patient/EEG/spectrogram links; balanced deterministic split; train/tune/calibration/test count table; duplicate and overlap audit; montage validation; raw STFT smoke; supplied-spectrum timestamp extraction; uniform/foveated time-bin tests; one-shard memory measurement; full bounded cache; train-only normalizer fit; cache completion manifest.

Acceptance tests: all requested windows retain exact labels and offsets; zero cross-partition component leakage; the two modalities align around their respective centers; foveated and uniform tensors have equal bytes; packed masks round-trip; partial shards are not accepted; cache stays below 6 billion bytes; out-of-range times do not wrap.

No test waveform figures or label-dependent examples are exported to the development role. Private data-quality review is logged separately from model tuning.

## 5. Notebook 02 — baselines and throughput

**File:** `02_baselines_and_budget.ipynb`. **GPU:** bounded development pool. **Inputs:** train/tune only, pretrained-weight manifest if approved. **Outputs:** B0/B1/B2/B3 development results, sample-exposure counts and runtime forecast.

Required cells: metric sanity checks; train prior; CPU feature baseline; compact B2 smoke and loss/backward check; HF checkpoint/revision/license record; B3 four-channel adaptation; batch-size profiling; fixed-budget fits; tune patient-KL comparison; final-six-fit cost forecast.

Acceptance tests: small weight download only; no hidden-pretraining overlap with HMS for the ImageNet comparator; no forced ImageNet color transform; targets never used as features; calibration/test labels inaccessible; failures remain in the ledger; final-budget projection below the remaining limit.

B1 uses fixed regional band-power summaries from each view and regularized soft-label regression, implemented against fractional targets rather than silently converting them to hard labels. The exact feature definition and training-only scaling are versioned.

## 6. Notebook 03 — mechanism ablations and protocol freeze

**File:** `03_ablation_and_freeze.ipynb`. **GPU:** bounded ablation pool. **Inputs:** development data and baseline ledger. **Outputs:** ablation table, fixed comparator, fixed epochs, referral-score choice and `protocol_lock.json`.

Required cells: equal-byte uniform/foveated comparison; fixed versus learned mixture; auxiliary coefficient 0 versus 0.1; input-only feature audit; model parameter/FLOP measurements; development paired-loss SD; feasibility/power scenarios; leakage audit rerun; preregistered final metric/figure/corruption list; configuration freeze.

Acceptance tests: no final-test access; no more than eight named development configurations overall; baseline selection is from tune results; augmentation and sample exposures are matched; center-region pooling uses the right coordinate map; no invented clinical margin; all known design deviations recorded before lock.

This notebook may reject the proposed method. Rejection is a valid result; it must not trigger an unbounded automatic architecture search.

## 7. Notebook 04 — final fits and calibration

**File:** `04_final_training_and_calibration.ipynb`. **GPU:** at most 7 final-fit hours plus allocated inference. **Inputs:** frozen protocol; train+tune; separate calibration-T/P. **Outputs:** two methods × three seeds, their calibrated predictors and completion manifests.

Required cells: GPU lock and budget check; resolved frozen config; final normalizer using train+tune only; seed101/202/303 sequential fits; restart/checkpoint verification; finish-schedule check; calibration-T scalar temperature; calibration-P coverage thresholds; immutable inference bundles.

Acceptance tests: six complete fits or explicit incomplete-study status; no choice of best test seed; calibration-T/P never used for gradient fitting of networks; no second calibration family; no competing GPU process; correct per-view missingness fallback; exportable seed101 chosen in advance.

## 8. Notebook 05 — locked test, advanced analyses and 20 figures

**File:** `05_locked_evaluation_and_figures.ipynb`. **GPU:** inference only. **Inputs:** frozen inference bundles and evaluator-approved locked labels. **Outputs:** private row predictions, matched score tables, bootstrap intervals, stress tests and V01–V20 manifests.

Required cells: one-time test gate; key alignment; raw and calibrated probabilities; patient/row KL; unique-majority hard metrics; calibration; disagreement/referral; paired bootstrap; vote-count and missingness strata; predefined corruption subset; evidence-use audit; resource plots; all 20 figures; claims checklist.

Acceptance tests: every plot joins to real prediction/run hashes; no placeholder numbers; no row-bootstrap replacement for patient inference; all exclusions and missing metrics visible; each figure’s denominator and axis units explicit; public figures have no patient IDs or unapproved individual traces.

## 9. Notebook 06 — release preparation

**File:** `06_release_and_reproducibility.ipynb`. **GPU:** off. **Inputs:** evaluated evidence and publication policy. **Outputs:** clean export candidate, scan reports, artifact manifest and publish decision.

Required cells: reproducibility audit; code/tests/environment record; executed-notebook output removal; aggregate privacy suppression; rights review; fresh allowlist export; content precheck; full secret scan; Git-history scan; approval/visibility validation; dry-run push report; optional explicitly authorized push.

Acceptance tests: no raw data, row predictions, checkpoints, credentials or symlinks exported by default; no wildcard `git add .`; no remote write without exact owner/repository/branch/visibility and approved export hash; push and release statuses remain `NOT_RUN` unless actually completed.

## 10. Cell conventions

Every notebook begins with purpose, inputs, outputs, permitted partitions, status and budget. Each expensive stage calls shared package code with a named config. Cells are idempotent: identical verified outputs are reused, and changed inputs create a new artifact hash.

Notebook metadata is stripped before public export. Notebook source must not contain hard-coded absolute personal paths, tokens, inline credentials, patient identifiers or clinical example outputs. Small synthetic fixtures are deterministic and unmistakably synthetic.

## 11. Minimum acceptance-test suite for Codex

Required additional tests include: raw-offset rounding; central10 placement; nonuniform time bins; supplied time-column coverage; channel order; frequency numeric sort; no-EKG feature path; missingness masks; train-only normalization; component leakage; duplicate labels; probability row sums; n=1 auxiliary masking; finite gradients; absent-view routing; byte ceiling; budget stop; resumability mismatch; test access denial; notebook output stripping; export traversal/symlink rejection; secret scanner failure; stale approval rejection.

A CPU synthetic pass does not certify GPU kernels, source units, clinical validity or medical accuracy. Keep these statuses separate in every report.
