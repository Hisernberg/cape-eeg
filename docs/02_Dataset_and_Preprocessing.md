# Dataset, alignment and compact-cache contract

CAPE-EEG | Data specification | 7 September 2026

## 1. Source of truth and acquisition

The requested source is the original HMS competition directory, not a replacement Alzheimer MRI dataset. The user will supply it locally. Only descriptive material is present in this conversation; no actual `train.csv` or Parquet recording has been inspected. The relevant schema and temporal definitions are supplied by the user. [S01]

A trustworthy 5–6 GB Hugging Face copy preserving patient IDs, all vote counts and both offsets was not verified in this research pass. Therefore `hf_dataset_repo: null` is deliberate. Do not invent a repository identifier or call an unrelated image dataset an equivalent substitute.

An HF mirror may be approved later only if its provenance, usage terms, revision, file inventory, checksums, label distributions, patient IDs and offset fidelity are verified against the original source. Repository access does not imply redistribution rights. HF model access is separate; the compact comparator has a verified model card. [S08, S11]

The active-cache budget is not the source-download budget. Measure the existing source size S, container/runtime size C and generated workspace size W separately. Total local disk consumption is S+C+W; a 5.6 GB cache does not make a larger source disappear.

## 2. Required files

```text
HMS_DATA_ROOT/
  train.csv
  train_eegs/<eeg_id>.parquet
  train_spectrograms/<spectrogram_id>.parquet
  test.csv                         # optional inference-format smoke only
  test_eegs/<eeg_id>.parquet        # optional
  test_spectrograms/<id>.parquet    # optional
  sample_submission.csv            # optional format reference
  example_figures/                 # optional private inspection only
```

The tiny downloaded test example is not a labeled validation cohort. The 2024 competition has ended; the current project is a local research experiment, not a promise of a new scored submission. [S01]

Use exactly this label order everywhere:

`seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote`

Preserve `patient_id`, `eeg_id`, `eeg_sub_id`, `spectrogram_id`, `spectrogram_sub_id`, `label_id`, both offsets, raw integer votes and the original consensus field in private metadata. Consensus is a convenience label, not a substitute for q. Do not infer that `other` means a healthy control.

## 3. Schema and provenance gate

Validate integer nonnegative votes, positive vote sum, unique `label_id`, nonmissing identifiers, finite nonnegative offsets and existence of referenced files. Duplicate label IDs fail the intake gate. Distinct labels at identical windows are retained and audited; do not blindly merge their vote counts because shared annotators cannot be identified.

The supplied spectrogram offset name contains a spelling irregularity: `spectogram_label_offset_seconds`. Accept that exact field and the correctly spelled `spectrogram_label_offset_seconds` as aliases. If both exist, their numerical values must agree. Store one canonical name while retaining the source spelling in the ingestion report.

Audit patients per EEG and per spectrogram. A recording shared across patient IDs must not enter two partitions: merge connected records into the same split component and flag the anomaly. Exact and near-duplicate input hashes across components require review before the split is released. No private identifiers appear in public CSV tables by default.

Create `source_manifest.json`, `cohort_summary.json`, `exclusions.csv`, `file_checksums.jsonl`, `schema_report.json` and an immutable input hash. Full checksumming is one sequential CPU/I/O pass; do not hash every source file at every notebook run.

## 4. Exact temporal alignment

For an EEG label offset e seconds, use source samples `[round(200e), round(200(e+50)))`. Sampling is 200 Hz. Confirm offsets lie on the sample grid within 1e-6 sample before rounding; a non-grid offset requires a documented source-convention review. This is 10,000 samples for a complete window. The labeled center is relative seconds `[20,30)`, i.e. samples `[4000,6000)` within that window. [S01]

For a supplied spectrogram label offset s, the long-context interval is `[s,s+600)` seconds and its labeled center is `[s+295,s+305)`. The corresponding 50-second central contextual interval is `[s+275,s+325)`. These are relative to the spectrogram’s own time axis, not necessarily the EEG file origin. [S01]

Read the spectrogram `time` column. Select timestamps using the interval; do not assume `offset//2` is universally correct. Check monotonicity, spacing, frequency suffixes and coverage. A commonly encountered two-second spacing is a measured property, not a hard-coded universal rule.

In the full converter, short/truncated edges are padded as missing and accompanied by a validity mask. The supplied `eeg_window_bounds` helper deliberately accepts complete windows only; the converter must implement and test an explicit missing-edge path rather than bypass that rejection by shifting the window. Do not repeat the last row, wrap around, silently move the window, or shift the target to a nearby event. Where source conventions are ambiguous, flag `ALIGNMENT_UNVERIFIED` and stop cache finalization.

Use deterministic synthetic time ramps and impulse windows to test center placement. Private spot-checks cover early, middle, late and partially missing offsets before full conversion. A plot that looks plausible is not sufficient proof of correct indexing.

## 5. Raw EEG preprocessing

Read one recording at a time using selected Parquet columns. Do not load every raw EEG into a Python dictionary. Process all requested offsets while that recording is open, then release it. A two-file LRU cache is the maximum default.

The regional bipolar chains are a fixed design choice:

| Region | Four bipolar differences |
|---|---|
| LL | Fp1−F7, F7−T3, T3−T5, T5−O1 |
| RL | Fp2−F8, F8−T4, T4−T6, T6−O2 |
| LP | Fp1−F3, F3−C3, C3−P3, P3−O1 |
| RP | Fp2−F4, F4−C4, C4−P4, P4−O2 |

Validate the actual channel spellings before use. Do not reinterpret renamed electrodes without an explicit montage mapping. The EKG lead is excluded from EEG spectra. Midline leads not in these chains are not used by the core; document that information loss rather than claiming all spatial information is retained.

For each 50-second window, identify nonfinite values before interpolation. Linear interpolation is permitted only for interior gaps of at most 0.25 seconds; preserve the original missingness mask. Longer gaps are zero-filled only for numerical computation and are marked invalid. Remove each valid channel’s median. Do not use a dataset-wide normalizer before splitting.

Compute a Hann-window STFT with sample rate 200 Hz, window length 256 samples, hop 64 samples, FFT length 256, `boundary=None`, and no end padding. Form power in float32. Retain frequencies in 0.5–40 Hz and interpolate/integrate onto 64 fixed frequency bins; the extra bins do not imply improved physical frequency resolution.

For each region average linear power across valid bipolar leads, perform the temporal integration in Section 6, and only then apply `log(max(power, 1e-8))`. Do not average already logged power when constructing the default cache. A time-frequency cell is valid only when the contributing window has at least 90% observed samples and at least two of its four bipolar leads are valid. Thresholds are prespecified engineering rules, not validated clinical criteria.

Core preprocessing uses no extra notch or high-order bandpass. Additional filtering would change the representation and is reserved for a clearly named sensitivity test, not hidden cleaning. A microvolt amplitude interpretation is withheld until source units are verified.

## 6. Foveated versus uniform local view

Both local encodings store four regions × 64 frequency bins × 32 temporal bins. For the foveated view, temporal bins cover:

- Left context `[0,20)`: eight equal-duration bins.
- Labeled center `[20,30)`: sixteen equal-duration bins.
- Right context `[30,50)`: eight equal-duration bins.

Aggregate STFT power using overlap-weighted bin integration before the log transform. Respect actual STFT window center times. Never treat the foveated x-axis as uniformly spaced seconds in a visualization. The target occupies columns `[8,24)` and is marked in the cache schema.

The uniform comparator uses 32 equal-duration bins across `[0,50)` and is otherwise identical. Its center-pooling map must be computed from its actual time intervals, not copied as columns 8:24. The reference model accepts a `center_fraction` appropriate to the encoding; this difference is tested.

The foveated and uniform representations are not cached in full simultaneously. Retain the default full cache and build a development-only alternate cache under the workspace cap. A final uniform baseline may rebuild/replace the local cache by content hash, while reusing the unchanged context cache and private row ordering.

## 7. Supplied long-context spectrogram

Parse region prefixes and numerical frequency suffixes; sort by frequency numerically. Preserve the source frequency range in the schema. Do not invent high-frequency information absent from supplied spectra. Each view has its own frequency coordinates and normalizer.

Extract the 600-second interval, clip finite positive power below at 1e-8, integrate valid power into 64 temporal and 64 frequency bins per region, then log-transform. Preserve invalid-cell masks. Negative or unexpected quantities trigger a units/domain review instead of blindly applying logarithms.

There is no assumption that the raw-derived spectrum and supplied spectrum have identical preprocessing, amplitude scaling or frequency coverage. Their stems and normalization are separate. Fusion occurs after encoding, not by subtracting supposedly equivalent spectral pixels.

## 8. Storage arithmetic

For one label row:

`float16 values = 4 × 64 × (32+64) × 2 = 49,152 bytes`

`bit-packed validity = 4 × 64 × 96 / 8 = 3,072 bytes`

`total tensor payload = 52,224 bytes`

At an illustrative 106,800 rows, values occupy 5,249,433,600 bytes and masks 328,089,600 bytes. Combined payload is **5,577,523,200 bytes = 5.578 decimal GB ≈ 5.194 GiB**. The count is a literature-scale planning input, not an audited local count. [S02]

Budget up to 250,000,000 additional bytes for compact metadata, row indexes, hashes and small array headers. Expected total at that row count is below 5.83 GB. The hard active-cache limit is 6,000,000,000 bytes; count actual files and overhead before declaring success. No compression ratio is assumed. On a fixed training-only sample, compare float32 log-power values with their float16 round trip; report maximum/median absolute error and nonfinite counts. Float16 storage is accepted only after this CPU representation check; it is not an assumption of lossless storage.

Use shard groups of at most 2,048 labels with contiguous `.npy` arrays for focal values, context values and packed masks. Access by memory map, `allow_pickle=False`. Sort the private row index by `label_id`; processing order may differ and must use an explicit row map. Commit each shard via `.partial` plus atomic rename and checksum verification.

Do not create 100,000 PNGs, pickle all arrays, copy the cache into Git, or materialize float32 dataset replicas. Convert only each mini-batch to float32. The byte calculator supplied in this package is executable and does not require the medical data.

## 9. Normalization and missingness

The cache stores deterministic log-power values independent of fitted dataset statistics. Fit median/scale per view and region on training data only, with a bounded deterministic sample of valid cells. Store normalization parameters under the split and preprocessing hash. After final refit uses train+tune, refit normalization only on that combined training set.

For numerical input, normalize valid cells, clip to [-8,8], and set missing cells to zero **after** normalization. A zero-filled missing cell is not evidence of normal EEG. Retain valid fractions and masks for quality audits, subgroup reporting and controlled view dropout.

If either view is entirely unavailable, use the surviving-view path and report the subgroup. If both fail, exclude with an explicit reason before any outcome-based evaluation. Report all exclusions by partition and class; never remove difficult or high-entropy labels merely to improve performance.

## 10. Patient-independent split

Construct connected components linking patient IDs with EEG IDs and spectrogram IDs. Partition components deterministically with fixed seed 20260907 into 60/10/5/5/20 proportions, balancing component size and six-class vote mass as well as practicable. Allocation may consult label summaries to stratify, but may never consult model predictions.

Freeze once after verifying that all labels with the same patient, recording and overlapping context remain together. Exact split proportions are approximate when components have different sizes. If a class is absent from a required partition, document and resolve feasibility before any modeling; do not repeatedly redraw until the eventual score looks favorable.

The final bootstrap resamples patients only if components are patient-disjoint. Otherwise resample components and retain patient-equal weighting inside each replicate. Report both counts. Private split manifests are reproducible from source hashes and the allocation algorithm; public distribution tables suppress small cells rather than revealing identifiers.

## 11. Readiness checklist

Data intake passes only when identifiers/votes/offsets are valid, referenced files are accounted for, alignment tests pass, patient/recording overlap is zero across partitions, cache-byte prediction is within cap, exclusion rules are frozen, source terms are recorded, and the normalizer is train-only. Unknown rights block public dataset or weight release; unknown alignment blocks training.

The supplied `audit_metadata.py` performs a **CSV-level pre-audit only**. It does not inspect Parquet values, verify clinical units, create the final stratified split or certify leakage freedom. Those implementation tasks and their acceptance tests are specified for Codex in Document 06.
