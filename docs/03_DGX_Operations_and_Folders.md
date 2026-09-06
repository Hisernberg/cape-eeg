# DGX operations and optimized folder plan

CAPE-EEG | Single DGX Spark / GB10 | 7 September 2026

## 1. Environment policy

Use the existing functioning ARM64/Blackwell PyTorch environment when it passes a local GPU smoke test. NVIDIA documents NGC as a source of DGX Spark environments, but a newer container is not automatically compatible with the machine’s installed driver. The 26.08 release notes reference CUDA 13.4.1; do not replace a working CUDA 13.0-era setup blindly. [S09, S10]

The prior device description is a GB10 Spark-class system with 128 GB shared memory. It is contextual information, not a live hardware inspection. Record `uname -m`, driver, GPU name, compute capability, Torch/CUDA versions, free system memory, free disk and container image digest locally. No password, token, environment dump or full home-directory listing belongs in this report.

Prefer a pinned working NGC image digest. If no working environment exists, first inspect the current compatibility documentation and the ARM64 manifest, then perform a small matrix multiply and convolution backward test. Do not rebuild PyTorch, install x86-only wheels or change the host driver as an automatic fallback.

## 2. One workspace, three trust zones

```text
~/research/cape-eeg/
  repo/                              # code + publishable specifications only
    AGENTS.md
    CODEX_EXECUTION.md
    START_HERE.md
    configs/
    src/cape_eeg/
    scripts/
    tests/
    notebooks/                       # source notebooks; no executed outputs
    docs/                            # Markdown authoritative documentation
    refs/
    templates/
    .github/workflows/               # CPU synthetic tests only
  private/                           # 0700; never a Git repository
    provenance/
    manifests/                       # IDs, source paths, split allocation
    cache/<preprocess_hash>/
    normalization/<split_hash>/
    runs/<run_id>/
      config.resolved.yaml
      status.json
      events.jsonl
      resource_usage.csv
      checkpoints/best.safetensors
      checkpoints/last.pt
      predictions/                   # individual rows remain private
    evaluation/<protocol_hash>/
    figures_private/
    approvals/
  public_export/                     # approved allowlist snapshot, not symlinks
  scratch/                           # disposable bounded intermediates
```

The original dataset remains outside this workspace at the directory the user supplies, referenced by `HMS_DATA_ROOT`. It is mounted read-only. Never copy the source dataset into `repo/` or pretend its external location eliminates its disk cost.

Store code in `repo/`, patient-derived information in `private/`, and explicitly approved publishable files in `public_export/`. A filename such as `anonymized.csv` does not confer permission to publish it. Even salted identifiers and spectrogram examples remain patient-derived data.

## 3. Executable initial setup

The following are implemented starter commands, not unimplemented training commands:

```bash
cd /path/to/CAPE_EEG_DGX_Master_Pack
python scripts/init_workspace.py --root "$HOME/research/cape-eeg"
python scripts/preflight.py --out "$HOME/research/cape-eeg/private/provenance/preflight.json"
python scripts/budget.py --rows 106800
python -m pytest -q
```

Copy the supplied repository files into the workspace’s `repo/` only after reviewing its contents. Configure the local dataset path without credentials:

```bash
export CAPE_ROOT="$HOME/research/cape-eeg"
export HMS_DATA_ROOT="/your/existing/hms-dataset"
python scripts/audit_metadata.py \
  --train-csv "$HMS_DATA_ROOT/train.csv" \
  --out "$CAPE_ROOT/private/provenance/metadata_pre_audit.json"
```

The audit writes private summary evidence. It does not train a model. Document 06 and `CODEX_EXECUTION.md` define implementation work required before end-to-end notebook execution.

## 4. Process and resource limits

| Resource | Target | Stop / gate |
|---|---|---|
| Concurrent GPU jobs | 1 | Lock file blocks second job |
| GPU allocation | 4–8 GiB or lower | Stop above 12 GiB |
| Process-tree RSS | ≤24 GiB | Stop above 32 GiB |
| System available memory | Keep ≥32 GiB | Pause below 32 GiB |
| Loader workers | 2 initially | Increase only after benchmark |
| CPU threads | 4 initially | No unrestricted thread fan-out |
| Active derived cache | ≤6.0 decimal GB | Preallocate only after byte check |
| Generated workspace | Aim ≤15 GiB | Hard stop at 30 GiB |
| Source data | Measured S | No assumed 5–6 GB source size |
| Containers / environment | Measured C | No silent repeated image pulls |
| Minimum free disk reserve | 20 GiB | Do not start a stage below reserve |
| Total GPU-wall time | ≤12 h | Incomplete jobs remain incomplete |

Spark has shared CPU/GPU memory. CUDA allocation and host RSS are overlapping accounting views, not two independent memory pools; track them alongside system availability. CUDA allocator statistics may miss allocations made by other processes. The supervisor must monitor process-tree and system state, not merely `max_memory_allocated()`.

The model is small, so start conservatively. There is no reason to reserve most of the 128 GB system or force the GPU to run continuously after the bounded experiment finishes.

## 5. CPU-first caching

Group offsets by recording, read each Parquet once where feasible, and process windows sequentially. Two producer processes are the starting point. STFT is CPU work; do not keep a GPU session open merely for CSV parsing, checksumming, plotting or bootstrap statistics.

Use memory-mapped float16 shards and packed masks. At most one shard’s temporary float32 arrays may exist per worker. Shared context files are opened on demand with a bounded LRU. Log cache progress by shard and counts, not by printing patient identifiers into public notebooks.

Estimate preprocessing duration from a fixed 64-recording CPU sample, stratified by file size and number of label offsets. Extrapolation is a planning estimate and must be replaced by actual time. A pathological large file is investigated explicitly rather than loaded into unbounded memory.

## 6. Job identity and resumption

A run ID combines phase, model, seed, split hash, preprocessing hash and configuration hash. The resolved config and source-code commit are immutable after the first training batch. A changed parameter creates a new run rather than overwriting an old metric row.

A run writes one of `PLANNED`, `RUNNING`, `PASS`, `FAIL`, `INCOMPLETE`, `NOT_RUN` or `BLOCKED`. Each status includes a reason, timestamp and artifact checksums. Resume only if all hashes match; otherwise start a new run. Never convert an interrupted run into `PASS` based on an existing partial checkpoint.

Use `.partial` output names and atomic replacement for completed shards and prediction tables. Maintain a single GPU lock using `flock`. The lock and supervisor implementation must be acceptance-tested on local synthetic jobs before real training.

## 7. Budget ledger

Record elapsed wall time for every GPU process, including warmup, failed attempts, calibration inference and stress tests. One sequential GPU process makes the accounting unambiguous. GPU-hours here mean elapsed time reserved by the project’s GPU jobs, not a utilization-adjusted estimate.

Measure median batch time after warmup, cache read rate, projected epoch time, peak memory and achieved examples/second. Predict the six final-fit cost before launching any of them. A faster baseline and slower candidate receive the same declared training exposure where possible and each has measured actual cost.

Do not shorten the candidate’s training halfway through the confirmatory comparison and call it a fair result. If the schedule does not fit, return to the development stage, amend and refreeze before test access, or stop with a bounded pilot report.

## 8. Checkpoint lifecycle

Retain best and last checkpoints for only the active training job. On completion keep one final inference weight artifact, its exact config and measured validation evidence. Last-state optimizer checkpoints may be deleted only after a verified successful run and a local retention decision; never delete original source data automatically.

Retain all final seeds’ predictions privately even when only seed 101 is selected for deployment. Predictions are much smaller than repeated model snapshots. Avoid saving full activations, every-epoch checkpoints, TensorBoard image dumps or duplicate caches. Export embeddings only for a fixed development sample when an analysis actually needs them.

Weights are excluded from public release by default pending dataset/pretraining-rights review. Use safe tensor serialization for inference weights; resume-state pickle formats are trusted-local only, never downloaded and unpickled from unknown sources.

## 9. Notebook operation

A notebook is a thin interface to tested package functions, not a separate implementation of data loading and training. Its first cell prints the resolved run ID, approved partition and resource caps without showing secrets. Long stages can be launched from the notebook as a bounded process and monitored through the manifest.

Rerunning a cell reuses a verified artifact with the same hash or creates a new named run. It must not retrain silently, append duplicate CSV rows, access test labels, or delete arbitrary folders. Private executed notebooks are separate from clean public notebook sources.

## 10. Network and credential boundaries

Separate network-enabled setup/publishing from network-disabled data processing/training. HF access is limited to approved model files pinned by revision. Download only a config and a safetensors checkpoint as needed, not an unrestricted repository snapshot. [S11]

Provide GitHub/HF credentials through the local credential manager or process-scoped environment configured outside the repository. Never paste them into notebook cells, Markdown, YAML, command arguments, URLs, model cards, terminal transcripts or Codex prompts. The token string is not required in this chat.

No GitHub upload occurs in training or evaluation code. Publication is a distinct, human-authorized stage with separate permissions, visibility approval, two scans and a reviewed immutable export hash.

## 11. Failure recovery

Out-of-memory: terminate the job, preserve failure evidence, lower physical batch size while preserving effective batch size, and rerun as a new development configuration. Corrupt cache: rebuild only the failed shard from source hashes. NaN loss: stop immediately and trace invalid input/normalization; do not replace the score with zero.

Unavailable GPU kernels: stop and inspect compatibility; do not launch a multi-hour source build. Missing data: record exclusions and resolve the source inventory. Token leak: revoke/rotate the credential and follow repository incident handling before any further push. A failed test gate is never bypassed to maintain an automation schedule.

## 12. Required final operational evidence

The run is operationally complete only with a measured environment snapshot, source/split/cache hashes, a passed leakage audit, finite training logs, actual compute/memory records, final prediction files, a locked evaluation report, a 20-figure manifest, and a publication decision report. For unrun stages the correct entry is `NOT_RUN`, with null metrics.
