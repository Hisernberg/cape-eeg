# Start here — CAPE-EEG on DGX Spark

This is a finalized experiment-design package plus CPU-tested starter utilities and seven notebook blueprints. It is not a trained medical model or a completed end-to-end experiment.

## Read in this order

01 Research Master Plan → 02 Dataset/Preprocessing → 03 DGX/Folders → 04 Evaluation → 05 Twenty Visualizations → 06 Notebooks → 07 GitHub → 08 Design Review. PDF editions are in pdfs/; editable authoritative sources are in docs/. Verified references are in refs/REFERENCES.md; cited sources are reproduced in the relevant PDF editions.

## What to place locally

Provide train.csv, train_eegs/ and train_spectrograms/ in an existing read-only directory and set HMS_DATA_ROOT to it. Test examples are optional format checks only. No HF dataset mirror has been certified; the original HMS data is the chosen source. The 5–6 GB target is a derived two-view cache, not a claim about original-download size.

## First actions

Run the synthetic tests, byte calculator and preflight in CODEX_EXECUTION.md. Then give Codex AGENTS.md, CODEX_EXECUTION.md and the full docs/ directory. The seven notebook files are implementation blueprints; their placeholder stages do not train or fabricate results.

## What is included

Eight separate PDF/Markdown plans; 20 figure specifications and a manifest; seven notebook blueprints; a detailed configuration; small model and metric reference code; local workspace/preflight/CSV-audit/byte/export utilities; synthetic tests; research references; and evidence/status templates.

## Current limits

No real HMS recordings, measured DGX throughput, medical scores, final statistical conclusions, source-redistribution approval or GitHub push were produced here. See evidence/local_validation.json for the exact local synthetic checks and evidence/implementation_status.json for what still needs implementation.

## Credentials and publication

Configure HF/GitHub authentication locally outside the repository. Do not paste token strings into chat, notebooks or Markdown. GitHub publication means screened reproducibility artifacts; patient data and individual predictions stay private. Public visibility and weight release require separate approval.

## Minimal local setup

From the extracted package root, run the commands in CODEX_EXECUTION.md. Set `HMS_DATA_ROOT` to the existing original dataset directory and `CAPE_ROOT` to the project workspace before source-dependent steps. The initializer creates empty directories; it does not move this package or copy your dataset. Place the source/config/docs/tests/notebooks under the workspace `repo/` after inspection, while keeping data and future run artifacts outside it.

The included synthetic validation passed **58 tests**. Candidate parameter count is **164,198**. These measurements do not establish clinical scores, source-data compatibility or DGX speed.
