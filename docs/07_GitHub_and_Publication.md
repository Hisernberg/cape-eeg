# GitHub automation and publication runbook

CAPE-EEG | Reproducibility without patient-data exposure | 7 September 2026

## 1. Publication objective

Upload the complete **reproducible research implementation**, not every local byte. A public repository should let an authorized researcher acquire the source independently, rebuild the cache, recover the split and repeat the experiment. It should not redistribute patient material or credentials merely because automation makes uploading easy.

No GitHub account, repository or token was used in preparing this package. Actual repository creation and upload remain `NOT_RUN`. The user will configure access locally in Codex. Existing connected accounts are not authorization to create a public repository.

## 2. What belongs in each location

| Artifact | Default location | Public status |
|---|---|---|
| Source code, tests, config schema, notebook source | repo/ | Eligible after scan |
| Markdown documentation and source register | repo/ | Eligible after scan |
| Synthetic fixtures with explicit provenance | tests/ | Eligible |
| Aggregate scores, confidence intervals, approved plots | private then export | Eligible after privacy review |
| Raw EEG, raw spectra, compact caches | source/private | Never auto-publish |
| train.csv, IDs, splits with IDs, per-row predictions | private | Never auto-publish |
| Individual traces, spectrograms, case explanations | private | Separate explicit rights/privacy review |
| Inference weights and model derivatives | private | Separate licensing/rights decision |
| Credentials, approval files, environment secrets | local credential store/private | Never publish |
| Executed notebooks, profiler traces, optimizer state | private | Not in public repo |

Hashed patient identifiers may remain linkable. They are not automatically public-safe. Prefer deterministic split code and a public manifest hash over individual split rows; restricted reviewers can reproduce the rows with authorized source access.

## 3. Repository shape

```text
cape-eeg/
  README.md
  AGENTS.md
  CODEX_EXECUTION.md
  CITATION.cff
  LICENSE                         # only after author selects an appropriate license
  DATA_ACCESS.md
  MODEL_CARD.md
  REPRODUCIBILITY.md
  pyproject.toml
  configs/
  src/cape_eeg/
  scripts/
  tests/
  notebooks/                      # output-free sources
  docs/                           # Markdown; avoid duplicate PDF histories
  refs/
  results/aggregate/              # approved, suppression-checked
  figures/                        # approved aggregate plots only
  .github/workflows/ci.yml
  .gitignore
```

Keep generated documentation PDFs in a tagged release or an explicitly reviewed docs directory, not repeatedly in Git history. Working repository target is under 25 MiB, excluding release assets. This project’s public-file cap is 10 MiB and export-snapshot cap 100 MiB. GitHub’s general Git-file ceiling is larger; it is not a reason to store datasets there. [S12]

## 4. Local credentials

Use a fine-grained GitHub credential with minimum access to the exact target repository. Prefer the authenticated local CLI/credential manager. HF access may be read-only for the single approved model. No credentials are required for CPU synthetic tests.

Never put token strings in Markdown, notebooks, YAML, shell history, remote URLs, source files or uploaded logs. Do not ask Codex to remember a token in its prompt. Do not call `printenv` or include unrestricted subprocess environments in telemetry. Provider secret scanning is supplementary, not preventive permission. [S13]

Network credentials are present only in setup/publish processes. Training and evaluation run without them. Disable credential inheritance when launching model jobs.

## 5. Build an allowlisted export

Create a fresh `public_export/` directory outside `repo/` and outside `private/`. Copy only approved relative paths from an explicit allowlist. Resolve paths, reject traversal, reject symlinks, reject hidden nested repositories and reject files above the cap. Do not follow a data symlink just because it sits inside an otherwise allowed folder.

Clear every code-cell output and execution count from notebook copies. Remove unnecessary metadata. Export only aggregate CSV/JSON whose schema has been approved; a suffix `.csv` is not sufficient. Do not pass an entire private result folder to `rsync`, `tar`, Git LFS or a release upload action.

For public aggregate tables, suppress cells with fewer than ten independent patients/components under a prespecified project policy. This is conservative data minimization, not a legal anonymization guarantee. Check complementary counts so suppressed values cannot be trivially reconstructed.

The supplied `export_precheck.py` implements a conservative filesystem/notebook/extension precheck and manifest. It is **not** a complete secret scanner, privacy review or approved publisher.

## 6. Two scans, two permission decisions

Scan A checks the full fresh export content using a mature secret scanner pinned to an approved version. Scanner errors, missing executable or unavailable rules fail closed. In addition to known credential patterns, inspect notebooks, Git remote configuration and archive contents.

Scan B checks every reachable Git object/history that will be pushed, not only the current working tree. A deleted token may remain in commits. Run the content scan again after staging to detect changes between export and commit. Keep scan logs private and expose only sanitized pass/fail evidence.

Decision 1 authorizes a push to the exact owner/repository/branch with the chosen visibility. Decision 2 separately authorizes making a repository public and/or releasing trained weights. Private repository approval does not authorize public release.

## 7. Immutable approval contract

```json
{
  "owner": "SET_LOCALLY",
  "repository": "cape-eeg",
  "branch": "main",
  "visibility": "private",
  "export_sha256": "SHA256_OF_APPROVED_EXPORT_MANIFEST",
  "content_scan": "PASS",
  "history_scan": "PASS",
  "data_rights_review": "PASS_FOR_THIS_EXPORT_ONLY",
  "public_visibility_approved": false,
  "weights_release_approved": false,
  "push_approved": false
}
```

This is a template, not an approval. Store the completed record in `private/approvals/`. Before push, independently recompute the export hash, verify repository destination and visibility, rerun scans, and verify the approval applies to this exact snapshot. A stale approval must block the push.

## 8. Automation state machine

```text
PREPARE_EXPORT
 -> NOTEBOOK_STRIP
 -> FILE_AND_PRIVACY_PRECHECK
 -> CONTENT_SECRET_SCAN
 -> REVIEW_EXPORT_MANIFEST
 -> DESTINATION_AND_VISIBILITY_APPROVAL
 -> COMMIT_APPROVED_FILES
 -> HISTORY_SECRET_SCAN
 -> RECHECK_HASH_AND_APPROVAL
 -> PUSH_DRY_RUN
 -> EXPLICITLY_AUTHORIZED_PUSH
 -> VERIFY_REMOTE_COMMIT
```

There is no `git add .` step, no force push, no automatic visibility change, no deleting an existing remote and no silent replacement of a user repository. A dry run is labeled a dry run, not a successful upload. Record the remote commit only after it is actually verified.

When Codex has access to an approved GitHub connector or authenticated CLI, use its documented operations and exact destination. Missing permissions or required fields mean `BLOCKED`; do not guess an account from an old profile. Repository setup remains separate from model execution.

## 9. Continuous integration

CI runs synthetic CPU tests, import checks, formatting, notebook-output checks, forbidden-file checks and documentation link validation. It never downloads HMS data, trains a model or uses the user’s private evaluation credentials.

Use least-privilege job permissions and pin third-party actions by audited commit before enabling workflows. Do not publish an unverified action hash as a trustworthy pin. The package includes a workflow blueprint only; enabling it requires local dependency/action verification.

Documentation rendering and release asset packaging occur in separate jobs only after approval. A CI badge means those listed tests passed, not that clinical performance or data rights are certified.

## 10. Release contents

Recommended first release: code, immutable configs, source citations, reconstruction instructions, environment digest, synthetic tests, sanitized aggregate tables, approved aggregate figures, protocol hash and report PDFs. Model weights are omitted initially.

The model card states research-only use, task and label semantics, source/partition provenance, missing external validation, ambiguity limitations, resource measurements, subgroup failures and forbidden clinical claims. The data card states source acquisition requirements, rights uncertainty, no dataset redistribution and exact cache reconstruction steps.

A manuscript reference score such as a published VIPEEGNet KL belongs in a clearly marked prior-work column, never in the measured experiment table. [S02]

## 11. Incident and rollback policy

If a secret is detected, stop publication and revoke/rotate it before remediation. Removing the current file is not sufficient if history contains it. Do not send leaked secrets into a public issue or bug report. Follow the provider’s documented incident procedure and retain only redacted local evidence. [S13]

If patient material is accidentally staged, unstage it and rebuild a fresh export. If it was pushed, stop, restrict access where authorized, document the exposure and obtain appropriate review; do not imply history rewriting guarantees all copies disappeared.

## 12. Completion definition

Publication is `PASS` only when authorized destination/visibility, exact export hash, two scan passes, rights review and verified remote commit all exist. Dataset upload and weight release have independent statuses. A local ZIP, a commit, a push dry run and a verified remote release are four different events.
