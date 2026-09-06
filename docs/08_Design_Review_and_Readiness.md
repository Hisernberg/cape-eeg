# Design-review loop and readiness gates

CAPE-EEG | Internal adversarial design review | 7 September 2026

## 1. What this review is

The three rounds below are structured self-review of the proposed design. They are not independent peer review, clinical validation, GPU experiments or evidence of improved model accuracy. Ratings are subjective planning scores with explicit weaknesses; no score is used as a substitute for measured results.

## 2. Round A — resolve the task and resource mismatch

Initial risk: the request mentions Alzheimer’s/rare disease, a 5–6 GB HF dataset and then specifies a different EEG competition. Treating these as interchangeable would produce an incoherent study.

Change: the named HMS dataset takes precedence. The task is six-class expert-vote EEG-pattern prediction. No Alzheimer, healthy-control or rare-disease labels are invented. The 5–6 GB requirement is explicitly a derived-cache target, while original source size and rights remain separately measured/reviewed.

Initial risk: compacting a 600-second image uniformly can blur the short target interval.

Change: use both raw EEG and provided spectra; allocate half of local temporal bins to the center ten seconds, and test a uniform encoding at exactly the same payload. Keep source timestamps and both offset conventions explicit.

Round A planning rating: **6.3/10**. Task alignment improves, but model novelty, statistical feasibility and the automated publication boundary still need work.

## 3. Round B — challenge the novelty and statistical claims

Initial risk: a new model name plus a small pretrained CNN and many metrics can look novel without isolating a contribution.

Change: define three incremental interventions: temporal foveation, learned local/context mixture, and an annotation-disagreement auxiliary target. Require matched-byte/exposure ablations and a stronger small pretrained control. Name the 2025 EEG precedent and identify 2026 sources as inspirations rather than evidence of success.

Initial risk: narrow fixed AUROC/noninferiority thresholds can be impossible to resolve, especially with correlated windows and small patient groups.

Change: use paired patient-mean KL as the sole confirmatory endpoint, with no invented clinical margin. Include development-based variability/MDE assessment, confidence intervals, a clear inconclusive outcome and explicit low-support class labels. Do not count 100,000 overlapping windows as 100,000 independent patients.

Initial risk: observed vote entropy depends on vote count, and low-vote labels can be wrongly declared certain.

Change: retain every positive-vote target for soft-label learning, use a defined pairwise disagreement statistic only for n>1, and separate it from Bayesian/clinical uncertainty claims.

Round B planning rating: **7.7/10**. The hypothesis is now testable, but local execution feasibility and model performance remain unmeasured.

## 4. Round C — challenge leakage, automation and budget credibility

Initial risk: tuning, calibration and final testing are mixed, and a notebook can accidentally use test outcomes to refine plots or thresholds.

Change: freeze train/tune/calibration-T/calibration-P/test roles, refit only on train+tune, isolate evaluator access, fix seed101 for deployment and average single-model losses across three seeds rather than selecting a winner on test.

Initial risk: “upload everything” could expose source recordings, patient IDs, tokens, notebook outputs or deleted secrets in Git history.

Change: separate repo/private/export directories; restrict paths and file types; reject symlinks; strip notebook outputs; perform content and history scans; bind approval to exact export hash/destination/visibility. Dataset and weight release are separately gated.

Initial risk: a large list of experiments quietly exceeds the low-resource requirement.

Change: allocate a hard 12 GPU-wall-hour ledger with only one GPU process, a fixed six-fit final comparison, at most eight development configurations, bounded masks/occlusion tests, CPU caching and bootstrap, and no large foundation-model teacher or ensemble.

Round C planning rating: **8.3/10**. This is a defensible bounded study design, not a proven top-tier research result.

## 5. Final rubric

| Dimension | Internal rating / 5 | Remaining weakness |
|---|---:|---|
| Clinical/task fidelity | 4.5 | Segment labels cannot establish event-level clinical utility |
| Novelty and mechanism isolation | 3.5 | Foveation/fusion/disagreement components have substantial prior art |
| Statistical validity | 4.3 | Actual patient/class counts and effect size are unknown |
| Resource discipline | 4.5 | DGX throughput and original-source disk size are unmeasured |
| Reproducibility/automation design | 4.5 | Full trainer/cache/evaluator still require implementation |
| Publication safeguards | 4.5 | Source terms and exact release destination remain locally unresolved |

These ratings are not averaged into a journal-acceptance estimate. The main scientific uncertainty is whether the compact representation improves useful prediction after fair controls, not whether enough diagrams can be produced.

## 6. Current evidence state

| Item | Status at package creation |
|---|---|
| User’s task/schema and overview reviewed | PASS_DESCRIPTION_ONLY |
| Relevant 2026 primary sources checked | PASS_SOURCE_REVIEW |
| Local HMS CSV/Parquet inspected | NOT_RUN |
| Suitable HF dataset mirror verified | NOT_VERIFIED |
| Source license/redistribution review | PENDING_LOCAL_REVIEW |
| Cache arithmetic | VERIFIED_ARITHMETIC; local byte count NOT_RUN |
| Starter code synthetic tests | See evidence/local_validation.json |
| Real DGX environment/kernel smoke | NOT_RUN |
| Real preprocessing/split leakage gate | NOT_RUN |
| Trained baseline/proposed models | NOT_RUN |
| Medical metrics / hypothesis test | NOT_RUN |
| GitHub creation / push / public release | NOT_RUN |

## 7. Hard go/no-go gates

G0 requires a valid local environment and explicit path/rights record. G1 requires coherent source metadata and exact time alignment. G2 requires independent split components, valid cache and train-only normalization. G3 requires finite CPU/GPU smoke and a feasible time forecast. G4 requires a frozen confirmatory protocol. G5 requires complete fits and locked calibration. G6 requires valid matched test predictions and honest uncertainty. G7 requires a screened, explicitly approved release.

Any failure stops only the affected phase and writes a reason. No automatic changes to clinical labels, test membership, resource caps or privacy settings are permitted to make a gate pass.

## 8. Negative-result and fallback paths

If the foveated representation loses to uniform encoding, retain that finding and reject the representation claim. If the learned mixture collapses to one view, inspect why and report the cheaper view-only alternative without claiming multimodal complementarity. If the auxiliary task worsens KL, report it as a failed hypothesis; do not publish only its favorable correlation plot.

If MobileNetV3-Small outperforms the compact candidate, the honest output is a strong low-resource baseline and negative mechanism study, unless a genuine measured efficiency trade-off justifies a narrower contribution. No fixed noninferiority margin is retrofitted after seeing the difference.

If source alignment cannot be verified, training does not begin. If six final fits do not fit the budget, report a pilot-stage experiment rather than an incomplete confirmatory comparison. If external validation is unavailable, narrow the claim to held-out patients within this dataset.

## 9. Publication bar

A strong manuscript should have: a reproducible cohort, a fair small-model baseline, a resolved primary effect with uncertainty, clean mechanism controls, a calibration/ambiguity analysis, an honest failure audit, measured resource accounting and restrained clinical conclusions. Independent external validation would materially strengthen it, but no substitute dataset is added without compatible task labels and rights.

Do not describe this plan as guaranteed Q1/Q2 work, a record-setting score, clinically deployable or state of the art. The package is designed to discover whether a substantive contribution exists, efficiently and reproducibly.
