# Evaluation, advanced metrics and statistics

CAPE-EEG | Prespecified analysis plan | 7 September 2026

> No medical-data scores are provided. Numeric values in this document are definitions, fixed analysis settings or labeled planning examples, not experimental results.

## 1. Estimands before metrics

The prediction unit is one labeled 10-second center, observed through a 50-second EEG and a 600-second spectral context. The independence unit is the patient, or a connected component if data-integrity auditing links several patient IDs. Repeated and overlapping rows do not create new independent patients. [S01]

The primary scientific estimand is the average patient’s expert-vote prediction loss, averaged over three fixed single-model training seeds. A secondary benchmark-style estimand gives each label row equal weight. Report both, because they answer different questions when patients contribute different numbers of windows.

The six classes are seizure, LPD, GPD, LRDA, GRDA and other. The primary outcome uses the full soft target q. Hard-label discrimination and accuracy are complementary, not substitutes for agreement with multiple expert votes.

## 2. Locked partitions and roles

| Partition | Initial share | Allowed use |
|---|---:|---|
| Train | 60% | Fit development models and normalization |
| Tune | 10% | Select comparator, epochs and bounded development choices |
| Calibration-T | 5% | Fit scalar temperature only after final network weights freeze |
| Calibration-P | 5% | Freeze review-score thresholds and specified operating points |
| Locked test | 20% | One final, read-only evaluation of frozen methods |

Final network training uses train+tune, approximately 70%, with the selected fixed epoch count. Neither calibration partition is added back into training. The test split, eligibility rules, metrics, hypothesis direction and reporting templates are signed by hash before any test predictions are examined.

The development role cannot read test labels or test-derived figures. The evaluator receives locked predictions and labels only after protocol freeze. Automation must fail if a development stage asks for the test partition. Data-quality intake may count source labels to construct the split, but it may not expose test outcomes to tuning.

## 3. Primary metric: KL divergence

For row i with positive total vote count n_i, q_ik=v_ik/n_i. Predictions are six finite nonnegative numbers summing to one. Clip p below at 1e-7 for numerical scoring and renormalize once; record this convention in the score manifest.

`KL_i = sum_{k:q_ik>0} q_ik * [log(q_ik) - log(p_ik)]`

Use natural logarithms. Terms with q=0 contribute exactly zero. Do not smooth q merely to avoid log(0). Loss is lower-is-better. Unit tests cover exact agreement, uniform prediction, zero-vote classes, invalid probability sums and severe misclassification.

For patient a, let I_a be its eligible rows:

`K_a,m,s = mean_{i in I_a} KL(q_i, p_i,m,s)`

For method m and seeds 101/202/303:

`K_m = mean_a mean_s K_a,m,s`

The primary paired difference is `Delta = K_P - K_B`. A negative value favors the proposed candidate. **Average losses across seeds, not probabilities**, unless reporting a separately labeled ensemble that is outside the core deployment plan.

Report the row-weighted KL as `mean_i KL_i` and each seed’s values separately. A comparator trained on different patients, a filtered high-vote subset, or different test rows is not a matched primary comparison.

## 4. Basic metrics

| Metric | Target / scope | Required caveat |
|---|---|---|
| Patient-mean KL | All eligible soft-label rows | Primary metric; lower is better |
| Row-mean KL | All eligible rows | Benchmark-like; overlapping rows correlated |
| Soft cross-entropy | Full q | Different numeric scale from KL |
| Soft-target squared error | sum_k(p_k−q_k)^2 | Label-distribution fidelity, not conventional hard Brier |
| Expected categorical Brier | 1−2 sum_k p_k q_k + sum_k p_k^2 | Expected score against an expert-drawn categorical label |
| Macro AUROC, one-vs-rest | Unique majority-label rows | No score for a class with no positive/negative cases |
| Macro AUPRC / average precision | Unique majority-label rows | Report prevalence and aggregation convention |
| Balanced accuracy | Unique majority-label rows | Mean class recall |
| Macro F1 | Unique majority-label rows | Undefined class cases explicitly marked |
| Accuracy, MCC, Cohen kappa | Unique majority-label rows | Descriptive; do not headline ambiguous-label accuracy |
| Sensitivity, specificity, PPV, NPV | Fixed per-class thresholds | Thresholds cannot be chosen on test |
| Confusion matrix | Unique majority-label rows | Raw counts and row-normalized versions |

Define a hard label only where exactly one class has the largest vote count. Tied maxima are retained for all soft metrics and separately counted, but excluded from unique-majority hard-label metrics. Never break ties silently by class-column order.

The source provides `expert_consensus` for convenience. Audit it against vote maxima; discrepancies are reported rather than silently replaced. Sensitivity analyses may use the provided consensus as a distinct target with a clearly labeled denominator. [S01]

## 5. Calibration metrics

Scalar temperature applies to final mixture probabilities using `p_T = softmax(log(p)/T)`, with T constrained to [0.25,4]. Fit T on calibration-T by patient-weighted soft cross-entropy. The chosen metric, bounds and patient weighting are fixed; report when the optimum hits a bound.

Report KL and both Brier conventions before and after temperature scaling. No isotonic regression, classwise temperature search or repeated calibration-method selection is allowed in the core small-calibration-set design.

Soft top-label ECE compares binned predicted confidence with the mean vote mass assigned to the predicted class. Use ten equal-count bins on test predictions, with binning independent of test outcomes. If a bin has fewer than 20 contributing independent groups, merge neighboring bins and report the rule. The quantity estimates consistency with expert-vote frequencies, not necessarily clinical truth.

Also report majority-label ECE on unique-majority rows and classwise reliability curves. Calibration errors depend on binning and sample size, so proper scoring rules remain the main calibration evidence. Do not claim that low ECE alone establishes safety.

## 6. Disagreement and review-prioritization metrics

Observed disagreement is `d=1−sum v_k(v_k−1)/(n(n−1))` for n>1. Record n=1 cases as `NOT_ESTIMABLE` for observed disagreement while retaining their classification scores. Report MSE and rank correlation of predicted d_hat with observed d, stratified by vote count.

Prespecified reference strata are vote count 1, 2–4, 5–9 and at least 10; target-entropy strata use development-defined quantiles frozen before test. These are engineering analysis bins, not clinical categories. The source’s qualitative idealized/proto/edge terminology is not converted into an exact new label without a validated rule. [S01]

Compare input-only review scores: predictive entropy, 1−maximum probability, and the auxiliary d_hat. Select one referral score using tune data; freeze its 80%, 90% and 95% acceptance-coverage thresholds using calibration-P without test labels.

Plot risk versus coverage using both KL and majority-label error. The primary operating point is 90% planned acceptance coverage; test coverage may differ. Report the actual number and fraction referred, accepted-case risk, and full-cohort risk together. A low risk after rejecting most difficult cases is not improved overall classification.

Area under the risk-coverage curve is descriptive, with the integration range [0.5,1.0] fixed to avoid unstable near-zero coverage. Do not compare two policies at different actual coverage without interpolation on the common prespecified grid.

## 7. Advanced robustness metrics

Predefine six conditions, applied without modifying reference labels: clean data; one regional input missing; 10% additional contiguous time-mask coverage; mild log-power gain shift; stronger gain shift; and a modest change in raw STFT window length. Gain shifts and artificial masking are representation-level perturbations, not verified physiological artifact simulators.

For raw-domain STFT sensitivity, recompute a **fixed label-blind subset of at most 1,024 test rows from at least 100 patients**, with seed and selection locked in advance. The clean control uses exactly the same subset. Each changed preprocessing condition has explicit units and parameters. Do not cherry-pick a severity after seeing its score.

Report Delta-KL relative to the clean matched input, worst-condition KL, label-flip fraction, Jensen–Shannon distance between clean/perturbed predictions, and change in referral rate. Confidence intervals respect patient/component clustering. Signal corruption may truly remove diagnostic evidence; robustness is not guaranteed label invariance.

The 2026 preprocessing and robustness papers motivate these tests, but do not validate this reduced suite or convert it into external-site validation. [S04, S07]

## 8. Evidence-use audits

For the fixed seed-101 model, compare intact predictions with the local view removed, the context view removed and individual regions removed. Report changes in KL and predictive distribution, not only an attractive saliency image.

For at most 48 private, prespecified cases, compute regional/time-bin occlusion and deletion curves. Case selection is balanced by development-defined class and ambiguity rules, not by visually impressive success cases. The target-center region and its temporal coordinates must be displayed accurately.

Attention/gate values are correlational diagnostics. A claimed explanatory importance should predict the effect of deletion better than a random deletion ordering. Even successful deletion tests do not establish a biological cause. This limited audit is inspired by evidence-focused EEG research, not a reproduction of its full model. [S06]

## 9. Computational metrics

Record trainable and total parameters, actual training wall time, total GPU-wall time including failures, batch size, peak allocated and reserved CUDA memory, process-tree RSS, cache bytes and end-to-end storage footprint.

Measure inference at batch 1 and 32 after 20 warmup batches and across at least 100 timed batches. Synchronize CUDA before and after measurement. Report median and p95 latency, examples/second, dtype and hardware. Separate cached-tensor inference from full raw-to-prediction latency.

Energy is reported only if a credible power sensor is available and the integration boundary is stated. Missing GPU power readings on a unified-memory system mean `NOT_MEASURED`, not an invented watt-hour estimate. FLOPs/MACs are supporting descriptors, not substitutes for measured DGX latency.

## 10. Primary inference and multiplicity

Bootstrap 2,000 resamples of independent patients/components using paired predictions. Compute the primary difference for the same rows and three seeds in each replicate. Pre-aggregate patient-level per-seed losses for CPU efficiency. Use percentile 2.5% and 97.5% limits and document the random seed.

The primary decision is superiority only: the interval for Delta must be wholly below zero. An interval crossing zero is inconclusive. Equivalence/noninferiority is not claimed unless a meaningful margin and power analysis are separately established before final testing.

Only one comparison is confirmatory. Ablations, subgroup analyses, individual-class comparisons and robustness conditions are secondary. Where p-values are provided for a family of secondary comparisons, use Holm adjustment; otherwise label their intervals exploratory. Do not interpret twenty plots as twenty independent confirmations.

Seed variation is reported separately from patient sampling uncertainty. The bootstrap conditions on these fitted models and does not account for all possible training datasets, sites or model-selection uncertainty. This limitation is explicit in the manuscript.

## 11. Sample-size feasibility

Before training, count independent groups and per-class positive/negative patients in every partition. A large number of overlapping rows does not resolve a small number of positive patients. If a class has fewer than 30 positive or 30 negative independent groups in the final test, mark its class-specific interval as low-support and do not use it for a strong clinical claim.

For calibration-P operating-point selection, require at least 30 relevant positive and 30 negative groups; otherwise omit the class-specific threshold claim. Coverage-based review thresholds can still be specified without claiming class sensitivity guarantees.

The paired-mean planning approximation is `MDE ≈ (1.96+0.84)*s_delta/sqrt(N_groups)`. At N=390 and paired SD 0.10/0.20/0.30 it is 0.014/0.028/0.043 KL. These values are hypothetical sensitivity calculations. Replace SD with a development estimate and examine heavy tails using bootstrap or simulation.

Do not set an unrealistically narrow success margin after looking at the expected cohort size. If an effect cannot be resolved, report the interval and limitation; absence of significance is not absence of a useful effect.

## 12. Hard limits on claims

No Alzheimer/rare-disease diagnosis, no seizure-onset latency, no false alarms per hour, no seizure-event sensitivity, no clinical decision-curve net benefit, no guaranteed conformal coverage and no hospital/site generalization claim are supported by this core experiment alone. Continuous timelines, action utilities, proper independent calibration or external cohorts would be needed for those respective analyses.

Conformal prediction is intentionally not included as a headline advanced metric: multiple correlated windows per patient and limited calibration groups would require a carefully specified patient-level target. Adding a named method without its assumptions would weaken the study.

## 13. Required final tables

Table 1: source/cohort/partition counts and exclusions. Table 2: all baseline and proposed metrics with confidence intervals and seed spread. Table 3: byte-/exposure-matched mechanism ablations. Table 4: uncertainty/referral and corruption results. Table 5: actual resource accounting.

Each metric record includes `run_id`, `method_id`, `seed`, `split_hash`, `preprocess_hash`, `protocol_hash`, `prediction_hash`, `metric`, `value`, `ci_low`, `ci_high`, `n_rows`, `n_patients`, `n_components`, `status` and `reason`. Unrun values are null, never 0.000 or a target score.

## 14. Final-test access gate

Before evaluating, verify all final fits finished the fixed schedule, all probability rows are finite and normalized, patient/recording overlap is zero, calibration objects are frozen, exclusion rules match intake, and every key aligns one-to-one across methods. Save a signed local protocol manifest and a test-access log.

A bug found after unblinding requires a documented correction and rerun history. Do not quietly retune and call the same set untouched. If the implementation bug affects conclusions, downgrade the original analysis and describe the repair transparently.
