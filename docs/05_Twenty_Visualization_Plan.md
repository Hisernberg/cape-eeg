# Twenty-visualization analysis plan

CAPE-EEG | Figure specifications, not fabricated results | 7 September 2026

## Shared rendering and evidence contract

This document specifies exactly 20 figure families. Each is generated only after its required inputs exist. No score curves, patient spectrograms or performance figures are synthesized to make a report look complete. An unavailable plot receives a manifest entry with status `NOT_RUN` and a reason.

Each family can have explicitly named companion outputs, such as raw-count versus normalized confusion matrices. They remain one planned analytical family; do not pad the figure count with repeated styling.

Save editable SVG or vector PDF where possible, plus 300-dpi PNG for manuscript insertion. Every caption includes the partition, cohort/row/group counts, model/seed policy, metric definition, interval method, source manifest hash and relevant exclusions. Use readable type, accessible line/marker differences and fixed axes for direct comparisons. Do not truncate axes to exaggerate gains.

Figures V01–V08 may use approved development or intake data. Figures V09–V20 that depend on final outcomes are created only after the protocol lock. Real single-case examples remain private unless separately cleared. Public documentation uses clearly labeled synthetic examples, not fake patient data.

The rendering pipeline must verify image bounds, nonempty outputs, finite plotted values, consistent legend labels, source-hash matching and absence of identifiers. A figure is not publishable merely because matplotlib saved it successfully.

## Figure manifest schema

`figure_id, title, data_hashes, run_ids, split, n_rows, n_patients, n_components, generation_code_commit, parameters, output_paths, status, privacy_review, caption`

Each figure’s sidecar JSON retains that provenance. No visual makes a stronger claim than the evaluation plan permits. The 2026 literature motivates preprocessing/evidence/robustness audits, but this is a bespoke reduced suite, not a reproduction of those papers. [S04, S06, S07]

## V01 — Cohort and exclusion flow

**Question.** Who entered the analysis, and who was excluded?

**Required inputs.** `source_manifest, cohort_summary, exclusions, split_summary`.

**Construction.** Count rows, distinct patients and connected components at intake, eligibility, partitioning and final prediction. Show each exclusion reason separately.

**Axes and denominator.** Flow counts, not model accuracy. Counts must reconcile at every transition.

**Uncertainty.** No confidence interval for deterministic counts. Show actual audited counts only.

**Acceptance check.** Every eligible label appears exactly once in an endpoint node; no patient/component appears in two partitions.

**Privacy and export.** Aggregate-only public export after small-cell suppression.

**Output family.** `figures/v01_*` plus a provenance sidecar. Initial status: `NOT_RUN`.


## V02 — Partition independence matrix

**Question.** Are train, tune, calibration and test genuinely separated?

**Required inputs.** `split_manifest, recording_component_map, duplicate_audit`.

**Construction.** Build a 5x5 intersection matrix of patient, EEG and spectrogram membership; publish aggregate intersections rather than identifiers.

**Axes and denominator.** Partition on both axes; off-diagonal overlaps must be zero.

**Uncertainty.** Deterministic audit; no statistical interval.

**Acceptance check.** Any nonzero forbidden overlap blocks training; do not make the color scale hide a small overlap.

**Privacy and export.** Public counts only; individual membership tables stay private.

**Output family.** `figures/v02_*` plus a provenance sidecar. Initial status: `NOT_RUN`.


## V03 — Six-class label distribution

**Question.** How much evidence supports each label and partition?

**Required inputs.** `raw_votes, split_manifest`.

**Construction.** Plot summed vote probability mass per class and unique-majority counts as separate chart outputs, normalized by eligible rows.

**Axes and denominator.** Class on x-axis; prevalence or count on y-axis, explicitly labeled.

**Uncertainty.** Report rows and independent patient counts alongside each estimate.

**Acceptance check.** Class order is seizure, LPD, GPD, LRDA, GRDA, other; vote mass sums to row count.

**Privacy and export.** Public aggregates with suppressed low-count cells.

**Output family.** `figures/v03_*` plus a provenance sidecar. Initial status: `NOT_RUN`.


## V04 — Vote count and ambiguity distribution

**Question.** How are annotation effort and disagreement related?

**Required inputs.** `raw_votes, vote_sum, target_entropy, pairwise_disagreement`.

**Construction.** Use fixed vote-count bins and display target-entropy distributions. Keep n=1 visible as unavailable for pairwise disagreement.

**Axes and denominator.** Vote-count band on x-axis; entropy in natural-log units or explicitly normalized entropy on y-axis.

**Uncertainty.** Patient-cluster summaries; label number of contributing patients.

**Acceptance check.** No fabricated zero disagreement for n=1; bin boundaries must match the evaluation protocol.

**Privacy and export.** Aggregate public figure; no label IDs.

**Output family.** `figures/v04_*` plus a provenance sidecar. Initial status: `NOT_RUN`.


## V05 — Temporal alignment and foveated allocation

**Question.** Does the cache preserve the correct central ten seconds?

**Required inputs.** `private aligned raw window, supplied spectral timestamps, cache_time_edges`.

**Construction.** Use one prespecified synthetic ramp for public documentation and up to six private real examples. Mark raw [20,30) and spectral [s+295,s+305).

**Axes and denominator.** Seconds on physical axes; foveated bin widths displayed nonuniformly, not as equal-duration pixels.

**Uncertainty.** This is a data/geometry audit, not outcome evidence.

**Acceptance check.** Compare plotted coordinates against independently calculated offsets; local columns 8:24 are the foveated target.

**Privacy and export.** Public version is synthetic; real traces require separate rights review.

**Output family.** `figures/v05_*` plus a provenance sidecar. Initial status: `NOT_RUN`.


## V06 — Missingness and signal-quality audit

**Question.** Does data quality differ across partitions or dominate errors?

**Required inputs.** `validity_masks, valid_fractions, split, exclusions, test_patient_loss`.

**Construction.** Show observed missingness distribution by partition; a separate post-lock chart relates quality bands to KL.

**Axes and denominator.** Valid fraction or missing fraction on x-axis; count or KL on y-axis.

**Uncertainty.** Patient-cluster intervals for performance by quality band; deterministic counts for intake.

**Acceptance check.** Do not use final-test quality/outcome associations to change exclusion rules.

**Privacy and export.** Aggregate public; per-example quality data stays private.

**Output family.** `figures/v06_*` plus a provenance sidecar. Initial status: `NOT_RUN`.


## V07 — Byte-budget representation comparison

**Question.** What information allocation is affordable without hiding storage costs?

**Required inputs.** `cache_manifest, candidate_encoding_spec, development_metrics`.

**Construction.** Compare uniform/foveated encodings with identical payloads; show any higher-resolution pilot only if actually run.

**Axes and denominator.** Measured cache GB on x-axis; development patient-KL on y-axis.

**Uncertainty.** Seed spread when available; unrun alternatives labeled NOT_RUN rather than plotted as predicted scores.

**Acceptance check.** Include validity masks and metadata; do not compare compressed disk bytes with uncompressed RAM bytes.

**Privacy and export.** Public engineering aggregate; no cached tensors.

**Output family.** `figures/v07_*` plus a provenance sidecar. Initial status: `NOT_RUN`.


## V08 — Learning curves and completed training exposure

**Question.** Did models converge under a comparable schedule?

**Required inputs.** `events.jsonl, tune_patient_metrics, cumulative_examples, run_status`.

**Construction.** Plot train loss and tune KL with clearly separate labels; report actual examples seen and elapsed GPU time.

**Axes and denominator.** Examples or elapsed minutes on x-axis; named loss on y-axis.

**Uncertainty.** Multiple seeds shown only where actually run; no made-up error band.

**Acceptance check.** Distinguish cross-entropy from KL; interrupted fits marked INCOMPLETE and never smoothed into completion.

**Privacy and export.** Public sanitized curves without paths or sample IDs.

**Output family.** `figures/v08_*` plus a provenance sidecar. Initial status: `NOT_RUN`.


## V09 — Primary paired patient-KL improvement

**Question.** Is the proposed method better for the same held-out patients?

**Required inputs.** `test_predictions for B/P and all three seeds, patient/component map`.

**Construction.** Aggregate KL per patient per seed, average across seeds, and display paired differences with overall paired cluster interval.

**Axes and denominator.** Delta-KL=P−B; negative favors P. Include a zero reference line.

**Uncertainty.** Prespecified 2,000 paired component/patient bootstrap resamples.

**Acceptance check.** Never select the best seed or drop patients only for one model; rows and keys must align exactly.

**Privacy and export.** Public distribution/summary only; private per-patient values are not uploaded.

**Output family.** `figures/v09_*` plus a provenance sidecar. Initial status: `NOT_RUN`.


## V10 — One-vs-rest ROC curves

**Question.** Which patterns are discriminable under unique-majority labels?

**Required inputs.** `test_probabilities, unique_majority_mask, labels, groups`.

**Construction.** Generate separate class ROC outputs and a declared macro summary; distinguish calibrated and uncalibrated variants.

**Axes and denominator.** False-positive rate on x-axis; true-positive rate on y-axis.

**Uncertainty.** Cluster-bootstrap AUROC intervals; label positives, negatives and independent groups.

**Acceptance check.** Tied maxima excluded only from these hard-label plots; absent classes reported NOT_ESTIMABLE.

**Privacy and export.** Public aggregate curves; no point-level identifiers.

**Output family.** `figures/v10_*` plus a provenance sidecar. Initial status: `NOT_RUN`.


## V11 — One-vs-rest precision-recall curves

**Question.** Does minority-class discrimination survive prevalence effects?

**Required inputs.** `same locked inputs as V10`.

**Construction.** Generate class-specific average-precision curves; include each class prevalence as a clearly identified reference.

**Axes and denominator.** Recall on x-axis; precision on y-axis.

**Uncertainty.** Patient/component bootstrap; specify average precision versus trapezoidal PR area.

**Acceptance check.** Do not use oversampled test data; all curves use the actual test class mix.

**Privacy and export.** Public aggregate.

**Output family.** `figures/v11_*` plus a provenance sidecar. Initial status: `NOT_RUN`.


## V12 — Confusion matrix with support

**Question.** Which harmful-activity patterns are confused?

**Required inputs.** `unique_majority_labels, argmax_predictions, exclusion_counts`.

**Construction.** Produce raw-count and row-normalized matrices as separately named outputs; display excluded tie count in caption.

**Axes and denominator.** True class on rows, predicted class on columns; fixed six-class order.

**Uncertainty.** Counts are descriptive; class-recall intervals accompany the table, not inferred from colors.

**Acceptance check.** Cells sum to the unique-majority evaluated row count; report patient-level support as well.

**Privacy and export.** Public aggregate with small-cell suppression where required.

**Output family.** `figures/v12_*` plus a provenance sidecar. Initial status: `NOT_RUN`.


## V13 — Calibration reliability

**Question.** Are stated probabilities consistent with expert vote frequencies?

**Required inputs.** `raw and temperature-scaled predictions, q, calibration_manifest`.

**Construction.** Plot confidence versus mean vote mass in predicted class using ten equal-count bins and the minimum-group merge rule.

**Axes and denominator.** Mean confidence on x-axis; observed soft agreement on y-axis; identity line.

**Uncertainty.** Patient-cluster bands and independent-group counts; proper scoring-rule results in caption.

**Acceptance check.** Temperature must come from calibration-T only; do not fit calibration to the plotted test points.

**Privacy and export.** Public binned aggregate only.

**Output family.** `figures/v13_*` plus a provenance sidecar. Initial status: `NOT_RUN`.


## V14 — Predicted versus observed disagreement

**Question.** Does the auxiliary head learn annotation ambiguity?

**Required inputs.** `d_hat, raw_votes, n, fixed vote strata`.

**Construction.** Use binned predicted disagreement against observed pairwise disagreement, separately for supported vote-count bands.

**Axes and denominator.** Predicted disagreement on x-axis; observed pairwise disagreement on y-axis.

**Uncertainty.** Cluster intervals and vote-count strata; n=1 is missing, not zero.

**Acceptance check.** Observed values may reach 1; do not force a 5/6 cap or claim a Bayesian uncertainty decomposition.

**Privacy and export.** Public binned aggregates; raw expert vote rows remain private.

**Output family.** `figures/v14_*` plus a provenance sidecar. Initial status: `NOT_RUN`.


## V15 — Review risk-coverage curves

**Question.** Can the model prioritize difficult cases without concealing referral volume?

**Required inputs.** `locked referral scores and thresholds, KL, majority errors, groups`.

**Construction.** Compare the prespecified input-only referral policies over acceptance coverage 0.5–1.0; mark fixed calibration-P operating points.

**Axes and denominator.** Actual acceptance coverage on x-axis; accepted-case KL or error on y-axis.

**Uncertainty.** Paired patient-cluster intervals; report actual number referred and whole-cohort risk.

**Acceptance check.** No threshold selected on test; comparisons use common coverage and include the 100% point.

**Privacy and export.** Public aggregate policy curves.

**Output family.** `figures/v15_*` plus a provenance sidecar. Initial status: `NOT_RUN`.


## V16 — Subgroup performance forest

**Question.** Are gains consistent across class, annotation support and missingness?

**Required inputs.** `test losses, fixed vote/quality strata, label-support counts`.

**Construction.** Plot baseline/candidate difference by prespecified subgroup, with one overall result and no new subgroup mining.

**Axes and denominator.** Delta-KL on x-axis; subgroup name on y-axis.

**Uncertainty.** Cluster intervals marked exploratory; low-support groups clearly identified.

**Acceptance check.** Do not invent age, sex, site or disease-stage metadata absent from the dataset.

**Privacy and export.** Public aggregate with suppression and complementary-count check.

**Output family.** `figures/v16_*` plus a provenance sidecar. Initial status: `NOT_RUN`.


## V17 — Preprocessing and corruption sensitivity

**Question.** Does performance remain stable under the fixed stress suite?

**Required inputs.** `clean/perturbed matched-subset predictions, perturbation_manifest`.

**Construction.** Plot change in KL and prediction JS versus declared severity; clean comparison uses precisely the same rows.

**Axes and denominator.** Perturbation severity in stated units on x-axis; Delta-KL/JS on y-axis.

**Uncertainty.** Paired cluster intervals; subset size and patient count stated.

**Acceptance check.** Six prespecified conditions only; artificial masking is not called validated clinical artifact simulation.

**Privacy and export.** Public aggregate; raw perturbed examples stay private.

**Output family.** `figures/v17_*` plus a provenance sidecar. Initial status: `NOT_RUN`.


## V18 — Evidence deletion and region dependence

**Question.** Does claimed important evidence actually affect the prediction?

**Required inputs.** `seed101 occlusion predictions, time/region masks, selected-case manifest`.

**Construction.** Compare ranked deletion against random deletion; aggregate local/context/region removal effects. Private maps show physical time coordinates.

**Axes and denominator.** Removed-evidence fraction on x-axis; score/loss change on y-axis.

**Uncertainty.** Case/patient resampling for aggregate exploratory results; case count explicitly limited.

**Acceptance check.** No saliency-only causal claim; no cherry-picked success examples; same masked area for comparisons.

**Privacy and export.** Aggregate public; individual case maps require separate approval.

**Output family.** `figures/v18_*` plus a provenance sidecar. Initial status: `NOT_RUN`.


## V19 — Mechanism-ablation effect plot

**Question.** Which change contributes beyond extra parameters or favorable preprocessing?

**Required inputs.** `B2,A1,A2,P development results; frozen final comparison where run`.

**Construction.** Plot incremental changes at fixed bytes and exposures: foveation, gate, auxiliary loss. Mark which comparisons are development-only.

**Axes and denominator.** Intervention on y-axis; paired Delta-KL on x-axis.

**Uncertainty.** Matched seeds/groups where available; a single-seed pilot has no invented seed interval.

**Acceptance check.** Do not label pilot effects confirmatory or silently omit an ablation that made results worse.

**Privacy and export.** Public sanitized aggregates and exact configs.

**Output family.** `figures/v19_*` plus a provenance sidecar. Initial status: `NOT_RUN`.


## V20 — Accuracy–resource Pareto and budget ledger

**Question.** Is predictive improvement worth the real DGX cost?

**Required inputs.** `measured KL, latency, parameters, CUDA/RSS, GPU ledger, cache manifest`.

**Construction.** Plot patient-KL versus measured end-to-end latency; separate table/companion outputs show memory, storage and cumulative GPU hours.

**Axes and denominator.** Measured latency in ms on x-axis; KL on y-axis; hardware and batch fixed.

**Uncertainty.** Timing medians/p95 and bootstrap KL intervals; measurement boundary stated.

**Acceptance check.** Separate cached inference from preprocessing; include failures in total compute; do not fabricate energy measurements.

**Privacy and export.** Public aggregate resource data with private paths redacted.

**Output family.** `figures/v20_*` plus a provenance sidecar. Initial status: `NOT_RUN`.
