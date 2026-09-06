# CAPE-EEG | Research master plan

**Byte-Budgeted Foveated Spectrogram Learning with Expert-Disagreement Audits**

Version 1.0 | 7 September 2026 | DGX Spark / GB10 | Research only

> Status: finalized experimental specification and starter utilities. HMS files, model training, DGX measurements, final test evaluation, and GitHub publication have NOT been run in this environment. Every proposed score remains `NOT_RUN`.

## 1. Final decision

Use the HMS Harmful Brain Activity Classification dataset supplied in the request. Study six-way segment-level probability estimation for seizure, LPD, GPD, LRDA, GRDA, and other. The target is the expert-vote distribution, not merely its majority class. This is a neurocritical-care EEG research project, not Alzheimer’s diagnosis, a rare-disease screening claim, or an event-onset detector. The source explicitly describes differing expert agreement and KL evaluation. [S01]

The central question is: **At a fixed cache byte budget and bounded single-device training budget, does preserving the labeled temporal center and learning how to combine local EEG-derived spectra with longer spectral context improve prediction of expert votes on unseen patients?**

CAPE-EEG is a working project name. Novelty is a falsifiable combination and evaluation protocol, not a verified claim of being the first method with local/global fusion or disagreement modeling.

## 2. Scope and constraints resolved

| Requirement | Final decision |
|---|---|
| Dataset | Your original local HMS files; no unverified mirror is substituted |
| Approximately 5–6 GB | Target the **derived training cache**, including validity masks, not the unknown original-download size |
| Modalities | Both supplied EEG waveforms and supplied long-context spectrograms are used |
| Small models | Custom compact convolutional candidate; one small ImageNet-pretrained comparator |
| Recent vision work | Borrow a lightweight-fusion design principle, not a large 2026 foundation model |
| Compute | One job at a time; 12 GPU-wall-hour hard study ceiling after local profiling |
| Research claim | Patient-held-out distributional performance, calibration, robustness and resource use |
| Publication | Code, configuration, sanitized aggregates and documentation; no raw patient material or credentials by default |
| Execution | Seven notebook specifications, bounded Codex phases, idempotent local artifacts |

No guaranteed accuracy, AUROC, KL, journal quartile, or acceptance probability is assigned. A polished repository is not evidence that the scientific hypothesis worked.

## 3. What recent research contributes

DPNeXt (17 July 2026) uses economical multiscale fusion in image dense prediction. We adopt only the general principle of inexpensive fusion; its task and backbone differ from EEG. [S03]

Same Brain, Different Prediction (8 May 2026) studies prediction changes induced by preprocessing. This motivates a small, declared preprocessing-sensitivity suite rather than choosing the most favorable preprocessing after examining the test set. [S04]

NeuroAtlas (14 May 2026) compares EEG and general time-series foundation models across diverse tasks. Its findings motivate small task-matched controls rather than assuming a foundation model will be superior. [S05]

The auditable EEG evidence-bottleneck preprint (25 August 2026) motivates deletion and view-drop checks on evidence use. CAPE-EEG does not reproduce its product-of-experts model. [S06]

RobustSeiz (3 September 2026) motivates controlled perturbation reporting. Here that means segment-level stress tests; onset delay and false alarms per hour cannot be inferred from the supplied label structure. [S07]

VIPEEGNet is an important **prior baseline**, not new novelty: the 2025 preprint reports vision-inspired harmful-activity classification and KL results on its evaluation cohorts. Those published scores cannot be directly compared with our new patient split. [S02]

## 4. Proposed scientific contributions

**C1 — Byte-matched temporal foveation.** Allocate half of the 32 local temporal bins to the labeled 10-second center, and the other half to the remaining 40 seconds. Compare against uniform 32-bin local compression with the same number of values, same patients, model trunk, training exposure and seeds.

**C2 — Cheap, inspectable modality fusion.** Separate small input stems handle raw-derived and supplied spectrogram distributions; a shared convolutional trunk feeds local and contextual probability heads. A small gate learns their mixture without patient identifiers, annotation counts or labels as input. Test it against a fixed 50:50 mixture.

**C3 — Annotation-aware uncertainty auditing.** Predict an observed pairwise expert-disagreement statistic using a training-only auxiliary target, while preserving unweighted soft-label classification. Assess whether the resulting input-only score helps prioritize review of difficult segments. This is not a calibrated Bayesian decomposition of clinical uncertainty.

**C4 — Reproducibility at a fixed resource budget.** Report all eligible labels, independent patient counts, cache bytes, actual runtime, memory, seed spread, and a frozen final-test protocol. Engineering quality supports the science but is not itself a medical advance.

## 5. End-to-end experiment

```text
Read-only original HMS directory
    -> schema, rights, offset and file-integrity audits
    -> immutable patient/recording connected-component split
    -> local 50 s EEG -> bipolar regions -> power spectra -> foveated 32 bins
    -> provided 600 s spectrogram -> regional context -> 64 bins
    -> 64 frequency bins/view + float16 values + bit-packed validity masks
    -> small baselines and bounded mechanism pilots on development data
    -> freeze method, comparator, epochs, metrics and corruption settings
    -> refit baseline and candidate on train+tune, three fixed seeds
    -> temperature on calibration-T; referral policy on calibration-P
    -> one locked test run; patient-cluster uncertainty; 20 figure recipes
    -> private evidence archive -> screened reproducibility release
```

The source labels the middle 10 seconds of a 50-second EEG window. Its long-context spectrogram covers 10 minutes. Time alignment is specified exactly in Document 02; treating both offsets as the same clock is prohibited. [S01]

## 6. Architecture contract

Input L is `float32[B,4,64,32]` after loading float16 cache data. Input C is `float32[B,4,64,64]`. The four channels retain left lateral, right lateral, left parasagittal and right parasagittal order. Validity is retained separately; two valid-fraction features may enter the gate.

Each view has a 3x3 4-to-24-channel stem. The shared trunk uses depthwise-separable blocks with channels 24→48→64→96→128 and two 128-channel residual blocks. Frequency/time strides are `(2,1), (2,2), (2,2), (1,1), (1,1), (1,1)`. Group normalization avoids dependence on tiny-batch running statistics.

Local pooling combines global mean, global maximum, and center-region mean; context pooling combines global mean and maximum. Each is projected to 128 dimensions. Independent linear heads output six logits. With local and context probabilities pL and pC:

`g = sigmoid(MLP([uL, uC, valid_L, valid_C, JS(pL,pC)]))`

`p = (1-g) * pL + g * pC`

An unavailable local view forces g=1; an unavailable contextual view forces g=0. If both views are unusable, the sample is quarantined rather than silently imputed as a valid medical observation. The gate is not described as an explanation without intervention audits.

The supplied reference model implements this contract for synthetic testing. Count its parameters from the code rather than relying on an estimate. The supplied full candidate has **164,198 parameters**, measured from the included reference code; the fixed-fusion/no-auxiliary control has 147,524. These are architecture counts, not trained-model evidence. Any implementation change requires recounting. A small model may underfit; this is tested against MobileNetV3-Small rather than concealed.

## 7. Losses and labels

For class k and row i, let v_ik be nonnegative expert votes, n_i their sum and q_ik=v_ik/n_i. Reject n_i=0; retain n_i=1 for classification.

`L_soft(i) = -sum_k q_ik log p_ik`

This has the same model optimum as KL(q || p), because target entropy is constant for a fixed example. The primary score uses KL, not cross-entropy with the name changed. There is no class weighting, vote-count weighting, majority-only training, focal loss or label sharpening in the default classification loss.

For n_i>1:

`d_i = 1 - sum_k v_ik(v_ik-1) / [n_i(n_i-1)]`

This is the fraction of distinct ordered annotator-vote pairs that disagree. Under an exchangeable independent-vote model it estimates population pairwise disagreement; those assumptions are not guaranteed. For n_i=1 the auxiliary target is unavailable, not zero. Finite-sample d_i can equal 1, so it must not be clipped to the six-class population maximum of 5/6.

`L = mean(L_soft) + 0.1 * mean_{n_i>1} (d_hat_i-d_i)^2`

If a mini-batch has no n>1 examples, set the auxiliary contribution to a differentiable zero; never take the mean of an empty tensor. The coefficient is fixed before final testing. The ablation sets it to zero. Annotation counts and observed d are supervision only; they are prohibited inference features. Predicted disagreement is not equivalent to seizure probability or epistemic uncertainty.

## 8. Baselines and ablations

| ID | Model / intervention | Purpose |
|---|---|---|
| B0 | Training-set empirical soft class prior | Detect scoring/preprocessing errors |
| B1 | CPU compact spectral-feature regression | Check whether deep learning adds value |
| B2 | Compact trunk, uniform local bins, fixed 50:50 fusion, no auxiliary head | Byte-matched conventional control |
| B3 | HF MobileNetV3-Small, four input channels, six outputs, same cached evidence | Stronger small pretrained comparator |
| A1 | B2 plus foveated local bins | Isolate temporal allocation |
| A2 | A1 plus learned gate, auxiliary loss off | Isolate fusion |
| P | A2 plus disagreement auxiliary loss | Full prespecified candidate |
| A3 | P with local-only / context-only inference | Audit actual modality dependence |

B3 uses `timm/mobilenetv3_small_100.lamb_in1k`; the original model card reports about 2.5M parameters before adaptation. Its adapted parameter count and load-time provenance must be measured. ImageNet RGB normalization is not automatically suitable for EEG. [S08]

Select the stronger B2/B3 comparator **using tune data only**. The primary final comparison is P versus that locked comparator. Mechanism claims about foveation additionally require the matched B2→A1 comparison; a win against B3 alone does not isolate the mechanism.

The B3 input concatenates the two views along time after separate normalization, then resizes from 64x96 to 128x192 without a center crop. Its artificial view boundary is documented. No random rotation or image color manipulation is used. A sensitivity check can test two shared-backbone view passes only within the fixed pilot budget.

## 9. Training contract

Start with physical batch 32, effective batch 64, AdamW, learning rate 0.001 for the compact model, weight decay 0.01, gradient norm cap 1.0, cosine decay, and at most 12 epochs. For B3 use backbone learning rate 0.0001, head learning rate 0.001 and one initial frozen-backbone epoch. These are initial choices, not asserted optimal settings.

The only allowed optimizer pilot changes are learning rate multiplied by 0.3 and a common shorter epoch count. Maximum eight named development configurations; no open-ended search. Compare effective sample exposures, not just epoch labels.

Mixed precision is allowed only after a finite-output and loss-parity smoke test. Keep softmax, logarithms, KL, target calculations and metrics in float32 or float64. Default to eager execution. Do not add compile overhead, TensorRT, quantization, large teacher distillation, self-supervised pretraining or multi-GPU orchestration without measured need and a protocol amendment.

Core augmentation is limited to mild log-power gain jitter, small frequency masking that does not erase the entire evidence, and occasional single-region dropout. Disable temporal translation, vertical/horizontal flips, broad center masking and rotations. Mixup is off in the core because it complicates disagreement-target semantics.

## 10. Locked evaluation design

Split independent patient/recording connected components once into approximately 60% train, 10% tune, 5% calibration-T, 5% calibration-P and 20% locked test. At the reported literature scale of about 1,950 patients this is roughly 1,170/195/98/97/390; actual counts are measured, not assumed. [S02]

After selecting epochs and methods, refit each final model on train+tune (approximately 70%). Fit scalar temperature separately on calibration-T. Freeze referral thresholds on calibration-P. Neither set is used for training final network weights. Test labels are not available to the implementation/tuning role before protocol lock.

Use fixed final seeds 101, 202 and 303. The primary estimand is the difference in **mean patient KL, averaged over the three single-model seeds**, between P and the chosen baseline. This is an algorithm comparison, not a three-model ensemble. The deployment exemplar is seed 101 selected in advance, never the seed with the best test result.

## 11. Success, uncertainty and stopping

The primary hypothesis is superiority: Δ = KL(P)−KL(B) < 0. A two-sided 95% patient/component-cluster bootstrap interval wholly below zero supports a held-out improvement claim. An interval crossing zero is inconclusive, not proof of equivalence. No arbitrary noninferiority margin is imposed.

Report absolute and relative effect sizes even when significant. Choose a minimum practically meaningful improvement only with domain input and a documented development-stage justification; no clinical utility threshold is invented here. Calibration, class recall and failure cases must accompany a headline improvement.

Before full training, estimate paired patient-loss variation on development data. Illustratively, with 390 independent patients and paired standard deviation 0.10/0.20/0.30, a normal approximation gives an 80%-power detectable difference of about 0.014/0.028/0.043. These are arithmetic scenarios, not observed power. Exact class-specific and clustered feasibility depends on the actual cohort.

A performance win, a resource win and a publication-ready clinical claim are different outcomes. A compact model may justify an efficiency paper only if predictive trade-offs are honestly quantified; prospective or external clinical validation is outside this core study.

## 12. Resource ceiling and priority order

| Stage | Maximum GPU-wall hours |
|---|---:|
| Hardware smoke and throughput pilot | 0.5 |
| Development baseline selection | 1.5 |
| Mechanism ablations | 1.5 |
| Final two methods × three seeds | 7.0 |
| Calibration, test inference, bounded stress tests | 0.5 |
| Contingency for diagnosed failures | 1.0 |
| **Total hard ceiling** | **12.0** |

These are allocations, not runtime predictions. One final fit has at most 70 minutes within the final-fit pool. A run that cannot finish the frozen training schedule is `INCOMPLETE`; it cannot be silently compared with a fully trained competitor. If projected cost exceeds the cap, stop before final testing and report a pilot-stage result.

Target GPU allocations of 4–8 GiB, stop at 12 GiB; host process-tree RSS target 24 GiB, stop at 32 GiB. Spark memory is shared, so additionally monitor total system availability. Start with two data-loader workers and four CPU threads. Details and storage accounting are in Document 03.

## 13. Publication and claim ladder

Level 0 is an auditable data and split manifest. Level 1 is a reproducible baseline with uncertainty. Level 2 is a matched-budget improvement with ablations. Level 3 is robustness and review-prioritization evidence. Clinical utility requires additional external or prospective evidence and cannot be reached by relabeling internal segment metrics.

A top-tier submission would require a convincing effect, a defensible novel mechanism, strong comparisons and transparent limitations. The current package provides the experiment to test that possibility. It does not certify the result in advance.

## 14. Reading order

Use Document 02 for offsets, cache and data quality; Document 03 for DGX execution; Document 04 for statistics; Document 05 for 20 visualizations; Document 06 for notebooks; Document 07 for GitHub; and Document 08 for the completed design-review loop. `AGENTS.md` and `CODEX_EXECUTION.md` are the machine-facing execution instructions. The verified source register accompanies the package.
