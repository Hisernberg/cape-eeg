# CAPE-EEG — complete evaluation report

Protocol hash `380269a4a6db7334` · split `4455edaac7cb7e5d` · preprocess `2406ce595fbd7457` · cache `af515c778666c076` · evaluated once on the locked test partition (21,305 windows, 391 patients = 391 independent components). Every value is measured; nothing is projected. Natural-log KL(q‖p) with p clipped at 1e-7; "patient-mean" weights every patient equally, "row-mean" weights every window equally. Calibrated = scalar temperature fitted on calibration-T only.

## 1. Primary confirmatory result

| Estimand | Δ = P − B3 | 95 % CI (2,000 paired patient bootstrap) | Decision |
|---|---:|---:|---|
| Patient-mean KL, calibrated (primary) | **+0.0657** | [+0.0334, +0.0973] | P inferior |
| Patient-mean KL, raw (secondary) | +0.0485 | [+0.0059, +0.0899] | P inferior |
| P+MSF − B3, calibrated (exploratory) | +0.0896 | [+0.0572, +0.1217] | inferior |
| P+MSF − P, calibrated (exploratory) | +0.0239 | [+0.0013, +0.0453] | MSF block hurts |

Per-seed patient-mean KL (calibrated): P 0.9235 / 0.9546 / 0.9244; B3 0.8776 / 0.8564 / 0.8715. 35.3 % of patients are better predicted by P. The development-stage paired SD was 0.58, giving a minimum detectable difference of 0.083 at 80 % power; the observed effect (+0.066) sits inside a clearly resolved interval, so the negative result is not a power failure.

## 2. Full metric table (locked test, three-seed means; per-seed values in `aggregate/table2_test_metrics_by_seed.csv`)

| Metric | P raw | P cal | B3 raw | B3 cal | P+MSF cal |
|---|---:|---:|---:|---:|---:|
| Patient-mean KL | 0.9910 | 0.9342 | 0.9425 | 0.8685 | 0.9580 |
| Row-mean KL | 0.9606 | 0.9056 | 0.8687 | 0.8066 | 0.9247 |
| Soft cross-entropy | 1.3215 | 1.2666 | 1.2297 | 1.1675 | 1.2857 |
| Soft-target squared error | 0.4213 | 0.4048 | 0.3717 | 0.3571 | 0.4130 |
| Expected categorical Brier | 0.6418 | 0.6254 | 0.5922 | 0.5776 | 0.6336 |
| KL, windows with ≥10 votes | 0.8192 | 0.7444 | 0.7544 | 0.6656 | 0.7375 |
| KL, single-vote windows | – | 0.939 | – | 0.817 | – |
| Macro AUROC (one-vs-rest, unique-majority rows n = 20,482, 823 ties excluded) | 0.8564 | 0.8570 | 0.8838 | 0.8842 | 0.8487 |
| Macro AUPRC | – | 0.5796 | – | 0.6557 | 0.5605 |
| Accuracy | – | 0.5392 | – | 0.5914 | 0.5243 |
| Balanced accuracy | 0.5227 | 0.5227 | 0.5761 | 0.5761 | 0.5064 |
| Macro F1 | – | 0.5204 | – | 0.5800 | 0.5022 |
| MCC | – | 0.4455 | – | 0.5090 | 0.4280 |
| Cohen κ | – | 0.4401 | – | 0.5034 | 0.4217 |
| Soft top-label ECE (10 equal-count bins) | 0.1181 | 0.0514 | 0.1072 | 0.0360 | 0.0400 |
| Majority-label ECE | – | 0.0267 | – | 0.0168 | 0.0363 |
| Temperature (calibration-T; none at a bound) | 1.221 / 1.386 / 1.210 | | 1.333 / 1.279 / 1.299 | | 1.419 / 1.429 / 1.349 |

Per-class AUROC, seed 101, calibrated (Seizure / LPD / GPD / LRDA / GRDA / Other): P 0.885 / 0.873 / 0.925 / 0.801 / 0.900 / 0.782; B3 0.915 / 0.903 / 0.932 / 0.813 / 0.915 / 0.819. Per-class recall, seed 101: P 0.731 / 0.513 / 0.592 / 0.237 / 0.643 / 0.507; B3 0.755 / 0.554 / 0.535 / 0.306 / 0.674 / 0.608. LRDA is the hardest class for both models (AUROC ≈ 0.80–0.85, recall < 0.32); its confusions go to GRDA and Other.

Confusion matrix, P seed 101, rows = true unique-majority class, columns = predicted (Seizure, LPD, GPD, LRDA, GRDA, Other):

```
Seizure  3188  156  127  157  195  540
LPD       521 1274  140  102  127  321
GPD       427   85 1860   33  303  435
LRDA      400  421   58  742  648  868
GRDA      137   54  207  190 2320  700
Other     648  292  160  168  578 1900
```

## 3. Development evidence (tune, seed 101; drives every frozen choice)

| Config | Params | Best tune patient-KL | Schedule / best epoch | Role |
|---|---:|---:|---|---|
| B0 prior | 0 | 1.4101 | – | scoring sanity |
| B1 band-power regression | 450 | 1.2405 | LBFGS | is deep learning needed? yes: −0.25 KL |
| B2 uniform bins, fixed fusion | 147,524 | 0.9862 | 12 ep / 2 | byte-matched control |
| A1 = B2 + foveation | 147,524 | 1.0011 | 12 ep / 2 | foveation: +0.015 (null) |
| A2 = A1 + learned gate | 155,877 | 0.9903 | 12 ep / 1 | gate: −0.011 (null) |
| P = A2 + disagreement aux | 164,134 | 0.9828 | 12 ep / 2 | aux loss: −0.007 (null) |
| P+MSF | 208,070 | 0.9875 | 12 ep / 1 | exploratory: +0.005 |
| B3 MobileNetV3-Small | 1,524,150 | 0.8959 | 12 ep / 3 | comparator |
| P, complete 3-epoch cosine | 164,134 | 0.9270 | 3 ep / 3 | pilot 7 → frozen schedule |
| B3, complete 3-epoch cosine | 1,524,150 | 0.9266 | 3 ep / 3 | pilot 8 → frozen schedule |

Referral-score selection on tune (AURC over coverage 0.5–1.0, lower is better): entropy 0.4017, 1−max 0.4067, predicted disagreement 0.4355 → entropy frozen.

## 4. Uncertainty, calibration and review prioritisation

* Temperature scaling reduces patient-mean KL by 0.046–0.081 for every model and seed and cuts soft ECE from 0.10–0.14 to 0.03–0.06. Both architectures are over-confident when trained with soft cross-entropy on windows dominated by 2–4 annotators.
* Referral at the 90 % planned operating point (thresholds from calibration-P): P seed 101 refers 2,519 windows (actual coverage 88.2 %), accepted-case patient KL 0.911 vs 0.923 whole-cohort; B3 seed 101 refers 1,929 (coverage 90.9 %), 0.874 vs 0.878. AURC (KL, coverage 0.5–1): P 0.401 / 0.406 / 0.415, B3 0.365 / 0.366 / 0.372 across seeds. Referral trims the tail but cannot repair the gap between the models.
* Disagreement head (P): Spearman with observed pairwise annotator disagreement 0.305 / 0.263 / 0.250 across seeds (MSE 0.088–0.094, 20,795 estimable windows); within vote strata 0.23–0.24 (2–4, 5–9, ≥10 votes). As a referral score it is worse than entropy (AURC 0.444–0.448 vs 0.401–0.415). Verdict: the head learns something real but weak, and it does not improve KL (P vs A2 on tune −0.007).
* Gate: mean 0.45 (5th–95th percentile 0.08–0.85) on seed 101, 0.72 on seed 202, 0.43 on seed 303 — the mixture is seed-dependent, which is one reason for P's higher seed spread (sd 0.018 vs 0.011).

## 5. Subgroups (exploratory, calibrated Δ = P − B3, paired patient bootstrap)

| Stratum | n rows | groups | KL P | KL B3 | Δ [95 % CI] |
|---|---:|---:|---:|---:|---|
| 1 vote | 510 | 28 | 0.939 | 0.817 | +0.123 [−0.032, +0.289] low support |
| 2–4 votes | 11,990 | 353 | 1.057 | 1.000 | +0.057 [+0.013, +0.099] |
| 5–9 votes | 315 | 56 | 0.491 | 0.473 | +0.018 [−0.043, +0.074] |
| ≥10 votes | 8,490 | 239 | 0.744 | 0.666 | +0.079 [+0.027, +0.129] |
| entropy = 0 (unanimous) | 10,877 | 357 | 1.046 | 0.973 | +0.073 [+0.028, +0.116] |
| entropy mid | 2,380 | 168 | 0.736 | 0.643 | +0.094 [+0.051, +0.141] |
| entropy high | 8,048 | 250 | 0.815 | 0.796 | +0.019 [−0.033, +0.071] |
| valid fraction < 0.95 | 1,523 | 134 | 1.036 | 0.917 | +0.119 [+0.029, +0.206] |
| valid fraction ≥ 0.95 | 19,782 | 381 | 0.917 | 0.853 | +0.064 [+0.030, +0.096] |
| majority Seizure | 4,363 | 191 | 0.983 | 0.928 | +0.056 [−0.007, +0.114] |
| majority LPD | 2,485 | 52 | 1.171 | 0.992 | +0.179 [+0.046, +0.308] |
| majority GPD | 3,143 | 65 | 0.983 | 0.901 | +0.082 [−0.062, +0.235] |
| majority LRDA | 3,137 | 93 | 1.580 | 1.609 | −0.028 [−0.154, +0.093] |
| majority GRDA | 3,608 | 140 | 0.979 | 0.940 | +0.039 [−0.030, +0.108] |
| majority Other | 3,746 | 239 | 0.777 | 0.709 | +0.068 [+0.024, +0.108] |
| tied majority | 823 | 97 | 0.760 | 0.718 | +0.042 [−0.021, +0.106] |

The comparator's advantage is concentrated in confident (unanimous or ≥10-vote) windows and in LPD; on the most ambiguous windows (high target entropy) and on LRDA the two models are statistically indistinguishable.

## 6. Robustness (777 label-blind test rows from 391 patients, ΔKL vs clean, patient bootstrap)

| Condition | P ΔKL [CI] | P flips / JS | B3 ΔKL [CI] | B3 flips / JS |
|---|---|---|---|---|
| Left-lateral region removed (both views) | +0.084 [+0.038, +0.131] | 0.31 / 0.20 | +0.053 [+0.002, +0.102] | 0.33 / 0.20 |
| +10 % contiguous time mask | −0.000 [−0.026, +0.024] | 0.16 / 0.10 | −0.007 [−0.035, +0.023] | 0.17 / 0.11 |
| +0.5 log-power gain | +0.008 [−0.010, +0.027] | 0.12 / 0.08 | −0.003 [−0.022, +0.014] | 0.10 / 0.07 |
| +1.5 log-power gain | +0.119 [+0.071, +0.167] | 0.29 / 0.19 | +0.040 [−0.005, +0.083] | 0.27 / 0.16 |
| STFT window 512 (was 256) | +0.053 [+0.024, +0.082] | 0.18 / 0.11 | +0.142 [+0.083, +0.201] | 0.30 / 0.19 |

Both models are insensitive to the small time mask and mild gain shift. P is more sensitive to strong amplitude shifts; the pretrained comparator is markedly more sensitive to the preprocessing (STFT resolution) change, consistent with the preprocessing-fragility literature the plan cites.

## 7. Evidence-use audit (P, seed 101, full test)

| Intervention | ΔKL [CI] | JS | label flips |
|---|---|---:|---:|
| Local 50 s view removed | +0.408 [+0.35, +0.47] | 0.220 | 31.5 % |
| Context 600 s view removed | +0.090 [+0.06, +0.13] | 0.181 | 28.8 % |
| Region LL removed | +0.097 | 0.199 | 32.9 % |
| Region RL removed | +0.117 | 0.216 | 36.7 % |
| Region LP removed | +0.204 | 0.238 | 42.5 % |
| Region RP removed | +0.066 | 0.177 | 30.2 % |

Ranked (|grad × input|) versus random deletion of local (region, time-bin) blocks on 48 prespecified cases — mean KL at 0 / 10 / 20 / 30 / 50 / 70 / 90 % removed: ranked 0.80 / 0.91 / 0.94 / 1.08 / 1.13 / 1.15 / 1.26; random 0.80 / 0.87 / 0.92 / 1.01 / 1.07 / 1.12 / 1.17. The ranking is informative at every fraction; the model's evidence use is auditable, not decorative.

## 8. Resources (Table 5)

| Item | Value |
|---|---|
| Total GPU wall time | 0.722 h of the 12 h ceiling (baselines 0.171, ablations 0.362, final 0.188) |
| GPU jobs / failed or incomplete | 17 / 0 |
| Final fit wall time | P 1.11 min, B3 0.98 min (3 epochs on 74,959 rows) |
| Peak CUDA allocation / process RSS | 0.44 GiB / 7.6 GiB |
| Training throughput | 2,200–3,700 examples/s (compact), 2,300–3,200 (B3 after the frozen epoch) |
| Cached inference, median (p95) | P 1.36 (1.38) ms @ batch 1, 2.62 (3.03) ms @ batch 32 = 12,200 ex/s; B3 1.50 (1.53) / 1.87 (2.17) ms = 17,100 ex/s |
| Raw Parquet → prediction, batch 1 | ≈ 45 ms (I/O and STFT dominated, both models) |
| Active cache / uniform alternate / source | 5.58 GB / 1.86 GB / 25.9 GB; conversion 786 s with 16 CPU workers, worker RSS 0.77 GiB |
| Energy | NOT_MEASURED |

## 9. Rating against the plan's own rubric (docs/08) and claim ladder

| Dimension | Score /5 | Evidence |
|---|---:|---|
| Clinical/task fidelity | 4.5 | Exact offsets verified with synthetic impulses, both modalities aligned on their own clocks, all votes retained, ties handled explicitly; segment-level only |
| Novelty and mechanism isolation | 3.0 | Mechanisms are isolated cleanly at matched bytes/exposure, but all three are null and the exploratory MSF block hurts; the pretrained baseline wins |
| Statistical validity | 4.5 | Locked protocol, single confirmatory test, patient-cluster bootstrap, seeds averaged as losses, MDE reported, low-support strata flagged |
| Resource discipline | 5.0 | 6 % of the GPU ceiling, one job at a time, every job ledgered, cache under cap with masks counted |
| Reproducibility/automation | 4.5 | 155 synthetic tests, hashed artefacts, idempotent drivers, executed notebooks, 20 provenance-tracked figures |
| Publication safeguards | 4.5 | Allowlisted export, identifier and secret scans, git-history scan, no raw data or weights |

Claim ladder reached: **Level 1 (reproducible baseline with uncertainty) and Level 3 evidence for the audits** (robustness, review prioritisation, evidence use). Level 2 (matched-budget improvement) is **not** reached: the confirmatory comparison is negative.

## 10. Analysis — what worked, what did not, and why

**What worked.** The engineering contract held end to end: alignment tests, leak-free split, byte-exact cache, bf16 parity, complete schedules, one test access, aggregate-only export. Temperature scaling and entropy-based referral behave as expected. Evidence deletion shows the compact model relies on the labelled raw-EEG window (as designed) and that its saliency ordering is meaningful.

**What did not.** (1) Every model memorises patient-specific structure within two to three epochs on 64k highly correlated windows; the 12-epoch schedule from the plan is too long and the allowed 3-epoch pilot was decisive (P: 0.983 → 0.927 on tune). (2) At that schedule the 164k-parameter candidate ties the 1.5M-parameter pretrained comparator on tune but loses on the locked test by 0.066 KL with a higher seed spread; ImageNet features transfer well to the upsampled 4-channel log-spectrogram image, and the compact trunk cannot match them from 1,170 training patients. (3) Foveating the local time axis, learning the view mixture and predicting annotator disagreement each change tune KL by ≤ 0.015 — the labelled centre is already well resolved by 32 uniform bins over 50 s, the gate mostly re-weights two views that agree, and the disagreement target is too noisy (Spearman 0.3) to regularise the classifier. (4) The DPNeXt-inspired multi-scale block adds capacity and overfits faster.

**Where the candidate is competitive.** High-entropy windows and LRDA (no measurable difference), preprocessing shifts (STFT window: +0.05 vs +0.14), parameter count (9× smaller) and batch-1 latency; batch-32 throughput favours the fused MobileNet kernels.

**Recommendations for the paper.** Present the study as a rigorous negative result with a reusable auditing protocol: a small pretrained CNN on a byte-budgeted two-view cache is a strong, cheap baseline for expert-vote EEG classification, and the three compact mechanisms should not be claimed. The natural next hypotheses within the same protocol are patient-level regularisation (fewer correlated windows per patient per epoch, stronger stochastic depth/dropout) and pretraining the compact trunk on the unlabeled portions of the 17,300 recordings; both would be new named configurations requiring a new lock.

## 11. Post-lock development analyses (added after review; tune partition only, test untouched)

Three-seed (101/202/303) re-run of the mechanism ladder and two confound controls under the frozen complete 3-epoch schedule, last-epoch weights, no checkpoint selection. B3-scratch = MobileNetV3-Small with random initialisation; B3-half = `mobilenetv3_small_050` with ImageNet weights (574,518 parameters). 19 GPU jobs, 0.36 GPU-hours (study total 1.08 h over 36 jobs).

| Config | Params | Tune patient-KL (mean ± SD) | Row KL |
|---|---:|---:|---:|
| A1 | 147,524 | 0.976 ± 0.037 | 0.945 |
| A2 | 155,877 | 0.979 ± 0.015 | 0.942 |
| B2 | 147,524 | 0.979 ± 0.016 | 0.966 |
| B3 | 1,524,150 | 0.913 ± 0.028 | 0.865 |
| B3H | 574,518 | 1.088 ± 0.024 | 1.055 |
| B3S | 1,524,150 | 1.103 ± 0.077 | 1.061 |
| P | 164,134 | 0.990 ± 0.063 | 0.979 |

| Paired contrast (same seed) | Δ tune KL (mean ± SD) | per seed |
|---|---:|---|
| A1 - B2 | -0.002 ± 0.038 | [np.float64(0.0112), np.float64(0.0276), np.float64(-0.0455)] |
| A2 - A1 | +0.002 ± 0.052 | [np.float64(-0.0404), np.float64(-0.0134), np.float64(0.0602)] |
| P - A2 | +0.012 ± 0.048 | [np.float64(-0.0381), np.float64(0.0148), np.float64(0.0585)] |
| P - B2 | +0.012 ± 0.072 | [np.float64(-0.0673), np.float64(0.0291), np.float64(0.0733)] |
| B3 - B3S | -0.190 ± 0.068 | [np.float64(-0.1228), np.float64(-0.1888), np.float64(-0.2597)] |
| B3 - B3H | -0.175 ± 0.052 | [np.float64(-0.1488), np.float64(-0.2347), np.float64(-0.1414)] |
| B3 - B2 | -0.066 ± 0.017 | [np.float64(-0.0677), np.float64(-0.0816), np.float64(-0.0485)] |
| P - B3 | +0.078 ± 0.067 | [np.float64(0.0004), np.float64(0.1106), np.float64(0.1217)] |
| B3S - P | +0.113 ± 0.031 | [np.float64(0.1225), np.float64(0.0782), np.float64(0.138)] |
| B3H - P | +0.097 ± 0.068 | [np.float64(0.1484), np.float64(0.124), np.float64(0.0197)] |

**Reading.** None of foveation, the learned gate or the disagreement loss changes tune loss by more than the seed spread (all |Δ| ≤ 0.012 with SD 0.04–0.07). ImageNet pretraining is worth 0.19 KL and full width 0.17 KL to the comparator; without either, MobileNetV3-Small is 0.11 (scratch) and 0.10 (half width) *worse* than the compact candidate. The locked-test advantage of B3 is therefore a transfer-learning effect at full width, not an architecture effect. The candidate's own seed spread (0.063) is the largest of the seven configurations.

## 12. Revision analyses: byte-budget sweep, learning curve, Dirichlet–multinomial head, energy (tune only; test untouched)

Nineteen plus thirty-three further development runs (GPU ledger now 1.99 h over 72 jobs, 0 failures) under the frozen complete 3-epoch schedule, last-epoch weights, three seeds.

### Byte-budget sweep (fixed fusion, no auxiliary head; coarser encodings are exact unions of the cached 32-bin cells)

| Local time bins | Local bytes / window | Encoding | Tune patient-KL |
|---:|---:|---|---:|
| 8 | 4,352 | foveated | 0.999 ± 0.034 |
| 8 | 4,352 | uniform | 1.014 ± 0.033 |
| 16 | 8,704 | foveated | 1.007 ± 0.021 |
| 16 | 8,704 | uniform | 1.004 ± 0.017 |
| 32 | 17,408 | foveated | 0.976 ± 0.037 |
| 32 | 17,408 | uniform | 0.979 ± 0.016 |

| Bins | Foveated − uniform (mean ± SD) | per seed |
|---:|---:|---|
| 8 | -0.015 ± 0.054 | [np.float64(0.0272), np.float64(0.0038), np.float64(-0.0752)] |
| 16 | +0.003 ± 0.004 | [np.float64(0.004), np.float64(0.0063), np.float64(-0.002)] |
| 32 | -0.002 ± 0.038 | [np.float64(0.0112), np.float64(0.0276), np.float64(-0.0455)] |

The foveation null is not a resolution artefact: foveated and uniform encodings are indistinguishable at every budget, and a four-fold compression of the local view costs only ~0.035 KL.

### Patient-count learning curve (P and B3; deterministic patient subsets; full tune evaluation)

| Model | Fraction | Train patients | Train rows | Tune patient-KL |
|---|---:|---:|---:|---:|
| B3 | 0.25 | 292 | 14,120 | 1.196 ± 0.080 |
| B3 | 0.50 | 585 | 33,503 | 0.977 ± 0.010 |
| B3 | 0.75 | 878 | 47,428 | 0.923 ± 0.023 |
| B3 | 1.00 | 1170 | 64,292 | 0.913 ± 0.028 |
| P | 0.25 | 292 | 14,120 | 1.187 ± 0.062 |
| P | 0.50 | 585 | 33,503 | 1.060 ± 0.044 |
| P | 0.75 | 878 | 47,428 | 1.052 ± 0.057 |
| P | 1.00 | 1170 | 64,292 | 0.990 ± 0.063 |

Both models are still improving at the full 1,170 patients; the comparator's margin is absent at a quarter of the patients and opens as patients are added. The task is patient-limited.

### Dirichlet–multinomial vote head (CAPE-EEG v2) versus the scalar disagreement head

| Config | Tune patient-KL | Spearman(d̂, observed d) | AURC entropy | AURC d̂ |
|---|---:|---:|---:|---:|
| A2 | 0.979 ± 0.015 |  | 0.432 |  |
| P | 0.990 ± 0.063 | 0.309 ± 0.022 | 0.446 | 0.477 |
| P_DM | 0.997 ± 0.044 | 0.310 ± 0.033 | 0.450 | 0.472 |

The probabilistic head is an equal classifier and an equal disagreement predictor; entropy remains the better referral score. The limit is the information in the auxiliary signal, not its parameterisation.

### Inference energy and CPU latency (frozen seed-101 models; boundary in `inference_measurements.json`)

| | P | B3 |
|---|---:|---:|
| GPU batch 32 throughput | 9883 win/s | 15262 win/s |
| Mean package power under load (idle 12.3 W) | 41.4 W | 46.9 W |
| Incremental energy per window | 2.94 mJ | 2.27 mJ |
| Energy per window incl. idle | 4.19 mJ | 3.08 mJ |
| CPU-only latency, 1 thread, batch 1 | 8.8 ms | 10.8 ms |
| CPU-only latency, 4 threads, batch 1 | 13.9 ms | 13.8 ms |

### External validation
Not performed: the only public corpus with periodic-discharge labels (TUEV) requires registered access. The fixed label mapping, alignment rule and lock procedure are in `docs/09_External_Validation_Plan.md`.
