# Model card — CAPE-EEG candidate P (seed 101) and comparator B3

**Status:** research artefact; weights not released (rights review pending). Every number below is a measured value from the locked evaluation of protocol `380269a4a6db7334`; nothing is projected.

## Task and labels
Six-way probability estimation of the expert-vote distribution over seizure, LPD, GPD, LRDA, GRDA and other for a labelled 10-second EEG centre, observed through a 50 s raw EEG window and a 600 s spectrogram context. Targets are vote fractions; single-annotator windows are retained. The target is expert agreement, not ground-truth pathology.

## Training data and partitions
HMS Harmful Brain Activity Classification (Kaggle 2024): 106,800 labelled windows, 1,950 patients, 17,089 EEG files, 11,138 spectrograms. Patient/recording connected components split deterministically (seed 20260907) into train 64,292 rows / 1,170 patients; tune 10,667 / 196; calibration-T 5,289 / 95; calibration-P 5,247 / 98; locked test 21,305 / 391. Zero cross-partition overlap of patients, EEG ids or spectrogram ids. Final weights were refit on train+tune for a complete 3-epoch cosine schedule with seeds 101/202/303; seed 101 was designated the deployment exemplar before the test run.

## Architectures
* **P:** 164,134 parameters; two 3×3 stems, shared depthwise-separable trunk (24-48-64-96-128, GroupNorm), local mean/max/centre pooling, context mean/max pooling, six-way heads, learned gate on `[u_L, u_C, valid_L, valid_C, JS]`, disagreement head.
* **B3:** timm `mobilenetv3_small_100.lamb_in1k` (HF revision `1824797e7887cbec1990e4adbd6675960a36c589`) with 4 input channels and 6 outputs, 1,524,150 parameters; both views concatenated along time (64×96) and resized to 128×192; no ImageNet colour normalisation.

## Performance on the locked test (temperature-calibrated, three-seed mean)
| | P | B3 |
|---|---:|---:|
| Patient-mean KL | 0.934 ± 0.018 | 0.869 ± 0.011 |
| Row-mean KL | 0.906 | 0.807 |
| KL, windows with ≥ 10 votes | 0.744 | 0.666 |
| Macro AUROC (unique-majority rows) | 0.857 | 0.884 |
| Per-class AUROC seed 101 (Sz, LPD, GPD, LRDA, GRDA, Other) | 0.885 / 0.873 / 0.925 / 0.801 / 0.900 / 0.782 | 0.915 / 0.903 / 0.932 / 0.813 / 0.915 / 0.819 |
| Balanced accuracy | 0.523 | 0.576 |
| Soft top-label ECE | 0.051 | 0.036 |
| Expected Brier | 0.625 | 0.578 |
| Temperature (calibration-T) | 1.21 / 1.39 / 1.21 | 1.33 / 1.28 / 1.30 |

Primary paired difference P − B3: +0.066 [+0.033, +0.097] (2,000 patient bootstrap replicates) — the candidate is inferior. Subgroups (exploratory): the gap is largest for LPD-majority windows (+0.18 [+0.05, +0.31]) and windows with < 95 % valid cells (+0.12); LRDA shows no difference (−0.03 [−0.15, +0.09]); single-vote windows have low support (28 patients).

## Uncertainty and referral
Frozen referral score: predictive entropy (chosen on tune among entropy, 1−max, predicted disagreement). Thresholds fixed on calibration-P. At the 90 % planned operating point P refers 2,519 of 21,305 test windows (actual coverage 88.2 %) and lowers accepted-case patient KL from 0.923 to 0.911. Predicted disagreement correlates with observed pairwise annotator disagreement at Spearman 0.31 (MSE 0.088); it is not a calibrated epistemic uncertainty.

## Evidence use (seed 101, full test)
Removing the raw-EEG local view: ΔKL +0.41; removing the 10-minute context: +0.09; removing one region: LL +0.10, RL +0.12, LP +0.20, RP +0.07. Saliency-ranked deletion of local (region, time-bin) blocks raises KL faster than random deletion at every fraction on 48 prespecified cases. Gate mean 0.45 (5th–95th percentile 0.08–0.85): the model mixes views rather than collapsing to one.

## Robustness (777 label-blind test rows, ΔKL versus clean)
| Condition | P | B3 |
|---|---:|---:|
| Left-lateral region removed | +0.084 | +0.053 |
| +10 % contiguous time mask | 0.000 | −0.007 |
| +0.5 log-power gain | +0.008 | −0.003 |
| +1.5 log-power gain | +0.119 | +0.040 |
| STFT window 512 instead of 256 | +0.053 | +0.142 |

## Resources
Final fit 1.1 min (P) / 1.0 min (B3) on a GB10; peak CUDA allocation 0.44 GiB; process RSS 7.6 GiB; cached inference median 1.36 ms (batch 1) and 2.62 ms (batch 32) for P, 1.50 / 1.87 ms for B3; raw Parquet-to-prediction ≈ 45 ms per window; whole study 0.72 GPU-hours over 17 jobs with no failures; derived cache 5.58 GB (+1.86 GB uniform alternate); energy not measured.

## Forbidden claims
No diagnosis, no seizure-onset latency, no false-alarm rate per hour, no clinical net benefit, no site generalisation, no conformal coverage guarantee. Do not use outside research.
