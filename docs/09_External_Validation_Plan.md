# External validation plan (blocked on data access)

The single improvement that would most strengthen the study is validation on a second corpus under a new protocol lock. This document specifies exactly how it will be done so that no decision is left to be made after seeing the data. It is a plan, not a result: **no second corpus was available in this study**, because the only public EEG corpus with lateralised/generalised periodic-discharge labels, the TUH EEG Event Corpus (TUEV, Temple University), requires registered access (signed data-use form and password-protected download) that cannot be obtained automatically.

## Candidate corpus
TUEV (`tuh_eeg_events`, v2.0.x): EDF recordings at 250 Hz with per-channel event annotations in six classes: SPSW (spike/sharp wave), GPED (generalised periodic epileptiform discharge), PLED (periodic lateralised epileptiform discharge), EYEM (eye movement), ARTF (artifact), BCKG (background). Annotations are single-expert hard labels, not vote distributions.

## Label mapping (fixed in advance)
| TUEV class | HMS class used for evaluation | Note |
|---|---|---|
| PLED | LPD | periodic lateralised discharge |
| GPED | GPD | periodic generalised discharge |
| BCKG | Other | background |
| SPSW, EYEM, ARTF | excluded | no HMS counterpart (HMS "Other" is not an artifact class) |
| — | Seizure, LRDA, GRDA | absent in TUEV; class-specific metrics NOT_ESTIMABLE |

Evaluation therefore uses the three mapped classes with a **hard-label** target (one-hot), scored by KL to the one-hot vector (= negative log-probability of the labelled class) and by one-vs-rest AUROC among the three classes, with the model's six-way probabilities renormalised over {LPD, GPD, Other}. This is a transfer test of discrimination, not of vote-distribution fidelity.

## Alignment and montage
TUEV EDF files use the 10–20 system with TCP bipolar references; the sixteen bipolar chains of `src/cape_eeg/data/montage.py` are formed from the referential channels after resampling to 200 Hz (`scipy.signal.resample_poly`, 4:5). A 50 s window is centred on each annotated event (events shorter than 10 s are centred; events longer than 10 s use their first 10 s as the labelled centre). No supplied 10-minute spectrogram exists; the context view is computed from the raw recording with the same STFT and integrated to 64×64 bins over the 600 s surrounding the window, and the absence of the competition's spectrogram pipeline is declared as a distribution shift.

## Protocol
1. Freeze this document's hash into `protocol_lock_external.json` before any TUEV file is read for evaluation.
2. Apply the seed-101 frozen P and B3 inference bundles (weights, normalisers fitted on HMS train+tune, temperature from HMS Calibration-T) **without any refitting**.
3. Score once. Primary external estimand: paired patient-mean KL to the one-hot label over the three mapped classes, P − B3, patient bootstrap (2,000 replicates, seed 20260907). Secondary: AUROC per mapped class, calibration on the mapped classes, and the six-condition corruption suite.
4. Report the result whatever its direction; no threshold, temperature or montage decision may be changed after step 3.

## What is already implemented
The cache converter (`scripts/build_cache.py`) accepts any table with `patient_id, eeg_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_label_offset_seconds` and Parquet recordings with the HMS column names; an EDF-to-Parquet adapter with the mapping above is the only missing piece (about 100 lines) and is intentionally left unwritten until the data can be tested against.
