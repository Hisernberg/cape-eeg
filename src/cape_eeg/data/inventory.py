"""Source inventory and schema gate (docs/02 section 3). Read-only on the source directory."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..contracts import (LABELS, SPEC_OFFSET_ALIASES, EEG_SAMPLE_RATE_HZ, RAW_WINDOW_SAMPLES, CONTEXT_SECONDS,
                         file_sha256, stable_hash, vote_targets, pairwise_disagreement)

ID_COLUMNS = ["eeg_id", "eeg_sub_id", "spectrogram_id", "spectrogram_sub_id", "label_id", "patient_id"]


def load_train_csv(path: Path) -> tuple[pd.DataFrame, dict]:
    """Load train.csv, canonicalise the spectrogram offset alias, and run the schema gate."""
    df = pd.read_csv(path)
    report = {"path_name": Path(path).name, "n_rows": int(len(df)), "columns": list(df.columns)}
    present = [c for c in SPEC_OFFSET_ALIASES if c in df.columns]
    if not present:
        raise ValueError("no spectrogram offset column found")
    report["spectrogram_offset_source_spelling"] = present
    if len(present) == 2 and not np.allclose(df[present[0]], df[present[1]]):
        raise ValueError("both spectrogram offset spellings exist and disagree")
    df = df.rename(columns={present[0]: "spectrogram_label_offset_seconds"})
    if len(present) == 2:
        df = df.drop(columns=[present[1]])
    missing = [c for c in ID_COLUMNS + LABELS + ["eeg_label_offset_seconds", "expert_consensus"] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    for c in ID_COLUMNS:
        if df[c].isna().any():
            raise ValueError(f"missing identifiers in {c}")
    if not df["label_id"].is_unique:
        raise ValueError("duplicate label_id values fail the intake gate")
    votes = df[LABELS].to_numpy()
    vote_targets(votes)  # raises on negative/non-integer/zero-sum rows
    for c in ["eeg_label_offset_seconds", "spectrogram_label_offset_seconds"]:
        v = df[c].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(v)) or np.any(v < 0):
            raise ValueError(f"invalid offsets in {c}")
    e = df["eeg_label_offset_seconds"].to_numpy(dtype=np.float64) * EEG_SAMPLE_RATE_HZ
    report["eeg_offsets_off_grid"] = int(np.sum(np.abs(e - np.round(e)) > 1e-6))
    if report["eeg_offsets_off_grid"]:
        raise ValueError("EEG offsets off the 200 Hz grid: ALIGNMENT_UNVERIFIED")
    n = votes.sum(1)
    report.update(
        n_patients=int(df.patient_id.nunique()), n_eeg=int(df.eeg_id.nunique()), n_spectrograms=int(df.spectrogram_id.nunique()),
        vote_sum_min=int(n.min()), vote_sum_max=int(n.max()), n_single_vote=int((n == 1).sum()),
        n_vote_ge10=int((n >= 10).sum()), vote_mass_per_class={k: float(votes[:, i].sum()) for i, k in enumerate(LABELS)},
        tied_max_rows=int(((votes == votes.max(1, keepdims=True)).sum(1) > 1).sum()),
        identical_eeg_windows=int(df.duplicated(subset=["eeg_id", "eeg_label_offset_seconds"], keep=False).sum()),
        patients_per_eeg_max=int(df.groupby("eeg_id").patient_id.nunique().max()),
        patients_per_spectrogram_max=int(df.groupby("spectrogram_id").patient_id.nunique().max()),
        spectrograms_per_eeg_max=int(df.groupby("eeg_id").spectrogram_id.nunique().max()),
        eegs_per_spectrogram_max=int(df.groupby("spectrogram_id").eeg_id.nunique().max()),
    )
    # consensus audit: consensus must equal the unique argmax where unique
    names = np.array(["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"])
    mx = votes.max(1, keepdims=True); unique = (votes == mx).sum(1) == 1
    am = names[votes.argmax(1)]
    report["consensus_mismatch_unique_rows"] = int(((am != df.expert_consensus.to_numpy()) & unique).sum())
    report["consensus_follows_column_order_on_ties"] = bool(np.all(am[~unique] == df.expert_consensus.to_numpy()[~unique]))
    d = pairwise_disagreement(votes)
    report["pairwise_disagreement_available_rows"] = int(np.isfinite(d).sum())
    report["train_csv_sha256"] = file_sha256(path)
    return df, report


def _meta(path: str) -> tuple[str, int, int]:
    m = pq.read_metadata(path)
    return os.path.basename(path), int(m.num_rows), int(os.path.getsize(path))


def _sha(path: str) -> tuple[str, str]:
    return os.path.basename(path), file_sha256(path)


def inventory_files(df: pd.DataFrame, data_root: Path, workers: int = 8, checksums: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Check every referenced Parquet exists, record rows/bytes, and verify offsets fit (or flag edges)."""
    eeg_ids = sorted(df.eeg_id.unique()); spec_ids = sorted(df.spectrogram_id.unique())
    eeg_paths = [str(data_root / "train_eegs" / f"{i}.parquet") for i in eeg_ids]
    spec_paths = [str(data_root / "train_spectrograms" / f"{i}.parquet") for i in spec_ids]
    missing = [p for p in eeg_paths + spec_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} referenced source files missing, e.g. {os.path.basename(missing[0])}")
    with ProcessPoolExecutor(workers) as ex:
        eeg_meta = list(ex.map(_meta, eeg_paths, chunksize=64))
        spec_meta = list(ex.map(_meta, spec_paths, chunksize=64))
        eeg_sha = dict(ex.map(_sha, eeg_paths, chunksize=32)) if checksums else {}
        spec_sha = dict(ex.map(_sha, spec_paths, chunksize=32)) if checksums else {}
    eeg_tab = pd.DataFrame({"eeg_id": eeg_ids, "eeg_rows": [m[1] for m in eeg_meta], "eeg_bytes": [m[2] for m in eeg_meta]})
    eeg_tab["eeg_sha256"] = [eeg_sha.get(f"{i}.parquet") for i in eeg_ids]
    spec_tab = pd.DataFrame({"spectrogram_id": spec_ids, "spec_rows": [m[1] for m in spec_meta], "spec_bytes": [m[2] for m in spec_meta]})
    spec_tab["spec_sha256"] = [spec_sha.get(f"{i}.parquet") for i in spec_ids]
    x = df.merge(eeg_tab[["eeg_id", "eeg_rows"]], on="eeg_id").merge(spec_tab[["spectrogram_id", "spec_rows"]], on="spectrogram_id")
    end = x.eeg_label_offset_seconds * EEG_SAMPLE_RATE_HZ + RAW_WINDOW_SAMPLES
    x["eeg_edge_missing_samples"] = np.clip(end - x.eeg_rows, 0, None).astype(int)
    summary = {
        "n_eeg_files": len(eeg_ids), "n_spectrogram_files": len(spec_ids),
        "eeg_bytes_total": int(eeg_tab.eeg_bytes.sum()), "spectrogram_bytes_total": int(spec_tab.spec_bytes.sum()),
        "unreferenced_eeg_files": int(len(list((data_root / "train_eegs").glob("*.parquet"))) - len(eeg_ids)),
        "unreferenced_spectrogram_files": int(len(list((data_root / "train_spectrograms").glob("*.parquet"))) - len(spec_ids)),
        "eeg_rows_min": int(eeg_tab.eeg_rows.min()), "eeg_rows_max": int(eeg_tab.eeg_rows.max()),
        "rows_with_truncated_eeg_window": int((x.eeg_edge_missing_samples > 0).sum()),
        "rows_with_short_spectrogram_context": int(((x.spectrogram_label_offset_seconds + CONTEXT_SECONDS) > x.spec_rows * 2 + 1).sum()),
        "checksums": checksums,
        "source_manifest_hash": stable_hash({"eeg": eeg_sha, "spec": spec_sha}) if checksums else None,
    }
    return eeg_tab, spec_tab, summary
