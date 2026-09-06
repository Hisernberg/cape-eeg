#!/usr/bin/env python3
"""B0 (train prior) and B1 (band-power soft logistic regression) on train -> tune, CPU only.
With --final, refit on train+tune and predict calibration/test partitions (evaluator stage)."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from cape_eeg.paths import resolve_workspace
from cape_eeg.status import write_json, read_json, Ledger
from cape_eeg.data.cache import CacheReader
from cape_eeg.data.dataset import CacheDataset, select_rows
from cape_eeg.data.normalization import fit_normalizer, save_normalizer, load_normalizer, Normalizer
from cape_eeg.data.spectral import ContextSpectralEncoder
from cape_eeg.baselines import PriorBaseline, SoftLogisticRegression, b1_features
from cape_eeg.metrics import kl_rows, patient_weighted_mean

ap = argparse.ArgumentParser(); ap.add_argument("--final", action="store_true"); ap.add_argument("--role", default="development")
a = ap.parse_args()
ws = resolve_workspace(); led = Ledger(ws.provenance / "ledger.jsonl")
cache_dir = sorted(ws.cache.glob("*/manifest.json"))[-1].parent; reader = CacheReader(cache_dir)
split_hash = read_json(ws.manifests / "split_summary.json")["split_hash"]; phash = reader.manifest["preprocess_hash"]
train_parts = ["train", "tune"] if a.final else ["train"]
norm_path = ws.normalization / split_hash / f"{phash}_{'+'.join(train_parts)}_foveated.json"
tr = select_rows(reader.index, train_parts, "development")
norm = load_normalizer(norm_path) if norm_path.exists() else save_normalizer(fit_normalizer(reader, tr), norm_path) and load_normalizer(norm_path)
N = Normalizer(norm)
ctx_range = tuple(float(x) for x in (0.49, 20.02))  # measured from the supplied spectrogram axis (0.59..19.92 Hz, 0.195 Hz spacing)
eval_parts = ["calibration_t", "calibration_p", "test"] if a.final else ["tune"]
role = "evaluator" if a.final else "development"
tds = CacheDataset(reader, tr, N)

def features(ds):
    X = []; t0 = time.time()
    for b in ds.batches(512, shuffle=False):
        X.append(b1_features(b, ctx_range))
    return np.concatenate(X)

t0 = time.time(); Xtr = features(tds); print(f"B1 features train {Xtr.shape} in {time.time()-t0:.0f}s")
b0 = PriorBaseline().fit(tds.q); b1 = SoftLogisticRegression().fit(Xtr, tds.q)
out_dir = ws.runs / f"{'final' if a.final else 'dev'}_cpu_baselines_{split_hash[:8]}_{phash[:8]}"; out_dir.mkdir(parents=True, exist_ok=True)
results = {"B0": {}, "B1": {}, "train_prior": b0.p.tolist(), "n_features": int(Xtr.shape[1]), "train_parts": train_parts}
for part in eval_parts:
    rows = select_rows(reader.index, [part], role); ds = CacheDataset(reader, rows, N); X = features(ds)
    for name, model_p in [("B0", b0.predict(len(ds))), ("B1", b1.predict(X))]:
        kl = kl_rows(ds.q.astype(np.float64), model_p.astype(np.float64))
        results[name][part] = {"patient_kl": float(patient_weighted_mean(kl, ds.patient)), "row_kl": float(kl.mean()), "n_rows": int(len(ds)), "n_patients": int(len(np.unique(ds.patient)))}
        pd.DataFrame({"label_id": ds.label_id, **{f"p{k}": model_p[:, k] for k in range(6)}}).to_parquet(out_dir / f"{name}_{part}_predictions.parquet", index=False)
        print(f"{name} {part}: patient KL {results[name][part]['patient_kl']:.4f} row KL {results[name][part]['row_kl']:.4f} (n={len(ds)})")
write_json(out_dir / "results.json", results)
led.append(out_dir.name, "PASS", "CPU baselines B0/B1 completed", **{k: v for k, v in results.items() if k in ("B0", "B1")})
