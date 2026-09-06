#!/usr/bin/env python3
"""Notebook-01 backend (part 2): bounded paired spectral cache conversion with .partial shards."""
from __future__ import annotations
import argparse, sys, time, resource, os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"  # one BLAS thread per conversion worker; must precede numpy import
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from cape_eeg.paths import resolve_workspace
from cape_eeg.status import write_json, Ledger, utc_now
from cape_eeg.contracts import preprocess_signature, stable_hash
from cape_eeg.data.inventory import load_train_csv
from cape_eeg.data.cache import build_row_index, convert, finalize

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", type=int, default=0, help="convert only N recordings (and their spectrograms) into a scratch cache")
ap.add_argument("--workers", type=int, default=12); ap.add_argument("--no-alt", action="store_true")
a = ap.parse_args()
ws = resolve_workspace(); led = Ledger(ws.provenance / "ledger.jsonl")
df, _ = load_train_csv(ws.data / "train.csv")
split = pd.read_parquet(ws.manifests / "split_manifest.parquet") if (ws.manifests / "split_manifest.parquet").exists() else None
if split is not None:
    df = df.merge(split[["label_id", "component", "partition"]], on="label_id", how="left")
index = build_row_index(df)
phash = stable_hash(preprocess_signature())
if a.smoke:
    rng = np.random.default_rng(0); eeg_ids = set(rng.choice(index.eeg_id.unique(), a.smoke, replace=False).tolist())
    sub = index[index.eeg_id.isin(eeg_ids)].copy(); sub = sub.sort_values("label_id").reset_index(drop=True)
    sub["row"] = np.arange(len(sub)); sub["shard"] = sub.row // 2048; sub["pos"] = sub.row % 2048
    cache_dir = ws.root / "scratch" / f"cache_smoke_{phash}"; index = sub
else:
    cache_dir = ws.cache / phash
print("cache dir:", cache_dir.name, "rows:", len(index), "workers:", a.workers)
t0 = time.time()
q = convert(index, ws.data, cache_dir, workers=a.workers, alt_uniform=not a.no_alt)
man = finalize(cache_dir, index, q, alt_uniform=not a.no_alt)
peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
child_rss_gib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20
rec = {"elapsed_seconds": round(time.time() - t0, 1), "main_peak_rss_gib": round(peak_rss_gib, 2), "worker_peak_rss_gib": round(child_rss_gib, 2),
       "rows": len(index), "smoke": a.smoke, "cache_dir": cache_dir.name, "workers": a.workers, "finished": utc_now()}
write_json(cache_dir / "conversion_resources.json", rec)
if not a.smoke:
    led.append("I2_cache", "PASS" if man["within_cap"] else "FAIL", "cache conversion finished", preprocess_hash=phash, cache_hash=man["cache_hash"], **rec)
print("resources:", rec)
