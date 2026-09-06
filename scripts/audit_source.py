#!/usr/bin/env python3
"""Notebook-00 backend: schema gate, file inventory, checksums, alignment spot checks, byte estimate."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from cape_eeg.paths import resolve_workspace, redact
from cape_eeg.status import write_json, environment_snapshot, Ledger
from cape_eeg.contracts import cache_bytes, preprocess_signature, stable_hash
from cape_eeg.data.inventory import load_train_csv, inventory_files
from cape_eeg.data.montage import validate_columns, montage_signature
import pyarrow.parquet as pq

ap = argparse.ArgumentParser(); ap.add_argument("--no-checksums", action="store_true"); ap.add_argument("--workers", type=int, default=12)
a = ap.parse_args()
ws = resolve_workspace(); led = Ledger(ws.provenance / "ledger.jsonl")
t0 = time.time()
env = environment_snapshot(); write_json(ws.provenance / "preflight.json", env)
df, schema = load_train_csv(ws.data / "train.csv")
write_json(ws.provenance / "schema_report.json", schema)
print("schema gate PASS:", {k: schema[k] for k in ["n_rows", "n_patients", "n_eeg", "n_spectrograms", "vote_sum_min", "tied_max_rows"]})
eeg_tab, spec_tab, summary = inventory_files(df, ws.data, workers=a.workers, checksums=not a.no_checksums)
eeg_tab.to_parquet(ws.provenance / "eeg_files.parquet", index=False); spec_tab.to_parquet(ws.provenance / "spectrogram_files.parquet", index=False)
cols = pq.read_schema(ws.data / "train_eegs" / f"{df.eeg_id.iloc[0]}.parquet").names
summary["montage"] = validate_columns(cols); summary["montage_signature_hash"] = stable_hash(montage_signature())
summary["source_bytes_total"] = summary["eeg_bytes_total"] + summary["spectrogram_bytes_total"] + int((ws.data / "train.csv").stat().st_size)
summary["byte_estimate"] = cache_bytes(len(df)).as_dict(); summary["preprocess_hash"] = stable_hash(preprocess_signature())
summary["elapsed_seconds"] = round(time.time() - t0, 1)
write_json(ws.provenance / "source_manifest.json", summary)
led.append("I1_source_audit", "PASS", "schema gate and inventory completed", n_rows=len(df), source_manifest_hash=summary["source_manifest_hash"])
print("inventory:", {k: v for k, v in summary.items() if k not in ("montage", "byte_estimate")})
print("byte estimate:", summary["byte_estimate"])
