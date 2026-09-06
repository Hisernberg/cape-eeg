#!/usr/bin/env python3
"""Notebook-01 backend (part 1): connected components, deterministic allocation, leakage audit."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cape_eeg.paths import resolve_workspace
from cape_eeg.status import write_json, Ledger
from cape_eeg.data.inventory import load_train_csv
from cape_eeg.data.splits import allocate, leakage_audit
ws = resolve_workspace(); led = Ledger(ws.provenance / "ledger.jsonl")
df, _ = load_train_csv(ws.data / "train.csv")
d, summary = allocate(df)
audit = leakage_audit(d)
if audit["status"] != "PASS":
    led.append("I1_split", "FAIL", "cross-partition overlap detected"); raise SystemExit("BLOCKED: leakage")
d[["label_id", "patient_id", "eeg_id", "spectrogram_id", "component", "partition"]].to_parquet(ws.manifests / "split_manifest.parquet", index=False)
write_json(ws.manifests / "split_summary.json", summary); write_json(ws.manifests / "leakage_audit.json", audit)
led.append("I1_split", "PASS", "frozen split written", split_hash=summary["split_hash"])
for p, s in summary.items():
    if isinstance(s, dict) and "rows" in s:
        print(f"{p:14s} rows {s['rows']:6d} ({s['row_share']:.3f}) patients {s['patients']:4d} comps {s['components']:4d} class share {[round(v,3) for v in s['class_vote_share'].values()]}")
print("split_hash", summary["split_hash"], "leakage", audit["status"], "forbidden", audit["forbidden_overlap"])
