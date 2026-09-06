#!/usr/bin/env python3
"""Write per-row predictions (private) for a finished run on the requested partitions.

Development role may predict train/tune only. The evaluator role (--role evaluator) is used once
for calibration-T/P and the locked test after the protocol lock (checked against protocol_lock.json).
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, torch
from cape_eeg.paths import resolve_workspace
from cape_eeg.status import write_json, read_json, Ledger, utc_now
from cape_eeg.contracts import stable_hash
from cape_eeg.data.cache import CacheReader
from cape_eeg.data.dataset import CacheDataset, select_rows
from cape_eeg.data.normalization import load_normalizer, Normalizer
from cape_eeg.model import build_model
from cape_eeg.training.engine import predict, load_weights

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="run directory name under private/runs")
ap.add_argument("--partitions", nargs="+", default=["tune"])
ap.add_argument("--role", default="development", choices=["development", "evaluator"])
ap.add_argument("--weights", default="best", choices=["best", "last"])
ap.add_argument("--views", nargs="+", default=["full"], help="full | local | context (forced-view predictions)")
a = ap.parse_args()
ws = resolve_workspace(); led = Ledger(ws.provenance / "ledger.jsonl")
run_dir = ws.runs / a.run; cfg = read_json(run_dir / "config.resolved.json"); status = read_json(run_dir / "status.json")
if status["status"] != "PASS":
    sys.exit(f"BLOCKED: run status is {status['status']} ({status.get('reason','')}); incomplete runs are not evaluated")
if a.role == "evaluator":
    lock = read_json(ws.manifests / "protocol_lock.json")
    if lock is None:
        sys.exit("BLOCKED: evaluator role requires protocol_lock.json")
    (ws.manifests / "test_access_log.jsonl").open("a").write(__import__("json").dumps({"run": a.run, "partitions": a.partitions, "timestamp": utc_now(), "protocol_hash": lock["protocol_hash"]}) + "\n")
reader = CacheReader(sorted(ws.cache.glob("*/manifest.json"))[-1].parent, alt_uniform=(cfg["encoding"] == "uniform"))
assert reader.manifest["cache_hash"] == cfg["cache_hash"], "cache hash mismatch with the run"
norm = load_normalizer(ws.normalization / cfg["split_hash"] / f"{cfg['preprocess_hash']}_{'+'.join(cfg['train_partitions'])}_{cfg['encoding']}.json")
assert norm["normalizer_hash"] == cfg["normalizer_hash"], "normalizer mismatch"
N = Normalizer(norm); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(cfg["config_id"]); load_weights(model, run_dir / "checkpoints" / f"{a.weights}.safetensors"); model.to(device).eval()
(run_dir / "predictions").mkdir(exist_ok=True)
for part in a.partitions:
    rows = select_rows(reader.index, [part], a.role); ds = CacheDataset(reader, rows, N)
    for view in a.views:
        t0 = time.time(); r = predict(model, ds, device, force_view=None if view == "full" else view)
        df = pd.DataFrame({"label_id": ds.label_id, "patient_id": ds.patient, "component": ds.component, "n_votes": ds.n_votes,
                           **{f"p{k}": r["p"][:, k] for k in range(6)}, **{f"pL{k}": r["pL"][:, k] for k in range(6)}, **{f"pC{k}": r["pC"][:, k] for k in range(6)},
                           "gate": r["g"], "js": r["js"], "d_hat": r["d_hat"] if r["d_hat"] is not None else np.nan,
                           **{f"v{k}": ds.votes[:, k] for k in range(6)}})
        out = run_dir / "predictions" / f"{part}_{view}_{a.weights}.parquet"; df.to_parquet(out, index=False)
        print(f"{a.run} {part} {view}: {len(df)} rows in {time.time()-t0:.1f}s -> {out.name}; finite={np.isfinite(df[[f'p{k}' for k in range(6)]].to_numpy()).all()}")
write_json(run_dir / "predictions" / "manifest.json", {"run": a.run, "weights": a.weights, "partitions": a.partitions, "views": a.views, "role": a.role, "timestamp": utc_now(),
           "prediction_hash": stable_hash([str(p.name) + str(p.stat().st_size) for p in sorted((run_dir / "predictions").glob("*.parquet"))])})
