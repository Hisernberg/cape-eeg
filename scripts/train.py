#!/usr/bin/env python3
"""Train one named configuration for one seed in a bounded, ledgered GPU run.

Phases:
  dev    : fit on train, select epoch on tune (development pool; test partition inaccessible)
  final  : fit on train+tune for the frozen epoch count; no selection; last weights are final
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, torch
from cape_eeg.paths import resolve_workspace
from cape_eeg.status import write_json, read_json, Ledger, run_id as make_run_id, git_commit
from cape_eeg.contracts import stable_hash
from cape_eeg.data.cache import CacheReader
from cape_eeg.data.dataset import CacheDataset, select_rows, subsample_patients
from cape_eeg.data.normalization import fit_normalizer, save_normalizer, load_normalizer, Normalizer
from cape_eeg.model import build_model, count_parameters, CONFIGS, DESCRIPTIONS
from cape_eeg.training.engine import fit, DEFAULT_TRAINING
from cape_eeg.training.supervisor import Supervisor, GpuLock, GpuLedger

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, choices=list(CONFIGS) + ["B3", "B3S", "B3H"])
ap.add_argument("--phase", default="dev", choices=["dev", "final", "smoke"])
ap.add_argument("--seed", type=int, default=101)
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--lr-mult", type=float, default=1.0)
ap.add_argument("--batch", type=int, default=None)
ap.add_argument("--stage", default="baselines", help="ledger stage name: smoke|baselines|ablations|final")
ap.add_argument("--max-minutes", type=float, default=None)
ap.add_argument("--cache", default=None, help="override cache dir (smoke)")
ap.add_argument("--tag", default="")
ap.add_argument("--patient-fraction", type=float, default=1.0, help="deterministic patient-level subsample of the training partition (learning curve)")
a = ap.parse_args()

ws = resolve_workspace(); led = Ledger(ws.provenance / "ledger.jsonl")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoding = CONFIGS.get(a.config, {}).get("encoding", "foveated")
time_bins = CONFIGS.get(a.config, {}).get("time_bins", 32)
cache_dir = Path(a.cache) if a.cache else sorted(ws.cache.glob("*/manifest.json"))[-1].parent
reader = CacheReader(cache_dir, alt_uniform=(encoding == "uniform"))
split_hash = read_json(ws.manifests / "split_summary.json")["split_hash"]
phash = reader.manifest["preprocess_hash"]
train_parts = ["train"] if a.phase != "final" else ["train", "tune"]
role = "development"
train_rows = select_rows(reader.index, train_parts, role)
tune_rows = select_rows(reader.index, ["tune"], role) if a.phase != "final" else None
if a.phase == "smoke":
    train_rows = train_rows[:2048]; tune_rows = tune_rows[:512]
if a.patient_fraction < 1.0:
    train_rows = subsample_patients(reader.index[reader.index.row.isin(train_rows)], a.patient_fraction, a.seed)
# normalizer: train-only (dev) or train+tune (final), per encoding
norm_name = f"{phash}_{'+'.join(train_parts)}_{encoding}.json"
norm_path = ws.normalization / split_hash / norm_name
if norm_path.exists():
    norm = load_normalizer(norm_path)
else:
    norm = fit_normalizer(reader, train_rows); save_normalizer(norm, norm_path)
normalizer = Normalizer(norm)
train_ds = CacheDataset(reader, train_rows, normalizer, time_bins=time_bins); tune_ds = CacheDataset(reader, tune_rows, normalizer, time_bins=time_bins) if tune_rows is not None else None
cfg = dict(DEFAULT_TRAINING)
cfg["compact_lr"] *= a.lr_mult; cfg["head_lr"] *= a.lr_mult; cfg["pretrained_backbone_lr"] *= a.lr_mult
if a.batch: cfg["physical_batch"] = a.batch
if a.config in ("B3", "B3H"): cfg["frozen_backbone_epochs"] = 1
if a.config == "B3S": cfg["pretrained_backbone_lr"] = cfg["head_lr"]  # no pretrained features to protect: one learning rate
epochs = a.epochs or cfg["max_epochs"]
cfg_for_id = {**cfg, "epochs": epochs, "config": a.config, "phase": a.phase, "encoding": encoding, "time_bins": time_bins, "patient_fraction": a.patient_fraction, "train_parts": train_parts, "tag": a.tag}
rid = make_run_id(a.phase, a.config, a.seed, split_hash, phash, cfg_for_id)
run_dir = ws.runs / rid
if (run_dir / "status.json").exists() and read_json(run_dir / "status.json").get("status") == "PASS":
    print(f"run {rid} already PASS; reuse (hashes match)"); sys.exit(0)
model = build_model(a.config); params = count_parameters(model)
ledger = GpuLedger(ws.runs / "gpu_ledger.jsonl")
if ledger.remaining_hours() <= 0:
    led.append(rid, "BLOCKED", "GPU ledger exhausted"); sys.exit("BLOCKED: 12 GPU-hour ceiling reached")
print(f"run {rid}\n  {a.config}: {DESCRIPTIONS[a.config]}\n  params {params} train rows {len(train_ds)} (quarantined {train_ds.n_quarantined}) tune rows {len(tune_ds) if tune_ds else 0}"
      f"\n  epochs {epochs} lr_mult {a.lr_mult} batch {cfg['physical_batch']} ledger used {ledger.total_hours():.2f} h of 12")
write_json(run_dir / "config.resolved.json", {"run_id": rid, "config_id": a.config, "description": DESCRIPTIONS[a.config], "seed": a.seed, "phase": a.phase,
    "split_hash": split_hash, "preprocess_hash": phash, "cache_hash": reader.manifest["cache_hash"], "normalizer_hash": norm["normalizer_hash"],
    "encoding": encoding, "time_bins": time_bins, "patient_fraction": a.patient_fraction, "n_train_patients": int(reader.index[reader.index.row.isin(train_rows)].patient_id.nunique()), "train_partitions": train_parts, "epochs": epochs, "training": cfg, "parameters": params, "git_commit": git_commit(ws.repo),
    "n_train_rows": len(train_ds), "n_tune_rows": len(tune_ds) if tune_ds else 0, "quarantined_rows": train_ds.n_quarantined})
sup = Supervisor(max_minutes=a.max_minutes)
t0 = time.time()
with GpuLock(ws.runs / "gpu.lock"):
    status = fit(model, train_ds, tune_ds, run_dir, cfg, a.seed, device, sup, epochs=epochs)
ledger.record(rid, a.stage, time.time() - t0, status["status"], config=a.config, seed=a.seed, phase=a.phase)
led.append(rid, status["status"], status.get("reason", ""), stage=a.stage, config=a.config, seed=a.seed, phase=a.phase, wall_min=round((time.time() - t0) / 60, 2))
print("status:", status["status"], status.get("reason", ""), "| best:", status.get("best"), "| peak:", {k: round(v, 2) for k, v in status.get("peak", {}).items()})
