#!/usr/bin/env python3
"""Render every planned figure (V01-V20) whose inputs exist; write figures/manifest.json and sidecars.

Idempotent: re-running after more artefacts appear re-renders those figures and overwrites their
outputs. Figures with missing inputs receive a NOT_RUN manifest record with the missing keys as the
reason; no plot is fabricated. Each saved PNG is verified non-empty, every plotted array finite, and
public captions/parameters are checked for identifiers and filesystem paths (a failure raises).
"""
from __future__ import annotations
import argparse, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cape_eeg.paths import resolve_workspace
from cape_eeg.status import write_json, read_json, git_commit, utc_now
from cape_eeg.visualization.registry import FIGURES, FIGURES_BY_ID, resolve_artefacts, missing_inputs
from cape_eeg.visualization.figures import render_figure, FigureVerificationError

MANIFEST_KEYS = ["figure_id", "title", "data_hashes", "run_ids", "split", "n_rows", "n_patients", "n_components", "generation_code_commit", "parameters", "output_paths", "status", "privacy_review", "caption"]

ap = argparse.ArgumentParser()
ap.add_argument("--only", default=None, help="comma-separated figure ids, e.g. V01,V03")
ap.add_argument("--out", default=None, help="override the public figure directory (default: <repo>/figures)")
ap.add_argument("--list", action="store_true", help="list figures and input availability, render nothing")
a = ap.parse_args()

ws = resolve_workspace()
out_dir = Path(a.out) if a.out else ws.figures_public
out_dir.mkdir(parents=True, exist_ok=True)
resolved = resolve_artefacts(ws)
wanted = [x.strip().upper() for x in a.only.split(",")] if a.only else [f.figure_id for f in FIGURES]
unknown = [w for w in wanted if w not in FIGURES_BY_ID]
if unknown:
    sys.exit(f"unknown figure id(s): {unknown}")

if a.list:
    for f in FIGURES:
        miss = missing_inputs(f, resolved)
        print(f"{f.figure_id}  {'READY   ' if not miss else 'NOT_RUN '} {f.title:55s} missing: {', '.join(miss) if miss else '-'}")
    sys.exit(0)

manifest_path = out_dir / "manifest.json"
existing = {r["figure_id"]: r for r in (read_json(manifest_path) or [])} if a.only else {}
records, failures = {}, []
t0 = time.time()
for fid in wanted:
    spec = FIGURES_BY_ID[fid]; t1 = time.time()
    try:
        rec = render_figure(fid, ws, out_dir=out_dir, private_dir=ws.figures_private, resolved=resolved)
    except Exception as e:  # keep rendering the rest; report at the end with a non-zero exit
        tb = traceback.format_exc(limit=3)
        rec = {"figure_id": fid, "title": spec.title, "data_hashes": {}, "run_ids": [], "split": None, "n_rows": None, "n_patients": None, "n_components": None,
               "generation_code_commit": git_commit(ws.repo), "parameters": {"required_inputs": list(spec.required)}, "output_paths": [], "private_output_paths": [],
               "status": "FAIL", "reason": f"{type(e).__name__}: {e}"[:500], "privacy_review": "not_applicable", "caption": f"{spec.title}: FAIL.", "generated": utc_now()}
        failures.append((fid, rec["reason"], tb))
    records[fid] = rec
    sidecar = out_dir / f"{spec.stem}.json"
    write_json(sidecar, rec)
    n_out = len(rec.get("output_paths", []))
    print(f"{fid}  {rec['status']:8s} {spec.title:55s} {n_out:2d} files  {time.time() - t1:5.1f}s  {rec.get('reason', '')}")

merged = existing | records
ordered = [merged[f.figure_id] for f in FIGURES if f.figure_id in merged]
public_records = [{k: r.get(k) for k in MANIFEST_KEYS} | {"reason": r.get("reason", ""), "question": FIGURES_BY_ID[r["figure_id"]].question, "generated": r.get("generated")} for r in ordered]
write_json(manifest_path, public_records)
n_pass = sum(r["status"] == "PASS" for r in ordered); n_nr = sum(r["status"] == "NOT_RUN" for r in ordered); n_fail = sum(r["status"] == "FAIL" for r in ordered)
print(f"\nmanifest: {manifest_path.relative_to(ws.repo) if manifest_path.is_relative_to(ws.repo) else manifest_path}  PASS {n_pass}  NOT_RUN {n_nr}  FAIL {n_fail}  ({time.time() - t0:.0f}s)")
for fid, reason, tb in failures:
    print(f"\nFAIL {fid}: {reason}\n{tb}")
sys.exit(1 if failures else 0)
