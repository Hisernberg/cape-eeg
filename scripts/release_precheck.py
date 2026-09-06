#!/usr/bin/env python3
"""Notebook-06 backend: content precheck (allowlist, forbidden files, sizes, secret + identifier scans) and manifest."""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cape_eeg.paths import resolve_workspace
from cape_eeg.status import write_json, Ledger
from cape_eeg.release.export import precheck, git_history_scan
ws = resolve_workspace(); led = Ledger(ws.provenance / "ledger.jsonl")
rep = precheck(ws.repo)
hist = git_history_scan(ws.repo) if (ws.repo / ".git").exists() else {"status": "NOT_RUN", "reason": "no git repository yet"}
write_json(ws.approvals / "export_precheck.json", {k: v for k, v in rep.items() if k != "files"} | {"history_scan": {k: v for k, v in hist.items() if k != "hits"} | {"n_hits": len(hist.get("hits", []))}})
write_json(ws.approvals / "export_manifest.json", rep)
print(f"content precheck: {rep['status']} files={rep['n_files']} bytes={rep['total_bytes']} export_sha256={rep['export_sha256'][:16]}...")
for p in rep["problems"][:30]: print("  -", p)
print(f"history scan: {hist['status']} objects={hist.get('n_objects')} hits={len(hist.get('hits', []))}")
led.append("I7_export_precheck", rep["status"], "; ".join(rep["problems"][:5]), export_sha256=rep["export_sha256"], history_scan=hist["status"])
sys.exit(0 if rep["status"] == "PASS" and hist["status"] in ("PASS", "NOT_RUN") else 1)
