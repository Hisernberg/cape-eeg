#!/usr/bin/env python3
"""Notebook-05 backend: one-time locked test evaluation of the frozen final fits.

Runs the final-test gate, produces evaluator-role predictions, calibrates (calibration-T),
freezes referral thresholds (calibration-P), scores the locked test once, bootstraps the primary
difference, runs strata / robustness / evidence / latency analyses, and exports sanitized
aggregate tables to results/aggregate/. Per-row artefacts stay in private/evaluation/<protocol_hash>/.
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, torch
from cape_eeg.paths import resolve_workspace
from cape_eeg.status import write_json, read_json, Ledger, utc_now
from cape_eeg.contracts import CLASS_NAMES, LABELS
from cape_eeg.data.cache import CacheReader
from cape_eeg.model import DESCRIPTIONS
from cape_eeg.evaluation import pipeline as PL

ap = argparse.ArgumentParser(); ap.add_argument("--skip-heavy", action="store_true", help="skip robustness/evidence/latency (debug only)")
a = ap.parse_args()
ws = resolve_workspace(); led = Ledger(ws.provenance / "ledger.jsonl"); log = print
lock = read_json(ws.manifests / "protocol_lock.json")
if lock is None: sys.exit("BLOCKED: no protocol lock")
out_dir = ws.evaluation / lock["protocol_hash"]; out_dir.mkdir(parents=True, exist_ok=True)
reader = CacheReader(ws.cache / lock["preprocess_hash"])
runs = PL.final_runs(ws, lock)
g = PL.gate(ws, lock, runs, log); write_json(out_dir / "gate.json", g)
if g["status"] != "PASS": led.append("I6_final_evaluation", "BLOCKED", "; ".join(g["problems"])); sys.exit("BLOCKED")
PL.produce_predictions(ws, runs, log)
res, preds = PL.calibrate_and_score(ws, lock, runs, out_dir, log)
res["primary"] = PL.primary_and_secondary(lock, preds, out_dir, log)
res["strata"] = PL.strata_analysis(lock, preds, reader, out_dir, log)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not a.skip_heavy:
    res["robustness_evidence"] = PL.robustness_and_evidence(ws, lock, runs, reader, out_dir, device, log)
    res["latency"] = PL.latency_benchmark(ws, lock, runs, reader, device, out_dir, log)
res["resources"] = PL.resource_ledger(ws, runs, reader, out_dir)
res["protocol"] = lock; res["timestamp"] = utc_now(); res["gate"] = g
write_json(out_dir / "summary.json", res)

# ---------------------------------------------------------------- sanitized aggregate export (no identifiers)
pub = ws.results_public; pub.mkdir(parents=True, exist_ok=True)
split = read_json(ws.manifests / "split_summary.json"); src = read_json(ws.provenance / "source_manifest.json"); schema = read_json(ws.provenance / "schema_report.json")
t1 = [{"partition": p, **{k: split[p][k] for k in ["rows", "patients", "components", "eegs", "spectrograms", "row_share"]}, **{f"share_{c}": split[p]["class_vote_share"][l] for c, l in zip(CLASS_NAMES, LABELS)}, **{f"majority_{c}": split[p]["unique_majority_counts"][l] for c, l in zip(CLASS_NAMES, LABELS)}, "ties": split[p]["unique_majority_counts"]["ties"]} for p in ["train", "tune", "calibration_t", "calibration_p", "test"]]
pd.DataFrame(t1).to_csv(pub / "table1_cohort_partitions.csv", index=False)
rows = []
for cid, mr in res["methods"].items():
    for seed, s in mr["seeds"].items():
        for cal in ["raw", "calibrated"]:
            x = s[cal]; rows.append({"method": cid, "description": DESCRIPTIONS[cid], "seed": seed, "calibration": cal, "temperature": s["temperature"]["T"], "patient_kl": x["patient_kl"], "component_kl": x["component_kl"], "row_kl": x["row_kl"], "soft_ce": x["soft_ce"], "expected_brier": x["expected_brier"], "soft_squared_error": x["soft_squared_error"],
                         "macro_auroc": x["hard"]["macro_auroc"], "macro_auprc": x["hard"]["macro_auprc"], "balanced_accuracy": x["hard"]["balanced_accuracy"], "macro_f1": x["hard"]["macro_f1"], "accuracy": x["hard"]["accuracy"], "mcc": x["hard"]["mcc"], "cohen_kappa": x["hard"]["cohen_kappa"],
                         "soft_top_label_ece": x["soft_top_label_ece"]["ece"], "majority_label_ece": x["majority_label_ece"]["ece"], "kl_votes_10plus": x["kl_by_vote_stratum"]["10+"]["patient_kl"], "kl_votes_1": x["kl_by_vote_stratum"]["1"]["patient_kl"],
                         "disagreement_spearman": ((x.get("disagreement") or {}).get("overall") or {}).get("spearman"), "gate_mean": x.get("gate_mean"), "n_rows": x["n_rows"], "n_patients": x["n_patients"], "params": s["parameters"]["total"]})
t2 = pd.DataFrame(rows); t2.to_csv(pub / "table2_test_metrics_by_seed.csv", index=False)
agg = t2.groupby(["method", "calibration"]).agg(patient_kl_mean=("patient_kl", "mean"), patient_kl_sd=("patient_kl", "std"), row_kl_mean=("row_kl", "mean"), row_kl_sd=("row_kl", "std"), macro_auroc_mean=("macro_auroc", "mean"), balanced_accuracy_mean=("balanced_accuracy", "mean"), soft_top_label_ece_mean=("soft_top_label_ece", "mean"), expected_brier_mean=("expected_brier", "mean"), kl_votes_10plus_mean=("kl_votes_10plus", "mean"), params=("params", "first")).reset_index()
pr = res["primary"]["primary_calibrated"]; agg["primary_delta_vs_comparator"] = None; agg["primary_ci_low"] = None; agg["primary_ci_high"] = None; agg["primary_decision"] = None
m = (agg.method == lock["candidate"]) & (agg.calibration == "calibrated"); agg.loc[m, ["primary_delta_vs_comparator", "primary_ci_low", "primary_ci_high", "primary_decision"]] = [pr["point_estimate"], pr["ci_low"], pr["ci_high"], pr["decision"]]
agg.to_csv(pub / "table2_test_metrics_summary.csv", index=False)
dev = pd.read_csv(ws.evaluation / "development" / "development_table.csv"); dev.to_csv(pub / "table3_development_and_ablations.csv", index=False)
t4 = []
for name, s in res["strata"].items(): t4.append({"analysis": "stratum", "item": name, **{k: s.get(k) for k in ["delta", "ci_low", "ci_high", "n_rows", "n_groups", "kl_candidate", "kl_comparator", "status", "low_support"]}})
if "robustness_evidence" in res:
    for cid in [lock["candidate"], lock["comparator"]]:
        for k, r in res["robustness_evidence"][cid]["robustness"].items(): t4.append({"analysis": f"robustness_{cid}", "item": k, "delta": r.get("delta_kl"), "ci_low": r.get("ci_low"), "ci_high": r.get("ci_high"), "kl_candidate": r.get("kl"), "status": r.get("status"), "extra": json.dumps({x: r.get(x) for x in ["label_flip_fraction", "js_distance", "majority_error", "severity", "units"]})})
    for k, r in res["robustness_evidence"]["evidence"]["view_region_deletion"].items(): t4.append({"analysis": "evidence_deletion", "item": k, "delta": r["delta_kl"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "kl_candidate": r["kl"], "status": "PASS", "extra": json.dumps({"js": r["js_distance"], "flip": r["label_flip_fraction"]})})
for cid, mr in res["methods"].items():
    s = mr["seeds"][lock["deployment_seed"]]
    for cov, r in s["referral"].items(): t4.append({"analysis": f"referral_{cid}_seed{lock['deployment_seed']}", "item": f"coverage_{cov}", "status": "PASS", "extra": json.dumps(r)})
pd.DataFrame(t4).to_csv(pub / "table4_strata_robustness_referral.csv", index=False)
r5 = res["resources"]; t5 = [{"item": "gpu_hours_total", "value": r5["gpu_hours_total"]}, {"item": "gpu_hours_ceiling", "value": 12.0}, {"item": "n_gpu_jobs", "value": r5["n_gpu_jobs"]}, {"item": "failed_or_incomplete_jobs", "value": r5["failed_or_incomplete_jobs"]}, {"item": "active_cache_bytes", "value": r5["cache"]["active_cache_bytes"]}, {"item": "alt_uniform_cache_bytes", "value": r5["cache"]["alt_uniform_bytes"]}, {"item": "source_bytes", "value": r5["source_bytes"]}, {"item": "conversion_seconds", "value": r5["conversion"]["elapsed_seconds"]}]
for k, v in r5["gpu_hours_by_stage"].items(): t5.append({"item": f"gpu_hours_{k}", "value": v})
for k, f in r5["final_fits"].items(): t5 += [{"item": f"final_fit_{k}_wall_min", "value": f["wall_min"]}, {"item": f"final_fit_{k}_peak_cuda_gib", "value": f["peak"]["cuda_alloc_gib"]}, {"item": f"final_fit_{k}_peak_rss_gib", "value": f["peak"]["rss_gib"]}, {"item": f"final_fit_{k}_params", "value": f["params"]}]
if "latency" in res:
    for cid in [lock["candidate"], lock["comparator"]]:
        for b in ["batch_1", "batch_32", "raw_to_prediction_batch_1"]: t5 += [{"item": f"latency_{cid}_{b}_median_ms", "value": res["latency"][cid][b]["median_ms"]}, {"item": f"latency_{cid}_{b}_p95_ms", "value": res["latency"][cid][b]["p95_ms"]}]
pd.DataFrame(t5).to_csv(pub / "table5_resources.csv", index=False)
public = {k: res[k] for k in ["primary", "strata", "resources", "protocol", "timestamp", "gate"] if k in res}
public["methods"] = {cid: {"mean_over_seeds": mr["mean_over_seeds"], "seeds": {s: {k: v for k, v in x.items() if k in ("temperature", "temperature_effect", "raw", "calibrated", "referral_thresholds", "referral", "risk_coverage_kl", "risk_coverage_majority_error", "alt_score_aurc", "parameters", "wall_seconds", "peak")} for s, x in mr["seeds"].items()}} for cid, mr in res["methods"].items()}
if "robustness_evidence" in res: public["robustness_evidence"] = res["robustness_evidence"]
if "latency" in res: public["latency"] = res["latency"]
write_json(pub / "final_evaluation_summary.json", public)
led.append("I6_final_evaluation", "PASS", "locked test evaluated once", protocol_hash=lock["protocol_hash"], primary=pr["point_estimate"], ci=[pr["ci_low"], pr["ci_high"]], decision=pr["decision"])
print("exported aggregate tables to", pub)
