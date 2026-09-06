#!/usr/bin/env python3
"""Notebook-02/03 backend: development comparison on tune, comparator selection, protocol lock.

Reads every PASS development run's tune predictions (best checkpoint), builds the development
table, selects the comparator between B2 and B3 on tune only, estimates paired-loss variability
for the feasibility scenarios, chooses the referral score for P, and writes protocol_lock.json.
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from cape_eeg.paths import resolve_workspace
from cape_eeg.status import write_json, read_json, Ledger, utc_now, git_commit
from cape_eeg.contracts import stable_hash
from cape_eeg import metrics as M
from cape_eeg.evaluation.report import load_predictions, align, summarize, per_group_losses
from cape_eeg.evaluation.bootstrap import mde_paired
from cape_eeg.evaluation.robustness import CONDITIONS
from cape_eeg.model import DESCRIPTIONS

ap = argparse.ArgumentParser(); ap.add_argument("--lock", action="store_true", help="write protocol_lock.json (irreversible for this study)")
ap.add_argument("--final-epochs", type=int, default=None)
ap.add_argument("--schedule-epochs", type=int, default=None, help="restrict comparator/candidate selection to runs trained with this complete schedule (the common shorter epoch-count pilot) and freeze it as the final schedule")
a = ap.parse_args()
ws = resolve_workspace(); led = Ledger(ws.provenance / "ledger.jsonl")
rows = []; preds = {}
for run_dir in sorted(ws.runs.glob("dev_*")):
    st = read_json(run_dir / "status.json"); cfg = read_json(run_dir / "config.resolved.json")
    if st is None or cfg is None: continue
    rec = {"run_id": run_dir.name, "config": cfg["config_id"], "seed": cfg["seed"], "epochs": cfg["epochs"], "lr_mult": round(cfg["training"]["compact_lr"] / 1e-3, 3),
           "params": cfg["parameters"]["total"], "status": st["status"], "reason": st.get("reason", ""), "best_epoch": (st.get("best") or {}).get("epoch"),
           "tune_patient_kl_best": (st.get("best") or {}).get("tune_patient_kl"), "wall_min": round(st.get("wall_seconds", 0) / 60, 1),
           "examples_seen": st.get("examples_seen"), "peak_cuda_gib": round((st.get("peak") or {}).get("cuda_alloc_gib", 0), 2), "peak_rss_gib": round((st.get("peak") or {}).get("rss_gib", 0), 2),
           "tag": cfg["training"].get("tag", "")}
    p = run_dir / "predictions" / "tune_full_best.parquet"
    if st["status"] == "PASS" and p.exists():
        df = load_predictions(p); s = summarize(df); preds[(cfg["config_id"], cfg["seed"], run_dir.name)] = df
        rec.update(tune_patient_kl=s["patient_kl"], tune_row_kl=s["row_kl"], tune_macro_auroc=s["hard"]["macro_auroc"], tune_balanced_acc=s["hard"]["balanced_accuracy"],
                   tune_soft_ece=s["soft_top_label_ece"]["ece"], gate_mean=s.get("gate_mean"), tune_disagreement_spearman=(s.get("disagreement") or {}).get("overall", {}).get("spearman") if s.get("disagreement") else None,
                   kl_vote10plus=s["kl_by_vote_stratum"]["10+"]["patient_kl"])
    rows.append(rec)
cpu = None
for d in sorted(ws.runs.glob("dev_cpu_baselines_*")):
    cpu = read_json(d / "results.json")
    for b in ["B0", "B1"]:
        rows.append({"run_id": d.name, "config": b, "seed": 0, "status": "PASS", "tune_patient_kl": cpu[b]["tune"]["patient_kl"], "tune_row_kl": cpu[b]["tune"]["row_kl"], "params": 0 if b == "B0" else cpu["n_features"] * 6 + 6})
table = pd.DataFrame(rows).sort_values(["config", "seed"]); table["description"] = table.config.map(DESCRIPTIONS)
out_dir = ws.evaluation / "development"; out_dir.mkdir(parents=True, exist_ok=True)
table.to_csv(out_dir / "development_table.csv", index=False)
print(table[["config", "seed", "status", "epochs", "lr_mult", "best_epoch", "tune_patient_kl", "tune_row_kl", "tune_macro_auroc", "wall_min", "params"]].to_string(index=False))
# comparator selection (tune only) among PASS B2/B3 default-lr runs
sched = (table.epochs == a.schedule_epochs) if a.schedule_epochs else (table.epochs.notna())
ok = table[(table.status == "PASS") & table.config.isin(["B2", "B3"]) & table.tune_patient_kl.notna() & sched]
best_by = ok.sort_values("tune_patient_kl").groupby("config").head(1)
comparator = best_by.sort_values("tune_patient_kl").iloc[0] if len(best_by) else None
# candidate best P run
pk = table[(table.status == "PASS") & (table.config == "P") & table.tune_patient_kl.notna() & sched].sort_values("tune_patient_kl")
best_p = pk.iloc[0] if len(pk) else None
lock = {"created": utc_now(), "git_commit": git_commit(ws.repo), "split_hash": read_json(ws.manifests / "split_summary.json")["split_hash"],
        "preprocess_hash": sorted(ws.cache.glob("*/manifest.json"))[-1].parent.name, "n_development_configs": int(table[table.config.isin(["B2","B3","A1","A2","P","P_MSF"])].drop_duplicates(["config","lr_mult","epochs"]).shape[0])}
if comparator is not None and best_p is not None:
    lock["comparator"] = comparator.config; lock["comparator_run"] = comparator.run_id; lock["comparator_tune_patient_kl"] = float(comparator.tune_patient_kl)
    lock["candidate"] = "P"; lock["candidate_run"] = best_p.run_id; lock["candidate_tune_patient_kl"] = float(best_p.tune_patient_kl)
    lock["candidate_lr_mult"] = float(best_p.lr_mult); lock["comparator_lr_mult"] = float(comparator.lr_mult)
    be = [int(x) for x in [best_p.best_epoch, comparator.best_epoch] if x == x and x is not None]
    lock["final_epochs"] = int(a.final_epochs or a.schedule_epochs or max(be))
    lock["schedule_matched_selection"] = bool(a.schedule_epochs)
    lock["selection_rule"] = "best tune patient-KL among PASS runs trained with the complete final schedule" if a.schedule_epochs else "best tune patient-KL among PASS runs; final epochs = max of the selected best epochs"
    # paired loss SD on tune between P and comparator -> feasibility scenarios
    al = align({"p": preds[("P", int(best_p.seed), best_p.run_id)], "b": preds[(comparator.config, int(comparator.seed), comparator.run_id)]})
    gp, mp = per_group_losses(al["p"]); gb, mb = per_group_losses(al["b"]); delta = mp - mb
    n_test = read_json(ws.manifests / "split_summary.json")["test"]["components"]
    lock["development_paired_sd"] = float(delta.std(ddof=1)); lock["development_paired_mean"] = float(delta.mean()); lock["n_tune_components"] = int(len(delta))
    lock["mde_test_at_dev_sd"] = float(mde_paired(delta.std(ddof=1), n_test)); lock["n_test_components"] = int(n_test)
    lock["mde_scenarios"] = {str(sd): float(mde_paired(sd, n_test)) for sd in (0.10, 0.20, 0.30)}
    # referral score choice for P on tune: lowest AURC over [0.5,1]
    dfp = al["p"]; v = dfp[[f"v{k}" for k in range(6)]].to_numpy(); q = M.votes_to_targets(v); pp = dfp[[f"p{k}" for k in range(6)]].to_numpy(); kl = M.kl_rows(q, pp)
    scores = {"entropy": M.predictive_entropy(pp), "one_minus_max": M.one_minus_max_prob(pp)}
    if np.isfinite(dfp.d_hat.to_numpy()).all(): scores["d_hat"] = dfp.d_hat.to_numpy()
    aurc = {k: float(M.risk_coverage(s, kl)["aurc"]) for k, s in scores.items()}
    lock["referral_score_aurc_tune"] = aurc; lock["referral_score"] = min(aurc, key=aurc.get)
    # entropy quantile strata frozen on development data
    ent = -(q * np.log(np.clip(q, 1e-12, None))).sum(1); lock["target_entropy_tertiles"] = [float(x) for x in np.quantile(ent, [1/3, 2/3])]
lock.update(primary_metric="three_seed_mean_component_KL_difference (P - comparator), natural log, eps 1e-7, 2000 paired component bootstrap, seed 20260907", primary_on="temperature_calibrated_predictions (raw reported as secondary)",
            final_seeds=[101, 202, 303], deployment_seed=101, final_refit_partitions=["train", "tune"], temperature_partition="calibration_t", referral_partition="calibration_p",
            referral_coverages=[0.80, 0.90, 0.95], vote_strata=["1", "2-4", "5-9", "10+"], corruption_conditions=[c["id"] for c in CONDITIONS], robustness_subset={"max_rows": 1024, "min_patients": 100, "seed": 20260907},
            figures=[f"V{i:02d}" for i in range(1, 21)], exploratory_configs=["P_MSF"], notes="Hypothesis: superiority of P over the tune-selected comparator on patient/component-mean KL. Interval crossing zero is inconclusive. No noninferiority margin.")
lock["protocol_hash"] = stable_hash({k: v for k, v in lock.items() if k not in ("created", "protocol_hash")})
write_json(out_dir / "protocol_candidate.json", lock)
print(json.dumps({k: lock.get(k) for k in ["comparator", "candidate_run", "final_epochs", "development_paired_sd", "mde_test_at_dev_sd", "referral_score", "referral_score_aurc_tune", "n_development_configs"]}, indent=1))
if a.lock:
    if (ws.manifests / "protocol_lock.json").exists():
        sys.exit("protocol_lock.json already exists; a lock is never overwritten")
    write_json(ws.manifests / "protocol_lock.json", lock); led.append("I4_protocol_lock", "PASS", "protocol frozen", protocol_hash=lock["protocol_hash"])
    print("PROTOCOL LOCKED", lock["protocol_hash"])
