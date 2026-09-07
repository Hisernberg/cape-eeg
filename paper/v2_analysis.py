#!/usr/bin/env python3
"""Post-lock development analyses for the revision (tune partition only; test untouched):
byte-budget sweep, Dirichlet-multinomial head, patient-count learning curve, inference measurements.
Writes results/aggregate/table6_*.csv, paper/figures/fig_bytesweep.pdf, fig_learningcurve.pdf and paper/numbers_v2.tex."""
from __future__ import annotations
import json, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]; REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO / "src"))
from cape_eeg import metrics as M
OUT = REPO / "paper" / "figures"; PUB = REPO / "results" / "aggregate"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
CP, CB, CR = "#1F77B4", "#E8710A", "#7F7F7F"; W1 = 3.45

rows = []
for rd in sorted(glob.glob(str(ROOT / "private/runs/dev_*"))):
    if not Path(rd, "config.resolved.json").exists(): continue
    cfg = json.load(open(rd + "/config.resolved.json")); st = json.load(open(rd + "/status.json"))
    if cfg["epochs"] != 3 or st["status"] != "PASS" or cfg["phase"] != "dev": continue
    ev = [json.loads(l) for l in open(rd + "/events.jsonl")]
    rows.append({"run": Path(rd).name, "config": cfg["config_id"], "seed": cfg["seed"], "encoding": cfg["encoding"], "time_bins": cfg.get("time_bins", 32), "patient_fraction": cfg.get("patient_fraction", 1.0),
                 "n_train_patients": cfg.get("n_train_patients"), "n_train_rows": cfg["n_train_rows"], "params": cfg["parameters"]["total"], "tune_kl_last": ev[-1]["tune_patient_kl"], "tune_rowkl_last": ev[-1]["tune_row_kl"], "wall_min": st["wall_seconds"] / 60})
t = pd.DataFrame(rows)
# runs created before n_train_patients was recorded: derive it from the fraction of the 1,170 training patients
t["n_train_patients"] = t.apply(lambda r: int(r.n_train_patients) if r.n_train_patients == r.n_train_patients and r.n_train_patients is not None else int(round(r.patient_fraction * 1170)), axis=1)
mac = []
def f(x, p=3): return f"{x:.{p}f}"
def sgn(x, p=3): return f"{x:+.{p}f}"

# ---------------------------------------------------------------- byte-budget sweep
sw = t[(t.patient_fraction == 1.0) & t.config.isin(["B2", "A1", "B2_t16", "A1_t16", "B2_t8", "A1_t8"])].copy()
sw["local_bytes"] = 4 * 64 * sw.time_bins * 2 + 4 * 64 * sw.time_bins // 8
g = sw.groupby(["encoding", "time_bins"]).agg(kl_mean=("tune_kl_last", "mean"), kl_sd=("tune_kl_last", "std"), n=("seed", "count"), local_bytes=("local_bytes", "first")).reset_index().sort_values(["time_bins", "encoding"])
pair = []
for tb in sorted(sw.time_bins.unique()):
    a = sw[(sw.encoding == "foveated") & (sw.time_bins == tb)].set_index("seed").tune_kl_last; b = sw[(sw.encoding == "uniform") & (sw.time_bins == tb)].set_index("seed").tune_kl_last
    ks = sorted(set(a.index) & set(b.index)); d = np.array([a[k] - b[k] for k in ks])
    pair.append({"time_bins": tb, "delta_fov_minus_uniform": d.mean(), "sd": d.std(ddof=1) if len(d) > 1 else np.nan, "n_seeds": len(d), "per_seed": list(np.round(d, 4))})
    mac += [f"\\newcommand{{\\dfov{['eight','sixteen','thirtytwo'][[8,16,32].index(tb)]}}}{{{sgn(d.mean())}}}", f"\\newcommand{{\\dsdfov{['eight','sixteen','thirtytwo'][[8,16,32].index(tb)]}}}{{{f(d.std(ddof=1)) if len(d) > 1 else 'n/a'}}}"]
pair = pd.DataFrame(pair); g.to_csv(PUB / "table6_byte_sweep.csv", index=False); pair.to_csv(PUB / "table6_byte_sweep_paired.csv", index=False)
for _, r in g.iterrows():
    key = ("Fov" if r.encoding == "foveated" else "Uni") + ['eight', 'sixteen', 'thirtytwo'][[8, 16, 32].index(int(r.time_bins))]
    mac += [f"\\newcommand{{\\kl{key}}}{{{f(r.kl_mean)}}}", f"\\newcommand{{\\sd{key}}}{{{f(r.kl_sd)}}}"]
fig, ax = plt.subplots(figsize=(W1, 2.5))
for enc, col, mk, lab in [("uniform", CR, "s", "uniform bins (B2 family)"), ("foveated", CP, "o", "foveated bins (A1 family)")]:
    gg = g[g.encoding == enc].sort_values("time_bins"); ax.errorbar(gg.time_bins, gg.kl_mean, yerr=gg.kl_sd, fmt="-" + mk, color=col, capsize=3, lw=1.8, ms=6, label=lab)
ax.set_xscale("log", base=2); ax.set_xticks([8, 16, 32]); ax.set_xticklabels(["8\n(12.5 KB)", "16\n(25 KB)", "32\n(49 KB)"]); ax.set_xlabel("Local time bins (local bytes per window)")
ax.set_ylabel("Tune patient-mean KL"); ax.legend(frameon=False, fontsize=8.5); ax.set_title("Byte-budget sweep (3 seeds, mean ± SD)", fontsize=10); fig.tight_layout(); fig.savefig(OUT / "fig_bytesweep.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- learning curve
lc = t[t.config.isin(["P", "B3"])].copy()
lg = lc.groupby(["config", "patient_fraction"]).agg(kl_mean=("tune_kl_last", "mean"), kl_sd=("tune_kl_last", "std"), n=("seed", "count"), n_patients=("n_train_patients", "first"), n_rows=("n_train_rows", "first")).reset_index().sort_values(["config", "patient_fraction"])
lg.to_csv(PUB / "table6_learning_curve.csv", index=False)
fig, ax = plt.subplots(figsize=(W1, 2.5))
for cid, col, mk in [("P", CP, "o"), ("B3", CB, "s")]:
    gg = lg[lg.config == cid]; ax.errorbar(gg.n_patients, gg.kl_mean, yerr=gg.kl_sd, fmt="-" + mk, color=col, capsize=3, lw=1.8, ms=6, label=f"{cid}")
    for _, r in gg.iterrows():
        mac.append(f"\\newcommand{{\\lc{cid.replace('3', 'three')}{['q', 'h', 'tq', 'full'][[0.25, 0.5, 0.75, 1.0].index(float(r.patient_fraction))]}}}{{{f(r.kl_mean)}}}")
        mac.append(f"\\newcommand{{\\lcsd{cid.replace('3', 'three')}{['q', 'h', 'tq', 'full'][[0.25, 0.5, 0.75, 1.0].index(float(r.patient_fraction))]}}}{{{f(r.kl_sd)}}}")
ax.set_xlabel("Training patients"); ax.set_ylabel("Tune patient-mean KL"); ax.legend(frameon=False); ax.set_title("Patient-count learning curve (3 seeds, 3-epoch schedule)", fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT / "fig_learningcurve.pdf", bbox_inches="tight"); plt.close(fig)
# slope of the last segment (75% -> 100%) per model, in KL per doubling of patients
for cid in ["P", "B3"]:
    gg = lg[lg.config == cid].sort_values("patient_fraction"); x = np.log2(gg.n_patients.to_numpy()); y = gg.kl_mean.to_numpy(); slope = np.polyfit(x, y, 1)[0]
    mac.append(f"\\newcommand{{\\lcslope{cid.replace('3', 'three')}}}{{{sgn(slope)}}}")

# ---------------------------------------------------------------- Dirichlet-multinomial head vs P vs A2 (tune, last-epoch predictions)
dm_rows = []
for cid in ["A2", "P", "P_DM"]:
    for _, r in t[(t.config == cid) & (t.patient_fraction == 1.0)].iterrows():
        p = ROOT / "private/runs" / r.run / "predictions" / "tune_full_last.parquet"
        if not p.exists(): continue
        df = pd.read_parquet(p); v = df[[f"v{k}" for k in range(6)]].to_numpy(); q = M.votes_to_targets(v); pp = df[[f"p{k}" for k in range(6)]].to_numpy(); kl = M.kl_rows(q, pp)
        rec = {"config": cid, "seed": r.seed, "tune_kl": float(M.patient_weighted_mean(kl, df.patient_id.to_numpy())), "aurc_entropy": float(M.risk_coverage(M.predictive_entropy(pp), kl)["aurc"])}
        if np.isfinite(df.d_hat.to_numpy()).all():
            dmm = M.disagreement_metrics(df.d_hat.to_numpy(dtype=np.float64), v); rec["spearman_d"] = dmm["overall"]["spearman"]; rec["mse_d"] = dmm["overall"]["mse"]; rec["aurc_dhat"] = float(M.risk_coverage(df.d_hat.to_numpy(dtype=np.float64), kl)["aurc"])
        dm_rows.append(rec)
dm = pd.DataFrame(dm_rows)
if len(dm):
    dg = dm.groupby("config").agg(tune_kl_mean=("tune_kl", "mean"), tune_kl_sd=("tune_kl", "std"), spearman_mean=("spearman_d", "mean"), spearman_sd=("spearman_d", "std"), aurc_entropy=("aurc_entropy", "mean"), aurc_dhat=("aurc_dhat", "mean"), n=("seed", "count")).reset_index()
    dg.to_csv(PUB / "table6_dirichlet_multinomial.csv", index=False); dm.to_csv(PUB / "table6_dirichlet_multinomial_runs.csv", index=False)
    for _, r in dg.iterrows():
        key = r.config.replace("_", "").replace("2", "two")
        mac += [f"\\newcommand{{\\dmkl{key}}}{{{f(r.tune_kl_mean)}}}", f"\\newcommand{{\\dmklsd{key}}}{{{f(r.tune_kl_sd)}}}", f"\\newcommand{{\\dmrho{key}}}{{{f(r.spearman_mean) if r.spearman_mean == r.spearman_mean else 'n/a'}}}", f"\\newcommand{{\\dmaurcE{key}}}{{{f(r.aurc_entropy)}}}", f"\\newcommand{{\\dmaurcD{key}}}{{{f(r.aurc_dhat) if r.aurc_dhat == r.aurc_dhat else 'n/a'}}}"]
    a = dm[dm.config == "P_DM"].set_index("seed").tune_kl; b = dm[dm.config == "P"].set_index("seed").tune_kl; ks = sorted(set(a.index) & set(b.index)); d = np.array([a[k] - b[k] for k in ks])
    mac += [f"\\newcommand{{\\ddmvsp}}{{{sgn(d.mean())}}}", f"\\newcommand{{\\dsddmvsp}}{{{f(d.std(ddof=1)) if len(d) > 1 else 'n/a'}}}"]

# ---------------------------------------------------------------- inference measurements
im = json.load(open(PUB / "inference_measurements.json")) if (PUB / "inference_measurements.json").exists() else None
if im:
    for cid in ["P", "B3"]:
        k = cid.replace("3", "three"); r = im[cid]
        mac += [f"\\newcommand{{\\cpuone{k}}}{{{r['cpu_threads_1_batch_1']['median_ms']:.1f}}}", f"\\newcommand{{\\cpufour{k}}}{{{r['cpu_threads_4_batch_1']['median_ms']:.1f}}}",
                f"\\newcommand{{\\ewin{k}}}{{{r['gpu_batch_32']['energy_mj_per_window_incremental']:.3f}}}", f"\\newcommand{{\\ewintot{k}}}{{{r['gpu_batch_32']['energy_mj_per_window_total']:.2f}}}",
                f"\\newcommand{{\\loadw{k}}}{{{r['gpu_batch_32']['mean_power_w']:.1f}}}", f"\\newcommand{{\\winps{k}}}{{{r['gpu_batch_32']['windows_per_s']:.0f}}}"]
    mac.append(f"\\newcommand{{\\idlew}}{{{im['idle_power_w']:.1f}}}")
(REPO / "paper" / "numbers_v2.tex").write_text("% generated by v2_analysis.py; do not edit\n" + "\n".join(mac) + "\n")
print(g.round(4).to_string(index=False)); print(pair.round(4).to_string(index=False)); print(lg.round(4).to_string(index=False))
if len(dm): print(dg.round(4).to_string(index=False))
print("macros", len(mac))
