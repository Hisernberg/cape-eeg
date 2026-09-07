#!/usr/bin/env python3
"""Paper-specific figures: vector PDF, large fonts, one message per panel. Reads only aggregate/private
evaluation artefacts of the locked protocol; writes paper/figures/fig_*.pdf (no identifiers)."""
from __future__ import annotations
import json, sys, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]            # workspace root (holds private/)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
OUT = Path(__file__).resolve().parent / "figures"; OUT.mkdir(exist_ok=True)
S = json.load(open(ROOT / "private/evaluation/380269a4a6db7334/summary.json"))
DEV = pd.read_csv(ROOT / "private/evaluation/development/development_table.csv")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11.5, "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
CP, CB, CG, CR = "#1F77B4", "#E8710A", "#2CA02C", "#7F7F7F"
W1 = 3.45  # single column width (in)

# ---------------------------------------------------------------- 1. primary
d = pd.read_csv(ROOT / "private/figures_private/v09_per_component_delta_calibrated.csv")
pr = S["primary"]["primary_calibrated"]
fig, ax = plt.subplots(figsize=(W1, 2.3))
ax.hist(d.delta_kl, bins=np.arange(-1.7, 1.75, 0.1), color=CP, alpha=0.85, edgecolor="white")
ax.axvline(0, color="k", ls="--", lw=1.2); ax.axvspan(pr["ci_low"], pr["ci_high"], color=CB, alpha=0.35, label="95% CI of the mean")
ax.axvline(pr["point_estimate"], color=CB, lw=2, label=f"mean $\\Delta$ = {pr['point_estimate']:+.3f}")
ax.set_xlabel("Paired $\\Delta$ patient-mean KL (P $-$ B3), nats"); ax.set_ylabel("Patients")
ax.set_title(f"{pr['n_groups']} test patients, 3 seeds; {100*(d.delta_kl<0).mean():.0f}% favour P", fontsize=10.5)
ax.legend(loc="upper left", frameon=False); fig.tight_layout(); fig.savefig(OUT / "fig_primary.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- 2. ablation ladder (dev, 3 seeds, frozen schedule) + test
def seed_vals(cfg, ep):
    t = DEV[(DEV.config == cfg) & (DEV.epochs == ep) & (DEV.status == "PASS")].sort_values("seed"); return dict(zip(t.seed.astype(int), t.tune_patient_kl))
def paired(a, b, ep=3):
    va, vb = seed_vals(a, ep), seed_vals(b, ep); ks = sorted(set(va) & set(vb)); dd = np.array([va[k] - vb[k] for k in ks])
    return float(dd.mean()), (float(dd.std(ddof=1)) if len(dd) > 1 else np.nan), len(dd)
rows = []
for name, a, b in [("Foveated bins (A1 $-$ B2)", "A1", "B2"), ("Learned gate (A2 $-$ A1)", "A2", "A1"), ("Disagreement loss (P $-$ A2)", "P", "A2"), ("All three (P $-$ B2)", "P", "B2"),
                   ("ImageNet pretraining (B3 $-$ B3 scratch)", "B3", "B3S"), ("Width 1.0 vs 0.5 (B3 $-$ B3 half)", "B3", "B3H"), ("Pretrained B3 vs control (B3 $-$ B2)", "B3", "B2")]:
    m, sd, n = paired(a, b); rows.append((name, m, sd, n, "dev"))
ex = S["primary"]
rows += [("Multi-scale block, test (P+MSF $-$ P)", ex["exploratory_P_MSF_vs_P_calibrated"]["point_estimate"], ex["exploratory_P_MSF_vs_P_calibrated"]["ci_low"], ex["exploratory_P_MSF_vs_P_calibrated"]["ci_high"], "test"),
         ("Primary, test (P $-$ B3)", pr["point_estimate"], pr["ci_low"], pr["ci_high"], "test")]
fig, ax = plt.subplots(figsize=(W1, 3.6)); y = np.arange(len(rows))[::-1]
for yi, (name, v, a1, a2, kind) in zip(y, rows):
    if kind == "dev":
        ax.errorbar(v, yi, xerr=(0 if np.isnan(a1) else a1), fmt="o", mfc="white", mec=CP, color=CP, mew=2, ms=7, capsize=3, lw=1.5)
    else:
        ax.errorbar(v, yi, xerr=[[v - a1], [a2 - v]], fmt="D", color=CB, ms=7, capsize=3, lw=2)
ax.axvline(0, color="k", ls="--", lw=1); ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
lim = max(abs(v) + (0 if np.isnan(a1) else a1) if k == "dev" else max(abs(a1), abs(a2)) for _, v, a1, a2, k in rows) * 1.15; ax.set_xlim(-lim, lim); ax.set_xticks([t for t in np.arange(-0.4, 0.41, 0.2) if abs(t) <= lim]); ax.set_xticklabels([f"{t:+.1f}" if abs(t) > 1e-9 else "0" for t in ax.get_xticks()], fontsize=9)
ax.set_xlabel("$\\Delta$ patient-mean KL (negative = first term better)", fontsize=9.5)
ax.plot([], [], "o", mfc="white", mec=CP, mew=2, label="development (tune): mean $\\pm$ SD, 3 seeds, 3-epoch schedule"); ax.plot([], [], "D", color=CB, label="locked test: 3 seeds, 95% patient-bootstrap CI")
ax.legend(loc="upper center", bbox_to_anchor=(0.3, -0.18), frameon=False, fontsize=7.5, ncol=1); fig.tight_layout(); fig.savefig(OUT / "fig_ablation.pdf", bbox_inches="tight"); plt.close(fig)
ab = pd.DataFrame([{"comparison": r[0].replace("$", ""), "mean_delta": r[1], "sd_or_ci_low": r[2], "n_or_ci_high": r[3], "kind": r[4]} for r in rows]); ab.to_csv(OUT.parent / "ablation_ladder_values.csv", index=False)

# ---------------------------------------------------------------- 3. learning curves
fig, ax = plt.subplots(figsize=(W1, 2.6))
styles = {"B2": (CR, "-"), "A1": ("#9467BD", "-"), "A2": ("#8C564B", "-"), "P": (CP, "-"), "P_MSF": ("#17BECF", "-"), "B3": (CB, "-"), "B3S": ("#D62728", "-"), "B3H": ("#BCBD22", "-")}
for rd in sorted(p for p in glob.glob(str(ROOT / "private/runs/dev_*")) if Path(p, "config.resolved.json").exists()):
    cfg = json.load(open(rd + "/config.resolved.json")); ev = [json.loads(l) for l in open(rd + "/events.jsonl")]
    if cfg["seed"] != 101 or cfg["config_id"] not in styles or cfg["config_id"] in ("B3S", "B3H") or cfg.get("patient_fraction", 1.0) < 1.0: continue
    c, ls = styles[cfg["config_id"]]; lab = cfg["config_id"].replace("_", "+") + (" (3 ep)" if cfg["epochs"] == 3 else "")
    ax.plot([e["epoch"] for e in ev], [e["tune_patient_kl"] for e in ev], ls if cfg["epochs"] == 12 else "--", color=c, lw=2 if cfg["config_id"] in ("P", "B3") else 1.3, marker="o" if cfg["epochs"] == 3 else None, ms=5, label=lab)
ax.set_xlabel("Epoch"); ax.set_ylabel("Tune patient-mean KL"); ax.set_ylim(0.85, 1.25); ax.set_xticks(range(1, 13))
ax.legend(ncol=2, frameon=False, fontsize=8, loc="upper left"); fig.tight_layout(); fig.savefig(OUT / "fig_learning.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- 4. calibration + risk-coverage
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"wspace": 0.4})
ax = axs[0]
for cid, col in [("P", CP), ("B3", CB)]:
    for cal, ls, lab in [("raw", "--", "raw"), ("calibrated", "-", "calibrated")]:
        e = S["methods"][cid]["seeds"]["101"][cal]["soft_top_label_ece"]
        ax.plot(e["bin_mean_confidence"], e["bin_mean_observed"], ls, color=col, marker="o", ms=4, lw=1.8, label=f"{cid} {lab} (ECE {e['ece']:.3f})")
ax.plot([0.2, 1], [0.2, 1], color="k", lw=0.8); ax.set_xlabel("Mean predicted confidence"); ax.set_ylabel("Observed vote mass, predicted class"); ax.set_title("Reliability (seed 101, test)", fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax = axs[1]
for cid, col in [("P", CP), ("B3", CB)]:
    rc = S["methods"][cid]["seeds"]["101"]["risk_coverage_kl"]
    ax.plot(rc["coverage_grid"], rc["risk_patient_weighted"], color=col, lw=2, label=f"{cid} (AURC {rc['aurc']:.3f})")
    ref = S["methods"][cid]["seeds"]["101"]["referral"]["0.9"]
    ax.plot(ref["actual_coverage"], ref["accepted_risk_patient_weighted"], "s", color=col, ms=7, mec="k")
ax.set_xlabel("Acceptance coverage (entropy score)"); ax.set_ylabel("Accepted-case patient-mean KL"); ax.set_title("Risk–coverage (squares: frozen 90% point)", fontsize=10)
ax.legend(frameon=False, fontsize=9, loc="upper left"); fig.tight_layout(); fig.savefig(OUT / "fig_calibration.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- 5. evidence deletion
ev = S["robustness_evidence"]["evidence"]; vr = ev["view_region_deletion"]
names = [("local_removed", "local 50 s view"), ("context_removed", "context 600 s view"), ("region_removed_LL", "region LL"), ("region_removed_RL", "region RL"), ("region_removed_LP", "region LP"), ("region_removed_RP", "region RP")]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.45})
ax = axs[0]; y = np.arange(len(names))[::-1]
for yi, (k, lab) in zip(y, names):
    v = vr[k]; ax.barh(yi, v["delta_kl"], color=CP if k.startswith("region") else CB, alpha=0.9)
    ax.errorbar(v["delta_kl"], yi, xerr=[[v["delta_kl"] - v["ci_low"]], [v["ci_high"] - v["delta_kl"]]], fmt="none", ecolor="k", capsize=3)
ax.set_yticks(y); ax.set_yticklabels([n[1] + " removed" for n in names], fontsize=9.5); ax.set_xlabel("$\\Delta$ patient-mean KL vs intact"); ax.set_title("Deletion on the full test (P, seed 101)", fontsize=11)
ax = axs[1]; cur = ev["ranked_vs_random_deletion"]
ax.plot(cur["fractions"], cur["ranked"]["kl"], "-o", color=CB, lw=2, label="saliency-ranked order"); ax.plot(cur["fractions"], cur["random"]["kl"], "--s", color=CR, lw=2, label="random order")
ax.set_xlabel("Fraction of local blocks removed"); ax.set_ylabel("Mean KL of 48 cases"); ax.set_title("Ranked vs random deletion", fontsize=11); ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig_evidence.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- 6. robustness grouped bars
conds = [("region_missing_LL", "LL region\nremoved"), ("time_mask_10pct", "10% time\nmask"), ("gain_shift_mild", "gain\n+0.5"), ("gain_shift_strong", "gain\n+1.5"), ("stft_window_512", "STFT\n512")]
fig, ax = plt.subplots(figsize=(W1, 2.3)); x = np.arange(len(conds)); w = 0.38
for i, (cid, col) in enumerate([("P", CP), ("B3", CB)]):
    r = S["robustness_evidence"][cid]["robustness"]
    v = [r[c]["delta_kl"] for c, _ in conds]; lo = [r[c]["delta_kl"] - r[c]["ci_low"] for c, _ in conds]; hi = [r[c]["ci_high"] - r[c]["delta_kl"] for c, _ in conds]
    ax.bar(x + (i - 0.5) * w, v, w, color=col, yerr=[lo, hi], capsize=3, label=cid)
ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels([c[1] for c in conds], fontsize=8.5); ax.set_ylabel("$\\Delta$ patient-mean KL vs clean")
ax.set_title("Stress suite (777 windows, 391 patients)", fontsize=10); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT / "fig_robustness.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- 7. alignment (synthetic impulse through the real encoder)
from cape_eeg.data.spectral import RawSpectralEncoder
from cape_eeg.data.montage import EXPECTED_SOURCE_COLUMNS
from cape_eeg.contracts import foveated_time_edges, uniform_time_edges
enc = RawSpectralEncoder(EXPECTED_SOURCE_COLUMNS); t = np.arange(10000) / 200; x = np.zeros((10000, 20), np.float32)
burst = (t >= 22) & (t < 28); x[burst, :] = np.sin(2 * np.pi * 3 * t[burst])[:, None] * 50
for c in [1, 4, 5, 12, 15, 16]: x[:, c] = 0
r = enc.encode(x, np.ones_like(x, bool)); fe, ue = foveated_time_edges(), uniform_time_edges(); f = enc.frequency_edges
fig, axs = plt.subplots(2, 1, figsize=(W1, 3.4), sharex=True)
for ax, vals, edges, title in [(axs[0], r["foveated"][0], fe, "Foveated: 8 + 16 + 8 bins"), (axs[1], r["uniform"][0], ue, "Uniform: 32 equal bins (same bytes)")]:
    ax.pcolormesh(edges, f, vals, cmap="magma", shading="flat"); ax.set_ylim(0.5, 12); ax.set_ylabel("Hz"); ax.set_title(title, fontsize=9.5)
    ax.axvline(20, color="cyan", lw=1.5); ax.axvline(30, color="cyan", lw=1.5)
    for e_ in edges: ax.axvline(e_, color="white", lw=0.3, alpha=0.6)
axs[1].set_xlabel("Seconds in the 50 s window (3 Hz burst, 22–28 s)", fontsize=9); fig.tight_layout(); fig.savefig(OUT / "fig_alignment.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- 8. pareto
lat = S["latency"]; fig, ax = plt.subplots(figsize=(W1, 2.5))
for cid, col, mk in [("P", CP, "o"), ("B3", CB, "s"), ("P_MSF", "#17BECF", "^")]:
    kl = S["methods"][cid]["mean_over_seeds"]["calibrated"]["patient_kl"]; params = S["methods"][cid]["seeds"]["101"]["parameters"]["total"]
    if cid in lat:
        for b, alpha in [("batch_1", 0.45), ("batch_32", 1.0)]:
            ax.scatter(lat[cid][b]["median_ms"], kl, s=40 + params / 8000, color=col, alpha=alpha, marker=mk, edgecolor="k", label=f"{cid} ({params/1e3:.0f}k params), {b.replace('_', ' ')}")
ax.set_xlabel("Median inference latency per batch (ms, GB10)"); ax.set_ylabel("Test patient-mean KL"); ax.set_xlim(1.0, 3.2)
ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0.0, -0.25), ncol=2); fig.tight_layout(); fig.savefig(OUT / "fig_pareto.pdf", bbox_inches="tight"); plt.close(fig)
print("paper figures written:", sorted(p.name for p in OUT.glob("fig_*.pdf")))
