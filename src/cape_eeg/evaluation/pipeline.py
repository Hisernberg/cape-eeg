"""Locked final evaluation pipeline (docs/04): calibration, referral, metrics, bootstrap, strata,
robustness, evidence audits, latency, resource ledger. Called once by scripts/final_evaluate.py.
All per-row artefacts stay private; aggregate tables are exported separately."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import metrics as M
from ..contracts import LABELS, CLASS_NAMES, stable_hash
from ..status import read_json, write_json, utc_now
from ..data.cache import CacheReader
from ..data.dataset import CacheDataset, select_rows
from ..data.normalization import load_normalizer, Normalizer
from ..model import build_model, count_parameters
from ..training.engine import load_weights, predict
from .report import load_predictions, align, summarize, per_group_losses, paired_primary, records_from_summary, P_COLS, V_COLS
from .calibrate import fit_temperature as _fit_temperature_raw, apply_temperature, temperature_effect_summary


def fit_temperature(p, q, groups, **kw):
    """calibrate.fit_temperature with a short 'T' alias for the fitted temperature."""
    out = _fit_temperature_raw(p, q, groups, **kw); out["T"] = out["temperature"]; return out
from .bootstrap import paired_cluster_bootstrap, cluster_bootstrap_statistic
from .robustness import CONDITIONS, select_subset, run_suite
from .evidence import view_and_region_deletion, select_cases, saliency_ranked_deletion


def _run(cmd: list[str], log):
    r = subprocess.run(cmd, capture_output=True, text=True)
    for line in (r.stdout + r.stderr).splitlines():
        if line and not any(w in line for w in ("Warning", "warn", "Found GPU", "Minimum and", "(8.0)")):
            log("   " + line)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd[:3])}")


def final_runs(ws, lock) -> dict[str, dict[int, Path]]:
    """{method_id: {seed: run_dir}} for the candidate, the comparator, and exploratory configs."""
    out = {}
    for cid in [lock["candidate"], lock["comparator"]] + lock.get("exploratory_configs", []):
        for rd in sorted(ws.runs.glob(f"final_{cid}_s*")):
            st = read_json(rd / "status.json"); cfg = read_json(rd / "config.resolved.json")
            if st and cfg and st["status"] == "PASS" and st.get("complete_schedule") and cfg["config_id"] == cid:
                out.setdefault(cid, {})[cfg["seed"]] = rd
    return out


def gate(ws, lock, runs, log) -> dict:
    """Final-test access gate (docs/04 section 14)."""
    problems = []
    for cid in [lock["candidate"], lock["comparator"]]:
        seeds = sorted(runs.get(cid, {}))
        if seeds != lock["final_seeds"]:
            problems.append(f"{cid}: complete final seeds {seeds} != {lock['final_seeds']}")
    la = read_json(ws.manifests / "leakage_audit.json")
    if la["status"] != "PASS":
        problems.append("leakage audit not PASS")
    g = {"status": "PASS" if not problems else "BLOCKED", "problems": problems, "timestamp": utc_now(), "protocol_hash": lock["protocol_hash"]}
    log(f"final-test gate: {g['status']} {problems}")
    return g


def produce_predictions(ws, runs, log):
    for cid, seeds in runs.items():
        for seed, rd in seeds.items():
            if not (rd / "predictions" / "test_full_last.parquet").exists():
                _run([sys.executable, str(ws.repo / "scripts" / "predict.py"), "--run", rd.name, "--partitions", "calibration_t", "calibration_p", "test",
                      "--role", "evaluator", "--weights", "last", "--views", "full"], log)


def calibrate_and_score(ws, lock, runs, out_dir: Path, log) -> dict:
    """Per method/seed: temperature on calibration-T, thresholds on calibration-P, metrics on test."""
    res = {"methods": {}}; records = []; preds = {}
    (out_dir / "test_predictions").mkdir(parents=True, exist_ok=True)
    for cid, seeds in runs.items():
        res["methods"][cid] = {"seeds": {}}
        for seed, rd in sorted(seeds.items()):
            cfg = read_json(rd / "config.resolved.json")
            ct = load_predictions(rd / "predictions" / "calibration_t_full_last.parquet"); cp = load_predictions(rd / "predictions" / "calibration_p_full_last.parquet"); te = load_predictions(rd / "predictions" / "test_full_last.parquet")
            qT = M.votes_to_targets(ct[V_COLS].to_numpy()); T = fit_temperature(ct[P_COLS].to_numpy(), qT, ct.patient_id.to_numpy())
            p_raw = te[P_COLS].to_numpy(dtype=np.float64); p_cal = apply_temperature(p_raw, T["T"])
            # referral score and thresholds on calibration-P (calibrated probabilities)
            def score_of(df, p):
                s = lock["referral_score"]
                if s == "entropy": return M.predictive_entropy(p)
                if s == "one_minus_max": return M.one_minus_max_prob(p)
                return df.d_hat.to_numpy(dtype=np.float64) if np.isfinite(df.d_hat.to_numpy()).all() else M.predictive_entropy(p)
            sp = score_of(cp, apply_temperature(cp[P_COLS].to_numpy(), T["T"]))
            thresholds = {str(c): float(M.threshold_for_coverage(sp, c)) for c in lock["referral_coverages"]}
            st = score_of(te, p_cal); qte = M.votes_to_targets(te[V_COLS].to_numpy()); kl_cal = M.kl_rows(qte, p_cal)
            lab, ties = M.unique_majority_labels(te[V_COLS].to_numpy()); maj_err = np.where(lab >= 0, (p_cal.argmax(1) != lab).astype(float), np.nan)
            referral = {c: M.referral_summary(st, thr, kl_cal, groups=te.component.to_numpy()) for c, thr in thresholds.items()}
            rc_kl = M.risk_coverage(st, kl_cal, groups=te.component.to_numpy()); rc_err = M.risk_coverage(st[lab >= 0], maj_err[lab >= 0], groups=te.component.to_numpy()[lab >= 0])
            # alternative input-only scores (descriptive)
            alt_scores = {"entropy": M.predictive_entropy(p_cal), "one_minus_max": M.one_minus_max_prob(p_cal)}
            if np.isfinite(te.d_hat.to_numpy()).all(): alt_scores["d_hat"] = te.d_hat.to_numpy(dtype=np.float64)
            rc_alt = {k: {"aurc": M.risk_coverage(s, kl_cal)["aurc"], "curve": M.risk_coverage(s, kl_cal)["risk"] if "risk" in M.risk_coverage(s, kl_cal) else None} for k, s in alt_scores.items()}
            raw = summarize(te, p_raw); cal = summarize(te, p_cal)
            teff = temperature_effect_summary(p_raw, qte, te.patient_id.to_numpy(), T["T"])
            base = {"run_id": rd.name, "method_id": cid, "seed": seed, "split_hash": cfg["split_hash"], "preprocess_hash": cfg["preprocess_hash"], "protocol_hash": lock["protocol_hash"], "prediction_hash": stable_hash(p_cal.round(6).tolist()[:2000])}
            records += [dict(r, calibration="raw") for r in records_from_summary(raw, base)] + [dict(r, calibration="temperature") for r in records_from_summary(cal, base)]
            res["methods"][cid]["seeds"][seed] = {"run_id": rd.name, "temperature": T, "temperature_effect": teff, "raw": raw, "calibrated": cal, "referral_thresholds": thresholds, "referral": referral,
                                                  "risk_coverage_kl": rc_kl, "risk_coverage_majority_error": rc_err, "alt_score_aurc": {k: v["aurc"] for k, v in rc_alt.items()}, "parameters": cfg["parameters"], "wall_seconds": read_json(rd / "status.json")["wall_seconds"], "peak": read_json(rd / "status.json")["peak"]}
            out = te.copy()
            for k in range(6): out[f"pcal{k}"] = p_cal[:, k]
            out["referral_score"] = st; out.to_parquet(out_dir / "test_predictions" / f"{cid}_s{seed}.parquet", index=False)
            preds[(cid, seed)] = out
            log(f"{cid} seed {seed}: T={T['T']:.3f} test patient-KL raw {raw['patient_kl']:.4f} cal {cal['patient_kl']:.4f} row-KL cal {cal['row_kl']:.4f} AUROC {cal['hard']['macro_auroc']}")
        seeds_res = res["methods"][cid]["seeds"]
        res["methods"][cid]["mean_over_seeds"] = {k: float(np.mean([s[c][k] for s in seeds_res.values()])) for c in ["raw", "calibrated"] for k in ["patient_kl", "row_kl"] for k in [k]} if False else {
            c: {k: float(np.mean([s[c][k] for s in seeds_res.values()])) for k in ["patient_kl", "component_kl", "row_kl", "soft_ce", "expected_brier", "soft_squared_error"]} for c in ["raw", "calibrated"]}
    pd.DataFrame(records).to_csv(out_dir / "metric_records.csv", index=False)
    return res, preds


def primary_and_secondary(lock, preds, out_dir, log) -> dict:
    cand, comp = lock["candidate"], lock["comparator"]
    def tables(cid, cal):
        return {s: (df.assign(**{f"p{k}": df[f"pcal{k}"] for k in range(6)}) if cal else df) for (c, s), df in preds.items() if c == cid}
    out = {"primary_calibrated": paired_primary(tables(cand, True), tables(comp, True)), "secondary_raw": paired_primary(tables(cand, False), tables(comp, False))}
    for cid in [c for (c, _) in preds if c not in (cand, comp)]:
        out[f"exploratory_{cid}_vs_{comp}_calibrated"] = paired_primary(tables(cid, True), tables(comp, True))
        out[f"exploratory_{cid}_vs_{cand}_calibrated"] = paired_primary(tables(cid, True), tables(cand, True))
    pr = out["primary_calibrated"]
    log(f"PRIMARY Delta(P - {comp}) = {pr['point_estimate']:+.4f} [{pr['ci_low']:+.4f}, {pr['ci_high']:+.4f}] -> {pr['decision']} (n_groups={pr['n_groups']})")
    write_json(out_dir / "bootstrap_primary.json", out)
    return out


def strata_analysis(lock, preds, reader, out_dir, log) -> dict:
    """Exploratory paired deltas by vote-count stratum, entropy tertile and missingness band (calibrated)."""
    cand, comp = lock["candidate"], lock["comparator"]
    quality = reader.quality.set_index("row"); index = reader.index.set_index("label_id")
    results = {}
    seeds = lock["final_seeds"]
    A = {s: preds[(cand, s)] for s in seeds}; B = {s: preds[(comp, s)] for s in seeds}
    ref = align({"a": A[seeds[0]], "b": B[seeds[0]]})["a"]
    v = ref[V_COLS].to_numpy(); n = v.sum(1); q = M.votes_to_targets(v); ent = -(q * np.log(np.clip(q, 1e-12, None))).sum(1)
    rows = index.loc[ref.label_id.to_numpy(), "row"].to_numpy(); vf = np.minimum(quality.loc[rows, "local_valid_fraction"].to_numpy(), quality.loc[rows, "context_valid_fraction"].to_numpy())
    t1, t2 = lock["target_entropy_tertiles"]
    lab, ties = M.unique_majority_labels(v)
    strata = {"votes_1": n == 1, "votes_2-4": (n >= 2) & (n <= 4), "votes_5-9": (n >= 5) & (n <= 9), "votes_10+": n >= 10,
              "entropy_low": ent <= t1, "entropy_mid": (ent > t1) & (ent <= t2), "entropy_high": ent > t2,
              "valid_fraction_lt_0.95": vf < 0.95, "valid_fraction_ge_0.95": vf >= 0.95}
    for k, name in enumerate(CLASS_NAMES): strata[f"majority_{name}"] = lab == k
    strata["tied_majority"] = ties
    for name, m in strata.items():
        if m.sum() < 10: results[name] = {"status": "NOT_ESTIMABLE", "n_rows": int(m.sum())}; continue
        a_g, b_g = [], []
        for s in seeds:
            al = align({"a": A[s], "b": B[s]}); da, db = al["a"][m].reset_index(drop=True), al["b"][m].reset_index(drop=True)
            pa = da[[f"pcal{k}" for k in range(6)]].to_numpy(); pb = db[[f"pcal{k}" for k in range(6)]].to_numpy()
            ga, ma = per_group_losses(da, pa); gb, mb = per_group_losses(db, pb); a_g.append(ma); b_g.append(mb)
        bs = paired_cluster_bootstrap(np.stack(a_g, 1), np.stack(b_g, 1))
        results[name] = {"delta": bs["point_estimate"], "ci_low": bs["ci_low"], "ci_high": bs["ci_high"], "n_rows": int(m.sum()), "n_groups": bs["n_groups"], "kl_candidate": float(np.mean(a_g)), "kl_comparator": float(np.mean(b_g)), "status": "PASS", "low_support": bs["n_groups"] < 30}
    write_json(out_dir / "strata.json", results)
    return results


def robustness_and_evidence(ws, lock, runs, reader, out_dir, device, log) -> dict:
    out = {}
    subset_rows = select_subset(reader.index, **{"n_rows": lock["robustness_subset"]["max_rows"], "min_patients": lock["robustness_subset"]["min_patients"], "seed": lock["robustness_subset"]["seed"]})
    write_json(out_dir / "robustness_subset.json", {"n_rows": int(len(subset_rows)), "n_patients": int(reader.index.set_index("row").loc[subset_rows].patient_id.nunique()), "seed": lock["robustness_subset"]["seed"], "hash": stable_hash(subset_rows.tolist())})
    for cid in [lock["candidate"], lock["comparator"]]:
        rd = runs[cid][lock["deployment_seed"]]; cfg = read_json(rd / "config.resolved.json")
        r = CacheReader(reader.dir, alt_uniform=(cfg["encoding"] == "uniform"))
        norm = load_normalizer(ws.normalization / cfg["split_hash"] / f"{cfg['preprocess_hash']}_{'+'.join(cfg['train_partitions'])}_{cfg['encoding']}.json"); N = Normalizer(norm)
        model = build_model(cid); load_weights(model, rd / "checkpoints" / "last.safetensors"); model.to(device).eval()
        T = fit_temperature(load_predictions(rd / "predictions" / "calibration_t_full_last.parquet")[P_COLS].to_numpy(), M.votes_to_targets(load_predictions(rd / "predictions" / "calibration_t_full_last.parquet")[V_COLS].to_numpy()), load_predictions(rd / "predictions" / "calibration_t_full_last.parquet").patient_id.to_numpy())["T"]
        ds = CacheDataset(r, subset_rows, N)
        conds = run_suite(model, ds, N, device, ws.data, log=log) if cfg["encoding"] == "foveated" else {c["id"]: predict(model, ds, device, perturb=None if c["id"] in ("clean",) else __import__("cape_eeg.evaluation.robustness", fromlist=["make_perturbation"]).make_perturbation(c["id"], N))["p"] for c in CONDITIONS if c["id"] != "stft_window_512"}
        q = ds.q.astype(np.float64); clean = apply_temperature(conds["clean"], T); kl_clean = M.kl_rows(q, clean); lab, _ = M.unique_majority_labels(ds.votes)
        rob = {}
        for c in CONDITIONS:
            if c["id"] not in conds: rob[c["id"]] = {"status": "NOT_RUN", "reason": "condition not applicable to the uniform-encoding comparator"}; continue
            p = apply_temperature(conds[c["id"]], T); kl = M.kl_rows(q, p)
            d = kl - kl_clean; g, dm = M.patient_mean(d, ds.component)
            bs = paired_cluster_bootstrap(M.patient_mean(kl, ds.component)[1], M.patient_mean(kl_clean, ds.component)[1])
            rob[c["id"]] = {**c, "kl": float(M.patient_weighted_mean(kl, ds.component)), "delta_kl": bs["point_estimate"], "ci_low": bs["ci_low"], "ci_high": bs["ci_high"],
                            "label_flip_fraction": float((p.argmax(1) != clean.argmax(1)).mean()), "js_distance": float(M.jensen_shannon_distance_rows(clean, p).mean()),
                            "majority_error": float((p.argmax(1)[lab >= 0] != lab[lab >= 0]).mean()), "status": "PASS"}
        out[cid] = {"robustness": rob, "n_rows": int(len(ds)), "n_patients": int(len(np.unique(ds.patient)))}
        log(f"robustness {cid}: " + ", ".join(f"{k}={v.get('delta_kl', 0):+.3f}" for k, v in rob.items() if v.get("status") == "PASS"))
        if cid == lock["candidate"]:
            test_rows = select_rows(reader.index, ["test"], "evaluator"); tds = CacheDataset(r, test_rows, N)
            dele = view_and_region_deletion(model, tds, device); qt = tds.q.astype(np.float64); base = apply_temperature(dele["intact"], T); klb = M.kl_rows(qt, base)
            ev = {}
            for k, p in dele.items():
                pc = apply_temperature(p, T); kl = M.kl_rows(qt, pc)
                bs = paired_cluster_bootstrap(M.patient_mean(kl, tds.component)[1], M.patient_mean(klb, tds.component)[1])
                ev[k] = {"kl": float(M.patient_weighted_mean(kl, tds.component)), "delta_kl": bs["point_estimate"], "ci_low": bs["ci_low"], "ci_high": bs["ci_high"], "js_distance": float(M.jensen_shannon_distance_rows(base, pc).mean()), "label_flip_fraction": float((pc.argmax(1) != base.argmax(1)).mean())}
            cases = select_cases(tds); curves = saliency_ranked_deletion(model, tds, cases, device)
            out["evidence"] = {"view_region_deletion": ev, "n_rows": int(len(tds)), "n_patients": int(len(np.unique(tds.patient))), "ranked_vs_random_deletion": curves, "n_cases": int(len(cases))}
            log(f"evidence deletion: local removed {ev['local_removed']['delta_kl']:+.3f}, context removed {ev['context_removed']['delta_kl']:+.3f}; ranked-vs-random at 30%: {curves['ranked']['kl'][3]:.3f} vs {curves['random']['kl'][3]:.3f}")
            # gate/uncertainty descriptives on test
            tp = predict(model, tds, device); out["gate_test"] = {"mean": float(tp["g"].mean()), "std": float(tp["g"].std()), "quantiles": [float(x) for x in np.quantile(tp["g"], [0.05, 0.25, 0.5, 0.75, 0.95])]}
    write_json(out_dir / "robustness_evidence.json", out)
    return out


def latency_benchmark(ws, lock, runs, reader, device, out_dir, log) -> dict:
    """Cached-tensor inference latency at batch 1 and 32 (20 warmup, 100 timed) + raw-to-prediction for batch 1."""
    import pandas as pd
    from ..data.spectral import RawSpectralEncoder, ContextSpectralEncoder
    from ..data.alignment import extract_eeg_window, select_spectrogram_rows
    res = {}
    for cid in [lock["candidate"], lock["comparator"]]:
        rd = runs[cid][lock["deployment_seed"]]; cfg = read_json(rd / "config.resolved.json")
        r = CacheReader(reader.dir, alt_uniform=(cfg["encoding"] == "uniform")); N = Normalizer(load_normalizer(ws.normalization / cfg["split_hash"] / f"{cfg['preprocess_hash']}_{'+'.join(cfg['train_partitions'])}_{cfg['encoding']}.json"))
        model = build_model(cid); load_weights(model, rd / "checkpoints" / "last.safetensors"); model.to(device).eval()
        rows = select_rows(reader.index, ["test"], "evaluator")[:256]; ds = CacheDataset(r, rows, N)
        res[cid] = {"parameters": cfg["parameters"], "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "dtype": "bf16_autocast"}
        for bs in (1, 32):
            b = ds.gather(np.arange(bs)); bd = {k: (torch.from_numpy(np.ascontiguousarray(v)).to(device) if hasattr(v, "shape") else v) for k, v in b.items()}
            times = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                for i in range(120):
                    if device.type == "cuda": torch.cuda.synchronize()
                    t0 = time.perf_counter(); model(bd["local"], bd["context"], bd["valid_l"], bd["valid_c"])
                    if device.type == "cuda": torch.cuda.synchronize()
                    if i >= 20: times.append((time.perf_counter() - t0) * 1000)
            res[cid][f"batch_{bs}"] = {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)), "examples_per_s": float(bs / (np.median(times) / 1000)), "n_timed": len(times)}
        # end-to-end raw -> prediction for batch 1 (parquet read + STFT + context binning + normalize + forward), 20 windows
        idx = reader.index.set_index("row").loc[rows[:20]]; enc = None; cenc = None; times = []
        for _, rw in idx.iterrows():
            t0 = time.perf_counter()
            X = pd.read_parquet(ws.data / "train_eegs" / f"{rw.eeg_id}.parquet"); enc = enc or RawSpectralEncoder(list(X.columns))
            w, obs, _ = extract_eeg_window(X.to_numpy(np.float32), rw.eeg_label_offset_seconds); e = enc.encode(w, obs)
            S = pd.read_parquet(ws.data / "train_spectrograms" / f"{rw.spectrogram_id}.parquet"); cenc = cenc or ContextSpectralEncoder(list(S.columns))
            keep, edges, _ = select_spectrogram_rows(S["time"].to_numpy(), rw.spectrogram_label_offset_seconds); c = cenc.encode(S.to_numpy()[keep], edges)
            key = "foveated" if cfg["encoding"] == "foveated" else "uniform"
            L = torch.from_numpy(N("local", e[key][None], e[f"{key}_mask"][None])).to(device); C = torch.from_numpy(N("context", c["context"][None], c["context_mask"][None])).to(device)
            with torch.no_grad(): model(L, C, torch.tensor([e[f"{key}_mask"].mean()], device=device), torch.tensor([c["context_mask"].mean()], device=device))
            if device.type == "cuda": torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        res[cid]["raw_to_prediction_batch_1"] = {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)), "n": len(times), "boundary": "parquet read + window + STFT + context binning + normalization + forward"}
        log(f"latency {cid}: b1 {res[cid]['batch_1']['median_ms']:.2f} ms, b32 {res[cid]['batch_32']['median_ms']:.2f} ms ({res[cid]['batch_32']['examples_per_s']:.0f} ex/s), raw->pred {res[cid]['raw_to_prediction_batch_1']['median_ms']:.0f} ms")
    res["energy"] = "NOT_MEASURED (no credible power sensor boundary on the unified-memory system)"
    write_json(out_dir / "latency.json", res)
    return res


def resource_ledger(ws, runs, reader, out_dir) -> dict:
    led = [json.loads(l) for l in (ws.runs / "gpu_ledger.jsonl").read_text().splitlines() if l.strip()]
    by_stage = {}
    for r in led: by_stage[r["stage"]] = by_stage.get(r["stage"], 0.0) + r["hours"]
    fits = {f"{cid}_s{seed}": {"wall_min": read_json(rd / "status.json")["wall_seconds"] / 60, "peak": read_json(rd / "status.json")["peak"], "params": read_json(rd / "config.resolved.json")["parameters"]["total"], "examples_seen": read_json(rd / "status.json")["examples_seen"]} for cid, seeds in runs.items() for seed, rd in seeds.items()}
    out = {"gpu_hours_total": sum(r["hours"] for r in led), "gpu_hours_by_stage": by_stage, "n_gpu_jobs": len(led), "failed_or_incomplete_jobs": sum(r["status"] != "PASS" for r in led),
           "ceiling_hours": 12.0, "final_fits": fits, "cache": {k: reader.manifest[k] for k in ["active_cache_bytes", "alt_uniform_bytes", "within_cap", "rows"]},
           "conversion": read_json(reader.dir / "conversion_resources.json"), "source_bytes": read_json(ws.provenance / "source_manifest.json")["source_bytes_total"]}
    write_json(out_dir / "resources.json", out)
    return out
