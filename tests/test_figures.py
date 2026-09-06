"""Unit tests for the visualization suite: registry integrity, NOT_RUN behaviour, and renderers on synthetic inputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cape_eeg.contracts import LABELS, PARTITIONS
from cape_eeg.paths import Workspace
from cape_eeg.status import write_json
from cape_eeg.visualization.registry import FIGURES, FIGURES_BY_ID, resolve_artefacts, missing_inputs
from cape_eeg.visualization.figures import render_figure, RENDERERS, SMALL_CELL

SEEDS = [101, 202, 303]


def make_ws(tmp_path: Path) -> Workspace:
    ws = Workspace(root=tmp_path, data=tmp_path, private=tmp_path / "private", repo=tmp_path / "repo")
    ws.ensure()
    return ws


def synthetic_votes(n_rows=600, n_patients=60, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pat = rng.integers(0, n_patients, n_rows)
    votes = np.zeros((n_rows, 6), dtype=np.int64)
    for i in range(n_rows):
        n = int(rng.choice([1, 3, 7, 12]))
        k = int(rng.integers(0, 6)); votes[i, k] += n // 2 + 1
        rest = n - votes[i, k]
        if rest > 0:
            votes[i] += rng.multinomial(rest, np.full(6, 1 / 6))
    df = pd.DataFrame(votes, columns=LABELS)
    df.insert(0, "label_id", np.arange(1_000, 1_000 + n_rows)); df.insert(1, "patient_id", 10_000 + pat)
    df["component"] = df.patient_id
    df["partition"] = [PARTITIONS[p % 5] for p in pat]
    return df


def write_intake(ws: Workspace, df: pd.DataFrame):
    df[["label_id", "patient_id"] + LABELS].to_csv(ws.data / "train.csv", index=False)
    df[["label_id", "patient_id", "component", "partition"]].assign(eeg_id=df.label_id, spectrogram_id=df.label_id).to_parquet(ws.manifests / "split_manifest.parquet", index=False)
    write_json(ws.manifests / "split_summary.json", {"split_hash": "deadbeefdeadbeef", "seed": 1} | {p: {"rows": int((df.partition == p).sum()), "patients": int(df.patient_id[df.partition == p].nunique()), "components": int(df.component[df.partition == p].nunique())} for p in PARTITIONS})


def write_lock_and_predictions(ws: Workspace, df: pd.DataFrame, seeds=SEEDS, deploy=101):
    lock = {"candidate": "P", "comparator": "B2", "final_seeds": seeds, "deployment_seed": deploy, "protocol_hash": "abc123", "referral_score": "entropy", "corruption_conditions": ["clean"], "target_entropy_tertiles": [0.3, 0.9]}
    write_json(ws.manifests / "protocol_lock.json", lock)
    pred_dir = ws.evaluation / "abc123" / "test_predictions"; pred_dir.mkdir(parents=True)
    test = df[df.partition == "test"].reset_index(drop=True)
    v = test[LABELS].to_numpy(); q = v / v.sum(1, keepdims=True)
    for method, noise in [("P", 0.6), ("B2", 1.2)]:
        for s in seeds:
            rng = np.random.default_rng(s + (0 if method == "P" else 7))
            logits = np.log(q + 0.05) + rng.normal(0, noise, q.shape); p = np.exp(logits); p /= p.sum(1, keepdims=True)
            out = pd.DataFrame({"label_id": test.label_id, "patient_id": test.patient_id, "component": test.component, "n_votes": v.sum(1)})
            for k in range(6): out[f"p{k}"] = p[:, k]
            for k in range(6): out[f"pL{k}"] = p[:, k]; out[f"pC{k}"] = p[:, k]
            out["gate"] = 0.5; out["js"] = 0.0; out["d_hat"] = rng.uniform(0, 1, len(test))
            for k in range(6): out[f"v{k}"] = v[:, k]
            for k in range(6): out[f"pcal{k}"] = p[:, k]
            out["referral_score"] = -(p * np.log(p)).sum(1)
            out.to_parquet(pred_dir / f"{method}_s{s}.parquet", index=False)
    return lock, test


# --------------------------------------------------------------------------------------
def test_registry_has_twenty_unique_ids():
    ids = [f.figure_id for f in FIGURES]
    assert len(ids) == 20 and len(set(ids)) == 20
    assert ids == [f"V{i:02d}" for i in range(1, 21)]
    assert set(RENDERERS) == set(FIGURES_BY_ID)
    for f in FIGURES:
        assert f.slug and f.title and f.question and isinstance(f.required, tuple)


def test_missing_input_yields_not_run_without_files(tmp_path):
    ws = make_ws(tmp_path)
    resolved = resolve_artefacts(ws)
    assert missing_inputs(FIGURES_BY_ID["V09"], resolved) == ["protocol_lock", "test_predictions_all_seeds"]
    rec = render_figure("V09", ws)
    assert rec["status"] == "NOT_RUN" and "missing inputs" in rec["reason"]
    assert rec["output_paths"] == [] and list(ws.figures_public.glob("*")) == []
    for fid in ("V01", "V03", "V12", "V16", "V20"):
        assert render_figure(fid, ws)["status"] == "NOT_RUN"
    assert list(ws.figures_public.glob("*")) == []


def test_v03_label_distribution_from_train_csv(tmp_path):
    ws = make_ws(tmp_path); df = synthetic_votes(); write_intake(ws, df)
    rec = render_figure("V03", ws)
    assert rec["status"] == "PASS", rec.get("reason")
    assert {"figures/v03_label_distribution_vote_mass.png", "figures/v03_label_distribution_majority_counts.png"} <= set(rec["output_paths"])
    for rel in rec["output_paths"]:
        assert (ws.repo / rel).stat().st_size > 1000
    assert rec["n_rows"] == len(df) and rec["n_patients"] == df.patient_id.nunique()
    shares = rec["parameters"]["per_partition"]["test"]["share"]
    assert abs(sum(shares) - 1.0) < 1e-9
    assert "seizure" not in rec["caption"].lower() or "Seizure, LPD, GPD, LRDA, GRDA, Other" in rec["caption"]
    assert "patient_id" not in rec["caption"] and str(tmp_path) not in rec["caption"]


def test_v12_confusion_matrix_synthetic(tmp_path):
    ws = make_ws(tmp_path); df = synthetic_votes(n_rows=900, n_patients=90); write_intake(ws, df)
    lock, test = write_lock_and_predictions(ws, df)
    rec = render_figure("V12", ws)
    assert rec["status"] == "PASS", rec.get("reason")
    r = rec["parameters"]["results"]["candidate"]
    cm = np.asarray(r["confusion_matrix"]); assert cm.shape == (6, 6)
    assert cm.sum() == r["n_rows_evaluated"] and r["n_rows_evaluated"] + r["n_ties_excluded"] == len(test)
    assert len(r["recall"]["point"]) == 6 and all(0 <= x <= 1 for x in r["recall"]["point"] if x is not None)
    names = {Path(p).name for p in rec["output_paths"]}
    assert {"v12_confusion_matrix_counts.png", "v12_confusion_matrix_row_normalized.png", "v12_confusion_matrix_row_normalized_comparator.png"} <= names
    # suppression rule: any populated cell with fewer than SMALL_CELL patients is counted as suppressed
    cp = np.asarray(r["cell_patients"]); assert r["suppressed_cells"] == 2 * int(((cm > 0) & (cp < SMALL_CELL)).sum())  # counts + row-normalized figures
    assert "tied maxima excluded" in rec["caption"]


def test_v09_paired_delta_synthetic(tmp_path):
    ws = make_ws(tmp_path); df = synthetic_votes(n_rows=900, n_patients=90); write_intake(ws, df)
    lock, test = write_lock_and_predictions(ws, df)
    rec = render_figure("V09", ws)
    assert rec["status"] == "PASS", rec.get("reason")
    pr = rec["parameters"]["results"]["primary_calibrated"]
    assert pr["n_groups"] == test.component.nunique() and pr["n_seeds"] == 3 and pr["n_replicates"] == 2000
    assert pr["ci_low"] <= pr["point_estimate"] <= pr["ci_high"]
    assert pr["point_estimate"] < 0  # the candidate was generated with less noise
    assert (ws.repo / "figures" / "v09_paired_delta_kl.png").exists() and (ws.repo / "figures" / "v09_paired_delta_kl_raw.svg").exists()
    priv = list(ws.figures_private.glob("v09_*.csv")); assert len(priv) == 1
    assert "$CAPE_ROOT" in rec["private_output_paths"][0] and str(tmp_path) not in json.dumps(rec)
    # a stale bootstrap_primary.json that disagrees must be detected
    write_json(ws.evaluation / "abc123" / "bootstrap_primary.json", {"primary_calibrated": {"point_estimate": 0.5, "ci_low": 0.4, "ci_high": 0.6}})
    with pytest.raises(Exception):
        render_figure("V09", ws)


def test_v05_synthetic_alignment_needs_no_data(tmp_path):
    ws = make_ws(tmp_path)
    rec = render_figure("V05", ws)
    assert rec["status"] == "PASS" and rec["parameters"]["synthetic"] is True
    assert rec["parameters"]["target_columns"] == [8, 24] and rec["parameters"]["context_centre_bins"] == [31, 32]


def test_v02_flags_nonzero_overlap_in_parameters(tmp_path):
    ws = make_ws(tmp_path); df = synthetic_votes(); write_intake(ws, df)
    mat = [[10, 0, 0, 0, 0], [0, 12, 0, 0, 0], [0, 0, 11, 0, 0], [0, 0, 0, 13, 0], [0, 0, 0, 0, 14]]
    write_json(ws.manifests / "leakage_audit.json", {"forbidden_overlap": 0, "status": "PASS", "patient_id": {"matrix": mat, "off_diagonal_total": 0}, "component": {"matrix": mat, "off_diagonal_total": 0}})
    rec = render_figure("V02", ws)
    assert rec["status"] == "PASS" and rec["parameters"]["forbidden_overlap"] == 0 and rec["parameters"]["keys"] == ["Patients", "Connected components"]


# --------------------------------------------------------------------------------------
# Full synthetic workspace: exercises every renderer end to end (post-lock ones included)
# --------------------------------------------------------------------------------------
def build_full_synthetic_workspace(root: Path, n_rows=1200, n_patients=120) -> Workspace:
    """Write every artefact the registry knows about, with the key structure produced by the real pipeline."""
    from cape_eeg.evaluation.report import paired_primary, per_group_losses, align
    from cape_eeg import metrics as M
    ws = make_ws(root); df = synthetic_votes(n_rows, n_patients); write_intake(ws, df)
    write_json(ws.provenance / "source_manifest.json", {"n_eeg_files": 10, "n_spectrogram_files": 5, "rows_with_truncated_eeg_window": 0, "rows_with_short_spectrogram_context": 0, "unreferenced_eeg_files": 1, "unreferenced_spectrogram_files": 0, "source_manifest_hash": "src0123456789ab", "source_bytes_total": 1000})
    write_json(ws.provenance / "schema_report.json", {"n_rows": len(df), "n_patients": int(df.patient_id.nunique()), "n_eeg": len(df), "n_spectrograms": len(df), "train_csv_sha256": "f" * 64})
    mat = [[int(df.patient_id[df.partition == p].nunique()) if i == j else 0 for j, _ in enumerate(PARTITIONS)] for i, p in enumerate(PARTITIONS)]
    write_json(ws.manifests / "leakage_audit.json", {"forbidden_overlap": 0, "status": "PASS"} | {k: {"matrix": mat, "off_diagonal_total": 0} for k in ["patient_id", "eeg_id", "spectrogram_id", "component"]})
    # cache: manifest, row index/quality, tiny shard files
    cdir = ws.cache / "cafe0123deadbeef"; (cdir / "shards").mkdir(parents=True); (cdir / "alt_uniform").mkdir()
    rng = np.random.default_rng(3)
    for s in range(2):
        for kind in ("local", "context"):
            np.save(cdir / "shards" / f"shard_{s:04d}_{kind}.npy", np.zeros((4, 4, 8, 8), np.float16)); np.save(cdir / "shards" / f"shard_{s:04d}_{kind}_mask.npy", np.zeros((4, 32), np.uint8))
        np.save(cdir / "alt_uniform" / f"shard_{s:04d}_local.npy", np.zeros((4, 4, 8, 8), np.float16)); np.save(cdir / "alt_uniform" / f"shard_{s:04d}_local_mask.npy", np.zeros((4, 32), np.uint8))
    idx = df[["label_id", "patient_id", "component", "partition"] + LABELS].copy(); idx.insert(0, "row", np.arange(len(df))); idx["eeg_id"] = idx.label_id; idx["spectrogram_id"] = idx.label_id
    idx.to_parquet(cdir / "row_index.parquet", index=False)
    q = pd.DataFrame({"row": np.arange(len(df)), "local_valid_fraction": np.clip(rng.beta(20, 1, len(df)), 0, 1), "context_valid_fraction": np.clip(rng.beta(10, 1, len(df)), 0, 1)})
    q.loc[0, ["local_valid_fraction", "context_valid_fraction"]] = 0.0; q.to_parquet(cdir / "row_quality.parquet", index=False)
    write_json(cdir / "manifest.json", {"preprocess_hash": "cafe0123deadbeef", "cache_hash": "c4c4c4c4", "rows": len(df), "active_cache_bytes": 5_000_000, "alt_uniform_bytes": 1_000_000, "within_cap": True})
    write_json(cdir / "conversion_resources.json", {"elapsed_seconds": 120.0})
    # runs + development table
    (ws.runs).mkdir(exist_ok=True)
    dev_rows = []
    for cfg, kl0 in [("B2", 0.80), ("B3", 0.78), ("A1", 0.76), ("A2", 0.74), ("P", 0.70)]:
        for seed in ([101, 202] if cfg in ("B2", "P") else [101]):
            rid = f"dev_{cfg}_s{seed}_aaaaaaaa_bbbbbbbb_cccccccc"; d = ws.runs / rid; d.mkdir()
            write_json(d / "config.resolved.json", {"run_id": rid, "config_id": cfg, "seed": seed, "phase": "dev", "epochs": 4, "training": {"compact_lr": 1e-3}, "parameters": {"total": 100000 + 1000 * len(cfg)}, "encoding": "uniform" if cfg == "B2" else "foveated"})
            ev = [{"epoch": e, "train_loss": 1.5 / e, "train_soft_ce": 1.4 / e, "examples_seen": 300 * e, "epoch_seconds": 10.0, "elapsed_min": e * 0.2, "tune_patient_kl": kl0 + 0.3 / e, "tune_row_kl": kl0 + 0.35 / e, "tune_gate_mean": 0.5} for e in range(1, 5)]
            (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in ev) + "\n")
            write_json(d / "status.json", {"status": "PASS", "complete_schedule": True, "wall_seconds": 40.0, "examples_seen": 1200, "peak": {"cuda_alloc_gib": 1.0, "rss_gib": 2.0}, "best": {"epoch": 4, "tune_patient_kl": kl0 + 0.075}})
            dev_rows.append({"run_id": rid, "config": cfg, "seed": seed, "epochs": 4, "lr_mult": 1.0, "params": 100000, "status": "PASS", "best_epoch": 4, "tune_patient_kl": kl0 + 0.075 + 0.01 * (seed == 202), "tune_row_kl": kl0 + 0.1, "tune_macro_auroc": 0.8, "wall_min": 0.7})
    (ws.runs / "gpu_ledger.jsonl").write_text("\n".join(json.dumps({"run_id": r["run_id"], "stage": "baselines", "hours": 0.01, "status": "PASS"}) for r in dev_rows) + "\n")
    (ws.evaluation / "development").mkdir(parents=True); pd.DataFrame(dev_rows).to_csv(ws.evaluation / "development" / "development_table.csv", index=False)
    # lock, predictions, evaluation outputs
    lock, test = write_lock_and_predictions(ws, df)
    lock.update(corruption_conditions=["clean", "region_missing_LL", "time_mask_10pct", "gain_shift_mild", "gain_shift_strong", "stft_window_512"], preprocess_hash="cafe0123deadbeef")
    write_json(ws.manifests / "protocol_lock.json", lock)
    ev_dir = ws.evaluation / "abc123"; pred_dir = ev_dir / "test_predictions"
    def load(cid, s): return pd.read_parquet(pred_dir / f"{cid}_s{s}.parquet").sort_values("label_id").reset_index(drop=True)
    preds = {(c, s): load(c, s) for c in ("P", "B2") for s in SEEDS}
    tables = lambda cid: {s: preds[(cid, s)] for s in SEEDS}
    write_json(ev_dir / "bootstrap_primary.json", {"primary_calibrated": paired_primary(tables("P"), tables("B2")), "secondary_raw": paired_primary(tables("P"), tables("B2"))})
    methods = {}
    for cid in ("P", "B2"):
        seeds = {}
        for s in SEEDS:
            d = preds[(cid, s)]; qv = M.votes_to_targets(d[[f"v{k}" for k in range(6)]].to_numpy()); p = d[[f"pcal{k}" for k in range(6)]].to_numpy(); kl = M.kl_rows(qv, p)
            score = d.referral_score.to_numpy()
            ref = {str(c): M.referral_summary(score, M.threshold_for_coverage(score, c), kl, groups=d.component.to_numpy()) for c in (0.8, 0.9, 0.95)}
            seeds[str(s)] = {"run_id": f"final_{cid}_s{s}", "temperature": {"T": 1.1 if cid == "P" else 1.3}, "calibrated": {"n_rows": len(d), "n_patients": int(d.patient_id.nunique()), "n_components": int(d.component.nunique()), "patient_kl": float(M.patient_weighted_mean(kl, d.patient_id.to_numpy()))},
                             "raw": {"n_rows": len(d)}, "referral": ref, "referral_thresholds": {k: v["threshold"] for k, v in ref.items()}}
        methods[cid] = {"seeds": seeds}
    write_json(ev_dir / "summary.json", {"methods": methods})
    v = test[LABELS].to_numpy(); n = v.sum(1); lab, ties = M.unique_majority_labels(v)
    strata = {}
    for name, mask in [("votes_1", n == 1), ("votes_2-4", (n >= 2) & (n <= 4)), ("votes_5-9", (n >= 5) & (n <= 9)), ("votes_10+", n >= 10), ("entropy_low", n <= 3), ("entropy_mid", (n > 3) & (n <= 7)), ("entropy_high", n > 7),
                       ("valid_fraction_lt_0.95", n % 2 == 0), ("valid_fraction_ge_0.95", n % 2 == 1)] + [(f"majority_{c}", lab == k) for k, c in enumerate(["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"])] + [("tied_majority", ties)]:
        ng = int(test.component[mask].nunique())
        strata[name] = {"status": "NOT_ESTIMABLE", "n_rows": int(mask.sum())} if mask.sum() < 10 else {"delta": -0.05 + 0.01 * len(name) % 3 * 0.01, "ci_low": -0.09, "ci_high": -0.01, "n_rows": int(mask.sum()), "n_groups": ng, "kl_candidate": 0.7, "kl_comparator": 0.75, "status": "PASS", "low_support": ng < 30}
    write_json(ev_dir / "strata.json", strata)
    conds = [{"id": "clean", "severity": 0.0, "units": "none"}, {"id": "region_missing_LL", "severity": 1.0, "units": "regions removed"}, {"id": "time_mask_10pct", "severity": 0.1, "units": "fraction of time axis"}, {"id": "gain_shift_mild", "severity": 0.5, "units": "log-power units"}, {"id": "gain_shift_strong", "severity": 1.5, "units": "log-power units"}, {"id": "stft_window_512", "severity": 512, "units": "STFT window samples"}]
    def rob(cid):
        out = {}
        for i, c in enumerate(conds):
            if cid == "B2" and c["id"] == "stft_window_512": out[c["id"]] = {"status": "NOT_RUN", "reason": "condition not applicable to the uniform-encoding comparator"}; continue
            out[c["id"]] = {**c, "description": c["id"], "kl": 0.7 + 0.02 * i, "delta_kl": 0.02 * i, "ci_low": 0.02 * i - 0.01, "ci_high": 0.02 * i + 0.01, "label_flip_fraction": 0.01 * i, "js_distance": 0.03 * i, "majority_error": 0.3, "status": "PASS"}
        return out
    fr = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    write_json(ev_dir / "robustness_evidence.json", {"P": {"robustness": rob("P"), "n_rows": 300, "n_patients": 40}, "B2": {"robustness": rob("B2"), "n_rows": 300, "n_patients": 40},
                                                     "evidence": {"view_region_deletion": {k: {"kl": 0.8, "delta_kl": 0.05 * (i + 1), "ci_low": 0.05 * (i + 1) - 0.02, "ci_high": 0.05 * (i + 1) + 0.02, "js_distance": 0.1, "label_flip_fraction": 0.05} for i, k in enumerate(["intact", "local_removed", "context_removed", "region_removed_LL", "region_removed_RL", "region_removed_LP", "region_removed_RP"])},
                                                                  "n_rows": len(test), "n_patients": int(test.patient_id.nunique()), "n_cases": 24,
                                                                  "ranked_vs_random_deletion": {"fractions": fr, "ranked": {"kl": [0.7 + 0.6 * f for f in fr], "js": [0.5 * f for f in fr]}, "random": {"kl": [0.7 + 0.3 * f for f in fr], "js": [0.25 * f for f in fr]}, "n_cases": 24, "baseline_kl": 0.7}}})
    write_json(ev_dir / "robustness_subset.json", {"n_rows": 300, "n_patients": 40, "seed": 1, "hash": "sub0"})
    write_json(ev_dir / "latency.json", {"P": {"parameters": {"total": 120000}, "device": "synthetic", "dtype": "bf16", "batch_1": {"median_ms": 2.0, "p95_ms": 2.5, "examples_per_s": 500, "n_timed": 100}, "batch_32": {"median_ms": 6.0, "p95_ms": 7.0, "examples_per_s": 5000, "n_timed": 100}, "raw_to_prediction_batch_1": {"median_ms": 300, "p95_ms": 400, "n": 20}},
                                         "B2": {"parameters": {"total": 100000}, "device": "synthetic", "dtype": "bf16", "batch_1": {"median_ms": 1.8, "p95_ms": 2.2, "examples_per_s": 550, "n_timed": 100}, "batch_32": {"median_ms": 5.0, "p95_ms": 6.0, "examples_per_s": 6000, "n_timed": 100}, "raw_to_prediction_batch_1": {"median_ms": 280, "p95_ms": 380, "n": 20}},
                                         "energy": "NOT_MEASURED"})
    write_json(ev_dir / "resources.json", {"gpu_hours_total": 3.2, "gpu_hours_by_stage": {"baselines": 2.0, "final": 1.2}, "n_gpu_jobs": 9, "failed_or_incomplete_jobs": 1, "ceiling_hours": 12.0, "source_bytes": 25e9, "conversion": {"elapsed_seconds": 900},
                                           "cache": {"active_cache_bytes": 5_000_000, "alt_uniform_bytes": 1_000_000, "within_cap": True, "rows": len(df)},
                                           "final_fits": {f"{c}_s{s}": {"wall_min": 30.0, "peak": {"cuda_alloc_gib": 1.5, "rss_gib": 3.0}, "params": 120000, "examples_seen": 10000} for c in ("P", "B2") for s in SEEDS}})
    return ws


@pytest.fixture(scope="module")
def full_ws(tmp_path_factory):
    return build_full_synthetic_workspace(tmp_path_factory.mktemp("fullws"))


def test_every_renderer_passes_on_the_full_synthetic_workspace(full_ws):
    resolved = resolve_artefacts(full_ws)
    assert all(missing_inputs(f, resolved) == [] for f in FIGURES), {f.figure_id: missing_inputs(f, resolved) for f in FIGURES}
    for f in FIGURES:
        rec = render_figure(f.figure_id, full_ws, resolved=resolved)
        assert rec["status"] == "PASS", (f.figure_id, rec.get("reason"))
        assert rec["output_paths"] and all((full_ws.repo / p).stat().st_size > 1000 for p in rec["output_paths"])
        for key in ("figure_id", "title", "data_hashes", "run_ids", "split", "n_rows", "n_patients", "n_components", "generation_code_commit", "parameters", "output_paths", "status", "privacy_review", "caption"):
            assert key in rec
        assert "patient_id" not in rec["caption"] and str(full_ws.root) not in json.dumps(rec)
    assert (full_ws.repo / "figures" / "v20_accuracy_resource_pareto_ledger_table.png").exists()
