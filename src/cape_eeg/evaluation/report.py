"""Turn prediction tables into metric records and comparison tables (docs/04 sections 3-6, 13)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import LABELS, CLASS_NAMES
from .. import metrics as M
from .bootstrap import paired_cluster_bootstrap, cluster_bootstrap_statistic

P_COLS = [f"p{k}" for k in range(6)]; V_COLS = [f"v{k}" for k in range(6)]


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.sort_values("label_id").reset_index(drop=True)


def align(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Restrict every table to the common label_id set, in the same order; refuse silent drops > 0."""
    keys = None
    for d in dfs.values():
        keys = set(d.label_id) if keys is None else keys & set(d.label_id)
    out = {}
    for k, d in dfs.items():
        if len(d) != len(keys):
            raise ValueError(f"prediction table {k} has {len(d)} rows but the common key set has {len(keys)}; keys must align one-to-one")
        out[k] = d[d.label_id.isin(keys)].sort_values("label_id").reset_index(drop=True)
    return out


def summarize(df: pd.DataFrame, p: np.ndarray | None = None) -> dict:
    """All basic + calibration + disagreement metrics for one prediction table."""
    v = df[V_COLS].to_numpy(dtype=np.int64); q = M.votes_to_targets(v)
    p = df[P_COLS].to_numpy(dtype=np.float64) if p is None else p
    groups = df.component.to_numpy()
    kl = M.kl_rows(q, p)
    out = {"n_rows": int(len(df)), "n_patients": int(df.patient_id.nunique()), "n_components": int(df.component.nunique()),
           "patient_kl": float(M.patient_weighted_mean(kl, df.patient_id.to_numpy())), "component_kl": float(M.patient_weighted_mean(kl, groups)), "row_kl": float(kl.mean()),
           "soft_ce": float(M.soft_cross_entropy_rows(q, p).mean()), "soft_squared_error": float(M.soft_squared_error_rows(q, p).mean()),
           "expected_brier": float(M.expected_brier_rows(q, p).mean())}
    hard = M.hard_metrics(v, p)
    out["hard"] = hard
    out["soft_top_label_ece"] = M.soft_top_label_ece(q, p, groups=groups)
    out["majority_label_ece"] = M.majority_label_ece(v, p)
    if "d_hat" in df.columns and np.isfinite(df.d_hat.to_numpy()).any():
        out["disagreement"] = M.disagreement_metrics(df.d_hat.to_numpy(dtype=np.float64), v)
    if "gate" in df.columns:
        out["gate_mean"] = float(df.gate.mean()); out["gate_std"] = float(df.gate.std())
    # vote-count strata
    n = v.sum(1); strata = {"1": n == 1, "2-4": (n >= 2) & (n <= 4), "5-9": (n >= 5) & (n <= 9), "10+": n >= 10}
    out["kl_by_vote_stratum"] = {k: {"row_kl": float(kl[m].mean()) if m.any() else None, "patient_kl": float(M.patient_weighted_mean(kl[m], df.patient_id.to_numpy()[m])) if m.any() else None, "n_rows": int(m.sum())} for k, m in strata.items()}
    return out


def per_group_losses(df: pd.DataFrame, p: np.ndarray | None = None, group_col: str = "component") -> tuple[np.ndarray, np.ndarray]:
    v = df[V_COLS].to_numpy(dtype=np.int64); q = M.votes_to_targets(v)
    p = df[P_COLS].to_numpy(dtype=np.float64) if p is None else p
    g, means = M.patient_mean(M.kl_rows(q, p), df[group_col].to_numpy())
    return g, means


def paired_primary(pred_a: dict[int, pd.DataFrame], pred_b: dict[int, pd.DataFrame], n_replicates: int = 2000, seed: int = 20260907) -> dict:
    """Primary estimand: mean over seeds of per-component mean KL, paired bootstrap of (A - B)."""
    seeds = sorted(pred_a); assert seeds == sorted(pred_b), "seed sets must match"
    A, B = [], []
    groups_ref = None
    for s in seeds:
        al = align({"a": pred_a[s], "b": pred_b[s]})
        ga, ma = per_group_losses(al["a"]); gb, mb = per_group_losses(al["b"])
        assert np.array_equal(ga, gb)
        if groups_ref is None: groups_ref = ga
        assert np.array_equal(groups_ref, ga)
        A.append(ma); B.append(mb)
    A = np.stack(A, 1); B = np.stack(B, 1)   # [G, S]
    bs = paired_cluster_bootstrap(A, B, n_replicates=n_replicates, seed=seed)
    bs.update(seeds=seeds, per_seed_a=[float(A[:, i].mean()) for i in range(len(seeds))], per_seed_b=[float(B[:, i].mean()) for i in range(len(seeds))],
              mean_a=float(A.mean()), mean_b=float(B.mean()), relative_change=float((A.mean() - B.mean()) / B.mean()),
              decision="superiority_supported" if bs["ci_high"] < 0 else ("inferior" if bs["ci_low"] > 0 else "inconclusive"))
    return bs


def bootstrap_scalar(df: pd.DataFrame, fn, n_replicates: int = 2000, seed: int = 20260907) -> dict:
    return cluster_bootstrap_statistic(fn, df.component.to_numpy(), n_replicates=n_replicates, seed=seed)


def records_from_summary(summary: dict, base: dict) -> list[dict]:
    recs = []
    flat = {"patient_kl": summary["patient_kl"], "component_kl": summary["component_kl"], "row_kl": summary["row_kl"], "soft_ce": summary["soft_ce"],
            "soft_squared_error": summary["soft_squared_error"], "expected_brier": summary["expected_brier"]}
    for k in ["accuracy", "balanced_accuracy", "macro_f1", "mcc", "cohen_kappa", "macro_auroc", "macro_auprc"]:
        flat[k] = summary["hard"].get(k)
    flat["soft_top_label_ece"] = summary["soft_top_label_ece"].get("ece"); flat["majority_label_ece"] = summary["majority_label_ece"].get("ece")
    for k, val in flat.items():
        recs.append(M.metric_record(metric=k, value=val, n_rows=summary["n_rows"], n_patients=summary["n_patients"], n_components=summary["n_components"],
                                    status="PASS" if val is not None else "NOT_ESTIMABLE", **base))
    return recs
