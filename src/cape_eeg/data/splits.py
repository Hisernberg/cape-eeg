"""Connected-component construction and deterministic stratified allocation (docs/02 section 10)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import LABELS, PARTITIONS, SPLIT_PROPORTIONS, SPLIT_SEED, stable_hash


class _UF:
    def __init__(self):
        self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def connected_components(df: pd.DataFrame) -> pd.Series:
    """Component id per row linking patient_id, eeg_id and spectrogram_id."""
    uf = _UF()
    for p, e, s in zip(df.patient_id.to_numpy(), df.eeg_id.to_numpy(), df.spectrogram_id.to_numpy()):
        uf.union(("p", int(p)), ("e", int(e))); uf.union(("p", int(p)), ("s", int(s)))
    roots = [uf.find(("p", int(p))) for p in df.patient_id.to_numpy()]
    codes = {r: i for i, r in enumerate(sorted(set(roots), key=lambda z: (z[0], z[1])))}
    return pd.Series([codes[r] for r in roots], index=df.index, name="component")


def allocate(df: pd.DataFrame, seed: int = SPLIT_SEED, proportions: dict = SPLIT_PROPORTIONS) -> tuple[pd.DataFrame, dict]:
    """Greedy deterministic stratified allocation of components to partitions.

    Components are visited in a seeded random order; each is placed in the partition whose
    weighted squared deviation from target shares (rows, per-class vote mass, components) is
    reduced most. Model predictions are never consulted.
    """
    comp = connected_components(df)
    d = df.assign(component=comp)
    votes = d[LABELS].to_numpy(dtype=np.float64)
    g = d.groupby("component")
    stats = pd.DataFrame({"rows": g.size(), "n_patients": g.patient_id.nunique()})
    for i, k in enumerate(LABELS):
        stats[k] = g.apply(lambda x: x[k].sum(), include_groups=False) if False else d.groupby("component")[k].sum()
    stats["mass"] = stats[LABELS].sum(1)
    for k in LABELS:
        stats[k] = stats[k] / stats["mass"] * stats["rows"]  # vote-mass rows per class
    dims = ["rows"] + LABELS + ["count"]
    stats["count"] = 1.0
    totals = stats[dims].sum(0).to_numpy()
    targets = np.asarray([proportions[p] for p in PARTITIONS])
    rng = np.random.default_rng(seed)
    order = stats.index.to_numpy()[rng.permutation(len(stats))]
    # visit largest first for coarse balance, then the rest randomly (deterministic given seed)
    sizes = stats.loc[order, "rows"].to_numpy()
    big = order[sizes >= np.quantile(sizes, 0.95)]; rest = order[sizes < np.quantile(sizes, 0.95)]
    order = np.concatenate([big[np.argsort(-stats.loc[big, "rows"].to_numpy(), kind="stable")], rest])
    current = np.zeros((len(PARTITIONS), len(dims)))
    assign = {}
    w = np.ones(len(dims)); w[0] = 2.0; w[-1] = 0.5
    for c in order:
        v = stats.loc[c, dims].to_numpy(dtype=np.float64)
        best, best_cost = None, None
        for j in range(len(PARTITIONS)):
            trial = current.copy(); trial[j] += v
            share = trial / totals[None, :]
            cost = (w[None, :] * (share - targets[:, None]) ** 2).sum()
            if best_cost is None or cost < best_cost - 1e-15:
                best, best_cost = j, cost
        assign[c] = PARTITIONS[best]; current[best] += v
    d["partition"] = d["component"].map(assign)
    summary = partition_summary(d)
    summary["seed"] = seed; summary["proportions"] = proportions
    summary["split_hash"] = stable_hash({"seed": seed, "assign": {int(k): v for k, v in sorted(assign.items())}})
    return d, summary


def partition_summary(d: pd.DataFrame) -> dict:
    out = {}
    for p in PARTITIONS:
        x = d[d.partition == p]
        v = x[LABELS].to_numpy(dtype=np.float64)
        out[p] = {"rows": int(len(x)), "patients": int(x.patient_id.nunique()), "components": int(x.component.nunique()),
                  "eegs": int(x.eeg_id.nunique()), "spectrograms": int(x.spectrogram_id.nunique()),
                  "row_share": round(len(x) / len(d), 4),
                  "class_vote_share": {k: round(float(v[:, i].sum() / v.sum()), 4) for i, k in enumerate(LABELS)},
                  "unique_majority_counts": _majority_counts(v)}
    return out


def _majority_counts(v: np.ndarray) -> dict:
    mx = v.max(1, keepdims=True); uniq = (v == mx).sum(1) == 1
    am = v.argmax(1)
    return {k: int(((am == i) & uniq).sum()) for i, k in enumerate(LABELS)} | {"ties": int((~uniq).sum())}


def leakage_audit(d: pd.DataFrame) -> dict:
    """5x5 intersection matrices for patient, eeg and spectrogram membership; off-diagonals must be zero."""
    out = {"forbidden_overlap": 0}
    for key in ["patient_id", "eeg_id", "spectrogram_id", "component"]:
        sets = {p: set(d.loc[d.partition == p, key].unique()) for p in PARTITIONS}
        mat = [[len(sets[a] & sets[b]) for b in PARTITIONS] for a in PARTITIONS]
        off = sum(mat[i][j] for i in range(5) for j in range(5) if i != j)
        out[key] = {"matrix": mat, "off_diagonal_total": int(off)}
        out["forbidden_overlap"] += int(off)
    out["status"] = "PASS" if out["forbidden_overlap"] == 0 else "FAIL"
    return out
