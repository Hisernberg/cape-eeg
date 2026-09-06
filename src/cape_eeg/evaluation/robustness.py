"""Prespecified six-condition stress suite (docs/04 section 7) on a fixed label-blind test subset.

Conditions are representation-level perturbations applied to normalized inputs, except the STFT
window-length condition, which recomputes the local view from raw EEG for the same rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..contracts import STFT, N_REGIONS, foveated_time_edges, context_time_edges
from ..data.alignment import extract_eeg_window
from ..data.spectral import RawSpectralEncoder
from ..data.dataset import CacheDataset, to_device
from ..data.normalization import Normalizer
from ..training.engine import predict

CONDITIONS = [
    {"id": "clean", "description": "unmodified cached inputs", "severity": 0.0, "units": "none"},
    {"id": "region_missing_LL", "description": "left-lateral region removed from both views (values zero, validity false)", "severity": 1.0, "units": "regions removed"},
    {"id": "time_mask_10pct", "description": "additional contiguous 10% time mask: local [5,10) s, context [60,120) s", "severity": 0.10, "units": "fraction of time axis"},
    {"id": "gain_shift_mild", "description": "+0.5 natural-log power units added to every valid cell of both views", "severity": 0.5, "units": "log-power units"},
    {"id": "gain_shift_strong", "description": "+1.5 natural-log power units added to every valid cell of both views", "severity": 1.5, "units": "log-power units"},
    {"id": "stft_window_512", "description": "local view recomputed with nperseg=nfft=512 (hop 64) instead of 256", "severity": 512, "units": "STFT window samples"},
]


def select_subset(index: pd.DataFrame, n_rows: int = 1024, min_patients: int = 100, seed: int = 20260907) -> np.ndarray:
    """Fixed label-blind subset of test rows: seeded patient-stratified draw, >= min_patients patients."""
    t = index[index.partition == "test"]
    rng = np.random.default_rng(seed)
    pats = rng.permutation(t.patient_id.unique())
    chosen = []
    per_patient = max(1, n_rows // len(pats))
    for p in pats:
        r = t[t.patient_id == p].row.to_numpy()
        chosen.extend(rng.choice(r, min(per_patient, len(r)), replace=False).tolist())
    chosen = np.array(sorted(set(chosen)))
    if len(chosen) > n_rows:
        chosen = np.sort(rng.choice(chosen, n_rows, replace=False))
    sub = t.set_index("row").loc[chosen]
    assert sub.patient_id.nunique() >= min_patients, "subset does not reach the minimum patient count"
    return chosen


def _col_range(edges: np.ndarray, lo: float, hi: float) -> tuple[int, int]:
    cols = np.where((edges[1:] > lo) & (edges[:-1] < hi))[0]
    return int(cols[0]), int(cols[-1]) + 1


def make_perturbation(cond_id: str, normalizer: Normalizer):
    """Return a callable batch -> batch (GPU tensors, in place) for representation-level conditions."""
    scale_l = torch.tensor(normalizer.scale["local"].reshape(-1), dtype=torch.float32)
    scale_c = torch.tensor(normalizer.scale["context"].reshape(-1), dtype=torch.float32)
    if cond_id == "clean" or cond_id == "stft_window_512":
        return None
    def f(b: dict) -> dict:
        L, Lm, C, Cm = b["local"], b["local_mask"], b["context"], b["context_mask"]
        if cond_id == "region_missing_LL":
            L[:, 0] = 0; C[:, 0] = 0; Lm[:, 0] = False; Cm[:, 0] = False
            b["valid_l"] = Lm.reshape(Lm.shape[0], -1).float().mean(1); b["valid_c"] = Cm.reshape(Cm.shape[0], -1).float().mean(1)
        elif cond_id == "time_mask_10pct":
            a0, a1 = _col_range(foveated_time_edges(), 5.0, 10.0); c0, c1 = _col_range(context_time_edges(), 60.0, 120.0)
            L[:, :, :, a0:a1] = 0; Lm[:, :, :, a0:a1] = False; C[:, :, :, c0:c1] = 0; Cm[:, :, :, c0:c1] = False
            b["valid_l"] = Lm.reshape(Lm.shape[0], -1).float().mean(1); b["valid_c"] = Cm.reshape(Cm.shape[0], -1).float().mean(1)
        elif cond_id.startswith("gain_shift"):
            delta = 0.5 if cond_id.endswith("mild") else 1.5
            L += (delta / scale_l.to(L.device))[None, :, None, None] * Lm.float()
            C += (delta / scale_c.to(C.device))[None, :, None, None] * Cm.float()
            L.clamp_(-8, 8); C.clamp_(-8, 8)
        b["local"], b["context"] = L * Lm.float(), C * Cm.float()
        return b
    return f


class STFTVariantDataset(CacheDataset):
    """Same rows/labels but the local view recomputed from raw EEG with a different STFT window."""

    def __init__(self, base: CacheDataset, data_root: Path, nperseg: int = 512, log=print):
        self.__dict__.update(base.__dict__)
        self.override = {}
        idx = base.reader.index.set_index("row").loc[base.rows]
        enc = None
        for eeg_id, g in idx.groupby("eeg_id"):
            X = pd.read_parquet(data_root / "train_eegs" / f"{eeg_id}.parquet")
            if enc is None:
                enc = RawSpectralEncoder(list(X.columns), nperseg=nperseg, hop=STFT["hop"], nfft=nperseg)
            Xv = X.to_numpy(dtype=np.float32)
            for row, off in zip(g.index, g.eeg_label_offset_seconds):
                w, obs, _ = extract_eeg_window(Xv, off); r = enc.encode(w, obs)
                self.override[int(row)] = (r["foveated"], r["foveated_mask"])
        log(f"STFT variant recomputed for {len(self.override)} rows (nperseg={nperseg})")

    def gather(self, sel):
        b = super().gather(sel)
        vals = np.stack([self.override[int(r)][0] for r in self.rows[sel]]); masks = np.stack([self.override[int(r)][1] for r in self.rows[sel]])
        b["local"] = self.norm("local", vals, masks); b["local_mask"] = masks
        b["valid_l"] = masks.reshape(len(sel), -1).mean(1).astype(np.float32)
        return b


def run_suite(model, ds: CacheDataset, normalizer: Normalizer, device, data_root: Path, log=print) -> dict[str, np.ndarray]:
    """Predictions per condition on the same rows (dict cond_id -> p [N,6])."""
    out = {}
    for c in CONDITIONS:
        if c["id"] == "stft_window_512":
            vds = STFTVariantDataset(ds, data_root, log=log); r = predict(model, vds, device)
        else:
            r = predict(model, ds, device, perturb=make_perturbation(c["id"], normalizer))
        out[c["id"]] = r["p"]; log(f"condition {c['id']}: mean max-prob {r['p'].max(1).mean():.3f}")
    return out
