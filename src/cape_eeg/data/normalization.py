"""Train-only per-view, per-region robust normalization (docs/02 section 9)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts import N_REGIONS, stable_hash
from ..status import write_json, read_json
from .spectral import unpack_mask
from .cache import CacheReader, LOCAL_SHAPE, CONTEXT_SHAPE

CLIP = 8.0


def fit_normalizer(reader: CacheReader, train_rows: np.ndarray, n_sample: int = 4000, seed: int = 20260907,
                   max_cells_per_region: int = 2_000_000) -> dict:
    """Median and 1.4826*MAD of valid log-power cells per view and region on a bounded train sample."""
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(np.asarray(train_rows), size=min(n_sample, len(train_rows)), replace=False))
    b = reader.rows(rows)
    lm = unpack_mask(b["local_mask"], (len(rows), *LOCAL_SHAPE)); cm = unpack_mask(b["context_mask"], (len(rows), *CONTEXT_SHAPE))
    out = {"views": {}, "n_rows": int(len(rows)), "seed": seed, "alt_uniform": bool(reader.alt),
           "preprocess_hash": reader.manifest["preprocess_hash"], "cache_hash": reader.manifest["cache_hash"]}
    for view, vals, mask in [("local", b["local"], lm), ("context", b["context"], cm)]:
        stats = []
        for r in range(N_REGIONS):
            v = vals[:, r].astype(np.float32)[mask[:, r]]
            if v.size > max_cells_per_region:
                v = v[rng.choice(v.size, max_cells_per_region, replace=False)]
            med = float(np.median(v)) if v.size else 0.0
            mad = float(np.median(np.abs(v - med))) * 1.4826 if v.size else 1.0
            if mad < 1e-3:
                mad = float(np.std(v)) if v.size and np.std(v) > 1e-3 else 1.0
            stats.append({"median": med, "scale": mad, "n_cells": int(v.size)})
        out["views"][view] = stats
    out["normalizer_hash"] = stable_hash(out)
    return out


def save_normalizer(norm: dict, path: Path):
    write_json(path, norm); return path


def load_normalizer(path: Path) -> dict:
    n = read_json(path)
    if n is None:
        raise FileNotFoundError(path)
    return n


class Normalizer:
    """Applies (x - median)/scale per region, clips to [-CLIP, CLIP], zeroes invalid cells AFTER normalization."""

    def __init__(self, norm: dict):
        self.norm = norm
        self.med = {v: np.asarray([s["median"] for s in st], np.float32)[None, :, None, None] for v, st in norm["views"].items()}
        self.scale = {v: np.asarray([s["scale"] for s in st], np.float32)[None, :, None, None] for v, st in norm["views"].items()}

    def __call__(self, view: str, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        x = (values.astype(np.float32) - self.med[view]) / self.scale[view]
        x = np.clip(x, -CLIP, CLIP)
        x[~mask] = 0.0
        return x
