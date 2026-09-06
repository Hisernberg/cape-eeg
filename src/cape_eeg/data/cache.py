"""Bounded paired spectral cache: float16 shards + packed validity, written by file-grouped workers.

Layout (private/cache/<preprocess_hash>/):
  row_index.parquet                     private row map sorted by label_id
  shards/shard_XXXX_{local,context}.npy float16 [n,4,64,32] / [n,4,64,64]
  shards/shard_XXXX_{local,context}_mask.npy uint8 packed little-endian bits
  alt_uniform/shard_XXXX_local{,_mask}.npy   development-only uniform local encoding
  row_quality.parquet, manifest.json
"""
from __future__ import annotations

import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..contracts import (LABELS, SHARD_ROWS, N_REGIONS, FREQUENCY_BINS, FOCAL_TIME_BINS, CONTEXT_TIME_BINS,
                         CACHE_MAX_BYTES, preprocess_signature, stable_hash, file_sha256, cache_bytes)
from ..status import write_json, utc_now
from .alignment import extract_eeg_window, select_spectrogram_rows
from .montage import validate_columns
from .spectral import RawSpectralEncoder, ContextSpectralEncoder, pack_mask

LOCAL_SHAPE = (N_REGIONS, FREQUENCY_BINS, FOCAL_TIME_BINS)
CONTEXT_SHAPE = (N_REGIONS, FREQUENCY_BINS, CONTEXT_TIME_BINS)
LOCAL_MASK_BYTES = int(np.prod(LOCAL_SHAPE)) // 8
CONTEXT_MASK_BYTES = int(np.prod(CONTEXT_SHAPE)) // 8

_G = {}  # per-worker state


def build_row_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.sort_values("label_id").reset_index(drop=True)
    idx["row"] = np.arange(len(idx)); idx["shard"] = idx["row"] // SHARD_ROWS; idx["pos"] = idx["row"] % SHARD_ROWS
    keep = ["row", "shard", "pos", "label_id", "patient_id", "eeg_id", "eeg_sub_id", "eeg_label_offset_seconds",
            "spectrogram_id", "spectrogram_sub_id", "spectrogram_label_offset_seconds", "expert_consensus"] + LABELS
    extra = [c for c in ["component", "partition"] if c in idx.columns]
    return idx[keep + extra]


def shard_paths(cache_dir: Path, shard: int, kind: str, alt: bool = False) -> tuple[Path, Path]:
    base = cache_dir / ("alt_uniform" if alt else "shards")
    return base / f"shard_{shard:04d}_{kind}.npy", base / f"shard_{shard:04d}_{kind}_mask.npy"


def _partial(p: Path) -> Path:
    return p.with_name(p.name + ".partial")


def preallocate(cache_dir: Path, n_rows: int, alt_uniform: bool = True) -> list[Path]:
    """Create .partial memmaps for every shard (header written, data zero-filled lazily)."""
    (cache_dir / "shards").mkdir(parents=True, exist_ok=True)
    if alt_uniform:
        (cache_dir / "alt_uniform").mkdir(parents=True, exist_ok=True)
    created = []
    for s in range((n_rows + SHARD_ROWS - 1) // SHARD_ROWS):
        n = min(SHARD_ROWS, n_rows - s * SHARD_ROWS)
        specs = [("local", False, (n, *LOCAL_SHAPE), np.float16), ("local", False, (n, LOCAL_MASK_BYTES), np.uint8),
                 ("context", False, (n, *CONTEXT_SHAPE), np.float16), ("context", False, (n, CONTEXT_MASK_BYTES), np.uint8)]
        if alt_uniform:
            specs += [("local", True, (n, *LOCAL_SHAPE), np.float16), ("local", True, (n, LOCAL_MASK_BYTES), np.uint8)]
        for kind, alt, shape, dtype in specs:
            val, msk = shard_paths(cache_dir, s, kind, alt)
            target = msk if dtype == np.uint8 else val
            p = _partial(target)
            if not p.exists() and not target.exists():
                mm = np.lib.format.open_memmap(p, mode="w+", dtype=dtype, shape=shape); del mm
            created.append(target)
    return created


def _open(path: Path, mode="r+"):
    key = (str(path), mode)
    if key not in _G:
        p = _partial(path) if _partial(path).exists() else path
        _G[key] = np.load(p, mmap_mode=mode)
    return _G[key]


def _init_worker(cache_dir: str, alt_uniform: bool):
    try:  # BLAS threads are fixed at import time; threadpoolctl can still limit them post hoc
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    _G.clear(); _G["cache_dir"] = Path(cache_dir); _G["alt"] = alt_uniform


def _local_task(args) -> list[dict]:
    """One EEG recording: all its label windows -> local foveated (+uniform) shards."""
    eeg_path, rows = args  # rows: list of (row, shard, pos, offset)
    cache_dir = _G["cache_dir"]
    tab = pq.read_table(eeg_path)
    cols = tab.column_names
    if "enc" not in _G or _G.get("enc_cols") != cols:
        validate_columns(cols); _G["enc"] = RawSpectralEncoder(cols); _G["enc_cols"] = cols
    enc = _G["enc"]
    X = np.column_stack([tab.column(c).to_numpy(zero_copy_only=False) for c in cols]).astype(np.float32)
    out = []
    for row, shard, pos, off in rows:
        w, obs, info = extract_eeg_window(X, off)
        r = enc.encode(w, obs)
        val, msk = shard_paths(cache_dir, shard, "local")
        _open(val)[pos] = r["foveated"].astype(np.float16); _open(msk)[pos] = pack_mask(r["foveated_mask"][None])[0]
        if _G["alt"]:
            val, msk = shard_paths(cache_dir, shard, "local", alt=True)
            _open(val)[pos] = r["uniform"].astype(np.float16); _open(msk)[pos] = pack_mask(r["uniform_mask"][None])[0]
        out.append({"row": row, "local_valid_fraction": float(r["foveated_mask"].mean()),
                    "uniform_valid_fraction": float(r["uniform_mask"].mean()),
                    "lead_valid_fraction": r["lead_valid_fraction"], "eeg_edge_missing_samples": info["edge_missing_samples"],
                    "eeg_observed_fraction": float(obs.mean())})
    for k, v in list(_G.items()):
        if isinstance(v, np.memmap): v.flush()
    return out


def _context_task(args) -> list[dict]:
    spec_path, rows = args  # rows: list of (row, shard, pos, offset)
    cache_dir = _G["cache_dir"]
    tab = pq.read_table(spec_path)
    cols = tab.column_names
    if "cenc" not in _G or _G.get("cenc_cols") != cols:
        _G["cenc"] = ContextSpectralEncoder(cols); _G["cenc_cols"] = cols
    cenc = _G["cenc"]
    t = tab.column("time").to_numpy().astype(np.float64)
    V = np.column_stack([tab.column(c).to_numpy(zero_copy_only=False) for c in cols]).astype(np.float32)
    out, memo = [], {}
    for row, shard, pos, off in rows:
        if off not in memo:
            keep, edges, info = select_spectrogram_rows(t, off)
            r = cenc.encode(V[keep], edges) if keep.size else {"context": np.zeros(CONTEXT_SHAPE, np.float32), "context_mask": np.zeros(CONTEXT_SHAPE, bool), "valid_fraction": 0.0}
            memo[off] = (r, info)
        r, info = memo[off]
        val, msk = shard_paths(cache_dir, shard, "context")
        _open(val)[pos] = r["context"].astype(np.float16); _open(msk)[pos] = pack_mask(r["context_mask"][None])[0]
        out.append({"row": row, "context_valid_fraction": r["valid_fraction"], "context_rows": info["n_rows"],
                    "context_coverage_seconds": info["coverage_seconds"]})
    for k, v in list(_G.items()):
        if isinstance(v, np.memmap): v.flush()
    return out


def _group(index: pd.DataFrame, id_col: str, off_col: str, folder: Path, limit_ids=None):
    tasks = []
    for fid, g in index.groupby(id_col, sort=True):
        if limit_ids is not None and fid not in limit_ids:
            continue
        tasks.append((str(folder / f"{fid}.parquet"), list(zip(g.row.to_numpy(), g.shard.to_numpy(), g.pos.to_numpy(), g[off_col].to_numpy(dtype=float)))))
    return tasks


def convert(index: pd.DataFrame, data_root: Path, cache_dir: Path, workers: int = 12, alt_uniform: bool = True,
            limit_eeg_ids=None, limit_spec_ids=None, log=print) -> pd.DataFrame:
    """Run both passes; returns the per-row quality table. Idempotent per row (rows are overwritten in place)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    preallocate(cache_dir, len(index), alt_uniform)
    quality = {}
    t0 = time.time()
    for name, fn, tasks in [("local", _local_task, _group(index, "eeg_id", "eeg_label_offset_seconds", data_root / "train_eegs", limit_eeg_ids)),
                            ("context", _context_task, _group(index, "spectrogram_id", "spectrogram_label_offset_seconds", data_root / "train_spectrograms", limit_spec_ids))]:
        done_rows, t1 = 0, time.time()
        with Pool(workers, initializer=_init_worker, initargs=(str(cache_dir), alt_uniform)) as pool:
            for i, res in enumerate(pool.imap_unordered(fn, tasks, chunksize=4)):
                for rec in res:
                    quality.setdefault(rec["row"], {}).update(rec)
                done_rows += len(res)
                if (i + 1) % 500 == 0 or i + 1 == len(tasks):
                    el = time.time() - t1
                    log(f"[{name}] files {i+1}/{len(tasks)} rows {done_rows} elapsed {el:.0f}s rate {done_rows/max(el,1e-9):.0f} rows/s")
    q = pd.DataFrame.from_dict(quality, orient="index").sort_index().reset_index(drop=True)
    log(f"conversion finished in {time.time()-t0:.0f}s for {len(q)} rows")
    return q


def finalize(cache_dir: Path, index: pd.DataFrame, quality: pd.DataFrame, alt_uniform: bool = True, log=print) -> dict:
    """Verify every row was written in both passes, rename .partial shards atomically, checksum, write manifest."""
    n = len(index)
    got = set(quality.row.to_numpy()) if len(quality) else set()
    need_cols = ["local_valid_fraction", "context_valid_fraction"]
    complete = len(got) == n and all(c in quality.columns for c in need_cols) and quality[need_cols].notna().all().all()
    if not complete:
        raise RuntimeError(f"cache incomplete: {len(got)}/{n} rows with both views; shards remain .partial")
    shards, total_bytes, alt_bytes = [], 0, 0
    for s in range((n + SHARD_ROWS - 1) // SHARD_ROWS):
        rec = {"shard": s}
        for kind, alt in [("local", False), ("context", False)] + ([("local", True)] if alt_uniform else []):
            for p in shard_paths(cache_dir, s, kind, alt):
                if _partial(p).exists():
                    os.replace(_partial(p), p)
                rec[f"{'alt_' if alt else ''}{p.name}"] = file_sha256(p)
                if alt: alt_bytes += p.stat().st_size
                else: total_bytes += p.stat().st_size
        shards.append(rec)
    index_path = cache_dir / "row_index.parquet"; index.to_parquet(index_path, index=False)
    quality_path = cache_dir / "row_quality.parquet"; quality.to_parquet(quality_path, index=False)
    meta_bytes = index_path.stat().st_size + quality_path.stat().st_size
    sig = preprocess_signature()
    manifest = {"preprocess_hash": stable_hash(sig), "preprocess_signature": sig, "rows": n, "shards": shards,
                "active_cache_bytes": total_bytes + meta_bytes, "active_payload_bytes": total_bytes, "metadata_bytes": meta_bytes,
                "alt_uniform_bytes": alt_bytes, "within_cap": (total_bytes + meta_bytes) <= CACHE_MAX_BYTES,
                "planned": cache_bytes(n).as_dict(), "created": utc_now(),
                "quality_summary": {c: {"mean": float(quality[c].mean()), "min": float(quality[c].min())} for c in quality.columns if c != "row" and quality[c].dtype.kind == "f"},
                "rows_local_all_invalid": int((quality.local_valid_fraction == 0).sum()),
                "rows_context_all_invalid": int((quality.context_valid_fraction == 0).sum()),
                "rows_both_invalid": int(((quality.local_valid_fraction == 0) & (quality.context_valid_fraction == 0)).sum())}
    manifest["cache_hash"] = stable_hash([r for r in shards])
    write_json(cache_dir / "manifest.json", manifest)
    log(f"cache manifest written: active {manifest['active_cache_bytes']/1e9:.3f} GB (cap 6.0), alt {alt_bytes/1e9:.3f} GB, within_cap={manifest['within_cap']}")
    return manifest


class CacheReader:
    """Memory-mapped read access: values float16 -> float32 per mini-batch only."""

    def __init__(self, cache_dir: Path, alt_uniform: bool = False):
        self.dir = Path(cache_dir)
        self.manifest = __import__("json").load(open(self.dir / "manifest.json"))
        self.index = pd.read_parquet(self.dir / "row_index.parquet")
        self.quality = pd.read_parquet(self.dir / "row_quality.parquet")
        self.n = len(self.index)
        self.alt = alt_uniform
        self._mm = {}

    def _get(self, shard: int, kind: str, mask: bool, alt: bool = False):
        key = (shard, kind, mask, alt)
        if key not in self._mm:
            val, msk = shard_paths(self.dir, shard, kind, alt)
            self._mm[key] = np.load(msk if mask else val, mmap_mode="r")
        return self._mm[key]

    def rows(self, rows: np.ndarray) -> dict:
        rows = np.asarray(rows)
        shards, pos = rows // SHARD_ROWS, rows % SHARD_ROWS
        out = {k: [] for k in ["local", "local_mask", "context", "context_mask"]}
        order = np.argsort(shards, kind="stable"); inv = np.empty_like(order); inv[order] = np.arange(len(order))
        for s in np.unique(shards):
            sel = pos[shards == s]
            out["local"].append(np.asarray(self._get(s, "local", False, self.alt)[sel]))
            out["local_mask"].append(np.asarray(self._get(s, "local", True, self.alt)[sel]))
            out["context"].append(np.asarray(self._get(s, "context", False)[sel]))
            out["context_mask"].append(np.asarray(self._get(s, "context", True)[sel]))
        res = {}
        for k, v in out.items():
            arr = np.concatenate(v, 0)   # grouped by ascending shard, stable within shard == order
            res[k] = arr[inv]
        return res
