"""Tiny end-to-end cache round trip on synthetic rows (tmp_path), plus Normalizer and the partition guard."""

import json

import numpy as np
import pandas as pd
import pytest

from cape_eeg.contracts import LABELS, SHARD_ROWS
from cape_eeg.data.cache import (build_row_index, preallocate, shard_paths, _partial, _open, _init_worker, _G,
                                 finalize, CacheReader, LOCAL_SHAPE, CONTEXT_SHAPE, LOCAL_MASK_BYTES, CONTEXT_MASK_BYTES)
from cape_eeg.data.dataset import select_rows, PartitionGuard
from cape_eeg.data.normalization import Normalizer, CLIP
from cape_eeg.data.spectral import pack_mask, unpack_mask

N_ROWS = 5
assert N_ROWS < SHARD_ROWS


def _synthetic_index(n=N_ROWS):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "label_id": [50, 10, 30, 20, 40][:n],                 # deliberately unsorted; build_row_index sorts
        "patient_id": [7, 7, 8, 9, 9][:n],
        "eeg_id": [1001, 1001, 1002, 1003, 1003][:n],
        "eeg_sub_id": [0, 1, 0, 0, 1][:n],
        "eeg_label_offset_seconds": [0.0, 4.0, 0.0, 0.0, 6.0][:n],
        "spectrogram_id": [2001, 2001, 2002, 2003, 2003][:n],
        "spectrogram_sub_id": [0, 1, 0, 0, 1][:n],
        "spectrogram_label_offset_seconds": [0.0, 4.0, 0.0, 0.0, 6.0][:n],
        "expert_consensus": ["Other", "Seizure", "LPD", "GPD", "Other"][:n],
        "partition": ["train", "train", "tune", "test", "test"][:n],
    })
    votes = rng.integers(0, 4, size=(n, 6)); votes[:, 0] += 1
    for i, k in enumerate(LABELS):
        df[k] = votes[:, i]
    return build_row_index(df)


def _synthetic_payload(n=N_ROWS, seed=2):
    rng = np.random.default_rng(seed)
    local = rng.normal(-3.0, 2.0, size=(n, *LOCAL_SHAPE)).astype(np.float16)
    context = rng.normal(-1.0, 1.0, size=(n, *CONTEXT_SHAPE)).astype(np.float16)
    uniform = rng.normal(2.0, 1.0, size=(n, *LOCAL_SHAPE)).astype(np.float16)
    lmask = rng.random((n, *LOCAL_SHAPE)) < 0.8
    cmask = rng.random((n, *CONTEXT_SHAPE)) < 0.9
    umask = rng.random((n, *LOCAL_SHAPE)) < 0.7
    return {"local": local, "context": context, "uniform": uniform,
            "local_mask": lmask, "context_mask": cmask, "uniform_mask": umask}


def _quality(payload, n=N_ROWS, complete=True):
    """Same column set as _local_task/_context_task produce, merged per row like convert() does."""
    q = {}
    for row in range(n):
        q[row] = {"row": row, "local_valid_fraction": float(payload["local_mask"][row].mean()),
                  "uniform_valid_fraction": float(payload["uniform_mask"][row].mean()),
                  "lead_valid_fraction": 1.0, "eeg_edge_missing_samples": 0, "eeg_observed_fraction": 1.0}
        if complete or row < n - 2:     # the incomplete table lacks the context pass on the last two rows
            q[row].update({"context_valid_fraction": float(payload["context_mask"][row].mean()),
                           "context_rows": 300, "context_coverage_seconds": 600.0})
    return pd.DataFrame.from_dict(q, orient="index").sort_index().reset_index(drop=True)


@pytest.fixture
def worker_state(tmp_path):
    """Point the module-level worker state at a tmp cache and restore it afterwards."""
    cache_dir = tmp_path / "cache"
    saved = dict(_G)
    _init_worker(str(cache_dir), True)
    try:
        yield cache_dir
    finally:
        for v in list(_G.values()):
            if isinstance(v, np.memmap):
                v.flush()
        _G.clear(); _G.update(saved)


def _write_all(cache_dir, index, payload):
    for row, shard, pos in zip(index.row, index.shard, index.pos):
        val, msk = shard_paths(cache_dir, int(shard), "local")
        _open(val)[pos] = payload["local"][row]; _open(msk)[pos] = pack_mask(payload["local_mask"][row][None])[0]
        val, msk = shard_paths(cache_dir, int(shard), "context")
        _open(val)[pos] = payload["context"][row]; _open(msk)[pos] = pack_mask(payload["context_mask"][row][None])[0]
        val, msk = shard_paths(cache_dir, int(shard), "local", alt=True)
        _open(val)[pos] = payload["uniform"][row]; _open(msk)[pos] = pack_mask(payload["uniform_mask"][row][None])[0]
    for v in list(_G.values()):
        if isinstance(v, np.memmap):
            v.flush()


def test_build_row_index_sorts_by_label_id():
    idx = _synthetic_index()
    assert list(idx.label_id) == [10, 20, 30, 40, 50]
    assert list(idx.row) == list(range(N_ROWS)) and (idx.shard == 0).all() and list(idx.pos) == list(range(N_ROWS))
    assert "partition" in idx.columns and all(k in idx.columns for k in LABELS)


def test_preallocate_creates_partial_memmaps(worker_state):
    cache_dir = worker_state
    created = preallocate(cache_dir, N_ROWS, alt_uniform=True)
    assert len(created) == 6
    for target in created:
        assert not target.exists() and _partial(target).exists()
    val, msk = shard_paths(cache_dir, 0, "local")
    assert np.load(_partial(val), mmap_mode="r").shape == (N_ROWS, *LOCAL_SHAPE)
    assert np.load(_partial(val), mmap_mode="r").dtype == np.float16
    assert np.load(_partial(msk), mmap_mode="r").shape == (N_ROWS, LOCAL_MASK_BYTES)
    val, msk = shard_paths(cache_dir, 0, "context")
    assert np.load(_partial(val), mmap_mode="r").shape == (N_ROWS, *CONTEXT_SHAPE)
    assert np.load(_partial(msk), mmap_mode="r").dtype == np.uint8
    # idempotent: a second call neither fails nor recreates
    assert preallocate(cache_dir, N_ROWS, alt_uniform=True) == created


def test_finalize_rejects_incomplete_quality_and_keeps_partials(worker_state):
    cache_dir = worker_state
    index = _synthetic_index(); payload = _synthetic_payload()
    created = preallocate(cache_dir, N_ROWS, alt_uniform=True)
    _write_all(cache_dir, index, payload)
    q_incomplete = _quality(payload, complete=False)
    assert q_incomplete.context_valid_fraction.isna().sum() == 2
    with pytest.raises(RuntimeError, match="incomplete"):
        finalize(cache_dir, index, q_incomplete, alt_uniform=True, log=lambda *a: None)
    for target in created:
        assert _partial(target).exists() and not target.exists()
    assert not (cache_dir / "manifest.json").exists()
    # a table missing the context column entirely, or with too few rows, is also rejected
    with pytest.raises(RuntimeError):
        finalize(cache_dir, index, q_incomplete.drop(columns=["context_valid_fraction"]), alt_uniform=True, log=lambda *a: None)
    with pytest.raises(RuntimeError):
        finalize(cache_dir, index, _quality(payload).iloc[:3], alt_uniform=True, log=lambda *a: None)
    with pytest.raises(RuntimeError):
        finalize(cache_dir, index, pd.DataFrame(columns=["row"]), alt_uniform=True, log=lambda *a: None)


def test_finalize_and_reader_roundtrip(worker_state):
    cache_dir = worker_state
    index = _synthetic_index(); payload = _synthetic_payload()
    created = preallocate(cache_dir, N_ROWS, alt_uniform=True)
    _write_all(cache_dir, index, payload)
    quality = _quality(payload)
    logs = []
    manifest = finalize(cache_dir, index, quality, alt_uniform=True, log=logs.append)
    assert logs and "manifest" in logs[0]
    for target in created:
        assert target.exists() and not _partial(target).exists()
    on_disk = json.loads((cache_dir / "manifest.json").read_text())
    assert on_disk["rows"] == N_ROWS and on_disk["within_cap"] is True
    assert on_disk["preprocess_hash"] == manifest["preprocess_hash"] and len(on_disk["cache_hash"]) == 16
    assert on_disk["planned"]["rows"] == N_ROWS
    assert on_disk["active_payload_bytes"] == sum(p.stat().st_size for p in created if "alt_uniform" not in str(p))
    assert on_disk["alt_uniform_bytes"] == sum(p.stat().st_size for p in created if "alt_uniform" in str(p))
    assert on_disk["rows_both_invalid"] == 0
    assert (cache_dir / "row_index.parquet").exists() and (cache_dir / "row_quality.parquet").exists()
    assert not (cache_dir / "manifest.json.partial").exists()

    reader = CacheReader(cache_dir)
    assert reader.n == N_ROWS and reader.manifest["rows"] == N_ROWS
    assert list(reader.index.row) == list(range(N_ROWS)) and len(reader.quality) == N_ROWS
    order = np.array([3, 0, 4, 1])
    b = reader.rows(order)
    assert b["local"].dtype == np.float16 and b["context"].dtype == np.float16
    assert b["local_mask"].dtype == np.uint8 and b["context_mask"].dtype == np.uint8
    assert b["local"].shape == (4, *LOCAL_SHAPE) and b["context"].shape == (4, *CONTEXT_SHAPE)
    assert b["local_mask"].shape == (4, LOCAL_MASK_BYTES) and b["context_mask"].shape == (4, CONTEXT_MASK_BYTES)
    np.testing.assert_array_equal(b["local"], payload["local"][order])
    np.testing.assert_array_equal(b["context"], payload["context"][order])
    np.testing.assert_array_equal(unpack_mask(b["local_mask"], (4, *LOCAL_SHAPE)), payload["local_mask"][order])
    np.testing.assert_array_equal(unpack_mask(b["context_mask"], (4, *CONTEXT_SHAPE)), payload["context_mask"][order])
    # a single row and a repeated row also come back in request order
    b1 = reader.rows(np.array([2, 2, 1]))
    np.testing.assert_array_equal(b1["local"], payload["local"][[2, 2, 1]])

    alt = CacheReader(cache_dir, alt_uniform=True)
    ba = alt.rows(order)
    np.testing.assert_array_equal(ba["local"], payload["uniform"][order])
    np.testing.assert_array_equal(unpack_mask(ba["local_mask"], (4, *LOCAL_SHAPE)), payload["uniform_mask"][order])
    np.testing.assert_array_equal(ba["context"], payload["context"][order])   # the context view is shared


# ------------------------------------------------------------------ Normalizer
def _norm_dict():
    return {"views": {"local": [{"median": -3.0, "scale": 2.0}, {"median": 0.0, "scale": 1.0},
                                {"median": 1.0, "scale": 0.5}, {"median": -1.0, "scale": 4.0}],
                      "context": [{"median": 0.0, "scale": 1.0}] * 4}}


def test_normalizer_clips_and_zeroes_invalid_after_normalization():
    norm = Normalizer(_norm_dict())
    x = np.zeros((2, 4, 64, 32), np.float16)
    x[0, 0] = -3.0            # exactly the median of region 0 -> 0
    x[0, 1] = 100.0           # (100-0)/1 -> clipped to +8
    x[0, 2] = -100.0          # (-100-1)/0.5 -> clipped to -8
    x[0, 3] = 3.0             # (3+1)/4 -> 1.0
    x[1] = 5.0
    mask = np.ones_like(x, dtype=bool)
    mask[1, 0, :, :] = False  # an invalid region whose raw value is far from the median
    mask[0, 3, 10, 5] = False
    y = norm("local", x, mask)
    assert y.dtype == np.float32
    assert np.all(y[0, 0] == 0.0)
    assert np.all(y[0, 1] == CLIP) and CLIP == 8.0
    assert np.all(y[0, 2] == -CLIP)
    np.testing.assert_allclose(y[0, 3][mask[0, 3]], 1.0)
    assert y[0, 3, 10, 5] == 0.0                       # invalid cell zeroed even though it would be 1.0
    assert np.all(y[1, 0] == 0.0)                      # invalid region zeroed, not (5+3)/2 = 4
    np.testing.assert_allclose(y[1, 1], 5.0)
    assert np.all(np.abs(y) <= CLIP)
    # zeroing happens after normalization: a valid cell at the median is 0 but was not "masked"
    z = norm("context", np.full((1, 4, 64, 64), 20.0, np.float16), np.ones((1, 4, 64, 64), bool))
    assert np.all(z == CLIP)


# ------------------------------------------------------------------ partition guard
def test_select_rows_guards_the_locked_test_partition():
    idx = _synthetic_index()
    with pytest.raises(PartitionGuard):
        select_rows(idx, ["test"])
    with pytest.raises(PartitionGuard):
        select_rows(idx, ["train", "test"], role="development")
    np.testing.assert_array_equal(select_rows(idx, ["train"]), idx.row[idx.partition == "train"].to_numpy())
    np.testing.assert_array_equal(select_rows(idx, ["train", "tune"], role="development"),
                                  idx.row[idx.partition.isin(["train", "tune"])].to_numpy())
    locked = select_rows(idx, ["test"], role="locked_evaluation")
    np.testing.assert_array_equal(locked, idx.row[idx.partition == "test"].to_numpy())
    assert len(locked) == 2
