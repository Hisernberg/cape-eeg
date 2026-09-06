"""Bounded mini-batch access to the cache: mmap gather -> float32 normalized tensors with masks."""
from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..contracts import LABELS, N_REGIONS, FOCAL_TIME_BINS, CONTEXT_TIME_BINS, FREQUENCY_BINS, vote_targets, pairwise_disagreement
from .cache import CacheReader, LOCAL_SHAPE, CONTEXT_SHAPE
from .normalization import Normalizer
from .spectral import unpack_mask

FORBIDDEN_FOR_DEVELOPMENT = ("test",)


class PartitionGuard(Exception):
    pass


def select_rows(index: pd.DataFrame, partitions: list[str], role: str = "development") -> np.ndarray:
    """Rows of the requested partitions. A development role can never request the locked test partition."""
    if role == "development" and any(p in FORBIDDEN_FOR_DEVELOPMENT for p in partitions):
        raise PartitionGuard(f"development role requested locked partition(s) {partitions}")
    m = index.partition.isin(partitions).to_numpy()
    return index.row.to_numpy()[m]


class CacheDataset:
    """Holds row ids + labels; yields normalized batches. Values are converted to float32 per batch only."""

    def __init__(self, reader: CacheReader, rows: np.ndarray, normalizer: Normalizer, quarantine_both_invalid: bool = True):
        self.reader, self.norm = reader, normalizer
        rows = np.asarray(rows)
        q = reader.quality.set_index("row").loc[rows]
        both_bad = (q.local_valid_fraction.to_numpy() == 0) & (q.context_valid_fraction.to_numpy() == 0)
        self.n_quarantined = int(both_bad.sum())
        if quarantine_both_invalid:
            rows = rows[~both_bad]
        self.rows = rows
        idx = reader.index.set_index("row").loc[rows]
        self.votes = idx[LABELS].to_numpy(dtype=np.int64)
        self.q = vote_targets(self.votes).astype(np.float32)
        self.d = pairwise_disagreement(self.votes).astype(np.float32)   # nan for n==1
        self.n_votes = self.votes.sum(1)
        self.patient = idx.patient_id.to_numpy(); self.label_id = idx.label_id.to_numpy()
        self.component = idx.component.to_numpy() if "component" in idx.columns else self.patient
        self.partition = idx.partition.to_numpy() if "partition" in idx.columns else np.array(["?"] * len(rows))

    def __len__(self):
        return len(self.rows)

    def gather(self, sel: np.ndarray) -> dict:
        """sel: positions into self.rows. Returns numpy dict (float32 values already normalized and masked)."""
        rows = self.rows[sel]
        b = self.reader.rows(rows)
        lm = unpack_mask(b["local_mask"], (len(rows), *LOCAL_SHAPE)); cm = unpack_mask(b["context_mask"], (len(rows), *CONTEXT_SHAPE))
        local = self.norm("local", b["local"], lm); context = self.norm("context", b["context"], cm)
        return {"local": local, "local_mask": lm, "context": context, "context_mask": cm,
                "valid_l": lm.reshape(len(rows), -1).mean(1).astype(np.float32),
                "valid_c": cm.reshape(len(rows), -1).mean(1).astype(np.float32),
                "q": self.q[sel], "votes": self.votes[sel], "d": self.d[sel], "n_votes": self.n_votes[sel], "pos": sel}

    def batches(self, batch_size: int, shuffle: bool, seed: int = 0, prefetch: int = 6, drop_last: bool = False):
        n = len(self)
        order = np.random.default_rng(seed).permutation(n) if shuffle else np.arange(n)
        if drop_last:
            order = order[: (n // batch_size) * batch_size]
        chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        qout: queue.Queue = queue.Queue(maxsize=prefetch)
        stop = object()

        def worker():
            try:
                for c in chunks:
                    qout.put(self.gather(np.sort(c)))
            except Exception as e:  # propagate
                qout.put(e)
            qout.put(stop)
        th = threading.Thread(target=worker, daemon=True); th.start()
        while True:
            item = qout.get()
            if item is stop:
                break
            if isinstance(item, Exception):
                raise item
            yield item
        th.join()


def to_device(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        if k in ("pos",):
            out[k] = v; continue
        t = torch.from_numpy(np.ascontiguousarray(v))
        out[k] = t.to(device, non_blocking=True)
    return out
