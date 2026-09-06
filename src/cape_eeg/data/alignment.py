"""Exact temporal alignment for raw EEG windows and supplied spectrogram intervals (docs/02 section 4)."""
from __future__ import annotations

import numpy as np

from ..contracts import (EEG_SAMPLE_RATE_HZ, RAW_WINDOW_SAMPLES, CONTEXT_SECONDS, eeg_window_bounds,
                         spectrogram_interval)


def extract_eeg_window(signal: np.ndarray, offset_seconds: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (window [10000, C] float32, observed mask [10000, C] bool, info).

    Truncated edges are padded with zeros and marked unobserved: the window is never shifted,
    wrapped or filled by repeating the last row. Nonfinite source values are also unobserved.
    """
    n = signal.shape[0]
    raw = float(offset_seconds) * EEG_SAMPLE_RATE_HZ
    if abs(raw - round(raw)) > 1e-6:
        raise ValueError(f"offset {offset_seconds} off the sample grid")
    start = int(round(raw)); stop = start + RAW_WINDOW_SAMPLES
    out = np.zeros((RAW_WINDOW_SAMPLES, signal.shape[1]), dtype=np.float32)
    obs = np.zeros_like(out, dtype=bool)
    s0, s1 = max(start, 0), min(stop, n)
    if s1 > s0:
        seg = np.asarray(signal[s0:s1], dtype=np.float32)
        fin = np.isfinite(seg)
        out[s0 - start:s1 - start] = np.where(fin, seg, 0.0)
        obs[s0 - start:s1 - start] = fin
    info = {"start": start, "stop": stop, "available": n, "edge_missing_samples": RAW_WINDOW_SAMPLES - (s1 - s0),
            "complete": (start >= 0 and stop <= n)}
    if info["complete"]:
        eeg_window_bounds(offset_seconds, n)  # contract check
    return out, obs, info


def select_spectrogram_rows(time_seconds: np.ndarray, offset_seconds: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Select supplied spectrogram rows whose interval overlaps [s, s+600).

    Row j with timestamp t_j is taken to cover [t_j - h/2, t_j + h/2) where h is the measured
    spacing (checked to be uniform). Returns (row indices, row edges [k+1] in seconds relative
    to s, info). Out-of-range parts are simply absent; they are masked downstream, never wrapped.
    """
    t = np.asarray(time_seconds, dtype=np.float64)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("empty time column")
    d = np.diff(t)
    if t.size > 1 and (np.any(d <= 0) or not np.allclose(d, d[0])):
        raise ValueError("spectrogram time column is not monotonic with uniform spacing")
    h = float(d[0]) if t.size > 1 else 2.0
    s, e = spectrogram_interval(offset_seconds)
    lo, hi = t - h / 2, t + h / 2
    keep = np.where((hi > s) & (lo < e))[0]
    edges = np.concatenate([[lo[keep[0]]], hi[keep]]) - s if keep.size else np.zeros(1)
    info = {"spacing": h, "n_rows": int(keep.size), "coverage_seconds": float(np.clip(hi[keep], s, e).sum() - np.clip(lo[keep], s, e).sum()) if keep.size else 0.0,
            "expected_seconds": float(CONTEXT_SECONDS), "first_time": float(t[keep[0]]) if keep.size else None}
    return keep, edges, info
