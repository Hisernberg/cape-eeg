"""Fixed contracts shared by every stage: label order, temporal geometry, byte rules.

Everything here is a deterministic constant or a pure function of its inputs. Nothing
reads data. Numbers are taken from docs/01 and docs/02 of the study package.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np

LABELS = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
CLASS_NAMES = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]
N_CLASSES = 6

EEG_SAMPLE_RATE_HZ = 200
RAW_WINDOW_SECONDS = 50
RAW_WINDOW_SAMPLES = EEG_SAMPLE_RATE_HZ * RAW_WINDOW_SECONDS  # 10,000
TARGET_SECONDS = (20.0, 30.0)  # labeled center inside the 50 s window
CONTEXT_SECONDS = 600
CONTEXT_CENTER_SECONDS = (295.0, 305.0)  # labeled center inside the 600 s context

REGIONS = ["LL", "RL", "LP", "RP"]
N_REGIONS = 4
FREQUENCY_BINS = 64
FOCAL_TIME_BINS = 32
CONTEXT_TIME_BINS = 64
FOVEATED_SEGMENTS = ((0.0, 20.0, 8), (20.0, 30.0, 16), (30.0, 50.0, 8))
FOCAL_TARGET_COLUMNS = (8, 24)  # columns [8,24) of the foveated local view are the target
RAW_FREQUENCY_RANGE_HZ = (0.5, 40.0)

STFT = dict(window="hann", nperseg=256, hop=64, nfft=256, boundary=None, padded=False)
MAX_INTERPOLATION_GAP_SECONDS = 0.25
VALID_WINDOW_FRACTION = 0.90
MINIMUM_VALID_BIPOLAR_LEADS = 2
MIN_BIN_VALID_WEIGHT_FRACTION = 0.50  # a binned cell is valid when >=50% of its weight comes from valid frames

SHARD_ROWS = 2048
CACHE_MAX_BYTES = 6_000_000_000
METADATA_BUDGET_BYTES = 250_000_000

SPLIT_SEED = 20260907
SPLIT_PROPORTIONS = {"train": 0.60, "tune": 0.10, "calibration_t": 0.05, "calibration_p": 0.05, "test": 0.20}
PARTITIONS = list(SPLIT_PROPORTIONS)

# Spectrogram offset field: source spelling irregularity is accepted as an alias.
SPEC_OFFSET_ALIASES = ("spectrogram_label_offset_seconds", "spectogram_label_offset_seconds")

OFFSET_ALIASES = {"spectrogram_label_offset_seconds": SPEC_OFFSET_ALIASES}


# --------------------------------------------------------------------------------------
# Temporal geometry
# --------------------------------------------------------------------------------------
def foveated_time_edges() -> np.ndarray:
    """Edges (seconds) of the 32 foveated local bins: 8 over [0,20), 16 over [20,30), 8 over [30,50)."""
    edges = [0.0]
    for start, stop, n in FOVEATED_SEGMENTS:
        seg = np.linspace(start, stop, n + 1)[1:]
        edges.extend(seg.tolist())
    e = np.asarray(edges, dtype=np.float64)
    assert e.shape == (FOCAL_TIME_BINS + 1,)
    return e


def uniform_time_edges(n_bins: int = FOCAL_TIME_BINS, duration: float = RAW_WINDOW_SECONDS) -> np.ndarray:
    return np.linspace(0.0, float(duration), n_bins + 1)


def context_time_edges() -> np.ndarray:
    return np.linspace(0.0, float(CONTEXT_SECONDS), CONTEXT_TIME_BINS + 1)


def center_weights_from_edges(edges: np.ndarray, center: tuple[float, float] = TARGET_SECONDS) -> np.ndarray:
    """Fraction of each time bin that overlaps the labeled center interval.

    Used for center-region pooling so the uniform comparator pools its true center instead of
    blindly copying columns 8:24.
    """
    lo = np.maximum(edges[:-1], center[0])
    hi = np.minimum(edges[1:], center[1])
    overlap = np.clip(hi - lo, 0.0, None)
    width = edges[1:] - edges[:-1]
    return overlap / width


def linear_frequency_edges(n_bins: int, f_lo: float, f_hi: float) -> np.ndarray:
    return np.linspace(float(f_lo), float(f_hi), n_bins + 1)


def overlap_matrix(src_edges: np.ndarray, dst_edges: np.ndarray) -> np.ndarray:
    """W[j,i] = length of overlap between source interval i and destination interval j.

    Sources and destinations are half-open intervals given by their edges. Rows are not
    normalized here; the caller normalizes by the valid weight actually present.
    """
    s_lo, s_hi = src_edges[:-1][None, :], src_edges[1:][None, :]
    d_lo, d_hi = dst_edges[:-1][:, None], dst_edges[1:][:, None]
    return np.clip(np.minimum(s_hi, d_hi) - np.maximum(s_lo, d_lo), 0.0, None)


# --------------------------------------------------------------------------------------
# Offsets and windows
# --------------------------------------------------------------------------------------
def eeg_window_bounds(offset_seconds: float, n_samples_available: int | None = None) -> tuple[int, int]:
    """Sample bounds [start, stop) for a 50 s window at EEG label offset e.

    Offsets must lie on the 200 Hz sample grid within 1e-6 sample. When the number of
    available samples is supplied, the window must fit completely; converters implement the
    explicit missing-edge path separately (docs/02 section 4).
    """
    raw = float(offset_seconds) * EEG_SAMPLE_RATE_HZ
    if not np.isfinite(raw) or raw < 0:
        raise ValueError(f"invalid EEG offset {offset_seconds!r}")
    if abs(raw - round(raw)) > 1e-6:
        raise ValueError(f"EEG offset {offset_seconds} is not on the 200 Hz sample grid")
    start = int(round(raw))
    stop = start + RAW_WINDOW_SAMPLES
    if n_samples_available is not None and stop > n_samples_available:
        raise ValueError(f"window [{start},{stop}) exceeds {n_samples_available} available samples")
    return start, stop


def spectrogram_interval(offset_seconds: float) -> tuple[float, float]:
    s = float(offset_seconds)
    if not np.isfinite(s) or s < 0:
        raise ValueError(f"invalid spectrogram offset {offset_seconds!r}")
    return s, s + CONTEXT_SECONDS


def spectrogram_center(offset_seconds: float) -> tuple[float, float]:
    s = float(offset_seconds)
    return s + CONTEXT_CENTER_SECONDS[0], s + CONTEXT_CENTER_SECONDS[1]


# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------
def vote_targets(votes: np.ndarray) -> np.ndarray:
    v = np.asarray(votes, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != N_CLASSES:
        raise ValueError("votes must be [N,6]")
    if np.any(v < 0) or np.any(v != np.round(v)):
        raise ValueError("votes must be nonnegative integers")
    n = v.sum(1)
    if np.any(n <= 0):
        raise ValueError("zero-vote rows are rejected")
    return v / n[:, None]


def pairwise_disagreement(votes: np.ndarray) -> np.ndarray:
    """d = 1 - sum_k v_k(v_k-1) / (n(n-1)) for n>1; NaN (unavailable) for n==1."""
    v = np.asarray(votes, dtype=np.float64)
    n = v.sum(1)
    out = np.full(v.shape[0], np.nan)
    m = n > 1
    out[m] = 1.0 - (v[m] * (v[m] - 1.0)).sum(1) / (n[m] * (n[m] - 1.0))
    return out


# --------------------------------------------------------------------------------------
# Byte arithmetic
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CacheBytes:
    rows: int
    values_bytes: int
    mask_bytes: int
    payload_bytes: int
    metadata_allowance_bytes: int
    total_allowance_bytes: int
    within_cap: bool

    def as_dict(self):
        return asdict(self)


def cache_bytes(rows: int, include_uniform_alternate: bool = False) -> CacheBytes:
    per_row_values = N_REGIONS * FREQUENCY_BINS * (FOCAL_TIME_BINS + CONTEXT_TIME_BINS) * 2
    per_row_mask = N_REGIONS * FREQUENCY_BINS * (FOCAL_TIME_BINS + CONTEXT_TIME_BINS) // 8
    if include_uniform_alternate:
        per_row_values += N_REGIONS * FREQUENCY_BINS * FOCAL_TIME_BINS * 2
        per_row_mask += N_REGIONS * FREQUENCY_BINS * FOCAL_TIME_BINS // 8
    values = per_row_values * rows
    masks = per_row_mask * rows
    payload = values + masks
    total = payload + METADATA_BUDGET_BYTES
    return CacheBytes(rows, values, masks, payload, METADATA_BUDGET_BYTES, total, total <= CACHE_MAX_BYTES)


# --------------------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------------------
def stable_hash(obj, length: int = 16) -> str:
    """Deterministic SHA-256 prefix of a JSON-serialisable object (sorted keys)."""
    payload = json.dumps(obj, sort_keys=True, default=_json_default, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:length]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "as_dict"):
        return o.as_dict()
    raise TypeError(f"not serialisable: {type(o)}")


def file_sha256(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def preprocess_signature() -> dict:
    """The exact preprocessing parameters that define a cache; hashed into preprocess_hash."""
    return dict(
        sample_rate=EEG_SAMPLE_RATE_HZ, raw_window=RAW_WINDOW_SECONDS, target=TARGET_SECONDS,
        context=CONTEXT_SECONDS, regions=REGIONS, frequency_bins=FREQUENCY_BINS,
        focal_time_bins=FOCAL_TIME_BINS, context_time_bins=CONTEXT_TIME_BINS,
        foveated=FOVEATED_SEGMENTS, stft=STFT, raw_freq=RAW_FREQUENCY_RANGE_HZ,
        max_gap=MAX_INTERPOLATION_GAP_SECONDS, valid_fraction=VALID_WINDOW_FRACTION,
        min_leads=MINIMUM_VALID_BIPOLAR_LEADS, min_bin_weight=MIN_BIN_VALID_WEIGHT_FRACTION,
        dtype="float16", validity="packbits_little", version=2,
    )
