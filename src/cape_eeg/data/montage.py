"""Regional bipolar montage (docs/02 section 5). Channel spellings are validated, never guessed."""
from __future__ import annotations

import numpy as np

from ..contracts import REGIONS

BIPOLAR_CHAINS = {
    "LL": [("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1")],
    "RL": [("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2")],
    "LP": [("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1")],
    "RP": [("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2")],
}
EXCLUDED_CHANNELS = ("EKG",)
UNUSED_MIDLINE = ("Fz", "Cz", "Pz")
REQUIRED_ELECTRODES = sorted({e for chain in BIPOLAR_CHAINS.values() for pair in chain for e in pair})
EXPECTED_SOURCE_COLUMNS = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz",
                           "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2", "EKG"]


def validate_columns(columns) -> dict:
    cols = list(columns)
    missing = [e for e in REQUIRED_ELECTRODES if e not in cols]
    extra = [c for c in cols if c not in EXPECTED_SOURCE_COLUMNS]
    if missing:
        raise ValueError(f"montage electrodes missing from source columns: {missing}")
    return {"n_columns": len(cols), "missing": missing, "unexpected": extra,
            "ekg_present": "EKG" in cols, "midline_unused": [c for c in UNUSED_MIDLINE if c in cols]}


def bipolar_index_pairs(columns) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays (a, b) so that leads = X[:, a] - X[:, b], ordered region-major (16 leads)."""
    idx = {c: i for i, c in enumerate(columns)}
    a, b = [], []
    for region in REGIONS:
        for e1, e2 in BIPOLAR_CHAINS[region]:
            a.append(idx[e1]); b.append(idx[e2])
    return np.asarray(a), np.asarray(b)


def montage_signature() -> dict:
    return {"chains": BIPOLAR_CHAINS, "excluded": list(EXCLUDED_CHANNELS), "regions": REGIONS}
