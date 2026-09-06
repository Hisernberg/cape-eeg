"""Synthetic tests for component construction, deterministic allocation and the leakage audit."""

import numpy as np
import pandas as pd
import pytest

from cape_eeg.contracts import LABELS, PARTITIONS, SPLIT_PROPORTIONS
from cape_eeg.data.splits import connected_components, allocate, leakage_audit, partition_summary

N_PATIENTS = 60


def _synthetic_frame(seed=20260907, n_patients=N_PATIENTS):
    """Clearly synthetic cohort: ids are small integers, votes are seeded random integers.

    Patient 0 and patient 1 share one spectrogram (spectrogram_id 100000) so that their
    rows must land in the same connected component.
    """
    rng = np.random.default_rng(seed)
    rows = []
    label_id = 0
    eeg_id = 1000
    for p in range(n_patients):
        for _ in range(int(rng.integers(1, 4))):            # 1..3 EEG recordings per patient
            spec_id = 100000 + eeg_id
            for sub in range(int(rng.integers(2, 7))):      # 2..6 labels per recording
                votes = rng.integers(0, 4, size=6)
                if votes.sum() == 0:
                    votes[int(rng.integers(0, 6))] = 1
                rows.append({"patient_id": p, "eeg_id": eeg_id, "eeg_sub_id": sub, "spectrogram_id": spec_id,
                             "label_id": label_id, **{k: int(v) for k, v in zip(LABELS, votes)}})
                label_id += 1
            eeg_id += 1
    df = pd.DataFrame(rows)
    # the shared spectrogram: one row of patient 1 reuses patient 0's first spectrogram
    shared = int(df.loc[df.patient_id == 0, "spectrogram_id"].iloc[0])
    idx = df.index[df.patient_id == 1][0]
    df.loc[idx, "spectrogram_id"] = shared
    return df


@pytest.fixture(scope="module")
def frame():
    return _synthetic_frame()


def test_frame_is_well_formed(frame):
    assert frame.patient_id.nunique() == N_PATIENTS
    assert frame.label_id.is_unique
    assert (frame[LABELS].sum(axis=1) > 0).all()
    assert 300 <= len(frame) <= 1200


def test_connected_components_merge_patients_sharing_a_spectrogram(frame):
    comp = connected_components(frame)
    assert comp.index.equals(frame.index) and comp.name == "component"
    c0 = set(comp[frame.patient_id == 0]); c1 = set(comp[frame.patient_id == 1])
    assert len(c0) == 1 and c0 == c1                       # merged through the shared spectrogram
    assert comp.nunique() == N_PATIENTS - 1                # every other patient is its own component
    # a component never splits a patient, an eeg or a spectrogram
    for key in ["patient_id", "eeg_id", "spectrogram_id"]:
        assert (frame.assign(component=comp).groupby(key).component.nunique() == 1).all()
    # components are labelled 0..K-1 and are invariant to row order
    assert sorted(comp.unique()) == list(range(N_PATIENTS - 1))
    shuffled = frame.sample(frac=1.0, random_state=3)
    comp_s = connected_components(shuffled).reindex(frame.index)
    same = pd.crosstab(comp, comp_s)
    assert (same.gt(0).sum(axis=1) == 1).all() and (same.gt(0).sum(axis=0) == 1).all()


def test_allocate_assigns_every_row_with_target_proportions(frame):
    d, summary = allocate(frame)
    assert len(d) == len(frame)
    assert d.partition.notna().all()
    assert set(d.partition.unique()) == set(PARTITIONS)
    assert "component" in d.columns
    shares = d.partition.value_counts(normalize=True)
    for p in PARTITIONS:
        assert abs(shares[p] - SPLIT_PROPORTIONS[p]) < 0.15, (p, shares[p])
    assert summary["seed"] == 20260907 and summary["proportions"] == SPLIT_PROPORTIONS
    assert len(summary["split_hash"]) == 16
    for p in PARTITIONS:
        s = summary[p]
        assert s["rows"] == int((d.partition == p).sum())
        assert s["patients"] == d.loc[d.partition == p, "patient_id"].nunique()
        assert abs(sum(s["class_vote_share"].values()) - 1.0) < 1e-3
    assert sum(summary[p]["rows"] for p in PARTITIONS) == len(frame)
    # a component is never split across partitions
    assert (d.groupby("component").partition.nunique() == 1).all()
    assert (d.groupby("patient_id").partition.nunique() == 1).all()


def test_allocate_is_deterministic_per_seed(frame):
    d1, s1 = allocate(frame, seed=123)
    d2, s2 = allocate(frame, seed=123)
    pd.testing.assert_series_equal(d1.partition, d2.partition)
    assert s1["split_hash"] == s2["split_hash"]
    differs = False
    for other in [124, 125, 126]:
        d3, s3 = allocate(frame, seed=other)
        if not d3.partition.equals(d1.partition):
            differs = True
            assert s3["split_hash"] != s1["split_hash"]
            break
    assert differs, "different seeds never changed the allocation"


def test_leakage_audit_passes_on_allocation(frame):
    d, _ = allocate(frame)
    audit = leakage_audit(d)
    assert audit["status"] == "PASS" and audit["forbidden_overlap"] == 0
    for key in ["patient_id", "eeg_id", "spectrogram_id", "component"]:
        mat = np.asarray(audit[key]["matrix"])
        assert mat.shape == (5, 5)
        assert audit[key]["off_diagonal_total"] == 0
        assert np.all(mat[~np.eye(5, dtype=bool)] == 0)
        assert np.diag(mat).sum() == d[key].nunique()


def test_leakage_audit_fails_on_deliberate_patient_leak(frame):
    d, _ = allocate(frame)
    leaky = d.copy()
    # pick a patient with at least two rows and move exactly one row into another partition
    counts = leaky.patient_id.value_counts()
    patient = int(counts[counts >= 2].index[0])
    rows = leaky.index[leaky.patient_id == patient]
    home = leaky.loc[rows[0], "partition"]
    other = [p for p in PARTITIONS if p != home][0]
    leaky.loc[rows[0], "partition"] = other
    audit = leakage_audit(leaky)
    assert audit["status"] == "FAIL"
    assert audit["forbidden_overlap"] > 0
    assert audit["patient_id"]["off_diagonal_total"] >= 2     # the overlap is counted in both directions
    i, j = PARTITIONS.index(home), PARTITIONS.index(other)
    assert audit["patient_id"]["matrix"][i][j] == 1 and audit["patient_id"]["matrix"][j][i] == 1
    assert audit["component"]["off_diagonal_total"] >= 2


def test_partition_summary_majority_counts_add_up(frame):
    d, _ = allocate(frame)
    summ = partition_summary(d)
    for p in PARTITIONS:
        mc = summ[p]["unique_majority_counts"]
        assert sum(mc.values()) == summ[p]["rows"]
