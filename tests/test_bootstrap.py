"""Deterministic synthetic tests for cape_eeg.evaluation.bootstrap."""

import math

import numpy as np
import pytest

from cape_eeg.evaluation import bootstrap as B


def _paired_case(seed=0, n_groups=300, delta_true=-0.05):
    rng = np.random.default_rng(seed)
    b = rng.gamma(2.0, 0.2, size=n_groups)
    a = b + delta_true + rng.normal(0.0, 0.05, size=n_groups)
    return a, b, delta_true


def test_paired_bootstrap_contains_truth_and_is_reproducible():
    a, b, delta_true = _paired_case()
    r1 = B.paired_cluster_bootstrap(a, b, n_replicates=1000, seed=123)
    r2 = B.paired_cluster_bootstrap(a, b, n_replicates=1000, seed=123)
    assert r1["ci_low"] == r2["ci_low"] and r1["ci_high"] == r2["ci_high"]
    assert r1["bootstrap_sd"] == r2["bootstrap_sd"]
    assert math.isclose(r1["point_estimate"], a.mean() - b.mean())
    assert r1["ci_low"] <= delta_true <= r1["ci_high"]
    assert r1["ci_low"] <= r1["point_estimate"] <= r1["ci_high"]
    assert r1["interval_wholly_below_zero"] is True and r1["interval_crosses_zero"] is False
    assert r1["n_groups"] == 300 and r1["n_replicates"] == 1000 and r1["seed"] == 123 and r1["n_seeds"] == 1
    assert r1["bootstrap_sd"] > 0 and r1["ci_method"] == "percentile"
    # Width is on the expected scale: sd(delta_g) / sqrt(G) ~ 0.05 / sqrt(300).
    half_width = (r1["ci_high"] - r1["ci_low"]) / 2
    assert 0.5 * 1.96 * 0.05 / math.sqrt(300) < half_width < 2.0 * 1.96 * 0.05 / math.sqrt(300)
    # A different seed gives a different replicate set but the same point estimate.
    r3 = B.paired_cluster_bootstrap(a, b, n_replicates=1000, seed=124)
    assert r3["bootstrap_sd"] != r1["bootstrap_sd"]
    assert r3["point_estimate"] == r1["point_estimate"]
    assert abs(r3["ci_low"] - r1["ci_low"]) < 0.005


def test_paired_bootstrap_null_case_crosses_zero():
    rng = np.random.default_rng(1)
    b = rng.gamma(2.0, 0.2, size=200)
    a = b + rng.normal(0.0, 0.05, size=200)
    out = B.paired_cluster_bootstrap(a, b, n_replicates=500, seed=7)
    assert out["interval_crosses_zero"] is True and out["interval_wholly_below_zero"] is False


def test_paired_bootstrap_seed_axis_averages_losses_per_group():
    a, b, _ = _paired_case(seed=2)
    rng = np.random.default_rng(9)
    noise = rng.normal(0.0, 0.01, size=(a.size, 3))
    a3 = a[:, None] + noise
    b3 = b[:, None] + noise[:, ::-1]
    out = B.paired_cluster_bootstrap(a3, b3, n_replicates=300, seed=5)
    assert out["n_seeds"] == 3
    assert math.isclose(out["point_estimate"], a3.mean(axis=1).mean() - b3.mean(axis=1).mean())
    # Identical to bootstrapping the seed-averaged per-group losses.
    ref = B.paired_cluster_bootstrap(a3.mean(axis=1), b3.mean(axis=1), n_replicates=300, seed=5)
    assert math.isclose(out["ci_low"], ref["ci_low"], abs_tol=1e-12)
    assert math.isclose(out["ci_high"], ref["ci_high"], abs_tol=1e-12)


def test_paired_bootstrap_validation():
    a, b, _ = _paired_case()
    with pytest.raises(ValueError):
        B.paired_cluster_bootstrap(a[:-1], b)
    with pytest.raises(ValueError):
        B.paired_cluster_bootstrap(a[:1], b[:1])
    bad = a.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError):
        B.paired_cluster_bootstrap(bad, b)
    with pytest.raises(ValueError):
        B.paired_cluster_bootstrap(a, b, n_replicates=0)


def test_cluster_bootstrap_statistic_mean_and_group_integrity():
    rng = np.random.default_rng(3)
    n_groups, rows_per_group = 60, 8
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    group_effect = rng.normal(0.0, 0.3, size=n_groups)
    x = 1.0 + group_effect[groups] + rng.normal(0.0, 0.1, size=groups.size)

    out = B.cluster_bootstrap_statistic(lambda idx: x[idx].mean(), groups, n_replicates=800, seed=11)
    again = B.cluster_bootstrap_statistic(lambda idx: x[idx].mean(), groups, n_replicates=800, seed=11)
    assert out == again
    assert math.isclose(out["point_estimate"], x.mean())
    assert out["ci_low"] <= 1.0 <= out["ci_high"]
    assert out["n_groups"] == n_groups and out["n_rows"] == groups.size and out["status"] == "PASS"
    assert out["n_nonfinite_replicates"] == 0

    # Each replicate contains exactly n_groups whole groups worth of rows.
    sizes = B.cluster_bootstrap_statistic(lambda idx: float(idx.size), groups, n_replicates=50, seed=1)
    assert sizes["ci_low"] == sizes["ci_high"] == float(groups.size)
    distinct = B.cluster_bootstrap_statistic(
        lambda idx: float(np.unique(groups[idx]).size), groups, n_replicates=50, seed=1
    )
    assert 0 < distinct["ci_low"] <= distinct["ci_high"] < n_groups

    # Vector statistics and non-finite replicates are handled.
    vec = B.cluster_bootstrap_statistic(lambda idx: np.array([x[idx].mean(), x[idx].std()]), groups, 100, 2)
    assert len(vec["ci_low"]) == 2 and len(vec["point_estimate"]) == 2
    nan_out = B.cluster_bootstrap_statistic(lambda idx: float("nan"), groups, n_replicates=20, seed=2)
    assert nan_out["status"] == "NOT_ESTIMABLE" and nan_out["ci_low"] is None


def test_cluster_bootstrap_unequal_group_sizes():
    groups = np.array([0, 0, 0, 1, 2, 2, 3, 3, 3, 3])
    x = np.arange(10, dtype=float)
    out = B.cluster_bootstrap_statistic(lambda idx: x[idx].mean(), groups, n_replicates=200, seed=0)
    assert math.isclose(out["point_estimate"], 4.5)
    assert out["n_groups"] == 4 and out["ci_low"] <= out["ci_high"]


def test_holm_adjust_known_example():
    adjusted = B.holm_adjust([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])
    np.testing.assert_allclose(B.holm_adjust([0.5, 0.6]), [1.0, 1.0])
    np.testing.assert_allclose(B.holm_adjust([0.2]), [0.2])
    assert B.holm_adjust([]).size == 0
    p = np.array([0.001, 0.02, 0.03, 0.2, 0.5])
    adj = B.holm_adjust(p)
    assert np.all(adj >= p) and np.all(np.diff(adj[np.argsort(p)]) >= 0)
    with pytest.raises(ValueError):
        B.holm_adjust([0.5, 1.5])


def test_mde_paired_matches_planning_table():
    assert math.isclose(B.mde_paired(0.10, 390), 0.0142, abs_tol=2e-4)
    assert math.isclose(B.mde_paired(0.20, 390), 0.0284, abs_tol=2e-4)
    assert math.isclose(B.mde_paired(0.30, 390), 0.0426, abs_tol=3e-4)
    assert math.isclose(B.mde_paired(0.10, 100), 2.0 * B.mde_paired(0.10, 400))  # scales as 1/sqrt(n)
    arr = B.mde_paired(np.array([0.1, 0.2]), 100)
    assert isinstance(arr, np.ndarray) and math.isclose(arr[1], 2 * arr[0])
    with pytest.raises(ValueError):
        B.mde_paired(0.1, 0)
