"""Deterministic synthetic tests for cape_eeg.metrics (no real data)."""

import math

import numpy as np
import pytest

from cape_eeg import metrics as M

LOG6 = math.log(6.0)


def _q(rows):
    return M.votes_to_targets(np.asarray(rows))


def _entropy(q):
    q = np.asarray(q, dtype=float)
    pos = q > 0
    return -(q[pos] * np.log(q[pos])).sum()


def _one_hot_ish(labels, conf):
    p = np.full((len(labels), 6), (1.0 - conf) / 5.0)
    p[np.arange(len(labels)), labels] = conf
    return p


# --------------------------------------------------------------------------- #
# Conventions, targets, predictions
# --------------------------------------------------------------------------- #


def test_label_order_fixed():
    assert M.LABELS == ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
    assert M.CLASS_NAMES == ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]
    assert M.N_CLASSES == 6
    assert M.SCORING_CONVENTION["logarithm"] == "natural"


def test_votes_to_targets_basic():
    q = M.votes_to_targets([[3, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 5]])
    assert q.dtype == np.float64
    np.testing.assert_allclose(q[0], [0.75, 0.25, 0, 0, 0, 0])
    np.testing.assert_allclose(q.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "bad",
    [
        [[0, 0, 0, 0, 0, 0]],
        [[1, -1, 0, 0, 0, 0]],
        [[1.5, 0, 0, 0, 0, 0]],
        [[1, 2, 3]],
        [[np.nan, 1, 0, 0, 0, 0]],
    ],
)
def test_votes_to_targets_rejects_invalid(bad):
    with pytest.raises((ValueError, TypeError)):
        M.votes_to_targets(bad)


def test_clip_and_renormalize_rows_sum_to_one_and_clip_floor():
    p = M.clip_and_renormalize([[1, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2]])
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert p[0].min() >= 1e-7 * 0.999
    np.testing.assert_allclose(p[1], 1 / 6)


@pytest.mark.parametrize(
    "bad",
    [
        [[np.nan, 0.5, 0.5, 0, 0, 0]],
        [[np.inf, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0]],
        [[-0.5, 1.5, 0, 0, 0, 0]],
        [[0.5, 0.5]],
    ],
)
def test_invalid_probability_rows_raise(bad):
    with pytest.raises(ValueError):
        M.clip_and_renormalize(bad)
    with pytest.raises(ValueError):
        M.kl_rows(_q([[1, 0, 0, 0, 0, 0]]), bad)


# --------------------------------------------------------------------------- #
# KL and soft metrics
# --------------------------------------------------------------------------- #


def test_kl_exact_agreement_is_zero():
    q = _q([[10, 0, 0, 0, 0, 0], [2, 3, 1, 0, 0, 4], [1, 1, 1, 1, 1, 1]])
    kl = M.kl_rows(q, q)
    assert np.all(kl >= 0)
    np.testing.assert_allclose(kl, 0.0, atol=1e-5)
    assert kl[2] == 0.0  # no clipping needed for a strictly positive row


def test_kl_uniform_prediction_equals_log6_minus_entropy():
    q = _q([[2, 3, 1, 0, 0, 4], [6, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
    p = np.full_like(q, 1 / 6)
    expected = np.array([LOG6 - _entropy(row) for row in q])
    np.testing.assert_allclose(M.kl_rows(q, p), expected, atol=1e-9)
    np.testing.assert_allclose(M.kl_rows(q, p)[2], 0.0, atol=1e-12)


def test_kl_zero_vote_classes_contribute_exactly_zero():
    q = _q([[5, 0, 0, 0, 0, 0]])
    p1 = [[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]]
    p2 = [[0.5, 0.4, 0.025, 0.025, 0.025, 0.025]]
    assert math.isclose(M.kl_rows(q, p1)[0], -math.log(0.5), abs_tol=1e-12)
    assert math.isclose(M.kl_rows(q, p2)[0], -math.log(0.5), abs_tol=1e-12)
    # p exactly zero at a zero-vote class must not produce inf/nan.
    kl3 = M.kl_rows(q, [[0.5, 0.5, 0, 0, 0, 0]])
    assert np.isfinite(kl3[0]) and math.isclose(kl3[0], math.log(2.0), abs_tol=1e-5)


def test_kl_severe_misclassification_is_large():
    q = _q([[4, 0, 0, 0, 0, 0]])
    kl = M.kl_rows(q, [[0, 1, 0, 0, 0, 0]])[0]
    assert kl > 10.0
    assert math.isclose(kl, -math.log(1e-7 / (1 + 5e-7)), rel_tol=1e-6)


def test_kl_row_count_mismatch_raises():
    with pytest.raises(ValueError):
        M.kl_rows(_q([[1, 0, 0, 0, 0, 0]]), np.full((2, 6), 1 / 6))


def test_soft_metric_definitions_and_identities():
    q = _q([[2, 3, 1, 0, 0, 4], [6, 0, 0, 0, 0, 0]])
    p = np.array([[0.3, 0.2, 0.1, 0.1, 0.1, 0.2], [0.6, 0.1, 0.1, 0.1, 0.05, 0.05]])
    kl = M.kl_rows(q, p)
    ce = M.soft_cross_entropy_rows(q, p)
    sse = M.soft_squared_error_rows(q, p)
    brier = M.expected_brier_rows(q, p)
    np.testing.assert_allclose(ce, kl + np.array([_entropy(r) for r in q]), atol=1e-12)
    np.testing.assert_allclose(sse, ((p - q) ** 2).sum(axis=1), atol=1e-12)
    np.testing.assert_allclose(brier, 1 - 2 * (p * q).sum(1) + (p ** 2).sum(1), atol=1e-12)
    np.testing.assert_allclose(brier, sse + 1 - (q ** 2).sum(1), atol=1e-12)


# --------------------------------------------------------------------------- #
# Row vs patient weighting
# --------------------------------------------------------------------------- #


def test_row_vs_patient_weighting_differ_with_unequal_group_sizes():
    values = np.array([1.0, 1.0, 1.0, 1.0, 5.0])
    groups = np.array(["a", "a", "a", "a", "b"])
    uniq, means = M.patient_mean(values, groups)
    assert list(uniq) == ["a", "b"]
    np.testing.assert_allclose(means, [1.0, 5.0])
    assert math.isclose(values.mean(), 1.8)
    assert math.isclose(M.patient_weighted_mean(values, groups), 3.0)


def test_patient_mean_two_dimensional_seeds():
    values = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]])
    groups = np.array([7, 7, 9])
    uniq, means = M.patient_mean(values, groups)
    np.testing.assert_array_equal(uniq, [7, 9])
    np.testing.assert_allclose(means, [[2.0, 3.0], [10.0, 20.0]])
    assert math.isclose(M.patient_weighted_mean(values, groups), (2 + 3 + 10 + 20) / 4)
    with pytest.raises(ValueError):
        M.patient_mean(values, groups[:2])


# --------------------------------------------------------------------------- #
# Hard labels
# --------------------------------------------------------------------------- #


def test_unique_majority_tie_handling():
    v = [[3, 3, 0, 0, 0, 0], [0, 4, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 2], [0, 2, 0, 2, 0, 2]]
    labels, tie = M.unique_majority_labels(v)
    np.testing.assert_array_equal(labels, [-1, 1, -1, 5, -1])
    np.testing.assert_array_equal(tie, [True, False, True, False, True])


def test_hard_metrics_perfect_predictions_and_tie_exclusion():
    labels = np.repeat(np.arange(6), 10)
    v = np.zeros((60, 6), dtype=int)
    v[np.arange(60), labels] = 3
    v = np.vstack([v, [[2, 2, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0]]])
    p = np.vstack([_one_hot_ish(labels, 0.9), np.full((2, 6), 1 / 6)])
    out = M.hard_metrics(v, p)
    assert out["status"] == "PASS"
    assert out["n_rows_evaluated"] == 60 and out["n_ties_excluded"] == 2 and out["n_rows_total"] == 62
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "mcc", "cohen_kappa", "macro_auroc", "macro_auprc"):
        assert math.isclose(out[key], 1.0, abs_tol=1e-12), key
    cm = np.asarray(out["confusion_matrix"])
    assert cm.shape == (6, 6) and cm.sum() == 60 and np.all(np.diag(cm) == 10)
    assert out["per_class"]["prevalence"] == [10 / 60] * 6
    assert out["n_classes_defined"]["auroc"] == 6


def test_hard_metrics_marks_undefined_classes_none():
    labels = np.array([0] * 8 + [1] * 8)
    v = np.zeros((16, 6), dtype=int)
    v[np.arange(16), labels] = 2
    rng = np.random.default_rng(3)
    p = rng.dirichlet(np.ones(6), size=16)
    p[np.arange(16), labels] += 2.0
    p /= p.sum(1, keepdims=True)
    out = M.hard_metrics(v, p)
    assert out["per_class"]["auroc"][2] is None and out["per_class"]["auprc"][5] is None
    assert out["per_class"]["recall"][3] is None
    assert out["per_class"]["f1"][4] is None
    assert out["n_classes_defined"]["auroc"] == 2
    assert out["macro_auroc"] is not None and 0 <= out["macro_auroc"] <= 1
    assert math.isclose(out["balanced_accuracy"], np.mean(out["per_class"]["recall"][:2]))
    assert math.isclose(out["accuracy"], 1.0)


def test_hard_metrics_all_ties_not_estimable():
    out = M.hard_metrics([[1, 1, 0, 0, 0, 0], [2, 0, 2, 0, 0, 0]], np.full((2, 6), 1 / 6))
    assert out["status"] == "NOT_ESTIMABLE" and out["accuracy"] is None and out["n_rows_evaluated"] == 0


def test_hard_metrics_never_breaks_ties_by_column_order():
    v = [[2, 2, 0, 0, 0, 0]] * 4 + [[0, 3, 0, 0, 0, 0]] * 4
    p = _one_hot_ish([0, 0, 0, 0, 1, 1, 1, 1], 0.8)
    out = M.hard_metrics(v, p)
    assert out["n_ties_excluded"] == 4 and out["n_rows_evaluated"] == 4
    assert out["per_class"]["n_positive"][0] == 0


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def test_soft_ece_zero_when_vote_mass_matches_confidence():
    rng = np.random.default_rng(11)
    p = rng.dirichlet(np.ones(6) * 0.7, size=500)
    out = M.soft_top_label_ece(p, p, n_bins=10)
    assert math.isclose(out["ece"], 0.0, abs_tol=1e-9)
    assert out["n_bins_final"] == 10 and out["n_merges"] == 0 and out["n_rows"] == 500
    assert all(lo <= hi for lo, hi in zip(out["bin_lower"], out["bin_upper"]))
    assert out["bin_upper"][:-1] <= out["bin_lower"][1:]
    assert sum(out["bin_count"]) == 500


def test_soft_ece_overconfident_gap():
    q = _q([[1, 1, 0, 0, 0, 0]] * 40)  # vote mass on predicted class 0 is 0.5
    p = _one_hot_ish([0] * 40, 0.9)
    out = M.soft_top_label_ece(q, p, n_bins=10)
    assert math.isclose(out["ece"], 0.4, abs_tol=1e-12)
    assert math.isclose(out["mce"], 0.4, abs_tol=1e-12)


def test_soft_ece_bin_merging_rule():
    rng = np.random.default_rng(5)
    n = 200
    p = rng.dirichlet(np.ones(6), size=n)
    q = rng.dirichlet(np.ones(6), size=n)
    # Ten groups in total: every bin has <20 groups, so all bins merge into one.
    few_groups = np.arange(n) % 10
    merged = M.soft_top_label_ece(q, p, n_bins=10, groups=few_groups, min_groups_per_bin=20)
    assert merged["n_bins_final"] == 1 and merged["n_merges"] == 9
    conf = p.max(1)
    obs = q[np.arange(n), p.argmax(1)]
    assert math.isclose(merged["ece"], abs(conf.mean() - obs.mean()), abs_tol=1e-12)
    assert merged["bin_n_groups"] == [10]
    # Every row its own group: 20 groups per bin, nothing merges.
    own_groups = np.arange(n)
    intact = M.soft_top_label_ece(q, p, n_bins=10, groups=own_groups, min_groups_per_bin=20)
    assert intact["n_bins_final"] == 10 and intact["n_merges"] == 0 and intact["bin_n_groups"] == [20] * 10
    # Only the lowest-confidence bin is sparse -> exactly one merge with its neighbour.
    order = np.argsort(conf, kind="stable")
    groups = np.arange(n)
    groups[order[:20]] = -1
    one_merge = M.soft_top_label_ece(q, p, n_bins=10, groups=groups, min_groups_per_bin=20)
    assert one_merge["n_bins_final"] == 9 and one_merge["n_merges"] == 1
    assert one_merge["bin_count"][0] == 40 and one_merge["bin_n_groups"][0] == 21
    assert "merge" in one_merge["merge_rule"]
    # min_groups_per_bin=1 never merges.
    none = M.soft_top_label_ece(q, p, n_bins=10, groups=few_groups, min_groups_per_bin=1)
    assert none["n_merges"] == 0
    # Without groups: ECE identical to the unmerged run.
    assert math.isclose(M.soft_top_label_ece(q, p, n_bins=10)["ece"], intact["ece"])


def test_majority_label_ece_basic_and_ties():
    labels = np.repeat(np.arange(6), 5)
    v = np.zeros((30, 6), dtype=int)
    v[np.arange(30), labels] = 4
    v = np.vstack([v, [[2, 2, 0, 0, 0, 0]]])
    p = np.vstack([_one_hot_ish(labels, 0.8), [[1 / 6] * 6]])
    out = M.majority_label_ece(v, p, n_bins=5)
    assert out["n_rows_evaluated"] == 30 and out["n_ties_excluded"] == 1
    assert math.isclose(out["ece"], 0.2, abs_tol=1e-12)
    assert out["status"] == "PASS"


# --------------------------------------------------------------------------- #
# Disagreement
# --------------------------------------------------------------------------- #


def test_pairwise_disagreement_values():
    v = [[1, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [2, 1, 0, 0, 0, 0], [3, 3, 0, 0, 0, 0]]
    d = M.pairwise_disagreement(v)
    assert np.isnan(d[0]) and d[0] != 0.0  # n=1: NOT_ESTIMABLE, never zero
    assert d[1] == 0.0
    assert d[2] == 1.0 and d[2] > 5 / 6  # reaches exactly 1.0, not clipped to 5/6
    assert math.isclose(d[3], 2 / 3)
    assert math.isclose(d[4], 1 - 12 / 30)


def test_disagreement_metrics_strata_and_statuses():
    v = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [2, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [3, 1, 0, 0, 0, 0],
            [4, 2, 1, 0, 0, 0],
            [5, 0, 0, 0, 0, 3],
            [6, 3, 1, 0, 0, 2],
            [10, 1, 0, 0, 0, 2],
        ]
    )
    d = M.pairwise_disagreement(v)
    d_hat = np.where(np.isnan(d), 0.5, d)
    out = M.disagreement_metrics(d_hat, v)
    assert out["n_rows_not_estimable"] == 2
    assert out["overall"]["status"] == "PASS"
    assert math.isclose(out["overall"]["mse"], 0.0, abs_tol=1e-15)
    assert math.isclose(out["overall"]["spearman"], 1.0)
    assert out["overall"]["n_rows_estimable"] == 7
    by_name = {s["stratum"]: s for s in out["strata"]}
    assert by_name["n=1"]["status"] == "NOT_ESTIMABLE" and by_name["n=1"]["mse"] is None
    assert by_name["n=1"]["n_rows"] == 2
    assert by_name["n=2-4"]["n_rows"] == 3 and by_name["n=2-4"]["status"] == "PASS"
    assert by_name["n=5-9"]["n_rows"] == 2
    assert by_name["n>=10"]["n_rows"] == 2 and by_name["n>=10"]["n_max"] is None
    # Predicted values that reverse the ranking give negative correlation.
    rev = M.disagreement_metrics(1.0 - d_hat, v)
    assert rev["overall"]["spearman"] < 0
    with pytest.raises(ValueError):
        M.disagreement_metrics(d_hat[:-1], v)


# --------------------------------------------------------------------------- #
# Referral scores and risk-coverage
# --------------------------------------------------------------------------- #


def test_predictive_entropy_and_one_minus_max():
    p = np.array([[1 / 6] * 6, [1, 0, 0, 0, 0, 0]])
    h = M.predictive_entropy(p)
    assert math.isclose(h[0], LOG6, abs_tol=1e-12)
    assert h[1] < 1e-4
    omm = M.one_minus_max_prob(p)
    assert math.isclose(omm[0], 5 / 6, abs_tol=1e-12)
    assert omm[1] < 1e-5


def test_risk_coverage_monotone_and_aurc_range():
    rng = np.random.default_rng(2)
    n = 400
    score = rng.permutation(np.linspace(0.0, 1.0, n))
    risk = score.copy()  # perfect ranking: selective risk must be non-decreasing
    groups = rng.integers(0, 40, size=n)
    out = M.risk_coverage(score, risk, groups=groups)
    r = np.asarray(out["risk"])
    assert np.all(np.diff(r) >= -1e-12)
    assert np.all(np.diff(out["n_accepted"]) >= 0)
    assert math.isclose(r[-1], risk.mean())
    assert math.isclose(out["full_cohort_risk"], risk.mean())
    assert 0.5 * r.min() <= out["aurc"] <= 0.5 * r.max()
    assert r.min() <= out["aurc_normalized"] <= r.max()
    assert out["integration_range"] == [0.5, 1.0]
    assert len(out["risk_patient_weighted"]) == len(out["coverage_grid"]) == 51
    assert out["n_accepted"][0] == 200 and out["n_accepted"][-1] == 400
    # Reversed ranking (high risk accepted first) gives a larger AURC.
    worse = M.risk_coverage(-score, risk)
    assert worse["aurc"] > out["aurc"]
    with pytest.raises(ValueError):
        M.risk_coverage(score, risk, coverage_grid=[0.9, 0.5])


def test_threshold_for_coverage_and_apply_threshold():
    rng = np.random.default_rng(4)
    score = rng.normal(size=400)
    thr = M.threshold_for_coverage(score, 0.9)
    accepted = M.apply_threshold(score, thr)
    assert accepted.sum() == 360
    assert M.apply_threshold(np.array([0.1, np.nan, 2.0]), 1.0).tolist() == [True, False, False]
    with pytest.raises(ValueError):
        M.threshold_for_coverage(score, 0.0)
    risk = rng.uniform(size=400)
    summary = M.referral_summary(score, thr, risk, groups=rng.integers(0, 30, size=400))
    assert summary["n_accepted"] == 360 and summary["n_referred"] == 40
    assert math.isclose(summary["actual_coverage"], 0.9) and math.isclose(summary["fraction_referred"], 0.1)
    assert math.isclose(summary["accepted_risk"], risk[accepted].mean())
    assert math.isclose(summary["whole_cohort_risk"], risk.mean())
    assert summary["accepted_risk_patient_weighted"] is not None and summary["status"] == "PASS"
    empty = M.referral_summary(score, score.min() - 1.0, risk)
    assert empty["n_accepted"] == 0 and empty["accepted_risk"] is None and empty["status"] == "NOT_ESTIMABLE"


# --------------------------------------------------------------------------- #
# Prediction shift and manifest records
# --------------------------------------------------------------------------- #


def test_jensen_shannon_distance_rows():
    p = np.array([[0.2, 0.3, 0.1, 0.1, 0.1, 0.2], [1, 0, 0, 0, 0, 0]])
    same = M.jensen_shannon_distance_rows(p, p)
    np.testing.assert_allclose(same, 0.0, atol=1e-9)
    disjoint = M.jensen_shannon_distance_rows([[1, 0, 0, 0, 0, 0]], [[0, 1, 0, 0, 0, 0]])
    assert math.isclose(disjoint[0], math.sqrt(math.log(2.0)), abs_tol=1e-4)
    assert np.all(disjoint <= math.sqrt(math.log(2.0)) + 1e-12)


def test_metric_record_keys_and_defaults():
    rec = M.metric_record(run_id="r1", metric="patient_mean_kl", value=np.float64(0.5), n_rows=np.int64(10))
    assert tuple(rec.keys()) == M.METRIC_RECORD_KEYS
    assert tuple(rec.keys()) == (
        "run_id", "method_id", "seed", "split_hash", "preprocess_hash", "protocol_hash", "prediction_hash",
        "metric", "value", "ci_low", "ci_high", "n_rows", "n_patients", "n_components", "status", "reason",
    )
    assert rec["status"] == "PASS" and rec["reason"] is None and rec["ci_low"] is None
    assert isinstance(rec["value"], float) and isinstance(rec["n_rows"], int)
    assert M.metric_record(value=float("nan"), status="NOT_RUN")["value"] is None
    with pytest.raises(ValueError):
        M.metric_record(unknown_key=1)
