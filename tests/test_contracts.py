"""Deterministic tests for the fixed contracts in cape_eeg.contracts (no data is read)."""

import math

import numpy as np
import pytest

from cape_eeg import contracts as C


# ------------------------------------------------------------------ temporal geometry
def test_foveated_edges_have_8_16_8_segments():
    e = C.foveated_time_edges()
    assert e.shape == (33,)
    assert e[0] == 0.0 and e[-1] == 50.0
    np.testing.assert_allclose(e[0:9], np.linspace(0.0, 20.0, 9))      # 8 bins over [0,20)
    np.testing.assert_allclose(e[8:25], np.linspace(20.0, 30.0, 17))   # 16 bins over [20,30)
    np.testing.assert_allclose(e[24:33], np.linspace(30.0, 50.0, 9))   # 8 bins over [30,50)
    widths = np.diff(e)
    np.testing.assert_allclose(widths[:8], 2.5)
    np.testing.assert_allclose(widths[8:24], 0.625)
    np.testing.assert_allclose(widths[24:], 2.5)
    assert np.all(np.diff(e) > 0)


def test_uniform_edges_are_equal_width():
    e = C.uniform_time_edges()
    assert e.shape == (33,)
    assert e[0] == 0.0 and e[-1] == 50.0
    np.testing.assert_allclose(np.diff(e), 50.0 / 32)
    e64 = C.context_time_edges()
    assert e64.shape == (65,)
    np.testing.assert_allclose(np.diff(e64), 600.0 / 64)


def test_center_weights_foveated_are_exact_indicator_of_columns_8_to_24():
    w = C.center_weights_from_edges(C.foveated_time_edges())
    expected = np.zeros(32)
    expected[8:24] = 1.0
    np.testing.assert_array_equal(w, expected)
    lo, hi = C.FOCAL_TARGET_COLUMNS
    assert (lo, hi) == (8, 24)


def test_center_weights_uniform_are_fractional_on_boundary_bins():
    e = C.uniform_time_edges()
    w = C.center_weights_from_edges(e)
    assert w.shape == (32,)
    assert np.all((w >= 0) & (w <= 1))
    overlapping = [i for i in range(32) if e[i + 1] > 20.0 and e[i] < 30.0]
    assert overlapping == list(range(12, 20))
    # boundary bins [18.75,20.3125) and [29.6875,31.25) overlap the center by 0.3125/1.5625 = 0.2
    np.testing.assert_allclose(w[12], 0.2)
    np.testing.assert_allclose(w[19], 0.2)
    np.testing.assert_allclose(w[13:19], 1.0)
    assert np.all(w[:12] == 0) and np.all(w[20:] == 0)
    # weighted widths integrate to exactly the 10 s center interval
    np.testing.assert_allclose((w * np.diff(e)).sum(), 10.0)


def test_overlap_matrix_rows_sum_to_covered_length():
    src = np.linspace(0.0, 50.0, 11)          # 10 source bins of 5 s
    dst = C.foveated_time_edges()             # 32 destination bins
    W = C.overlap_matrix(src, dst)
    assert W.shape == (32, 10)
    assert np.all(W >= 0)
    # every destination interval is fully covered by the source grid
    np.testing.assert_allclose(W.sum(1), np.diff(dst))
    # every source interval is fully covered by the destination grid
    np.testing.assert_allclose(W.sum(0), np.diff(src))
    # partial source coverage: rows sum to the length of the destination interval covered by sources
    src_short = np.linspace(0.0, 30.0, 7)
    W2 = C.overlap_matrix(src_short, dst)
    covered = np.clip(np.minimum(dst[1:], 30.0) - np.minimum(dst[:-1], 30.0), 0.0, None)
    np.testing.assert_allclose(W2.sum(1), covered)
    assert W2[-1].sum() == 0.0


# ------------------------------------------------------------------ offsets and windows
def test_eeg_window_bounds_returns_start_plus_10000():
    assert C.eeg_window_bounds(0.0) == (0, 10000)
    assert C.eeg_window_bounds(4.0) == (800, 10800)
    assert C.eeg_window_bounds(4.0, n_samples_available=10800) == (800, 10800)
    assert C.eeg_window_bounds(0.005) == (1, 10001)  # exactly one sample on the 200 Hz grid


@pytest.mark.parametrize("bad", [0.001, 0.0033, 1.0001])
def test_eeg_window_bounds_rejects_off_grid_offsets(bad):
    with pytest.raises(ValueError):
        C.eeg_window_bounds(bad)


def test_eeg_window_bounds_rejects_negative_and_nonfinite():
    for bad in [-1.0, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            C.eeg_window_bounds(bad)


def test_eeg_window_bounds_rejects_window_exceeding_available_samples():
    with pytest.raises(ValueError):
        C.eeg_window_bounds(10.0, n_samples_available=11000)   # [2000, 12000) > 11000
    assert C.eeg_window_bounds(5.0, n_samples_available=11000) == (1000, 11000)


def test_spectrogram_interval_and_center_arithmetic():
    assert C.spectrogram_interval(0.0) == (0.0, 600.0)
    assert C.spectrogram_interval(42.0) == (42.0, 642.0)
    assert C.spectrogram_center(0.0) == (295.0, 305.0)
    assert C.spectrogram_center(42.0) == (337.0, 347.0)
    lo, hi = C.spectrogram_center(7.0)
    assert hi - lo == 10.0
    s, e = C.spectrogram_interval(7.0)
    assert (lo + hi) / 2 == (s + e) / 2  # the labeled center sits in the middle of the 600 s context
    with pytest.raises(ValueError):
        C.spectrogram_interval(-1.0)
    with pytest.raises(ValueError):
        C.spectrogram_interval(float("nan"))


# ------------------------------------------------------------------ labels
def test_vote_targets_normalises_rows():
    v = np.array([[3, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 5]])
    q = C.vote_targets(v)
    np.testing.assert_allclose(q.sum(1), 1.0)
    np.testing.assert_allclose(q[0], [0.75, 0.25, 0, 0, 0, 0])
    np.testing.assert_allclose(q[1], [0, 0, 0, 0, 0, 1.0])


@pytest.mark.parametrize("bad", [
    [[0, 0, 0, 0, 0, 0]],          # zero-sum row
    [[2, -1, 0, 0, 0, 0]],         # negative
    [[1.5, 0, 0, 0, 0, 0]],        # non-integer
    [[1, 0, 0, 0, 0]],             # wrong width
])
def test_vote_targets_rejects_invalid_rows(bad):
    with pytest.raises(ValueError):
        C.vote_targets(np.asarray(bad, dtype=float))


def test_pairwise_disagreement_values():
    v = np.array([[1, 0, 0, 0, 0, 0],   # n=1 -> unavailable
                  [1, 1, 0, 0, 0, 0],   # two raters disagree -> exactly 1
                  [3, 0, 0, 0, 0, 0],   # unanimous -> 0
                  [1, 1, 1, 1, 1, 1],   # six distinct opinions -> exactly 1, not 5/6
                  [2, 2, 0, 0, 0, 0]])  # n=4: 1 - (2+2)/12 = 2/3
    d = C.pairwise_disagreement(v)
    assert np.isnan(d[0])
    assert d[1] == 1.0
    assert d[2] == 0.0
    assert d[3] == 1.0 and d[3] > 5 / 6
    np.testing.assert_allclose(d[4], 2 / 3)
    assert np.all(d[1:] >= 0) and np.all(d[1:] <= 1)


# ------------------------------------------------------------------ byte arithmetic
def test_cache_bytes_reproduces_planned_numbers():
    cb = C.cache_bytes(106800)
    assert cb.rows == 106800
    assert cb.values_bytes == 5_249_433_600
    assert cb.mask_bytes == 328_089_600
    assert cb.payload_bytes == 5_577_523_200
    assert cb.metadata_allowance_bytes == C.METADATA_BUDGET_BYTES
    assert cb.total_allowance_bytes == cb.payload_bytes + C.METADATA_BUDGET_BYTES
    assert cb.within_cap is True
    d = cb.as_dict()
    assert d["payload_bytes"] == 5_577_523_200 and d["within_cap"] is True


def test_cache_bytes_with_uniform_alternate_exceeds_cap():
    cb = C.cache_bytes(106800, include_uniform_alternate=True)
    assert cb.within_cap is False
    assert cb.total_allowance_bytes > C.CACHE_MAX_BYTES
    base = C.cache_bytes(106800)
    per_row_alt_values = C.N_REGIONS * C.FREQUENCY_BINS * C.FOCAL_TIME_BINS * 2
    per_row_alt_mask = C.N_REGIONS * C.FREQUENCY_BINS * C.FOCAL_TIME_BINS // 8
    assert cb.values_bytes - base.values_bytes == per_row_alt_values * 106800
    assert cb.mask_bytes - base.mask_bytes == per_row_alt_mask * 106800


def test_cache_bytes_scales_linearly():
    assert C.cache_bytes(0).payload_bytes == 0
    assert C.cache_bytes(2).payload_bytes == 2 * C.cache_bytes(1).payload_bytes


# ------------------------------------------------------------------ hashing
def test_stable_hash_is_order_independent_and_stable():
    a = C.stable_hash({"b": [1, 2], "a": "x"})
    b = C.stable_hash({"a": "x", "b": [1, 2]})
    assert a == b
    assert a == "721ef82f2d6c0997"           # pinned reference value (sha256 of the canonical JSON)
    assert C.stable_hash({"b": [1, 2], "a": "x"}, 8) == a[:8]
    assert len(C.stable_hash({"k": 1}, 32)) == 32
    assert C.stable_hash({"a": 1}) != C.stable_hash({"a": 2})
    # numpy scalars / arrays and dataclasses with as_dict are serialisable
    assert C.stable_hash({"n": np.int64(3), "f": np.float32(0.5), "arr": np.arange(3)}) == \
        C.stable_hash({"n": 3, "f": 0.5, "arr": [0, 1, 2]})
    assert C.stable_hash(C.cache_bytes(10)) == C.stable_hash(C.cache_bytes(10).as_dict())
    with pytest.raises(TypeError):
        C.stable_hash({"bad": object()})


def test_preprocess_signature_hash_is_deterministic():
    assert C.stable_hash(C.preprocess_signature()) == C.stable_hash(C.preprocess_signature())
    assert C.preprocess_signature()["dtype"] == "float16"
