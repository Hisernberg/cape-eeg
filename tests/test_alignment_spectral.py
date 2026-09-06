"""Synthetic tests for exact alignment (data.alignment) and spectral encoding (data.spectral).

Every signal here is generated in-test from a seeded generator; no dataset file is read.
"""

import numpy as np
import pytest

from cape_eeg import contracts as C
from cape_eeg.data.alignment import extract_eeg_window, select_spectrogram_rows
from cape_eeg.data.montage import EXPECTED_SOURCE_COLUMNS, BIPOLAR_CHAINS
from cape_eeg.data.spectral import RawSpectralEncoder, ContextSpectralEncoder, pack_mask, unpack_mask, LOG_FLOOR

N_SAMPLES = 11000  # 55 s at 200 Hz
N_CH = len(EXPECTED_SOURCE_COLUMNS)


@pytest.fixture(scope="module")
def signal():
    rng = np.random.default_rng(20260907)
    return rng.normal(0.0, 10.0, size=(N_SAMPLES, N_CH)).astype(np.float32)


# ------------------------------------------------------------------ extract_eeg_window
def test_complete_window_is_copied_exactly(signal):
    w, obs, info = extract_eeg_window(signal, 2.0)
    assert w.shape == (10000, N_CH) and obs.shape == (10000, N_CH)
    assert w.dtype == np.float32 and obs.dtype == bool
    np.testing.assert_array_equal(w, signal[400:10400])
    assert obs.all()
    assert info["complete"] is True and info["edge_missing_samples"] == 0
    assert (info["start"], info["stop"]) == (400, 10400)


def test_truncated_window_is_zero_padded_not_shifted(signal):
    w, obs, info = extract_eeg_window(signal, 10.0)   # [2000, 12000) but only 11000 samples exist
    assert info["complete"] is False
    assert info["edge_missing_samples"] == 1000
    np.testing.assert_array_equal(w[:9000], signal[2000:11000])   # not shifted
    assert np.all(w[9000:] == 0.0)                                 # padded with zeros, not wrapped
    assert obs[:9000].all()
    assert not obs[9000:].any()
    # a wrapped window would have copied the start of the recording here
    assert not np.array_equal(w[9000:9100], signal[:100])


def test_nan_samples_become_unobserved_and_zero(signal):
    x = signal.copy()
    x[500, 3] = np.nan
    x[600:610, 7] = np.inf
    w, obs, info = extract_eeg_window(x, 0.0)
    assert info["complete"] is True
    assert not obs[500, 3] and w[500, 3] == 0.0
    assert not obs[600:610, 7].any() and np.all(w[600:610, 7] == 0.0)
    expected = np.ones((10000, N_CH), bool); expected[500, 3] = False; expected[600:610, 7] = False
    np.testing.assert_array_equal(obs, expected)
    assert np.isfinite(w).all()


def test_off_grid_offset_rejected(signal):
    with pytest.raises(ValueError):
        extract_eeg_window(signal, 0.001)


# ------------------------------------------------------------------ select_spectrogram_rows
def test_select_rows_full_coverage():
    t = np.arange(1, 600, 2, dtype=np.float64)    # 1,3,5,...,599 -> 300 rows of 2 s
    assert t.size == 300
    keep, edges, info = select_spectrogram_rows(t, 0.0)
    assert keep.size == 300
    np.testing.assert_array_equal(keep, np.arange(300))
    assert edges.shape == (301,)
    assert edges[0] == 0.0 and edges[-1] == 600.0
    np.testing.assert_allclose(np.diff(edges), 2.0)
    assert info["spacing"] == 2.0 and info["n_rows"] == 300
    np.testing.assert_allclose(info["coverage_seconds"], 600.0)
    assert info["expected_seconds"] == 600.0 and info["first_time"] == 1.0


def test_select_rows_with_offset_selects_shifted_block():
    t = np.arange(1, 1200, 2, dtype=np.float64)   # 600 rows covering [0,1200)
    keep, edges, info = select_spectrogram_rows(t, 300.0)
    assert keep.size == 300 and keep[0] == 150
    assert edges[0] == 0.0 and edges[-1] == 600.0  # relative to the offset
    np.testing.assert_allclose(info["coverage_seconds"], 600.0)


def test_select_rows_rejects_nonuniform_or_nonmonotonic_time():
    with pytest.raises(ValueError):
        select_spectrogram_rows(np.array([1.0, 3.0, 5.0, 8.0]), 0.0)       # non-uniform spacing
    with pytest.raises(ValueError):
        select_spectrogram_rows(np.array([1.0, 3.0, 2.0, 5.0]), 0.0)       # non-monotonic
    with pytest.raises(ValueError):
        select_spectrogram_rows(np.array([1.0, 1.0, 1.0]), 0.0)            # zero spacing
    with pytest.raises(ValueError):
        select_spectrogram_rows(np.array([]), 0.0)


def test_select_rows_past_file_end_returns_partial_coverage():
    t = np.arange(1, 600, 2, dtype=np.float64)
    keep, edges, info = select_spectrogram_rows(t, 300.0)    # [300, 900) but the file ends at 600
    assert keep.size == 150
    assert keep[0] == 150 and keep[-1] == 299
    assert info["coverage_seconds"] < 600.0
    np.testing.assert_allclose(info["coverage_seconds"], 300.0)
    assert edges[0] == 0.0 and edges[-1] == 300.0            # nothing wrapped in from the start
    assert edges.shape == (151,)


# ------------------------------------------------------------------ RawSpectralEncoder
@pytest.fixture(scope="module")
def encoder():
    return RawSpectralEncoder(EXPECTED_SOURCE_COLUMNS)


# Electrodes carrying the test tone: alternate members of every bipolar chain so that all
# 16 leads (Fp1-F7, F7-T3, ...) equal +/- the tone rather than cancelling.
TONE_ELECTRODES = ["Fp1", "T3", "O1", "Fp2", "T4", "O2", "C3", "C4"]


def _tone_window(freq_hz=3.0, start_s=22.0, stop_s=28.0, amplitude=50.0, noise=0.01, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(10000) / C.EEG_SAMPLE_RATE_HZ
    x = rng.normal(0.0, noise, size=(10000, N_CH)).astype(np.float32)
    tone = amplitude * np.sin(2 * np.pi * freq_hz * t) * ((t >= start_s) & (t < stop_s))
    for e in TONE_ELECTRODES:
        x[:, EXPECTED_SOURCE_COLUMNS.index(e)] += tone.astype(np.float32)
    return x, np.ones_like(x, dtype=bool)


def test_all_leads_nonzero_for_tone_electrodes():
    for chain in BIPOLAR_CHAINS.values():
        for e1, e2 in chain:
            assert (e1 in TONE_ELECTRODES) != (e2 in TONE_ELECTRODES)


def test_tone_energy_peaks_in_center_columns_and_correct_frequency_bin(encoder):
    x, obs = _tone_window()
    r = encoder.encode(x, obs)
    fov, uni = r["foveated"], r["uniform"]
    assert fov.shape == (4, 64, 32) and uni.shape == (4, 64, 32)
    assert fov.dtype == np.float32
    freq_bin = np.searchsorted(encoder.frequency_edges, 3.0, side="right") - 1
    assert encoder.frequency_edges[freq_bin] <= 3.0 < encoder.frequency_edges[freq_bin + 1]
    for region in range(4):
        fb, tb = np.unravel_index(np.argmax(fov[region]), fov[region].shape)
        assert 8 <= tb < 24, f"region {region}: foveated peak column {tb} outside the center"
        assert fb == freq_bin, f"region {region}: peak frequency bin {fb} != {freq_bin}"
        # the whole 3 Hz row is hotter inside the tone interval than outside it
        assert fov[region, freq_bin, 12:20].min() > fov[region, freq_bin, :8].max()
        assert fov[region, freq_bin, 12:20].min() > fov[region, freq_bin, 24:].max()
        # the uniform encoding places the peak in the 1.5625 s bins overlapping [22,28)
        fbu, tbu = np.unravel_index(np.argmax(uni[region]), uni[region].shape)
        assert 14 <= tbu <= 17 and fbu == freq_bin


def test_foveated_and_uniform_outputs_are_byte_matched(encoder):
    x, obs = _tone_window()
    r = encoder.encode(x, obs)
    assert r["foveated"].shape == r["uniform"].shape == (4, 64, 32)
    assert r["foveated"].nbytes == r["uniform"].nbytes
    assert r["foveated_mask"].shape == r["uniform_mask"].shape == (4, 64, 32)
    assert r["foveated_mask"].nbytes == r["uniform_mask"].nbytes
    assert pack_mask(r["foveated_mask"][None]).nbytes == pack_mask(r["uniform_mask"][None]).nbytes == 4 * 64 * 32 // 8


def test_masks_all_true_for_fully_observed_window(encoder):
    x, obs = _tone_window()
    r = encoder.encode(x, obs)
    assert r["foveated_mask"].all() and r["uniform_mask"].all()
    assert r["lead_valid_fraction"] == 1.0 and r["region_valid_fraction"] == 1.0
    assert np.isfinite(r["foveated"]).all() and np.isfinite(r["uniform"]).all()


def test_missing_tail_invalidates_only_trailing_columns(encoder):
    x, obs = _tone_window()
    obs = obs.copy(); obs[9000:] = False        # last 5 s unobserved on every electrode
    r = encoder.encode(x, obs)
    for key in ["foveated_mask", "uniform_mask"]:
        m = r[key]
        col_valid = m.all(axis=(0, 1))          # [32]
        col_invalid = ~m.any(axis=(0, 1))
        assert np.array_equal(col_valid, ~col_invalid)   # validity is column-wise (per time bin)
        assert not col_valid[-1]                        # the last column is invalid
        assert col_valid[:24].all()                     # the target and everything before it is valid
        # invalid columns form a suffix: once invalid, every later column is invalid too
        first_bad = int(np.argmin(col_valid))
        assert not col_valid[first_bad:].any()
        assert r[key.replace("_mask", "")][~m].max() == 0.0 if (~m).any() else True
    assert r["region_valid_fraction"] < 1.0


def test_pack_unpack_roundtrip_is_exact():
    rng = np.random.default_rng(3)
    for shape in [(5, 4, 64, 32), (3, 4, 64, 64), (1, 4, 64, 32)]:
        m = rng.random(shape) < 0.5
        packed = pack_mask(m)
        assert packed.dtype == np.uint8 and packed.shape == (shape[0], int(np.prod(shape[1:])) // 8)
        np.testing.assert_array_equal(unpack_mask(packed, shape), m)
    m = np.zeros((2, 4, 64, 32), bool); m[1, 2, 10, 5] = True
    np.testing.assert_array_equal(unpack_mask(pack_mask(m), m.shape), m)


def test_frame_validity_rule_uses_valid_window_fraction(encoder):
    x, obs = _tone_window()
    obs = obs.copy(); obs[5000:5040, :] = False   # 40 missing samples = 0.2 s interior gap
    pw, lead_valid = encoder.power(x, obs)
    assert pw.shape == (16, 129, encoder.n_frames) and lead_valid.shape == (16, encoder.n_frames)
    # independent recomputation of the per-frame observed fraction on each bipolar pair
    hop, nperseg = encoder.hop, encoder.nperseg
    obs_pair = obs[:, encoder.a] & obs[:, encoder.b]
    expected = np.stack([obs_pair[hop * i: hop * i + nperseg].mean(0) >= C.VALID_WINDOW_FRACTION
                         for i in range(encoder.n_frames)], axis=1)
    np.testing.assert_array_equal(lead_valid, expected)
    # frames containing the whole 40-sample gap miss 15.6% > 10% -> invalid; far frames are valid
    assert not lead_valid[:, 75:79].any()
    assert lead_valid[:, :70].all() and lead_valid[:, 85:].all()
    # a gap of exactly 10% of a frame (25.6 samples) is the boundary: 25 missing samples stay valid
    obs2 = np.ones_like(obs); obs2[5000:5025, :] = False
    _, lv2 = encoder.power(x, obs2)
    assert lv2.all()
    assert np.isfinite(pw).all()


def test_small_gap_is_interpolated_but_long_gap_stays_zero(encoder):
    x, obs = _tone_window(noise=0.0, amplitude=0.0)
    x[:, 0] = np.linspace(0.0, 100.0, 10000, dtype=np.float32)   # a ramp on Fp1
    obs = obs.copy()
    obs[4000:4010, 0] = False            # 0.05 s gap: interpolated
    obs[6000:6200, 0] = False            # 1.0 s gap: stays zero
    x_gap = x.copy(); x_gap[~obs] = 0.0
    y = encoder._interpolate_small_gaps(x_gap, obs)
    np.testing.assert_allclose(y[4000:4010, 0], x[4000:4010, 0], atol=1e-3)
    assert np.all(y[6000:6200, 0] == 0.0)
    np.testing.assert_array_equal(y[:, 1:], x_gap[:, 1:])


# ------------------------------------------------------------------ ContextSpectralEncoder
SRC_FREQS = np.round(0.59 + 0.1953125 * np.arange(100), 2)   # 0.59 .. 19.92, 100 values per region


def _context_columns(seed=11):
    cols = ["time"] + [f"{r}_{f:.2f}" for r in ["LL", "RL", "LP", "RP"] for f in SRC_FREQS]
    rng = np.random.default_rng(seed)
    return [cols[i] for i in rng.permutation(len(cols))]     # deliberately shuffled column order


def _row_edges(n_rows=300):
    return np.arange(n_rows + 1, dtype=np.float64) * (600.0 / n_rows)


def test_context_encoder_sorts_frequencies_numerically_from_shuffled_columns():
    cols = _context_columns()
    enc = ContextSpectralEncoder(cols)
    np.testing.assert_allclose(enc.freqs, np.sort(SRC_FREQS))
    assert np.all(np.diff(enc.freqs) > 0)
    for r, region in enumerate(["LL", "RL", "LP", "RP"]):
        assert enc.region_cols[region] == [f"{region}_{f:.2f}" for f in SRC_FREQS]
        assert [cols[i] for i in enc.region_idx[r]] == enc.region_cols[region]
    # a power that grows with frequency must come out monotone along the binned axis
    values = np.zeros((300, len(cols)), np.float32)
    for c, name in enumerate(cols):
        if name != "time":
            values[:, c] = float(name.split("_", 1)[1])
    out = enc.encode(values, _row_edges())
    assert out["context_mask"].all()
    for r in range(4):
        assert np.all(np.diff(out["context"][r, :, 0]) > 0)


def test_context_encoder_constant_power_gives_log_power():
    cols = _context_columns()
    enc = ContextSpectralEncoder(cols)
    power = 12.5
    values = np.full((300, len(cols)), power, np.float32)
    values[:, cols.index("time")] = np.arange(300) * 2.0 + 1.0
    out = enc.encode(values, _row_edges())
    assert out["context"].shape == (4, 64, 64) and out["context_mask"].shape == (4, 64, 64)
    assert out["context_mask"].all() and out["valid_fraction"] == 1.0
    np.testing.assert_allclose(out["context"], np.log(power), rtol=1e-5)


def test_context_encoder_nan_cells_invalidate_dominated_bins():
    cols = _context_columns()
    enc = ContextSpectralEncoder(cols)
    power = 2.0
    values = np.full((300, len(cols)), power, np.float32)
    rows = slice(100, 200)                                  # seconds [200, 400)
    bad_freqs = [f for f in SRC_FREQS if 8.0 <= f <= 12.0]  # a contiguous chunk of source frequencies
    bad_cols = [cols.index(f"LL_{f:.2f}") for f in bad_freqs]
    values[rows, bad_cols] = np.nan
    out = enc.encode(values, _row_edges())
    ctx, mask = out["context"], out["context_mask"]
    assert np.isfinite(ctx).all()
    assert not mask[0].all() and mask[1:].all()             # only LL is touched
    fe, te = enc.frequency_edges, enc.time_edges
    f_inside = (fe[:-1] >= 8.0) & (fe[1:] <= 12.0 + 0.2)    # destination bins fully inside the NaN chunk
    t_inside = (te[:-1] >= 200.0) & (te[1:] <= 400.0)
    assert f_inside.sum() >= 5 and t_inside.sum() >= 10
    assert not mask[0][np.ix_(f_inside, t_inside)].any()
    f_far = fe[1:] < 6.0; t_far = te[1:] < 150.0
    assert mask[0][np.ix_(f_far, t_far)].all()
    assert mask[0][:, t_far].all() and mask[0][f_far, :].all()
    np.testing.assert_allclose(ctx[0][mask[0]], np.log(power), rtol=1e-5)
    assert np.all(ctx[0][~mask[0]] == 0.0)
    assert 0 < out["valid_fraction"] < 1


def test_context_encoder_negative_values_are_invalid_not_logged():
    cols = _context_columns()
    enc = ContextSpectralEncoder(cols)
    power = 2.0
    base = np.full((300, len(cols)), power, np.float32)
    bad_freqs = [f for f in SRC_FREQS if 8.0 <= f <= 12.0]
    bad_cols = [cols.index(f"RP_{f:.2f}") for f in bad_freqs]
    neg = base.copy(); neg[100:200, bad_cols] = -5.0
    nan = base.copy(); nan[100:200, bad_cols] = np.nan
    out_neg, out_nan = enc.encode(neg, _row_edges()), enc.encode(nan, _row_edges())
    assert np.isfinite(out_neg["context"]).all()
    np.testing.assert_array_equal(out_neg["context_mask"], out_nan["context_mask"])
    np.testing.assert_allclose(out_neg["context"], out_nan["context"], rtol=1e-6)
    assert not out_neg["context_mask"][3].all() and out_neg["context_mask"][:3].all()
    np.testing.assert_allclose(out_neg["context"][out_neg["context_mask"]], np.log(power), rtol=1e-5)


def test_context_encoder_partial_rows_leave_trailing_time_bins_invalid():
    cols = _context_columns()
    enc = ContextSpectralEncoder(cols)
    values = np.full((150, len(cols)), 3.0, np.float32)
    edges = np.arange(151, dtype=np.float64) * 2.0      # only the first 300 s were supplied
    out = enc.encode(values, edges)
    m = out["context_mask"]
    assert m[:, :, :32].all() and not m[:, :, 32:].any()
    assert np.all(out["context"][:, :, 32:] == 0.0)
    np.testing.assert_allclose(out["valid_fraction"], 0.5)


def test_context_encoder_rejects_inconsistent_or_missing_regions():
    with pytest.raises(ValueError):
        ContextSpectralEncoder(["time", "LL_1.00", "LL_2.00", "RL_1.00", "RL_3.00", "LP_1.00", "LP_2.00", "RP_1.00", "RP_2.00"])
    with pytest.raises(ValueError):
        ContextSpectralEncoder(["time"])
