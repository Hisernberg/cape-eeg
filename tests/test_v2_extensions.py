"""Tests for the revision extensions: coarser foveated edges, time-bin pooling, Dirichlet-multinomial loss, patient subsampling."""
import numpy as np, pandas as pd, torch, pytest
from cape_eeg.contracts import foveated_time_edges, uniform_time_edges, center_weights_from_edges
from cape_eeg.data.dataset import pool_time_bins, subsample_patients
from cape_eeg.training.losses import dirichlet_multinomial_nll, total_loss
from cape_eeg.model import build_model, CONFIGS


def test_foveated_edges_scale_and_nest():
    e32, e16, e8 = foveated_time_edges(32), foveated_time_edges(16), foveated_time_edges(8)
    assert e16.shape == (17,) and e8.shape == (9,)
    assert np.allclose(e16, e32[::2]) and np.allclose(e8, e32[::4])
    assert np.isclose(center_weights_from_edges(e16)[4:12].sum(), 8) and np.isclose(center_weights_from_edges(e8)[2:6].sum(), 4)
    with pytest.raises(ValueError):
        foveated_time_edges(12)


def test_pool_time_bins_is_linear_power_mean_and_mask_rule():
    rng = np.random.default_rng(0); v = np.log(rng.uniform(0.5, 5, (3, 4, 64, 32))).astype(np.float16); m = np.ones(v.shape, bool)
    m[0, 0, :, :3] = False  # 3 of the first 4 bins invalid -> merged cell invalid at factor 4, valid at factor 2 for pair (2,3)? pair (0,1) invalid
    nv, nm = pool_time_bins(v, m, 4)
    assert nv.shape == (3, 4, 64, 8) and not nm[0, 0, 0, 0] and nm[0, 1, 0, 0]
    expected = np.log(np.exp(v[1, 2, 5, 4:8].astype(np.float32)).mean()); assert np.isclose(nv[1, 2, 5, 1], expected, atol=1e-3)
    nv2, nm2 = pool_time_bins(v, m, 2); assert not nm2[0, 0, 0, 0] and nm2[0, 0, 0, 1]  # bins (2,3): one valid of two -> valid (>= half)
    assert np.all(nv[~nm] == 0)


def test_dirichlet_multinomial_nll_properties():
    votes = torch.tensor([[4, 0, 0, 0, 0, 0], [2, 2, 0, 0, 0, 0]]); p = torch.tensor([[0.9, 0.02, 0.02, 0.02, 0.02, 0.02], [0.5, 0.5, 0, 0, 0, 0]])
    c_hi, c_lo = torch.tensor([100.0, 100.0]), torch.tensor([1.0, 1.0])
    nll_hi = dirichlet_multinomial_nll(votes, p, c_hi); nll_lo = dirichlet_multinomial_nll(votes, p, c_lo)
    assert torch.isfinite(nll_hi) and torch.isfinite(nll_lo)
    # unanimous votes with a confident p are better explained by high concentration; split votes on a 50:50 p also (mean matches) -> overall high c wins here
    assert nll_hi < nll_lo
    m = build_model("P_DM"); out = m(torch.randn(2, 4, 64, 32), torch.randn(2, 4, 64, 64), torch.ones(2), torch.ones(2))
    assert out["dm_c"] is not None and torch.all(out["dm_c"] > 0) and torch.all((out["d_hat"] >= 0) & (out["d_hat"] <= 1))
    loss, parts = total_loss(out, {"q": votes.float() / votes.sum(1, keepdim=True), "votes": votes, "d": torch.tensor([0.0, 1.0])}, 0.1); loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_subsample_patients_deterministic_and_patient_level():
    idx = pd.DataFrame({"row": np.arange(100), "patient_id": np.repeat(np.arange(20), 5)})
    a = subsample_patients(idx, 0.5, 101); b = subsample_patients(idx, 0.5, 101); c = subsample_patients(idx, 0.5, 202)
    assert np.array_equal(a, b) and len(a) == 50 and len(set(idx.patient_id[a])) == 10 and not np.array_equal(a, c)


def test_sweep_configs_build_with_matching_time_bins():
    for cid in ["B2_t16", "A1_t16", "B2_t8", "A1_t8"]:
        m = build_model(cid); assert m.time_bins == CONFIGS[cid]["time_bins"]
        out = m(torch.randn(2, 4, 64, m.time_bins), torch.randn(2, 4, 64, 64), torch.ones(2), torch.ones(2)); assert torch.allclose(out["p"].sum(1), torch.ones(2), atol=1e-5)
