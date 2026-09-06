"""CPU tests for the compact model variants, the B3 comparator and the training losses."""

import math

import numpy as np
import pytest
import torch

from cape_eeg import model as M
from cape_eeg.training import losses as L

DEVICE = torch.device("cpu")
B = 4


def _batch(seed=0, n=B):
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    local = torch.randn(n, 4, 64, 32, generator=g)
    context = torch.randn(n, 4, 64, 64, generator=g)
    votes = torch.randint(0, 4, (n, 6), generator=g); votes[:, 0] += 1
    votes[0] = torch.tensor([1, 0, 0, 0, 0, 0])                  # an n=1 row: d is NaN
    q = votes.float() / votes.sum(1, keepdim=True)
    nv = votes.sum(1).float()
    d = 1 - (votes.float() * (votes.float() - 1)).sum(1) / (nv * (nv - 1))
    d[nv <= 1] = float("nan")
    return {"local": local, "context": context, "valid_l": torch.ones(n), "valid_c": torch.ones(n),
            "q": q, "votes": votes, "d": d, "n_votes": nv}


def _forward(model, b, **kw):
    return model(b["local"], b["context"], b["valid_l"], b["valid_c"], **kw)


@pytest.mark.parametrize("config_id", list(M.CONFIGS))
def test_config_forward_gives_valid_probabilities(config_id):
    torch.manual_seed(0)
    model = M.build_model(config_id).to(DEVICE).eval()
    b = _batch()
    with torch.no_grad():
        out = _forward(model, b)
    for key in ["p", "pL", "pC"]:
        p = out[key]
        assert p.shape == (B, 6) and torch.isfinite(p).all()
        assert torch.all(p >= 0)
        assert torch.allclose(p.sum(1), torch.ones(B), atol=1e-5)
    assert out["g"].shape == (B,) and torch.all((out["g"] >= 0) & (out["g"] <= 1))
    assert out["js"].shape == (B,) and torch.isfinite(out["js"]).all() and torch.all(out["js"] >= 0)
    cfg = M.CONFIGS[config_id]
    assert (out["d_hat"] is not None) == cfg["aux"]
    if cfg["aux"]:
        assert out["d_hat"].shape == (B,) and torch.all((out["d_hat"] >= 0) & (out["d_hat"] <= 1))
    assert model.encoding == cfg["encoding"]
    assert (model.gate is not None) == cfg["learned_gate"]


def test_b3_comparator_forward_without_pretrained_weights():
    pytest.importorskip("timm")
    from cape_eeg.baselines import MobileNetV3Comparator
    torch.manual_seed(0)
    model = MobileNetV3Comparator(pretrained=False).to(DEVICE).eval()
    b = _batch()
    with torch.no_grad():
        out = _forward(model, b)
    assert out["p"].shape == (B, 6) and torch.isfinite(out["p"]).all()
    assert torch.allclose(out["p"].sum(1), torch.ones(B), atol=1e-5)
    assert torch.all(out["g"] == 0.5) and out["d_hat"] is None
    assert len(list(model.backbone_parameters())) + len(list(model.head_parameters())) == len(list(model.parameters()))


def test_parameter_counts_match_the_report():
    assert M.count_parameters(M.build_model("B2"))["total"] == 147524
    assert M.count_parameters(M.build_model("A1"))["total"] == 147524      # same bytes, different bins
    assert M.count_parameters(M.build_model("P"))["total"] == 164134
    c = M.count_parameters(M.build_model("P"))
    assert c["trainable"] == c["total"]
    assert M.count_parameters(M.build_model("A2"))["total"] < 164134         # no auxiliary head
    assert M.count_parameters(M.build_model("P_MSF"))["total"] > 164134


@pytest.mark.parametrize("config_id", ["P", "A2", "B2"])
def test_absent_view_routing(config_id):
    torch.manual_seed(1)
    model = M.build_model(config_id).to(DEVICE).eval()
    b = _batch()
    with torch.no_grad():
        ref = _forward(model, b)
        no_local = _forward(model, {**b, "valid_l": torch.zeros(B)})
        no_context = _forward(model, {**b, "valid_c": torch.zeros(B)})
    assert torch.all(no_local["g"] == 1.0)
    assert torch.allclose(no_local["p"], no_local["pC"], atol=1e-6)
    assert torch.all(no_context["g"] == 0.0)
    assert torch.allclose(no_context["p"], no_context["pL"], atol=1e-6)
    # the per-view heads themselves do not depend on the validity scalars
    assert torch.allclose(no_local["pL"], ref["pL"]) and torch.allclose(no_context["pC"], ref["pC"])
    # both absent: the context rule is applied last and wins (gate 0)
    with torch.no_grad():
        both = _forward(model, {**b, "valid_l": torch.zeros(B), "valid_c": torch.zeros(B)})
    assert torch.all(both["g"] == 0.0)


@pytest.mark.parametrize("config_id", ["P", "B2"])
def test_force_view(config_id):
    torch.manual_seed(2)
    model = M.build_model(config_id).to(DEVICE).eval()
    b = _batch()
    with torch.no_grad():
        loc = _forward(model, b, force_view="local")
        ctx = _forward(model, b, force_view="context")
        # force_view overrides even the absent-view rule
        loc2 = _forward(model, {**b, "valid_l": torch.zeros(B)}, force_view="local")
    assert torch.all(loc["g"] == 0.0) and torch.allclose(loc["p"], loc["pL"], atol=1e-6)
    assert torch.all(ctx["g"] == 1.0) and torch.allclose(ctx["p"], ctx["pC"], atol=1e-6)
    assert torch.all(loc2["g"] == 0.0)
    assert not torch.allclose(loc["p"], ctx["p"])


def test_fixed_gate_is_half_and_mixture_is_the_average():
    for cid in ["A1", "B2"]:
        model = M.build_model(cid).to(DEVICE).eval()
        b = _batch()
        with torch.no_grad():
            out = _forward(model, b)
        assert torch.all(out["g"] == 0.5)
        assert torch.allclose(out["p"], 0.5 * out["pL"] + 0.5 * out["pC"], atol=1e-6)


def test_learned_gate_depends_on_input():
    torch.manual_seed(3)
    model = M.build_model("P").to(DEVICE).eval()
    with torch.no_grad():
        g1 = _forward(model, _batch(seed=1))["g"]; g2 = _forward(model, _batch(seed=2))["g"]
    assert not torch.allclose(g1, g2)
    assert torch.all((g1 > 0) & (g1 < 1))


def test_center_weights_pooled_onto_eight_columns():
    uni = M.build_model("B2")._center_weights(8)
    fov = M.build_model("A1")._center_weights(8)
    torch.testing.assert_close(uni, torch.tensor([0, 0, 0, .5, .5, 0, 0, 0], dtype=torch.float32))
    torch.testing.assert_close(fov, torch.tensor([0, 0, .25, .25, .25, .25, 0, 0], dtype=torch.float32))
    assert math.isclose(float(uni.sum()), 1.0, abs_tol=1e-6) and math.isclose(float(fov.sum()), 1.0, abs_tol=1e-6)
    full = M.build_model("A1").center_weights_full
    assert full.shape == (32,) and torch.all(full[8:24] == 1) and torch.all(full[:8] == 0) and torch.all(full[24:] == 0)
    torch.testing.assert_close(M.build_model("A1")._center_weights(32), full / full.sum())


@pytest.mark.parametrize("config_id", list(M.CONFIGS))
def test_backward_gives_finite_gradients_for_every_parameter(config_id):
    torch.manual_seed(4)
    model = M.build_model(config_id).to(DEVICE).train()
    b = _batch()
    out = _forward(model, b)
    aux_w = 0.1 if model.aux_on else 0.0
    loss, parts = L.total_loss(out, b, aux_w)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{config_id}: no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"{config_id}: nonfinite gradient in {name}"
    assert parts["loss"] == pytest.approx(parts["soft_ce"] + aux_w * parts["aux_mse"], abs=1e-6)


def test_js_divergence_properties():
    p = torch.tensor([[0.5, 0.5, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0]])
    q = torch.tensor([[0.5, 0.5, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0]])
    js = M.js_divergence(p, q)
    assert js[0].abs() < 1e-5
    assert js[1] == pytest.approx(math.log(2), abs=1e-4)      # maximal for disjoint support
    assert torch.allclose(M.js_divergence(p, q), M.js_divergence(q, p))


# ------------------------------------------------------------------ losses
def test_soft_cross_entropy_matches_hand_computation():
    p = torch.tensor([[0.7, 0.1, 0.05, 0.05, 0.05, 0.05]])
    q = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0, 0.0]])
    expected = -(0.5 * math.log(0.7) + 0.5 * math.log(0.1))
    assert L.soft_cross_entropy(p, q).item() == pytest.approx(expected, rel=1e-6)
    assert L.soft_cross_entropy(p, q).shape == (1,)
    # the clamp keeps an impossible prediction finite
    z = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    assert torch.isfinite(L.soft_cross_entropy(z, q)).all()
    assert L.soft_cross_entropy(z, q).item() == pytest.approx(-0.5 * math.log(L.EPS), rel=1e-6)


def test_kl_rows_zero_on_agreement_and_log6_minus_entropy_for_uniform():
    q = torch.tensor([[0.5, 0.5, 0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4, 0, 0], [1.0, 0, 0, 0, 0, 0]])
    assert torch.allclose(L.kl_rows(q, q), torch.zeros(3), atol=1e-6)
    uniform = torch.full((3, 6), 1 / 6)
    H = -(q * torch.log(q.clamp_min(1e-12))).sum(1)
    H[q.sum(1) == 0] = 0
    torch.testing.assert_close(L.kl_rows(q, uniform), math.log(6) - H, atol=1e-5, rtol=0)
    assert L.kl_rows(q, uniform)[2].item() == pytest.approx(math.log(6), abs=1e-5)
    assert torch.all(L.kl_rows(q, uniform) >= 0)
    # unnormalised predictions are renormalised inside kl_rows
    torch.testing.assert_close(L.kl_rows(q, 3 * uniform), L.kl_rows(q, uniform))


def test_masked_disagreement_mse_ignores_nan_rows_and_is_differentiable_when_empty():
    d_hat = torch.tensor([0.2, 0.9, 0.5, 0.1], requires_grad=True)
    d = torch.tensor([float("nan"), 1.0, 0.0, float("nan")])
    loss = L.masked_disagreement_mse(d_hat, d)
    assert loss.item() == pytest.approx(((0.9 - 1.0) ** 2 + (0.5 - 0.0) ** 2) / 2, rel=1e-6)
    loss.backward()
    assert d_hat.grad[0] == 0 and d_hat.grad[3] == 0 and d_hat.grad[1] != 0 and torch.isfinite(d_hat.grad).all()
    # no observed d anywhere: a differentiable zero, not NaN
    d_hat2 = torch.tensor([0.2, 0.9], requires_grad=True)
    empty = L.masked_disagreement_mse(d_hat2, torch.full((2,), float("nan")))
    assert empty.requires_grad and empty.item() == 0.0 and torch.isfinite(empty)
    empty.backward()
    assert torch.all(d_hat2.grad == 0)
    # models without an auxiliary head return a plain zero scalar
    none = L.masked_disagreement_mse(None, torch.tensor([0.5, float("nan")]))
    assert none.shape == () and none.item() == 0.0 and not none.requires_grad
    # every row observed
    full = L.masked_disagreement_mse(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0]))
    assert full.item() == pytest.approx(0.5)


def test_total_loss_with_zero_aux_weight_equals_soft_ce():
    torch.manual_seed(5)
    model = M.build_model("P").to(DEVICE).eval()
    b = _batch()
    with torch.no_grad():
        out = _forward(model, b)
    loss0, parts0 = L.total_loss(out, b, 0.0)
    ce = L.soft_cross_entropy(out["p"], b["q"]).mean()
    assert torch.allclose(loss0, ce)
    assert parts0["aux_mse"] == 0.0 and parts0["loss"] == pytest.approx(parts0["soft_ce"])
    loss1, parts1 = L.total_loss(out, b, 0.3)
    aux = L.masked_disagreement_mse(out["d_hat"], b["d"])
    assert torch.allclose(loss1, ce + 0.3 * aux)
    assert parts1["aux_mse"] == pytest.approx(aux.item()) and torch.isfinite(loss1)
