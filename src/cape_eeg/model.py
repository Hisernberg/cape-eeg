"""CAPE-EEG compact foveated-gate model (docs/01 section 6) and its ablation variants.

Variants are selected by constructor flags so that every ablation shares the same trunk code:
  learned_gate=False  -> fixed 50:50 mixture (B2, A1)
  aux=False           -> no disagreement head (B2, A1, A2)
  center_weights      -> pooling map for the encoding actually cached (foveated or uniform)
  multiscale=True     -> replaces the last residual block with an economical multi-scale fusion
                         block (CAPE-EEG+MSF; exploratory 2026-inspired extension, docs/01 S03)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import N_CLASSES, N_REGIONS, foveated_time_edges, uniform_time_edges, center_weights_from_edges

EPS = 1e-7


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(8 if c % 8 == 0 else 1, c)


class DSBlock(nn.Module):
    """Depthwise-separable block: dw3x3(stride) -> GN -> SiLU -> pw1x1 -> GN -> SiLU (+ residual)."""

    def __init__(self, cin: int, cout: int, stride=(1, 1), residual: bool = False):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False)
        self.n1 = _gn(cin); self.pw = nn.Conv2d(cin, cout, 1, bias=False); self.n2 = _gn(cout)
        self.residual = residual and cin == cout and tuple(stride) == (1, 1)

    def forward(self, x):
        y = F.silu(self.n1(self.dw(x)))
        y = F.silu(self.n2(self.pw(y)))
        return x + y if self.residual else y


class MultiScaleFusionBlock(nn.Module):
    """Economical multi-scale fusion: parallel depthwise convs with temporal dilations 1/2/3,
    concatenated, fused by a 1x1 projection and a lightweight channel gate, with a residual path.
    Inspired by lightweight multi-scale fusion in 2026 dense-prediction work (docs/01 S03); the
    task and backbone differ, so this is an exploratory named configuration, not a claim."""

    def __init__(self, c: int, dilations=(1, 2, 3), reduction: int = 4):
        super().__init__()
        self.branches = nn.ModuleList([nn.Conv2d(c, c, 3, padding=(1, d), dilation=(1, d), groups=c, bias=False) for d in dilations])
        self.n1 = _gn(c * len(dilations)); self.fuse = nn.Conv2d(c * len(dilations), c, 1, bias=False); self.n2 = _gn(c)
        self.gate = nn.Sequential(nn.Linear(c, c // reduction), nn.SiLU(), nn.Linear(c // reduction, c), nn.Sigmoid())

    def forward(self, x):
        y = torch.cat([b(x) for b in self.branches], 1)
        y = F.silu(self.n2(self.fuse(F.silu(self.n1(y)))))
        s = self.gate(y.mean((2, 3)))[:, :, None, None]
        return x + y * s


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (torch.log(a.clamp_min(EPS)) - torch.log(b.clamp_min(EPS)))).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


class CAPEModel(nn.Module):
    def __init__(self, channels=(24, 48, 64, 96, 128), embedding: int = 128, learned_gate: bool = True, aux: bool = True,
                 encoding: str = "foveated", multiscale: bool = False, fixed_gate_value: float = 0.5, time_bins: int = 32, dm: bool = False):
        super().__init__()
        c0, c1, c2, c3, c4 = channels
        self.stem_l = nn.Sequential(nn.Conv2d(N_REGIONS, c0, 3, padding=1, bias=False), _gn(c0), nn.SiLU())
        self.stem_c = nn.Sequential(nn.Conv2d(N_REGIONS, c0, 3, padding=1, bias=False), _gn(c0), nn.SiLU())
        last = MultiScaleFusionBlock(c4) if multiscale else DSBlock(c4, c4, (1, 1), residual=True)
        self.trunk = nn.Sequential(DSBlock(c0, c1, (2, 1)), DSBlock(c1, c2, (2, 2)), DSBlock(c2, c3, (2, 2)),
                                   DSBlock(c3, c4, (1, 1)), DSBlock(c4, c4, (1, 1), residual=True), last)
        self.proj_l = nn.Linear(3 * c4, embedding); self.proj_c = nn.Linear(2 * c4, embedding)
        self.head_l = nn.Linear(embedding, N_CLASSES); self.head_c = nn.Linear(embedding, N_CLASSES)
        self.learned_gate, self.aux_on, self.fixed_gate_value = learned_gate, aux, fixed_gate_value
        self.gate = nn.Sequential(nn.Linear(2 * embedding + 3, 32), nn.SiLU(), nn.Linear(32, 1)) if learned_gate else None
        self.aux = nn.Sequential(nn.Linear(2 * embedding, 32), nn.SiLU(), nn.Linear(32, 1)) if aux else None
        self.dm = dm and aux   # Dirichlet-multinomial concentration head instead of the scalar disagreement regressor
        edges = foveated_time_edges(time_bins) if encoding == "foveated" else uniform_time_edges(time_bins)
        w = torch.tensor(center_weights_from_edges(edges), dtype=torch.float32)
        self.register_buffer("center_weights_full", w)
        self.encoding, self.time_bins = encoding, time_bins

    def _center_weights(self, t_out: int) -> torch.Tensor:
        """Pool the 32-column center map onto the trunk's downsampled time axis (adaptive average)."""
        w = F.adaptive_avg_pool1d(self.center_weights_full[None, None, :], t_out)[0, 0]
        return w / w.sum().clamp_min(EPS)

    def encode_local(self, x):
        h = self.trunk(self.stem_l(x))                       # [B,C,f,t]
        mean = h.mean((2, 3)); mx = h.amax((2, 3))
        w = self._center_weights(h.shape[-1]).to(h.dtype)
        center = (h.mean(2) * w[None, None, :]).sum(-1)        # [B,C]
        return self.proj_l(torch.cat([mean, mx, center], 1))

    def encode_context(self, x):
        h = self.trunk(self.stem_c(x))
        return self.proj_c(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1))

    def forward(self, local, context, valid_l, valid_c, force_view: str | None = None):
        uL = self.encode_local(local); uC = self.encode_context(context)
        # heads, gate, mixture and auxiliary output stay in float32 regardless of autocast
        with torch.autocast(device_type=uL.device.type, enabled=False):
            uL = uL.float(); uC = uC.float(); valid_l = valid_l.float(); valid_c = valid_c.float()
            zL = self.head_l(uL); zC = self.head_c(uC)
            pL = F.softmax(zL, -1); pC = F.softmax(zC, -1)
            js = js_divergence(pL, pC)
            if self.learned_gate:
                g = torch.sigmoid(self.gate(torch.cat([uL, uC, valid_l[:, None], valid_c[:, None], js[:, None]], 1))).squeeze(1)
            else:
                g = torch.full_like(js, self.fixed_gate_value)
            # unavailable views force the surviving path
            g = torch.where(valid_l <= 0, torch.ones_like(g), g)
            g = torch.where(valid_c <= 0, torch.zeros_like(g), g)
            if force_view == "local":
                g = torch.zeros_like(g)
            elif force_view == "context":
                g = torch.ones_like(g)
            p = (1 - g)[:, None] * pL + g[:, None] * pC
            dm_c = None
            if self.aux_on and self.dm:
                # concentration c(x) of a Dirichlet-multinomial vote model v ~ DirMult(n, c * p);
                # expected pairwise disagreement under that model: 1 - sum_k p_k (c p_k + 1)/(c + 1)
                dm_c = F.softplus(self.aux(torch.cat([uL, uC], 1)).squeeze(1)) + 1e-3
                d_hat = 1.0 - (p * (dm_c[:, None] * p + 1.0) / (dm_c[:, None] + 1.0)).sum(-1)
            else:
                d_hat = torch.sigmoid(self.aux(torch.cat([uL, uC], 1))).squeeze(1) if self.aux_on else None
        return {"p": p, "pL": pL, "pC": pC, "g": g, "js": js, "d_hat": d_hat, "dm_c": dm_c, "uL": uL, "uC": uC}


def count_parameters(m: nn.Module) -> dict:
    total = sum(p.numel() for p in m.parameters()); trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


CONFIGS = {
    # id: (constructor kwargs, cache encoding, description)
    "B2": dict(learned_gate=False, aux=False, encoding="uniform", multiscale=False),
    "A1": dict(learned_gate=False, aux=False, encoding="foveated", multiscale=False),
    "A2": dict(learned_gate=True, aux=False, encoding="foveated", multiscale=False),
    "P": dict(learned_gate=True, aux=True, encoding="foveated", multiscale=False),
    "P_MSF": dict(learned_gate=True, aux=True, encoding="foveated", multiscale=True),
    # post-lock development configurations (never eligible for the locked comparison)
    "P_DM": dict(learned_gate=True, aux=True, encoding="foveated", multiscale=False, dm=True),
    "B2_t16": dict(learned_gate=False, aux=False, encoding="uniform", multiscale=False, time_bins=16),
    "B2_t8": dict(learned_gate=False, aux=False, encoding="uniform", multiscale=False, time_bins=8),
    "A1_t16": dict(learned_gate=False, aux=False, encoding="foveated", multiscale=False, time_bins=16),
    "A1_t8": dict(learned_gate=False, aux=False, encoding="foveated", multiscale=False, time_bins=8),
}
DESCRIPTIONS = {
    "B0": "training-set empirical soft class prior",
    "B1": "CPU band-power soft-label logistic regression",
    "B2": "compact trunk, uniform local bins, fixed 50:50 fusion, no auxiliary head (byte-matched control)",
    "B3": "timm MobileNetV3-Small (ImageNet, 4-channel adaptation) on the same cached evidence",
    "B3S": "MobileNetV3-Small trained from scratch (no ImageNet weights); post-lock development control for the pretraining confound",
    "B3H": "MobileNetV3-Small width 0.5 (ImageNet, 4-channel adaptation); post-lock development control for the scale confound",
    "A1": "B2 + foveated local bins",
    "A2": "A1 + learned gate (auxiliary loss off)",
    "P": "A2 + disagreement auxiliary loss (full prespecified candidate)",
    "P_MSF": "P with an economical multi-scale temporal fusion block replacing the last residual block (exploratory)",
    "P_DM": "CAPE-EEG v2 head: Dirichlet-multinomial vote model with a learned concentration c(x) replacing the scalar disagreement regressor (post-lock, development only)",
    "B2_t16": "B2 with 16 uniform local time bins (byte-budget sweep; half the local bytes)",
    "B2_t8": "B2 with 8 uniform local time bins (byte-budget sweep; quarter of the local bytes)",
    "A1_t16": "A1 with 16 foveated local time bins (4+8+4; byte-budget sweep)",
    "A1_t8": "A1 with 8 foveated local time bins (2+4+2; byte-budget sweep)",
}


def build_model(config_id: str) -> nn.Module:
    if config_id in ("B3", "B3S", "B3H"):
        from .baselines import MobileNetV3Comparator
        if config_id == "B3S":
            return MobileNetV3Comparator(pretrained=False)
        if config_id == "B3H":
            return MobileNetV3Comparator(pretrained=True, variant="mobilenetv3_small_050.lamb_in1k")
        return MobileNetV3Comparator()
    return CAPEModel(**CONFIGS[config_id])
