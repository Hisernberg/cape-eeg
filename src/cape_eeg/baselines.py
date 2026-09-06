"""Baselines B0 (prior), B1 (band-power soft regression) and B3 (timm MobileNetV3-Small comparator)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import N_CLASSES, N_REGIONS, FREQUENCY_BINS, FOCAL_TARGET_COLUMNS, RAW_FREQUENCY_RANGE_HZ, linear_frequency_edges

BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0), "gamma": (30.0, 40.0)}


# ---------------------------------------------------------------- B0
class PriorBaseline:
    def __init__(self):
        self.p = None
    def fit(self, q: np.ndarray):
        self.p = q.mean(0); return self
    def predict(self, n: int) -> np.ndarray:
        return np.repeat(self.p[None, :], n, 0)


# ---------------------------------------------------------------- B1
def band_masks(freq_lo: float, freq_hi: float) -> dict:
    edges = linear_frequency_edges(FREQUENCY_BINS, freq_lo, freq_hi)
    centers = (edges[:-1] + edges[1:]) / 2
    return {b: (centers >= lo) & (centers < hi) for b, (lo, hi) in BANDS.items() if ((centers >= lo) & (centers < hi)).any()}


def b1_features(batch: dict, ctx_freq_range: tuple[float, float]) -> np.ndarray:
    """Fixed regional band-power summaries per view; center-vs-context differences for the local view."""
    L, Lm = batch["local"], batch["local_mask"]; C, Cm = batch["context"], batch["context_mask"]
    feats = []
    lb = band_masks(*RAW_FREQUENCY_RANGE_HZ); cb = band_masks(*ctx_freq_range)
    c0, c1 = FOCAL_TARGET_COLUMNS
    def masked_mean(x, m, axes):
        w = m.astype(np.float32); return (x * w).sum(axes) / np.maximum(w.sum(axes), 1.0)
    for name, sel in lb.items():
        feats.append(masked_mean(L[:, :, sel, :], Lm[:, :, sel, :], (2, 3)))                       # [B,4]
        center = masked_mean(L[:, :, sel, c0:c1], Lm[:, :, sel, c0:c1], (2, 3))
        outer = masked_mean(np.concatenate([L[:, :, sel, :c0], L[:, :, sel, c1:]], 3), np.concatenate([Lm[:, :, sel, :c0], Lm[:, :, sel, c1:]], 3), (2, 3))
        feats.append(center - outer)
    for name, sel in cb.items():
        feats.append(masked_mean(C[:, :, sel, :], Cm[:, :, sel, :], (2, 3)))
        mid = slice(30, 34)  # the central 37.5 s of the 600 s context
        feats.append(masked_mean(C[:, :, sel, mid], Cm[:, :, sel, mid], (2, 3)) - masked_mean(C[:, :, sel, :], Cm[:, :, sel, :], (2, 3)))
    feats.append(batch["valid_l"][:, None]); feats.append(batch["valid_c"][:, None])
    return np.concatenate(feats, 1).astype(np.float32)


class SoftLogisticRegression:
    """Multinomial logistic regression on fractional targets (soft cross-entropy), L2-regularised, CPU."""

    def __init__(self, l2: float = 1e-3, epochs: int = 300, lr: float = 0.05, seed: int = 0):
        self.l2, self.epochs, self.lr, self.seed = l2, epochs, lr, seed
        self.mu = self.sd = None; self.W = None

    def fit(self, X: np.ndarray, q: np.ndarray, log=None):
        torch.manual_seed(self.seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xt = torch.tensor((X - self.mu) / self.sd); qt = torch.tensor(q)
        lin = nn.Linear(X.shape[1], N_CLASSES)
        opt = torch.optim.LBFGS(lin.parameters(), lr=1.0, max_iter=self.epochs, history_size=20, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            logp = F.log_softmax(lin(Xt), -1)
            loss = -(qt * logp).sum(1).mean() + self.l2 * (lin.weight ** 2).sum()
            loss.backward(); return loss
        opt.step(closure)
        self.lin = lin.eval(); return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return F.softmax(self.lin(torch.tensor((X - self.mu) / self.sd)), -1).numpy()


# ---------------------------------------------------------------- B3
class MobileNetV3Comparator(nn.Module):
    """timm mobilenetv3_small_100.lamb_in1k with 4 input channels and 6 outputs.

    The two views are concatenated along time after separate normalization (64x96) and resized
    bilinearly to 128x192 without a center crop. The artificial view boundary is documented.
    """
    HF_ID = "timm/mobilenetv3_small_100.lamb_in1k"

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import timm
        self.net = timm.create_model("mobilenetv3_small_100.lamb_in1k", pretrained=pretrained, in_chans=N_REGIONS, num_classes=N_CLASSES)
        self.learned_gate = False; self.aux_on = False

    def backbone_parameters(self):
        return [p for n, p in self.net.named_parameters() if not n.startswith("classifier")]

    def head_parameters(self):
        return [p for n, p in self.net.named_parameters() if n.startswith("classifier")]

    def forward(self, local, context, valid_l, valid_c, force_view=None):
        x = torch.cat([local, context], -1)                                  # [B,4,64,96]
        x = F.interpolate(x, size=(128, 192), mode="bilinear", align_corners=False)
        z = self.net(x).float()
        p = F.softmax(z, -1)
        return {"p": p, "pL": p, "pC": p, "g": torch.full((p.shape[0],), 0.5, device=p.device), "js": torch.zeros(p.shape[0], device=p.device), "d_hat": None}
