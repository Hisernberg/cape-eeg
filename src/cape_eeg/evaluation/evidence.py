"""Evidence-use audits (docs/04 section 8): view/region deletion and ranked-vs-random occlusion curves."""
from __future__ import annotations

import numpy as np
import torch

from ..contracts import N_REGIONS, FOCAL_TIME_BINS, FREQUENCY_BINS
from ..data.dataset import CacheDataset, to_device
from ..training.engine import predict
from ..training.losses import kl_rows


def view_and_region_deletion(model, ds: CacheDataset, device) -> dict[str, np.ndarray]:
    """Predictions with the local view removed, the context view removed, and each region removed."""
    out = {"intact": predict(model, ds, device)["p"], "local_removed": predict(model, ds, device, force_view="context")["p"],
           "context_removed": predict(model, ds, device, force_view="local")["p"]}
    for r, name in enumerate(["LL", "RL", "LP", "RP"]):
        def pert(b, r=r):
            b["local"][:, r] = 0; b["context"][:, r] = 0; b["local_mask"][:, r] = False; b["context_mask"][:, r] = False
            b["valid_l"] = b["local_mask"].reshape(b["local_mask"].shape[0], -1).float().mean(1); b["valid_c"] = b["context_mask"].reshape(b["context_mask"].shape[0], -1).float().mean(1)
            return b
        out[f"region_removed_{name}"] = predict(model, ds, device, perturb=pert)["p"]
    return out


def select_cases(ds: CacheDataset, per_cell: int = 4, seed: int = 20260907) -> np.ndarray:
    """<=48 cases: 6 classes x {low, high} target entropy x per_cell, from unique-majority rows, label-only selection."""
    rng = np.random.default_rng(seed)
    q = ds.q.astype(np.float64); ent = -(q * np.log(np.clip(q, 1e-12, None))).sum(1)
    mx = ds.votes.max(1, keepdims=True); uniq = (ds.votes == mx).sum(1) == 1; am = ds.votes.argmax(1)
    med = np.median(ent[uniq]); chosen = []
    for k in range(6):
        for band in ["low", "high"]:
            m = uniq & (am == k) & ((ent <= med) if band == "low" else (ent > med))
            cand = np.where(m)[0]
            if len(cand):
                chosen.extend(rng.choice(cand, min(per_cell, len(cand)), replace=False).tolist())
    return np.array(sorted(chosen))


def saliency_ranked_deletion(model, ds: CacheDataset, positions: np.ndarray, device, fractions=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9), seed: int = 20260907) -> dict:
    """Delete local-view (region, time-bin) blocks ranked by |grad x input| vs a random order; report loss/JS curves."""
    from ..metrics import jensen_shannon_distance_rows
    model.eval(); rng = np.random.default_rng(seed)
    b = to_device(ds.gather(positions), device)
    L = b["local"].clone().requires_grad_(True)
    out = model(L, b["context"], b["valid_l"], b["valid_c"])
    loss = -(b["q"] * torch.log(out["p"].clamp_min(1e-7))).sum(1).sum()
    grad, = torch.autograd.grad(loss, L)
    sal = (grad * L).abs().sum(2).detach()                 # [B,4,T] importance per (region, time-bin)
    B = sal.shape[0]; n_blocks = N_REGIONS * FOCAL_TIME_BINS
    ranked = torch.argsort(sal.reshape(B, -1), dim=1, descending=True).cpu().numpy()
    random = np.stack([rng.permutation(n_blocks) for _ in range(B)])
    base_p = out["p"].detach()
    base_kl = kl_rows(b["q"], base_p).cpu().numpy()
    curves = {"fractions": list(fractions), "ranked": {"kl": [], "js": []}, "random": {"kl": [], "js": []}, "n_cases": int(B), "baseline_kl": float(base_kl.mean())}
    with torch.no_grad():
        for order_name, order in [("ranked", ranked), ("random", random)]:
            for f in fractions:
                k = int(round(f * n_blocks))
                Lm = b["local_mask"].clone(); Lx = b["local"].clone()
                for i in range(B):
                    blocks = order[i, :k]
                    r_idx, t_idx = blocks // FOCAL_TIME_BINS, blocks % FOCAL_TIME_BINS
                    Lx[i, r_idx, :, t_idx] = 0; Lm[i, r_idx, :, t_idx] = False
                vl = Lm.reshape(B, -1).float().mean(1)
                o = model(Lx, b["context"], vl, b["valid_c"])
                curves[order_name]["kl"].append(float(kl_rows(b["q"], o["p"]).mean()))
                curves[order_name]["js"].append(float(jensen_shannon_distance_rows(base_p.cpu().numpy().astype(np.float64), o["p"].cpu().numpy().astype(np.float64)).mean()))
    return curves
