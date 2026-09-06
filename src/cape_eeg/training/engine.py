"""Bounded training engine: fixed schedule, tune-based selection in development, complete-schedule gate."""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..contracts import stable_hash
from ..status import write_json, utc_now
from ..data.dataset import CacheDataset, to_device
from ..metrics import kl_rows as np_kl_rows, patient_weighted_mean
from .losses import total_loss, kl_rows
from .supervisor import Supervisor, BudgetStop

DEFAULT_TRAINING = dict(physical_batch=64, effective_batch=64, max_epochs=12, optimizer="AdamW", compact_lr=1e-3,
                        pretrained_backbone_lr=1e-4, head_lr=1e-3, weight_decay=0.01, gradient_clip=1.0, warmup_steps=200,
                        aux_weight=0.1, amp="bf16", augment=True, gain_jitter=0.1, freq_mask_max=4, freq_mask_p=0.5,
                        region_dropout_p=0.1, frozen_backbone_epochs=0)


def augment(batch: dict, cfg: dict, gen: torch.Generator) -> dict:
    """Mild gain jitter, small frequency masking, occasional single-region dropout (GPU, in place)."""
    for view, mask_key in [("local", "local_mask"), ("context", "context_mask")]:
        x = batch[view]; m = batch[mask_key]; B = x.shape[0]
        if cfg["gain_jitter"] > 0:
            x = x + (torch.randn(B, 1, 1, 1, device=x.device, generator=gen) * cfg["gain_jitter"])
        if cfg["freq_mask_max"] > 0:
            apply = torch.rand(B, device=x.device, generator=gen) < cfg["freq_mask_p"]
            width = torch.randint(1, cfg["freq_mask_max"] + 1, (B,), device=x.device, generator=gen)
            start = torch.randint(0, x.shape[2] - cfg["freq_mask_max"], (B,), device=x.device, generator=gen)
            f = torch.arange(x.shape[2], device=x.device)[None, :]
            fm = ((f >= start[:, None]) & (f < (start + width)[:, None]) & apply[:, None])  # [B,F]
            x = x.masked_fill(fm[:, None, :, None], 0.0)
        if cfg["region_dropout_p"] > 0:
            drop = torch.rand(B, device=x.device, generator=gen) < cfg["region_dropout_p"]
            region = torch.randint(0, x.shape[1], (B,), device=x.device, generator=gen)
            rm = (torch.arange(x.shape[1], device=x.device)[None, :] == region[:, None]) & drop[:, None]  # [B,4]
            x = x.masked_fill(rm[:, :, None, None], 0.0)
        batch[view] = x * m.float()  # invalid cells stay zero
    return batch


def make_optimizer(model: nn.Module, cfg: dict):
    if hasattr(model, "backbone_parameters"):
        groups = [{"params": model.backbone_parameters(), "lr": cfg["pretrained_backbone_lr"]},
                  {"params": model.head_parameters(), "lr": cfg["head_lr"]}]
    else:
        groups = [{"params": model.parameters(), "lr": cfg["compact_lr"]}]
    return torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])


def lr_lambda(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


@torch.no_grad()
def predict(model: nn.Module, ds: CacheDataset, device, batch_size: int = 128, amp: str = "bf16", force_view=None,
            perturb=None) -> dict:
    model.eval()
    outs = {k: [] for k in ["p", "pL", "pC", "g", "d_hat", "js"]}; pos = []
    for b in ds.batches(batch_size, shuffle=False):
        b = to_device(b, device)
        if perturb is not None:
            b = perturb(b)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(amp == "bf16" and device.type == "cuda")):
            o = model(b["local"], b["context"], b["valid_l"], b["valid_c"], force_view=force_view)
        for k in outs:
            v = o.get(k)
            outs[k].append(v.float().cpu().numpy() if v is not None else None)
        pos.append(b["pos"])
    res = {k: (np.concatenate(v) if v[0] is not None else None) for k, v in outs.items()}
    res["pos"] = np.concatenate(pos)
    order = np.argsort(res["pos"], kind="stable")
    for k, v in res.items():
        if v is not None:
            res[k] = v[order]
    return res


def evaluate_tune(model, ds: CacheDataset, device, amp: str) -> dict:
    r = predict(model, ds, device, amp=amp)
    kl = np_kl_rows(ds.q.astype(np.float64), r["p"].astype(np.float64))
    return {"tune_patient_kl": float(patient_weighted_mean(kl, ds.patient)), "tune_row_kl": float(kl.mean()),
            "tune_gate_mean": float(r["g"].mean()), "n_rows": int(len(kl))}


def fit(model: nn.Module, train_ds: CacheDataset, tune_ds: CacheDataset | None, run_dir: Path, cfg: dict, seed: int,
        device, supervisor: Supervisor, log=print, epochs: int | None = None) -> dict:
    """Train for a fixed schedule. With tune_ds, select the best epoch by tune patient-KL; without, keep the last."""
    cfg = {**DEFAULT_TRAINING, **cfg}
    epochs = epochs or cfg["max_epochs"]
    run_dir.mkdir(parents=True, exist_ok=True); (run_dir / "checkpoints").mkdir(exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    model.to(device)
    opt = make_optimizer(model, cfg)
    accum = max(1, cfg["effective_batch"] // cfg["physical_batch"])
    steps_per_epoch = math.ceil(len(train_ds) / cfg["physical_batch"]) // accum
    total_steps = steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total_steps, cfg["warmup_steps"]))
    use_amp = cfg["amp"] == "bf16" and device.type == "cuda"
    status = {"status": "RUNNING", "started": utc_now(), "epochs_planned": epochs, "seed": seed, "config": cfg}
    write_json(run_dir / "status.json", status)
    events = open(run_dir / "events.jsonl", "a"); usage = open(run_dir / "resource_usage.csv", "a")
    uw = csv.writer(usage); uw.writerow(["epoch", "elapsed_min", "examples_seen", "cuda_max_alloc_gib", "cuda_reserved_gib", "rss_gib", "sys_available_gib", "examples_per_s"])
    best = {"tune_patient_kl": float("inf"), "epoch": None}; examples_seen = 0; step = 0; t_start = time.time()
    frozen = cfg.get("frozen_backbone_epochs", 0)
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            if hasattr(model, "backbone_parameters"):
                for p in model.backbone_parameters():
                    p.requires_grad_(epoch > frozen)
            t_ep = time.time(); losses = []; n_ep = 0
            for i, b in enumerate(train_ds.batches(cfg["physical_batch"], shuffle=True, seed=seed * 1000 + epoch)):
                b = to_device(b, device)
                if cfg["augment"]:
                    b = augment(b, cfg, gen)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    out = model(b["local"], b["context"], b["valid_l"], b["valid_c"])
                loss, parts = total_loss(out, b, cfg["aux_weight"] if getattr(model, "aux_on", False) else 0.0)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite loss at epoch {epoch} step {i}")
                (loss / accum).backward()
                if (i + 1) % accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                    opt.step(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
                losses.append(parts); n_ep += b["q"].shape[0]; examples_seen += b["q"].shape[0]
                if i % 50 == 0:
                    supervisor.check()
            ep_time = time.time() - t_ep
            rec = {"epoch": epoch, "train_loss": float(np.mean([l["loss"] for l in losses])), "train_soft_ce": float(np.mean([l["soft_ce"] for l in losses])),
                   "train_aux_mse": float(np.mean([l["aux_mse"] for l in losses])), "examples_seen": examples_seen, "epoch_seconds": ep_time,
                   "examples_per_s": n_ep / ep_time, "lr": sched.get_last_lr()[0], "elapsed_min": (time.time() - t_start) / 60}
            if tune_ds is not None:
                rec.update(evaluate_tune(model, tune_ds, device, cfg["amp"]))
                if rec["tune_patient_kl"] < best["tune_patient_kl"]:
                    best = {"tune_patient_kl": rec["tune_patient_kl"], "epoch": epoch}
                    _save_weights(model, run_dir / "checkpoints" / "best.safetensors")
            s = supervisor.sample()
            uw.writerow([epoch, round(s["elapsed_min"], 2), examples_seen, round(s.get("cuda_max_alloc_gib", 0), 3), round(s.get("cuda_reserved_gib", 0), 3), round(s["rss_gib"], 2), round(s["sys_available_gib"], 1), round(rec["examples_per_s"], 1)]); usage.flush()
            events.write(json.dumps(rec) + "\n"); events.flush()
            log(f"epoch {epoch}/{epochs} loss {rec['train_loss']:.4f} " + (f"tune pKL {rec['tune_patient_kl']:.4f} rowKL {rec['tune_row_kl']:.4f} g {rec['tune_gate_mean']:.2f} " if tune_ds is not None else "") + f"{rec['examples_per_s']:.0f} ex/s {ep_time:.0f}s")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "seed": seed}, run_dir / "checkpoints" / "last.pt")
        _save_weights(model, run_dir / "checkpoints" / "last.safetensors")
        status.update(status="PASS", finished=utc_now(), epochs_completed=epochs, best=best, examples_seen=examples_seen,
                      wall_seconds=time.time() - t_start, peak=supervisor.peak, complete_schedule=True)
    except BudgetStop as e:
        status.update(status="INCOMPLETE", reason=str(e), finished=utc_now(), wall_seconds=time.time() - t_start, peak=supervisor.peak, complete_schedule=False)
    except Exception as e:
        status.update(status="FAIL", reason=f"{type(e).__name__}: {e}"[:500], finished=utc_now(), wall_seconds=time.time() - t_start, peak=supervisor.peak, complete_schedule=False)
        write_json(run_dir / "status.json", status); events.close(); usage.close(); raise
    events.close(); usage.close()
    write_json(run_dir / "status.json", status)
    return status


def _save_weights(model: nn.Module, path: Path):
    try:
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(path))
    except ImportError:
        torch.save(model.state_dict(), str(path).replace(".safetensors", ".pt"))


def load_weights(model: nn.Module, path: Path):
    path = Path(path)
    if path.suffix == ".safetensors" and path.exists():
        from safetensors.torch import load_file
        model.load_state_dict(load_file(str(path)))
    else:
        alt = Path(str(path).replace(".safetensors", ".pt"))
        model.load_state_dict(torch.load(alt, map_location="cpu")["model"] if "last.pt" in alt.name else torch.load(alt, map_location="cpu"))
    return model
