#!/usr/bin/env python3
"""Inference measurements for the frozen final models (seed 101): GPU latency with concurrent power sampling
(nvidia-smi power.draw, idle baseline subtracted), CPU-only latency at 1 and 4 threads, and parameter/byte counts.
Boundary statement is written into the output. Uses cached test tensors only (no labels)."""
from __future__ import annotations
import json, subprocess, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, torch
from cape_eeg.paths import resolve_workspace
from cape_eeg.status import read_json, write_json, utc_now
from cape_eeg.data.cache import CacheReader
from cape_eeg.data.dataset import CacheDataset, select_rows
from cape_eeg.data.normalization import load_normalizer, Normalizer
from cape_eeg.model import build_model, count_parameters
from cape_eeg.training.engine import load_weights

ws = resolve_workspace(); lock = read_json(ws.manifests / "protocol_lock.json")
reader = CacheReader(ws.cache / lock["preprocess_hash"])

def power_sampler(stop, samples):
    while not stop.is_set():
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"], text=True, timeout=2).strip()
            samples.append((time.time(), float(out)))
        except Exception:
            pass
        time.sleep(0.1)

def sample_power(seconds: float) -> float:
    stop = threading.Event(); samples = []; th = threading.Thread(target=power_sampler, args=(stop, samples), daemon=True); th.start()
    time.sleep(seconds); stop.set(); th.join(timeout=2)
    return float(np.mean([p for _, p in samples])) if samples else float("nan")

def bench(model, bd, n_warm=20, n_timed=100, device="cuda", amp=True):
    times = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(amp and device == "cuda")):
        for i in range(n_warm + n_timed):
            if device == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter(); model(bd["local"], bd["context"], bd["valid_l"], bd["valid_c"])
            if device == "cuda": torch.cuda.synchronize()
            if i >= n_warm: times.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)), "n_timed": len(times)}

res = {"timestamp": utc_now(), "device": torch.cuda.get_device_name(0), "boundary": "GPU package power as reported by nvidia-smi power.draw on the GB10 unified-memory system, sampled at 10 Hz during a 20 s sustained inference loop; idle baseline (20 s, no work) subtracted; energy per window = (mean load power - idle power) / throughput. CPU figures use PyTorch eager float32 on the host ARM cores with the stated thread count; they are a no-accelerator reference, not an embedded-device measurement."}
res["idle_power_w"] = sample_power(20.0)
for cid in [lock["candidate"], lock["comparator"]]:
    rd = sorted(ws.runs.glob(f"final_{cid}_s{lock['deployment_seed']}_*"))[-1]; cfg = read_json(rd / "config.resolved.json")
    r = CacheReader(reader.dir, alt_uniform=(cfg["encoding"] == "uniform")); N = Normalizer(load_normalizer(ws.normalization / cfg["split_hash"] / f"{cfg['preprocess_hash']}_{'+'.join(cfg['train_partitions'])}_{cfg['encoding']}.json"))
    rows = select_rows(reader.index, ["test"], "evaluator")[:256]; ds = CacheDataset(r, rows, N)
    model = build_model(cid); load_weights(model, rd / "checkpoints" / "last.safetensors"); model.eval()
    out = {"parameters": count_parameters(model)["total"], "input_bytes_per_window_float16": 4 * 64 * 96 * 2 + 4 * 64 * 96 // 8}
    # GPU latency + energy
    model.to("cuda")
    for bs in (1, 32):
        b = ds.gather(np.arange(bs)); bd = {k: torch.from_numpy(np.ascontiguousarray(v)).to("cuda") for k, v in b.items() if hasattr(v, "shape")}
        lat = bench(model, bd, device="cuda")
        # sustained loop with power sampling
        stop = threading.Event(); samples = []; th = threading.Thread(target=power_sampler, args=(stop, samples), daemon=True); th.start()
        t0 = time.time(); n = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            while time.time() - t0 < 20.0:
                model(bd["local"], bd["context"], bd["valid_l"], bd["valid_c"]); n += 1
            torch.cuda.synchronize()
        el = time.time() - t0; stop.set(); th.join(timeout=2)
        load_w = float(np.mean([p for _, p in samples])) if samples else float("nan")
        win_per_s = n * bs / el
        out[f"gpu_batch_{bs}"] = {**lat, "windows_per_s": win_per_s, "mean_power_w": load_w, "idle_power_w": res["idle_power_w"], "incremental_power_w": load_w - res["idle_power_w"],
                                  "energy_mj_per_window_incremental": (load_w - res["idle_power_w"]) / win_per_s * 1000, "energy_mj_per_window_total": load_w / win_per_s * 1000, "n_power_samples": len(samples)}
        print(f"{cid} GPU b{bs}: {lat['median_ms']:.2f} ms, {win_per_s:.0f} win/s, power {load_w:.1f} W (idle {res['idle_power_w']:.1f}), {out[f'gpu_batch_{bs}']['energy_mj_per_window_incremental']:.3f} mJ/window incremental")
    # CPU latency
    model.to("cpu"); b = ds.gather(np.arange(1)); bd = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in b.items() if hasattr(v, "shape")}
    for th_n in (1, 4):
        torch.set_num_threads(th_n); lat = bench(model, bd, n_warm=10, n_timed=50, device="cpu", amp=False)
        out[f"cpu_threads_{th_n}_batch_1"] = lat; print(f"{cid} CPU {th_n} thread(s) b1: {lat['median_ms']:.2f} ms")
    res[cid] = out
write_json(ws.repo / "results" / "aggregate" / "inference_measurements.json", res)
print("written results/aggregate/inference_measurements.json")
