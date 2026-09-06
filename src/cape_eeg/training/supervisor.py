"""Resource supervisor and GPU budget ledger (docs/03 sections 4, 6, 7)."""
from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path

import torch

from ..status import utc_now


class BudgetStop(Exception):
    pass


def _rss_tree_gib() -> float:
    """RSS of this process and its children (from /proc), in GiB."""
    total = 0
    pid = os.getpid()
    try:
        with open(f"/proc/{pid}/status") as f:
            for l in f:
                if l.startswith("VmRSS"):
                    total += int(l.split()[1]); break
        for child in Path(f"/proc/{pid}/task").glob("*/children"):
            for c in child.read_text().split():
                try:
                    with open(f"/proc/{c}/status") as f:
                        for l in f:
                            if l.startswith("VmRSS"):
                                total += int(l.split()[1]); break
                except Exception:
                    pass
    except Exception:
        pass
    return total / 2**20


def _sys_available_gib() -> float:
    with open("/proc/meminfo") as f:
        for l in f:
            if l.startswith("MemAvailable"):
                return int(l.split()[1]) / 2**20
    return float("nan")


class Supervisor:
    def __init__(self, cuda_stop_gib=12.0, rss_stop_gib=32.0, sys_min_available_gib=32.0, max_minutes=None):
        self.cuda_stop, self.rss_stop, self.sys_min, self.max_minutes = cuda_stop_gib, rss_stop_gib, sys_min_available_gib, max_minutes
        self.t0 = time.time(); self.peak = {"cuda_alloc_gib": 0.0, "cuda_reserved_gib": 0.0, "rss_gib": 0.0, "sys_available_min_gib": float("inf")}

    def sample(self) -> dict:
        s = {"elapsed_min": (time.time() - self.t0) / 60, "rss_gib": _rss_tree_gib(), "sys_available_gib": _sys_available_gib()}
        if torch.cuda.is_available():
            s["cuda_alloc_gib"] = torch.cuda.memory_allocated() / 2**30
            s["cuda_reserved_gib"] = torch.cuda.memory_reserved() / 2**30
            s["cuda_max_alloc_gib"] = torch.cuda.max_memory_allocated() / 2**30
        self.peak["cuda_alloc_gib"] = max(self.peak["cuda_alloc_gib"], s.get("cuda_max_alloc_gib", 0))
        self.peak["cuda_reserved_gib"] = max(self.peak["cuda_reserved_gib"], s.get("cuda_reserved_gib", 0))
        self.peak["rss_gib"] = max(self.peak["rss_gib"], s["rss_gib"])
        self.peak["sys_available_min_gib"] = min(self.peak["sys_available_min_gib"], s["sys_available_gib"])
        return s

    def check(self) -> dict:
        s = self.sample()
        if s.get("cuda_max_alloc_gib", 0) > self.cuda_stop:
            raise BudgetStop(f"CUDA allocation {s['cuda_max_alloc_gib']:.1f} GiB exceeds stop {self.cuda_stop}")
        if s["rss_gib"] > self.rss_stop:
            raise BudgetStop(f"process-tree RSS {s['rss_gib']:.1f} GiB exceeds stop {self.rss_stop}")
        if s["sys_available_gib"] < self.sys_min:
            raise BudgetStop(f"system available memory {s['sys_available_gib']:.1f} GiB below {self.sys_min}")
        if self.max_minutes is not None and s["elapsed_min"] > self.max_minutes:
            raise BudgetStop(f"elapsed {s['elapsed_min']:.1f} min exceeds allocation {self.max_minutes}")
        return s


class GpuLock:
    """Single GPU job at a time via flock; a second job blocks with an explicit error."""

    def __init__(self, path: Path):
        self.path = Path(path); self.fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "w")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError("another GPU job holds the lock; concurrent GPU jobs are prohibited") from e
        self.fh.write(f"{os.getpid()} {utc_now()}\n"); self.fh.flush(); return self

    def __exit__(self, *a):
        fcntl.flock(self.fh, fcntl.LOCK_UN); self.fh.close()


class GpuLedger:
    """Append elapsed GPU wall time per job; total must stay under the study ceiling."""

    def __init__(self, path: Path, ceiling_hours: float = 12.0):
        self.path = Path(path); self.ceiling = ceiling_hours

    def total_hours(self) -> float:
        if not self.path.exists():
            return 0.0
        return sum(json.loads(l)["hours"] for l in self.path.read_text().splitlines() if l.strip())

    def remaining_hours(self) -> float:
        return self.ceiling - self.total_hours()

    def record(self, run_id: str, stage: str, seconds: float, status: str, **extra):
        rec = {"run_id": run_id, "stage": stage, "hours": seconds / 3600, "status": status, "timestamp": utc_now(), **extra}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec
