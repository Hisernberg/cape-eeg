"""Status ledger and run identity. Every stage writes an explicit status; nothing is implied."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .contracts import stable_hash

STATUSES = ("PLANNED", "RUNNING", "PASS", "FAIL", "INCOMPLETE", "NOT_RUN", "BLOCKED")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, obj, indent: int = 2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent, default=_default)
    os.replace(tmp, path)
    return path


def read_json(path: Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def _default(o):
    import numpy as np
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, Path): return str(o)
    if isinstance(o, set): return sorted(o)
    raise TypeError(str(type(o)))


def status_record(status: str, reason: str = "", **extra) -> dict:
    assert status in STATUSES, status
    rec = {"status": status, "reason": reason, "timestamp": utc_now()}
    rec.update(extra)
    return rec


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def run_id(phase: str, model: str, seed: int, split_hash: str, preprocess_hash: str, config: dict) -> str:
    return f"{phase}_{model}_s{seed}_{split_hash[:8]}_{preprocess_hash[:8]}_{stable_hash(config, 8)}"


class Ledger:
    """Append-only JSONL ledger with a summary JSON of the latest status per key."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, key: str, status: str, reason: str = "", **extra):
        rec = status_record(status, reason, key=key, **extra)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=_default) + "\n")
        return rec

    def latest(self) -> dict:
        out = {}
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        out[r["key"]] = r
        return out


def environment_snapshot() -> dict:
    """Hardware/software probe without secrets, home listings or environment dumps."""
    info = {"timestamp": utc_now(), "machine": platform.machine(), "python": platform.python_version(),
            "kernel": platform.release()}
    try:
        import numpy, pandas, scipy
        info.update(numpy=numpy.__version__, pandas=pandas.__version__, scipy=scipy.__version__)
    except Exception:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(0)
            info["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            info["cudnn"] = torch.backends.cudnn.version()
    except Exception as e:
        info["torch_error"] = str(e)[:200]
    try:
        import timm
        info["timm"] = timm.__version__
    except Exception:
        pass
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
                                      text=True, timeout=10).strip()
        info["driver"] = out.split(",")[0].strip()
    except Exception:
        info["driver"] = None
    try:
        with open("/proc/meminfo") as f:
            mem = {l.split(":")[0]: int(l.split()[1]) for l in f if ":" in l}
        info["mem_total_gib"] = round(mem.get("MemTotal", 0) / 2**20, 1)
        info["mem_available_gib"] = round(mem.get("MemAvailable", 0) / 2**20, 1)
    except Exception:
        pass
    try:
        st = os.statvfs(Path.home())
        info["disk_free_gib"] = round(st.f_bavail * st.f_frsize / 2**30, 1)
    except Exception:
        pass
    info["cpu_count"] = os.cpu_count()
    return info
