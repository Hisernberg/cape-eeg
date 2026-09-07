"""Allowlisted public export, notebook output stripping, identifier/secret scans (docs/07)."""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from ..contracts import stable_hash, file_sha256

ALLOWED_TOP = ["README.md", "EVALUATION_REPORT.md", "AGENTS.md", "CODEX_EXECUTION.md", "CITATION.cff", "LICENSE", "DATA_ACCESS.md", "MODEL_CARD.md", "REPRODUCIBILITY.md",
               "pyproject.toml", ".gitignore", "configs", "src", "scripts", "tests", "notebooks", "docs", "refs", "results/aggregate", "figures", ".github"]
ALLOWED_EXT = {".py", ".sh", ".md", ".yaml", ".yml", ".toml", ".cff", ".txt", ".json", ".csv", ".ipynb", ".png", ".svg", ".gitignore", ".cfg", ".ini", ""}
FORBIDDEN_NAMES = {"train.csv", "test.csv", "sample_submission.csv"}
FORBIDDEN_EXT = {".parquet", ".npy", ".pt", ".safetensors", ".pkl", ".pickle", ".zip", ".h5", ".hdf5", ".log"}
MAX_FILE_BYTES = 10 * 1024 * 1024; MAX_EXPORT_BYTES = 100 * 1024 * 1024
SECRET_PATTERNS = [r"ghp_[A-Za-z0-9]{30,}", r"github_pat_[A-Za-z0-9_]{30,}", r"KGAT_[A-Za-z0-9]{20,}", r"hf_[A-Za-z0-9]{30,}", r"sk-[A-Za-z0-9]{30,}",
                   r"AKIA[0-9A-Z]{16}", r"-----BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY-----", r"xox[baprs]-[A-Za-z0-9-]{10,}"]
# private identifiers that must not appear in public text/notebook outputs: 8-10 digit ids following these column words
IDENTIFIER_PATTERNS = [r"\b(patient_id|eeg_id|spectrogram_id|label_id)\b\s*[:=]\s*\d{5,}"]
TEXT_EXT = {".py", ".sh", ".md", ".yaml", ".yml", ".toml", ".cff", ".txt", ".json", ".csv", ".ipynb", ".svg", ".cfg", ".ini", ""}


def strip_notebook(src: Path, dst: Path, keep_outputs: bool = False):
    nb = json.loads(src.read_text())
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code":
            if not keep_outputs:
                c["outputs"] = []; c["execution_count"] = None
            c.get("metadata", {}).pop("execution", None)
    nb.get("metadata", {}).pop("widgets", None)
    dst.parent.mkdir(parents=True, exist_ok=True); dst.write_text(json.dumps(nb, indent=1))


def iter_allowed(repo: Path):
    for top in ALLOWED_TOP:
        p = repo / top
        if not p.exists():
            continue
        if p.is_file():
            yield p; continue
        for f in sorted(p.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts and ".pytest_cache" not in f.parts and ".ipynb_checkpoints" not in f.parts:
                yield f


def precheck(repo: Path, executed_notebooks: Path | None = None) -> dict:
    """Return a report with problems; fails closed on any forbidden file, symlink, size, secret or identifier."""
    problems, files, total = [], [], 0
    for f in iter_allowed(repo):
        rel = f.relative_to(repo)
        if f.is_symlink():
            problems.append(f"symlink: {rel}"); continue
        if f.name in FORBIDDEN_NAMES or f.suffix in FORBIDDEN_EXT:
            problems.append(f"forbidden file: {rel}"); continue
        if f.suffix not in ALLOWED_EXT:
            problems.append(f"extension not allowlisted: {rel}"); continue
        size = f.stat().st_size; total += size
        if size > MAX_FILE_BYTES:
            problems.append(f"file above 10 MiB: {rel} ({size} bytes)")
        if f.suffix in TEXT_EXT:
            try:
                txt = f.read_text(errors="ignore")
            except Exception:
                txt = ""
            for pat in SECRET_PATTERNS:
                if re.search(pat, txt):
                    problems.append(f"secret pattern {pat[:12]} in {rel}")
            for pat in IDENTIFIER_PATTERNS:
                if re.search(pat, txt):
                    problems.append(f"identifier pattern in {rel}")
            if f.suffix == ".ipynb":
                nb = json.loads(txt or "{}")
                for c in nb.get("cells", []):
                    for o in c.get("outputs", []) if c.get("cell_type") == "code" else []:
                        blob = json.dumps(o)
                        for pat in SECRET_PATTERNS + IDENTIFIER_PATTERNS:
                            if re.search(pat, blob):
                                problems.append(f"notebook output contains secret/identifier pattern: {rel}")
        files.append({"path": str(rel), "bytes": size, "sha256": file_sha256(f)})
    if total > MAX_EXPORT_BYTES:
        problems.append(f"export exceeds 100 MiB: {total}")
    for name in ["private", "scratch", "train_eegs", "train_spectrograms"]:
        if (repo / name).exists():
            problems.append(f"data/private directory inside repo: {name}")
    return {"status": "PASS" if not problems else "FAIL", "problems": problems, "n_files": len(files), "total_bytes": total, "export_sha256": stable_hash(files, 64), "files": files}


def git_history_scan(repo: Path) -> dict:
    """Scan every reachable blob in git history for secret patterns (fails closed if git is unavailable)."""
    import subprocess
    try:
        objs = subprocess.check_output(["git", "-C", str(repo), "rev-list", "--objects", "--all"], text=True).splitlines()
    except Exception as e:
        return {"status": "FAIL", "reason": f"git unavailable: {e}"}
    hits = []
    for line in objs:
        sha = line.split()[0]
        try:
            t = subprocess.check_output(["git", "-C", str(repo), "cat-file", "-t", sha], text=True).strip()
            if t != "blob":
                continue
            blob = subprocess.check_output(["git", "-C", str(repo), "cat-file", "-p", sha], text=False)[:5_000_000].decode(errors="ignore")
        except Exception:
            continue
        for pat in SECRET_PATTERNS:
            if re.search(pat, blob):
                hits.append({"object": sha, "pattern": pat[:12], "name": line.split(" ", 1)[1] if " " in line else ""})
    return {"status": "PASS" if not hits else "FAIL", "n_objects": len(objs), "hits": hits}
