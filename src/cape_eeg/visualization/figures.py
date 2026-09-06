"""Renderers for the twenty planned figure families (docs/05).

One renderer per figure id. Each renderer receives a :class:`FigureContext`, reads only the
artefacts resolved by the registry, saves 300-dpi PNG plus SVG under the public figure
directory, and returns a manifest record (schema in docs/05). Missing inputs never produce a
plot: :func:`render_figure` returns a ``NOT_RUN`` record without touching the filesystem.

Conventions enforced here:
* class order is always Seizure, LPD, GPD, LRDA, GRDA, Other;
* no identifiers (patient/eeg/label/spectrogram ids) or filesystem paths in public outputs;
* counts below :data:`SMALL_CELL` independent patients are never displayed as a labelled cell;
* fixed, non-truncated axes wherever two methods are compared;
* every plotted array is tracked and verified finite before a figure is accepted.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D

from .. import metrics as M
from ..contracts import (LABELS, CLASS_NAMES, N_CLASSES, PARTITIONS, REGIONS, FOCAL_TARGET_COLUMNS, TARGET_SECONDS,
                         CONTEXT_CENTER_SECONDS, CONTEXT_SECONDS, RAW_WINDOW_SECONDS, EEG_SAMPLE_RATE_HZ, RAW_WINDOW_SAMPLES,
                         foveated_time_edges, uniform_time_edges, context_time_edges, file_sha256, stable_hash)
from ..paths import Workspace
from ..status import read_json, git_commit, utc_now
from ..evaluation.bootstrap import paired_cluster_bootstrap, cluster_bootstrap_statistic
from .registry import FigureSpec, FIGURES_BY_ID, resolve_artefacts, missing_inputs, run_dirs_with_events

__all__ = ["FigureContext", "RENDERERS", "render_figure", "SMALL_CELL", "CLASS_COLORS", "PARTITION_COLORS"]

# --------------------------------------------------------------------------------------
# Style (colour-vision-safe palette validated with the dataviz palette checker)
# --------------------------------------------------------------------------------------
CLASS_COLORS = dict(zip(CLASS_NAMES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
PARTITION_COLORS = dict(zip(PARTITIONS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]))
PARTITION_LABELS = {"train": "Train", "tune": "Tune", "calibration_t": "Calibration-T", "calibration_p": "Calibration-P", "test": "Test"}
PARTITION_STYLES = dict(zip(PARTITIONS, ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]))
METHOD_COLORS = {"candidate": "#2a78d6", "comparator": "#eb6834", "alt1": "#1baf7a", "alt2": "#4a3aa7", "neutral": "#6b6b66"}
INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#d9d8d2"; SUPPRESSED = "#bdbcb6"
SMALL_CELL = 10                      # minimum independent patients for a displayed labelled cell
BOOT_REPLICATES = 2000; BOOT_SEED = 20260907
V_COLS = [f"v{k}" for k in range(6)]; P_COLS = [f"p{k}" for k in range(6)]; PCAL_COLS = [f"pcal{k}" for k in range(6)]
VOTE_STRATA = [("1", 1, 1), ("2-4", 2, 4), ("5-9", 5, 9), ("10+", 10, 10**9)]
ID_PATTERN = re.compile(r"\b(patient_id|eeg_id|label_id|spectrogram_id)\b")

RC = {"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
      "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
      "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
      "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
      "figure.dpi": 100, "savefig.dpi": 300, "svg.fonttype": "none", "figure.facecolor": "white", "axes.facecolor": "white",
      "lines.linewidth": 1.8, "axes.titleweight": "semibold"}


def fmt(n) -> str:
    return f"{int(round(float(n))):,}"


def suppress(n_patients: int) -> bool:
    return int(n_patients) < SMALL_CELL


class FigureVerificationError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------------------
@dataclass
class FigureContext:
    ws: Workspace
    spec: FigureSpec
    paths: dict
    out_dir: Path
    private_dir: Path
    tracked: dict = field(default_factory=dict)
    output_paths: list = field(default_factory=list)
    private_output_paths: list = field(default_factory=list)
    data_hashes: dict = field(default_factory=dict)
    run_ids: list = field(default_factory=list)

    # ---- inputs
    def path(self, key: str) -> Path:
        p = self.paths.get(key)
        if p is None:
            raise FileNotFoundError(f"artefact '{key}' is not available")
        return p

    def has(self, key: str) -> bool:
        return self.paths.get(key) is not None

    def hash_inputs(self, *keys: str):
        for k in keys:
            p = self.paths.get(k)
            if p is not None and p.is_file():
                self.data_hashes[k] = file_sha256(p)[:16]

    # ---- verification
    def track(self, name: str, values, allow_nan: bool = False):
        arr = np.asarray(values, dtype=np.float64).ravel()
        if allow_nan:
            arr = arr[np.isfinite(arr) | np.isnan(arr)]
            if np.any(np.isinf(arr)):
                raise FigureVerificationError(f"{self.spec.figure_id}: infinite values in '{name}'")
            arr = arr[np.isfinite(arr)]
        elif arr.size and not np.all(np.isfinite(arr)):
            raise FigureVerificationError(f"{self.spec.figure_id}: non-finite values in '{name}'")
        self.tracked[name] = arr
        return values

    # ---- outputs
    def save(self, fig, suffix: str | None = None) -> list[str]:
        stem = self.spec.stem + (f"_{suffix}" if suffix else "")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        out = []
        for ext in ("png", "svg"):
            p = self.out_dir / f"{stem}.{ext}"
            fig.savefig(p, bbox_inches="tight", facecolor="white")
            out.append(str(p.relative_to(self.ws.repo)) if p.is_relative_to(self.ws.repo) else p.name)
        plt.close(fig)
        self.output_paths.extend(out)
        return out

    def save_private(self, name: str, df: pd.DataFrame) -> str:
        self.private_dir.mkdir(parents=True, exist_ok=True)
        p = self.private_dir / name
        df.to_csv(p, index=False)
        rel = "$CAPE_ROOT/" + str(p.relative_to(self.ws.root)) if p.is_relative_to(self.ws.root) else name
        self.private_output_paths.append(rel)
        return rel

    def record(self, *, split, n_rows, n_patients, n_components, parameters, caption, privacy_review, status="PASS", reason="") -> dict:
        return {"figure_id": self.spec.figure_id, "title": self.spec.title, "question": self.spec.question,
                "data_hashes": dict(self.data_hashes), "run_ids": list(self.run_ids), "split": split,
                "n_rows": None if n_rows is None else int(n_rows), "n_patients": None if n_patients is None else int(n_patients),
                "n_components": None if n_components is None else int(n_components),
                "generation_code_commit": git_commit(self.ws.repo), "parameters": parameters, "output_paths": list(self.output_paths),
                "private_output_paths": list(self.private_output_paths), "status": status, "reason": reason,
                "privacy_review": privacy_review, "caption": caption, "generated": utc_now()}


def caption_text(title: str, *, partition: str, counts: str, seeds: str, metric: str, interval: str, hashes: dict, extra: str = "") -> str:
    h = "; ".join(f"{k} {v}" for k, v in hashes.items() if v)
    parts = [f"{title}.", f"Partition: {partition}.", f"Counts: {counts}.", f"Model/seed policy: {seeds}.", f"Metric: {metric}.",
             f"Uncertainty: {interval}.", f"Source hashes: {h}." if h else ""]
    if extra:
        parts.append(extra if extra.endswith(".") else extra + ".")
    return " ".join(p for p in parts if p)


def new_fig(w=7.0, h=4.0, **kw):
    plt.rcParams.update(RC)
    return plt.subplots(figsize=(w, h), **kw)


def hashes_from(ctx: FigureContext) -> dict:
    """Study-level hashes for captions: split, preprocess, protocol, source manifest."""
    out = {}
    if ctx.has("source_manifest"):
        out["source"] = (read_json(ctx.path("source_manifest")) or {}).get("source_manifest_hash")
    if ctx.has("split_summary"):
        out["split"] = (read_json(ctx.path("split_summary")) or {}).get("split_hash")
    if ctx.has("cache_manifest"):
        cm = read_json(ctx.path("cache_manifest")) or {}
        out["preprocess"] = cm.get("preprocess_hash"); out["cache"] = cm.get("cache_hash")
    if ctx.has("protocol_lock"):
        out["protocol"] = (read_json(ctx.path("protocol_lock")) or {}).get("protocol_hash")
    return {k: v for k, v in out.items() if v}


# --------------------------------------------------------------------------------------
# Shared loaders and statistics
# --------------------------------------------------------------------------------------
def load_votes(ctx: FigureContext) -> pd.DataFrame:
    """label_id, patient_id, component, partition, v0..v5 for every eligible row (cache row index or train.csv + split)."""
    p = ctx.path("raw_votes")
    if p.suffix == ".parquet":
        df = pd.read_parquet(p, columns=["label_id", "patient_id", "component", "partition"] + LABELS)
        ctx.hash_inputs("raw_votes")
    else:
        df = pd.read_csv(p, usecols=["label_id", "patient_id"] + LABELS)
        sm = pd.read_parquet(ctx.path("split_manifest"), columns=["label_id", "component", "partition"])
        df = df.merge(sm, on="label_id", how="inner")
        ctx.hash_inputs("split_manifest")
        ctx.data_hashes["raw_votes"] = "train_csv:" + (read_json(ctx.path("schema_report")) or {}).get("train_csv_sha256", "?")[:16] if ctx.has("schema_report") else "train_csv"
    df = df.rename(columns=dict(zip(LABELS, V_COLS)))
    for c in V_COLS:
        df[c] = df[c].astype(np.int64)
    return df.sort_values("label_id").reset_index(drop=True)


def load_lock(ctx: FigureContext) -> dict:
    ctx.hash_inputs("protocol_lock")
    return read_json(ctx.path("protocol_lock"))


def load_prediction(ctx: FigureContext, key: str, method: str, seed: int) -> pd.DataFrame:
    p = ctx.path(key) / f"{method}_s{seed}.parquet"
    df = pd.read_parquet(p).sort_values("label_id").reset_index(drop=True)
    ctx.data_hashes[f"predictions_{method}_s{seed}"] = file_sha256(p)[:16]
    return df


def align_pair(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(a) != len(b) or not np.array_equal(a.label_id.to_numpy(), b.label_id.to_numpy()):
        raise FigureVerificationError("prediction tables do not align one-to-one on label keys")
    return a, b


def targets(df: pd.DataFrame) -> np.ndarray:
    return M.votes_to_targets(df[V_COLS].to_numpy(dtype=np.int64))


def probs(df: pd.DataFrame, calibrated: bool) -> np.ndarray:
    return df[PCAL_COLS if calibrated else P_COLS].to_numpy(dtype=np.float64)


def entropy_rows(q: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(q > 0, q * np.log(np.where(q > 0, q, 1.0)), 0.0)
    return -t.sum(1)


def group_mean_ci(values: np.ndarray, groups: np.ndarray, n_rep: int = BOOT_REPLICATES, seed: int = BOOT_SEED) -> dict:
    """Patient-weighted mean with a percentile cluster bootstrap over independent groups."""
    _, means = M.patient_mean(values, groups)
    G = means.size
    if G < 2:
        return {"mean": float(means.mean()), "ci_low": None, "ci_high": None, "n_groups": int(G)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, size=(n_rep, G))
    reps = means[idx].mean(1)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"mean": float(means.mean()), "ci_low": float(lo), "ci_high": float(hi), "n_groups": int(G)}


def fast_auroc(y: np.ndarray, s: np.ndarray) -> float:
    from scipy.stats import rankdata
    n_pos = int(y.sum()); n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fast_average_precision(y: np.ndarray, s: np.ndarray) -> float:
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    ys = y[order].astype(np.float64)
    tp = np.cumsum(ys); prec = tp / np.arange(1, ys.size + 1)
    return float((prec * ys).sum() / n_pos)


def add_caption_note(fig, text: str):
    fig.text(0.0, -0.02, text, ha="left", va="top", fontsize=7, color=INK2, wrap=True)


def bar_labels(ax, bars, labels, fontsize=7, rotation=0):
    for b, t in zip(bars, labels):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), t, ha="center", va="bottom", fontsize=fontsize, color=INK2, rotation=rotation)


# ======================================================================================
# V01 - Cohort and exclusion flow
# ======================================================================================
def _flow_box(ax, x, y, w, h, title, lines, color="#eef3fb", edge="#2a78d6", fontsize=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=color, ec=edge, lw=1.0))
    ax.text(x + w / 2, y + h - 0.02, title, ha="center", va="top", fontsize=fontsize + 0.5, fontweight="semibold", color=INK)
    ax.text(x + w / 2, y + h - 0.02 - 0.045, "\n".join(lines), ha="center", va="top", fontsize=fontsize, color=INK2, linespacing=1.25)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, color=INK2, lw=0.9, shrinkA=0, shrinkB=0))


def render_v01(ctx: FigureContext) -> dict:
    src = read_json(ctx.path("source_manifest")); schema = read_json(ctx.path("schema_report")); split = read_json(ctx.path("split_summary"))
    ctx.hash_inputs("source_manifest", "schema_report", "split_summary")
    intake = {"rows": int(schema["n_rows"]), "patients": int(schema["n_patients"]), "eegs": int(schema["n_eeg"]), "spectrograms": int(schema["n_spectrograms"])}
    exclusions = [("Truncated 50 s EEG window", int(src.get("rows_with_truncated_eeg_window", 0))),
                  ("Short 600 s spectrogram context", int(src.get("rows_with_short_spectrogram_context", 0))),
                  ("Duplicate label id / zero-vote row (intake gate)", 0)]
    excluded_rows = sum(n for _, n in exclusions)
    eligible = {"rows": intake["rows"] - excluded_rows, "patients": intake["patients"]}
    part = {p: split[p] for p in PARTITIONS}
    rows_sum = sum(part[p]["rows"] for p in PARTITIONS); pat_sum = sum(part[p]["patients"] for p in PARTITIONS); comp_sum = sum(part[p]["components"] for p in PARTITIONS)
    if rows_sum != eligible["rows"] or pat_sum != eligible["patients"]:
        raise FigureVerificationError(f"V01 counts do not reconcile: partitions rows {rows_sum} vs eligible {eligible['rows']}, patients {pat_sum} vs {eligible['patients']}")
    # quarantine (both views invalid) per partition, when the cache exists
    quarantine = None
    if ctx.has("row_quality") and ctx.has("row_index"):
        q = pd.read_parquet(ctx.path("row_quality"), columns=["row", "local_valid_fraction", "context_valid_fraction"])
        idx = pd.read_parquet(ctx.path("row_index"), columns=["row", "patient_id", "partition"])
        m = idx.merge(q, on="row", how="inner")
        both = (m.local_valid_fraction == 0) & (m.context_valid_fraction == 0)
        quarantine = {p: {"rows": int((both & (m.partition == p)).sum()), "patients": int(m.patient_id[both & (m.partition == p)].nunique())} for p in PARTITIONS}
        ctx.hash_inputs("row_quality", "row_index", "cache_manifest")
    final_eval = None
    if ctx.has("eval_summary") and ctx.has("protocol_lock"):
        lock = read_json(ctx.path("protocol_lock")); ev = read_json(ctx.path("eval_summary"))
        try:
            s = ev["methods"][lock["candidate"]]["seeds"][str(lock["deployment_seed"])]["calibrated"]
        except KeyError:
            s = ev["methods"][lock["candidate"]]["seeds"][lock["deployment_seed"]]["calibrated"]
        final_eval = {"rows": int(s["n_rows"]), "patients": int(s["n_patients"]), "components": int(s["n_components"])}
        ctx.hash_inputs("eval_summary", "protocol_lock")
        if quarantine is not None and part["test"]["rows"] - quarantine["test"]["rows"] != final_eval["rows"]:
            raise FigureVerificationError("V01: test rows minus quarantined rows do not equal the evaluated rows")

    def cell(rows, patients, comps=None):
        if suppress(patients):
            return f"rows {fmt(rows)} · patients <{SMALL_CELL} (suppressed)"
        s = f"rows {fmt(rows)} · patients {fmt(patients)}"
        return s + (f" · components {fmt(comps)}" if comps is not None else "")

    fig, ax = new_fig(10.5, 6.2)
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(False)
    _flow_box(ax, 0.22, 0.86, 0.40, 0.12, "Source label rows (train.csv)",
              [cell(intake["rows"], intake["patients"]), f"EEG recordings {fmt(intake['eegs'])} · spectrograms {fmt(intake['spectrograms'])}"])
    _flow_box(ax, 0.655, 0.82, 0.335, 0.16, "Intake exclusions (label rows)",
              [f"{name}: {fmt(n)}" for name, n in exclusions] + [f"Unreferenced source files (not label rows):", f"EEG {fmt(src.get('unreferenced_eeg_files', 0))}, spectrogram {fmt(src.get('unreferenced_spectrogram_files', 0))}"],
              color="#fbf1ec", edge="#eb6834", fontsize=6.6)
    _arrow(ax, 0.62, 0.90, 0.655, 0.90)
    _arrow(ax, 0.42, 0.86, 0.42, 0.79)
    _flow_box(ax, 0.16, 0.66, 0.52, 0.13, "Eligible labels (after intake gate)",
              [cell(eligible["rows"], eligible["patients"], comp_sum), f"excluded rows {fmt(excluded_rows)}", "independent unit = patient/recording connected component"])
    _arrow(ax, 0.42, 0.66, 0.42, 0.60)
    ax.text(0.42, 0.585, "Deterministic component allocation (seed %d, split hash %s)" % (split.get("seed", 0), split.get("split_hash", "?")), ha="center", va="top", fontsize=7.5, color=INK2)
    xs = np.linspace(0.02, 0.80, 5)
    for x, p in zip(xs, PARTITIONS):
        _flow_box(ax, x, 0.37, 0.18, 0.16, PARTITION_LABELS[p],
                  [f"rows {fmt(part[p]['rows'])}", f"patients {fmt(part[p]['patients'])}" if not suppress(part[p]['patients']) else f"patients <{SMALL_CELL} (s)",
                   f"components {fmt(part[p]['components'])}", f"row share {part[p].get('row_share', part[p]['rows'] / max(rows_sum, 1)):.3f}"],
                  color="#f4f4f1", edge=PARTITION_COLORS[p])
        _arrow(ax, 0.42, 0.555, x + 0.09, 0.53)
    # quarantine and final prediction
    if quarantine is not None:
        qlines = [f"{PARTITION_LABELS[p]}: {fmt(quarantine[p]['rows'])} rows" for p in PARTITIONS]
        _flow_box(ax, 0.02, 0.08, 0.44, 0.22, "Quarantined before modelling (both views entirely invalid)", qlines + ["Removed with an explicit reason; never imputed as valid EEG"], color="#fbf1ec", edge="#eb6834", fontsize=7)
    else:
        _flow_box(ax, 0.02, 0.08, 0.44, 0.22, "Quarantine (both views entirely invalid)", ["PENDING: cache conversion not finalized", "counts appear once row_quality.parquet exists"], color="#f4f4f1", edge=INK2)
    for x in xs:
        _arrow(ax, x + 0.09, 0.37, 0.24, 0.30)
    if final_eval is not None:
        _flow_box(ax, 0.54, 0.08, 0.44, 0.22, "Final locked-test prediction (frozen candidate, deployment seed)",
                  [cell(final_eval["rows"], final_eval["patients"], final_eval["components"]), "every eligible test label scored exactly once", "excluded: quarantined rows only"], color="#eef3fb", fontsize=7)
    else:
        _flow_box(ax, 0.54, 0.08, 0.44, 0.22, "Final locked-test prediction", ["NOT_RUN: protocol lock / final evaluation pending", "no test outcome is displayed before the lock"], color="#f4f4f1", edge=INK2)
    _arrow(ax, xs[-1] + 0.09, 0.37, 0.76, 0.30)
    ax.set_title("V01  Cohort and exclusion flow (audited counts; no model outcome)", loc="left")
    ctx.track("counts", [intake["rows"], eligible["rows"], rows_sum, pat_sum, comp_sum])
    ctx.save(fig)
    hashes = hashes_from(ctx)
    cap = caption_text("Cohort and exclusion flow", partition="all partitions (intake to allocation)" + ("; locked test at the final node" if final_eval else ""),
                       counts=f"source rows {fmt(intake['rows'])}, patients {fmt(intake['patients'])}, eligible rows {fmt(eligible['rows'])}, components {fmt(comp_sum)}; excluded rows {fmt(excluded_rows)}",
                       seeds="no model; deterministic split seed %d" % split.get("seed", 0), metric="audited counts (rows, independent patients, connected components)",
                       interval="none (deterministic counts)", hashes=hashes,
                       extra=("Quarantine counts from the finalized cache" if quarantine else "Quarantine and final-prediction nodes pending") + f"; cells with fewer than {SMALL_CELL} patients are suppressed")
    return ctx.record(split="all", n_rows=eligible["rows"], n_patients=eligible["patients"], n_components=comp_sum,
                      parameters={"exclusions": dict(exclusions), "partitions": {p: {k: part[p][k] for k in ("rows", "patients", "components")} for p in PARTITIONS},
                                  "quarantine": quarantine, "final_evaluation": final_eval, "small_cell_threshold": SMALL_CELL},
                      caption=cap, privacy_review="aggregate_only; small-cell suppression applied; no identifiers")


# ======================================================================================
# V02 - Partition independence matrix
# ======================================================================================
def render_v02(ctx: FigureContext) -> dict:
    audit = read_json(ctx.path("leakage_audit")); split = read_json(ctx.path("split_summary"))
    ctx.hash_inputs("leakage_audit", "split_summary")
    keys = [k for k in ["patient_id", "eeg_id", "spectrogram_id", "component"] if k in audit]
    names = {"patient_id": "Patients", "eeg_id": "EEG recordings", "spectrogram_id": "Spectrograms", "component": "Connected components"}
    fig, axes = new_fig(9.2, 8.2, nrows=2, ncols=2)
    off_total = 0
    for ax, k in zip(axes.ravel(), keys):
        Mx = np.asarray(audit[k]["matrix"], dtype=np.int64)
        off = Mx.sum() - np.trace(Mx); off_total += off
        img = np.zeros(Mx.shape + (3,))
        for i in range(5):
            for j in range(5):
                if i == j: img[i, j] = matplotlib.colors.to_rgb("#c9dcf5")
                elif Mx[i, j] > 0: img[i, j] = matplotlib.colors.to_rgb("#e34948")
                else: img[i, j] = (1, 1, 1)
        ax.imshow(img, interpolation="nearest"); ax.grid(False)
        for i in range(5):
            for j in range(5):
                v = Mx[i, j]
                txt = fmt(v) if (i == j or v == 0 or not suppress(v) or k != "patient_id") else f"<{SMALL_CELL}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=("white" if (i != j and v > 0) else INK), fontweight=("bold" if (i != j and v > 0) else "normal"))
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels([PARTITION_LABELS[p] for p in PARTITIONS], rotation=35, ha="right"); ax.set_yticklabels([PARTITION_LABELS[p] for p in PARTITIONS])
        ax.set_title(f"{names[k]}: off-diagonal total = {fmt(off)}", color=("#e34948" if off else INK))
        for s in ax.spines.values(): s.set_visible(True); s.set_color(GRID)
        ctx.track(f"matrix_{k}", Mx)
    for ax in axes.ravel()[len(keys):]: ax.set_axis_off()
    axes[1, 0].set_xlabel("Partition (columns)"); axes[1, 1].set_xlabel("Partition (columns)")
    axes[0, 0].set_ylabel("Partition (rows)"); axes[1, 0].set_ylabel("Partition (rows)")
    fig.suptitle(f"V02  Partition independence: shared members between partitions (status {audit.get('status', '?')}, forbidden overlap {fmt(audit.get('forbidden_overlap', off_total))})", x=0.02, ha="left", fontsize=10, fontweight="semibold")
    fig.legend(handles=[Patch(fc="#c9dcf5", ec=GRID, label="diagonal: members of the partition"), Patch(fc="white", ec=GRID, label="0: no shared member"), Patch(fc="#e34948", ec=GRID, label="any nonzero overlap (blocks training)")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    ctx.save(fig)
    hashes = hashes_from(ctx)
    n_rows = sum(split[p]["rows"] for p in PARTITIONS); n_pat = sum(split[p]["patients"] for p in PARTITIONS); n_comp = sum(split[p]["components"] for p in PARTITIONS)
    cap = caption_text("Partition independence matrix", partition="all five partitions (both axes)",
                       counts=f"rows {fmt(n_rows)}, patients {fmt(n_pat)}, components {fmt(n_comp)}; keys audited: {', '.join(names[k] for k in keys)}",
                       seeds="no model; deterministic audit of the frozen split", metric="count of members shared by the row and column partition (diagonal = partition size); every off-diagonal cell must be zero",
                       interval="none (deterministic audit)", hashes=hashes, extra="Any nonzero forbidden overlap is drawn in saturated red irrespective of magnitude; individual membership tables stay private")
    return ctx.record(split="all", n_rows=n_rows, n_patients=n_pat, n_components=n_comp,
                      parameters={"keys": [names[k] for k in keys], "forbidden_overlap": audit.get("forbidden_overlap"), "audit_status": audit.get("status"), "off_diagonal_totals": {names[k]: int(audit[k].get("off_diagonal_total", 0)) for k in keys}},
                      caption=cap, privacy_review="aggregate_only; counts only, no identifiers")


# ======================================================================================
# V03 - Six-class label distribution
# ======================================================================================
def _class_partition_stats(df: pd.DataFrame) -> dict:
    v = df[V_COLS].to_numpy(dtype=np.int64); n = v.sum(1); q = v / n[:, None]
    labels, ties = M.unique_majority_labels(v)
    out = {"vote_mass_total": float(q.sum()), "n_rows": int(len(df)), "partitions": {}}
    for p in PARTITIONS:
        m = (df.partition == p).to_numpy()
        if not m.any():
            continue
        pat = df.patient_id.to_numpy()[m]
        rec = {"rows": int(m.sum()), "patients": int(np.unique(pat).size), "share": [], "share_patients": [], "majority": [], "majority_patients": [],
               "raw_vote_share": [float(v[m, k].sum() / v[m].sum()) for k in range(N_CLASSES)], "total_votes": int(v[m].sum()),
               "ties": int(ties[m].sum()), "tie_patients": int(np.unique(pat[ties[m]]).size)}
        for k in range(N_CLASSES):
            rec["share"].append(float(q[m, k].sum() / m.sum()))
            rec["share_patients"].append(int(np.unique(pat[v[m, k] > 0]).size))
            rec["majority"].append(int((labels[m] == k).sum()))
            rec["majority_patients"].append(int(np.unique(pat[labels[m] == k]).size))
        out["partitions"][p] = rec
    return out


def _class_panels(ctx, stats, key, key_pat, ylabel, title, suffix, include_ties=False):
    parts = [p for p in PARTITIONS if p in stats["partitions"]]
    fig, axes = new_fig(2.3 * len(parts) + 0.6, 3.9, ncols=len(parts), sharey=True)
    axes = np.atleast_1d(axes)
    names = CLASS_NAMES + (["Tied"] if include_ties else [])
    colors = [CLASS_COLORS[c] for c in CLASS_NAMES] + ([SUPPRESSED] if include_ties else [])
    for ax, p in zip(axes, parts):
        r = stats["partitions"][p]
        vals = list(r[key]) + ([r["ties"]] if include_ties else []); pats = list(r[key_pat]) + ([r["tie_patients"]] if include_ties else [])
        show = [0.0 if suppress(np_) else v for v, np_ in zip(vals, pats)]
        bars = ax.bar(range(len(names)), show, color=colors, width=0.72, edgecolor="white", linewidth=0.8)
        for b, v, np_ in zip(bars, vals, pats):
            if suppress(np_):
                b.set_hatch("////"); b.set_facecolor(SUPPRESSED); b.set_height(max(show) * 0.02 if max(show) > 0 else 0.0)
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), "s", ha="center", va="bottom", fontsize=7, color=INK2)
            else:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{fmt(np_)}\npts", ha="center", va="bottom", fontsize=6.3, color=INK2, linespacing=1.0)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_title(f"{PARTITION_LABELS[p]}\n{fmt(r['rows'])} rows · {fmt(r['patients'])} patients", fontsize=8.5, pad=6)
        ctx.track(f"{suffix}_{p}", show)
    ymax = max(max(list(stats["partitions"][p][key]) + ([stats["partitions"][p]["ties"]] if include_ties else [])) for p in parts)
    axes[0].set_ylim(0, ymax * 1.32)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    ctx.save(fig, suffix)


def render_v03(ctx: FigureContext) -> dict:
    df = load_votes(ctx)
    stats = _class_partition_stats(df)
    if abs(stats["vote_mass_total"] - stats["n_rows"]) > 1e-6:
        raise FigureVerificationError("V03: normalized vote mass does not sum to the row count")
    _class_panels(ctx, stats, "share", "share_patients", "Share of normalized vote mass (sum q_k / rows)",
                  "V03a  Vote-mass share per class and partition (labels above bars: patients contributing votes to the class)", "vote_mass")
    _class_panels(ctx, stats, "majority", "majority_patients", "Unique-majority rows (count)",
                  "V03b  Unique-majority row counts per class and partition (tied maxima shown separately, excluded from hard-label metrics)", "majority_counts", include_ties=True)
    hashes = hashes_from(ctx)
    n_pat = int(df.patient_id.nunique()); n_comp = int(df.component.nunique())
    cap = caption_text("Six-class label distribution", partition="all five partitions, one panel each",
                       counts=f"rows {fmt(len(df))}, patients {fmt(n_pat)}, components {fmt(n_comp)}; per-panel rows and patients in titles; per-bar labels give patients contributing to the class",
                       seeds="no model; expert votes only", metric="(a) share of normalized vote mass sum_i q_ik / rows (each row weighted equally, so it differs from the raw vote-count share sum_i v_ik / sum_i n_i recorded in the sidecar); (b) rows whose unique maximum vote is class k, tied maxima counted separately",
                       interval="none (descriptive counts); patient counts reported alongside every estimate", hashes=hashes,
                       extra=f"Class order fixed as Seizure, LPD, GPD, LRDA, GRDA, Other; vote mass sums to the row count ({stats['vote_mass_total']:.1f}); cells with fewer than {SMALL_CELL} patients are hatched and marked s")
    return ctx.record(split="all", n_rows=len(df), n_patients=n_pat, n_components=n_comp,
                      parameters={"class_order": CLASS_NAMES, "per_partition": stats["partitions"], "small_cell_threshold": SMALL_CELL},
                      caption=cap, privacy_review="aggregate_only; suppressed low-count cells; no identifiers")


# ======================================================================================
# V04 - Vote count and ambiguity distribution
# ======================================================================================
def render_v04(ctx: FigureContext) -> dict:
    df = load_votes(ctx)
    v = df[V_COLS].to_numpy(dtype=np.int64); n = v.sum(1); q = v / n[:, None]
    ent = entropy_rows(q); d = M.pairwise_disagreement(v); pat = df.patient_id.to_numpy()
    fig, axes = new_fig(9.6, 4.2, ncols=2)
    summary = {}
    pos = np.arange(len(VOTE_STRATA))
    for ax, (vals, ylabel, title, key) in zip(axes, [(ent, "Target entropy H(q) (nats, natural log)", "(a) Label ambiguity by vote-count band", "entropy"),
                                                     (d, "Observed pairwise disagreement d", "(b) Pairwise disagreement by vote-count band", "disagreement")]):
        data, positions, labels = [], [], []
        for i, (name, lo, hi) in enumerate(VOTE_STRATA):
            m = (n >= lo) & (n <= hi); ok = m & np.isfinite(vals)
            rec = summary.setdefault(name, {"n_rows": int(m.sum()), "n_patients": int(np.unique(pat[m]).size)})
            if key == "disagreement" and hi < 2:
                rec["disagreement"] = {"status": "NOT_ESTIMABLE", "reason": "single-vote rows have no pairwise disagreement", "n_rows": int(m.sum())}
                ax.add_patch(FancyBboxPatch((i - 0.32, 0.0), 0.64, 1.0, boxstyle="round,pad=0.0", fc="#f4f4f1", ec=GRID, hatch="////"))
                ax.text(i, 0.5, "NOT_ESTIMABLE\n(n = 1)", ha="center", va="center", fontsize=7.5, color=INK2)
                labels.append(f"{name}\n{fmt(m.sum())} rows\n{fmt(rec['n_patients'])} pts")
                continue
            if ok.sum() == 0:
                labels.append(f"{name}\n0 rows"); continue
            data.append(vals[ok]); positions.append(i)
            ci = group_mean_ci(vals[ok], pat[ok])
            rec[key] = {"patient_weighted_mean": ci["mean"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "n_groups": ci["n_groups"],
                        "row_median": float(np.median(vals[ok])), "row_mean": float(vals[ok].mean()), "max": float(vals[ok].max())}
            ax.errorbar([i + 0.22], [ci["mean"]], yerr=[[ci["mean"] - ci["ci_low"]], [ci["ci_high"] - ci["mean"]]], fmt="D", color=INK, ms=4, capsize=3, lw=1.0, zorder=5)
            labels.append(f"{name}\n{fmt(m.sum())} rows\n{fmt(rec['n_patients'])} pts")
            ctx.track(f"{key}_{name}", [ci["mean"], ci["ci_low"], ci["ci_high"]])
        if data:
            vp = ax.violinplot(data, positions=positions, widths=0.6, showmedians=True, showextrema=False)
            for b in vp["bodies"]: b.set_facecolor("#2a78d6"); b.set_alpha(0.45); b.set_edgecolor("#2a78d6")
            vp["cmedians"].set_color(INK); vp["cmedians"].set_linewidth(1.2)
        ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_xlabel("Vote-count band (prespecified strata)"); ax.set_ylabel(ylabel); ax.set_title(title, loc="left")
        ax.set_ylim(0, (math.log(6) * 1.05) if key == "entropy" else 1.05)
    axes[0].axhline(math.log(6), color=INK2, lw=0.8, ls=":"); axes[0].text(len(VOTE_STRATA) - 0.55, math.log(6) + 0.01, "max ln 6", fontsize=7, color=INK2, ha="right", va="bottom")
    fig.legend(handles=[Patch(fc="#2a78d6", alpha=0.45, label="row distribution (violin, median line)"), Line2D([], [], marker="D", color=INK, ls="", ms=4, label="patient-weighted mean, 95% patient-cluster bootstrap")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("V04  Vote count and ambiguity: entropy and pairwise disagreement by vote-count band", x=0.01, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    ctx.save(fig)
    hashes = hashes_from(ctx)
    n_pat = int(df.patient_id.nunique())
    cap = caption_text("Vote count and ambiguity distribution", partition="all partitions pooled (label-only intake statistics)",
                       counts=f"rows {fmt(len(df))}, patients {fmt(n_pat)}; per-band rows and patients under each tick; single-vote rows {fmt((n == 1).sum())}",
                       seeds="no model; expert votes only", metric="target entropy H(q) = -sum q_k ln q_k (nats) and pairwise disagreement d = 1 - sum v_k(v_k-1)/(n(n-1)), undefined for n = 1",
                       interval="patient-weighted mean with 2,000-replicate percentile patient-cluster bootstrap (seed %d)" % BOOT_SEED, hashes=hashes,
                       extra="Bands 1, 2-4, 5-9, 10+ match the evaluation protocol; n = 1 is shown as NOT_ESTIMABLE, never as zero disagreement")
    return ctx.record(split="all", n_rows=len(df), n_patients=n_pat, n_components=int(df.component.nunique()),
                      parameters={"strata": summary, "bootstrap": {"replicates": BOOT_REPLICATES, "seed": BOOT_SEED, "unit": "patient"}},
                      caption=cap, privacy_review="aggregate_only; no label ids")


# ======================================================================================
# V05 - Temporal alignment and foveated allocation (public version is synthetic)
# ======================================================================================
def synthetic_alignment_case(seed: int = BOOT_SEED) -> dict:
    """Prespecified synthetic 50 s raw window and 600 s context: ramped 10 Hz burst inside the labeled centre,
    impulses at 5 s and 45 s (raw) and a burst inside [295, 305) s (context)."""
    from ..data.montage import EXPECTED_SOURCE_COLUMNS
    from ..data.spectral import RawSpectralEncoder, ContextSpectralEncoder
    from ..data.alignment import select_spectrogram_rows
    rng = np.random.default_rng(seed)
    cols = list(EXPECTED_SOURCE_COLUMNS)
    t = np.arange(RAW_WINDOW_SAMPLES) / EEG_SAMPLE_RATE_HZ
    x = rng.normal(0.0, 4.0, (RAW_WINDOW_SAMPLES, len(cols))).astype(np.float32)
    gains = rng.uniform(0.3, 1.7, len(cols)).astype(np.float32)
    inside = (t >= TARGET_SECONDS[0]) & (t < TARGET_SECONDS[1])
    ramp = (t - TARGET_SECONDS[0]) / (TARGET_SECONDS[1] - TARGET_SECONDS[0])
    burst = 60.0 * ramp * np.sin(2 * np.pi * 10.0 * t)
    x[inside] += (burst[inside][:, None] * gains[None, :]).astype(np.float32)
    impulses = (8.6, 41.4)  # interior to a bin on both the foveated and the uniform grid (STFT frames spread +-0.64 s)
    for ts in impulses:
        i = int(round(ts * EEG_SAMPLE_RATE_HZ)); x[i:i + 4] += (400.0 * gains[None, :]).astype(np.float32)
    enc = RawSpectralEncoder(cols)
    r = enc.encode(x, np.ones_like(x, dtype=bool))
    # synthetic supplied spectrogram: 300 rows of 2 s, 4 regions x 100 frequency columns
    freqs = np.round(np.linspace(0.59, 19.92, 100), 2)
    scols = ["time"] + [f"{reg}_{f:.2f}" for reg in REGIONS for f in freqs]
    times = 1.0 + 2.0 * np.arange(300)
    vals = rng.uniform(0.5, 1.5, (300, 4 * 100)).astype(np.float32)
    cen = (times >= CONTEXT_CENTER_SECONDS[0]) & (times < CONTEXT_CENTER_SECONDS[1])
    fband = (freqs >= 8) & (freqs <= 12)
    for ri in range(4):
        blk = vals[:, ri * 100:(ri + 1) * 100]; blk[np.ix_(cen, fband)] += 40.0
    S = np.column_stack([times, vals]).astype(np.float32)
    cenc = ContextSpectralEncoder(scols)
    keep, edges, info = select_spectrogram_rows(times, 0.0)
    c = cenc.encode(S[keep], edges)
    return {"foveated": r["foveated"], "foveated_mask": r["foveated_mask"], "uniform": r["uniform"], "uniform_mask": r["uniform_mask"],
            "context": c["context"], "context_mask": c["context_mask"], "freq_edges": enc.frequency_edges, "context_freq_edges": cenc.frequency_edges,
            "impulses_s": impulses, "burst_band_hz": (9.0, 11.0), "context_rows": info["n_rows"]}


def render_v05(ctx: FigureContext) -> dict:
    case = synthetic_alignment_case()
    fe, ue, ce = foveated_time_edges(), uniform_time_edges(), context_time_edges()
    # independent offset checks
    fov_col = case["foveated"][0].mean(0); uni_col = case["uniform"][0].mean(0); ctx_col = case["context"][0].mean(0)
    fc = (case["freq_edges"][:-1] + case["freq_edges"][1:]) / 2; band = (fc >= case["burst_band_hz"][0]) & (fc <= case["burst_band_hz"][1])
    fov_band = case["foveated"][0][band].mean(0)
    a, b = FOCAL_TARGET_COLUMNS
    if not (np.all(fe[a:b + 1] >= TARGET_SECONDS[0] - 1e-9) and abs(fe[a] - TARGET_SECONDS[0]) < 1e-9 and abs(fe[b] - TARGET_SECONDS[1]) < 1e-9):
        raise FigureVerificationError("V05: foveated edges do not place columns 8:24 on [20, 30) s")
    target_cols = np.arange(a, b); other = np.setdiff1d(np.arange(fov_col.size), target_cols)
    if fov_band[target_cols].mean() <= fov_band[other].mean() + 1.0 or np.any(np.diff(fov_band[target_cols[:-1]]) < -0.15) or fov_band[target_cols[-1]] <= fov_band[other].mean() + 1.0:
        raise FigureVerificationError("V05: ramped burst not recovered in the foveated target columns")
    exp_fov_imp = [int(np.searchsorted(fe, s, side="right") - 1) for s in case["impulses_s"]]
    exp_uni_imp = [int(np.searchsorted(ue, s, side="right") - 1) for s in case["impulses_s"]]
    got_fov_imp = [int(np.argmax(np.where(np.isin(np.arange(32), other) & ((np.arange(32) < 16) if s < 25 else (np.arange(32) >= 16)), fov_col, -np.inf))) for s in case["impulses_s"]]
    got_uni_imp = [int(np.argmax(np.where((np.arange(32) < 16) if s < 25 else (np.arange(32) >= 16), uni_col, -np.inf))) for s in case["impulses_s"]]
    if got_fov_imp != exp_fov_imp or got_uni_imp != exp_uni_imp:
        raise FigureVerificationError(f"V05: impulse columns {got_fov_imp}/{got_uni_imp} differ from independently computed {exp_fov_imp}/{exp_uni_imp}")
    exp_ctx = np.where((ce[1:] > CONTEXT_CENTER_SECONDS[0]) & (ce[:-1] < CONTEXT_CENTER_SECONDS[1]))[0]
    got_ctx = np.argsort(ctx_col)[-exp_ctx.size:]
    if set(got_ctx.tolist()) != set(exp_ctx.tolist()):
        raise FigureVerificationError(f"V05: context burst bins {sorted(got_ctx.tolist())} differ from expected {exp_ctx.tolist()}")

    fig = plt.figure(figsize=(10.5, 7.6)); plt.rcParams.update(RC)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.05], hspace=0.55, wspace=0.18)
    vmin, vmax = float(np.percentile(case["foveated"][0], 2)), float(np.percentile(case["foveated"][0], 99.5))
    for row, (key, edges, title) in enumerate([("foveated", fe, "Foveated local view: 8 bins over [0,20) s (2.5 s), 16 over [20,30) s (0.625 s), 8 over [30,50) s (2.5 s)"),
                                                ("uniform", ue, "Uniform comparator: 32 bins of 1.5625 s")]):
        ax = fig.add_subplot(gs[row, :])
        Z = np.ma.masked_where(~case[f"{key}_mask"][0], case[key][0])
        pm = ax.pcolormesh(edges, case["freq_edges"], Z, cmap="viridis", vmin=vmin, vmax=vmax, shading="flat")
        ax.vlines(edges, case["freq_edges"][0], case["freq_edges"][-1], color="white", lw=0.5, alpha=0.7)  # true (nonuniform) bin boundaries
        ax.axvspan(*TARGET_SECONDS, color="none", ec="#e34948", lw=1.6, ls="--", zorder=4)
        box = dict(fc="white", ec="none", alpha=0.85, pad=1.5)
        ax.text(25, 37.5, "labeled centre [20, 30) s" + (f" = columns [{a},{b})" if key == "foveated" else ""), ha="center", va="top", fontsize=7.5, color="#e34948", bbox=box, zorder=5)
        for s in case["impulses_s"]:
            ax.axvline(s, color="white", lw=0.8, ls=":"); ax.text(s, 2.0, f"impulse {s:.1f} s", ha="center", va="bottom", fontsize=7, color=INK, bbox=box, zorder=5)
        ax.set_xlim(0, RAW_WINDOW_SECONDS); ax.set_ylim(case["freq_edges"][0], case["freq_edges"][-1]); ax.grid(False)
        ax.set_xlabel("Time within 50 s raw window (s)"); ax.set_ylabel("Frequency (Hz)"); ax.set_title(title, loc="left", fontsize=9)
        ax.set_xticks(edges, minor=True); ax.tick_params(which="minor", length=3, color=INK2)
        fig.colorbar(pm, ax=ax, pad=0.01, fraction=0.03, label="log power")
        ctx.track(f"{key}_values", case[key][0][case[f"{key}_mask"][0]])
    ax = fig.add_subplot(gs[2, :])
    Zc = np.ma.masked_where(~case["context_mask"][0], case["context"][0])
    pm = ax.pcolormesh(ce, case["context_freq_edges"], Zc, cmap="viridis", shading="flat")
    ax.axvspan(*CONTEXT_CENTER_SECONDS, color="none", ec="#e34948", lw=1.6, ls="--", zorder=4)
    ax.text(312, case["context_freq_edges"][-1] * 0.93, "labeled centre [s+295, s+305) s", ha="left", va="top", fontsize=7.5, color="#e34948", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5), zorder=5)
    ax.set_xlim(0, CONTEXT_SECONDS); ax.grid(False); ax.set_xlabel("Time within 600 s supplied spectrogram context (s from interval start s)"); ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Context view: 64 uniform bins of {CONTEXT_SECONDS / 64:.3f} s; centre falls in bins {exp_ctx.tolist()}", loc="left", fontsize=9)
    fig.colorbar(pm, ax=ax, pad=0.01, fraction=0.03, label="log power")
    ctx.track("context_values", case["context"][0][case["context_mask"][0]])
    fig.suptitle("V05  Temporal alignment audit on a SYNTHETIC ramp/impulse window (region LL; not patient data)", x=0.01, ha="left", fontsize=10, fontweight="semibold")
    ctx.save(fig)
    hashes = hashes_from(ctx)
    cap = caption_text("Temporal alignment and foveated allocation", partition="none (synthetic public example; real traces require separate rights review)",
                       counts="1 synthetic 50 s window (20 source channels, 10,000 samples at 200 Hz) and 1 synthetic 600 s context (%d rows of 2 s)" % case["context_rows"],
                       seeds="no model; deterministic synthetic generator seed %d" % BOOT_SEED, metric="log integrated power per (frequency, time) bin drawn with true nonuniform bin widths",
                       interval="none (geometry audit)", hashes=hashes,
                       extra=f"Independent checks passed: foveated columns [{a},{b}) span [20,30) s, ramp recovered monotonically, impulses land in columns {exp_fov_imp} (foveated) / {exp_uni_imp} (uniform), context burst in bins {exp_ctx.tolist()}")
    return ctx.record(split="none", n_rows=0, n_patients=0, n_components=0,
                      parameters={"synthetic": True, "foveated_edges_s": fe.tolist(), "uniform_edges_s": ue.tolist(), "context_bin_seconds": CONTEXT_SECONDS / 64,
                                  "target_columns": list(FOCAL_TARGET_COLUMNS), "impulse_columns_expected": {"foveated": exp_fov_imp, "uniform": exp_uni_imp}, "context_centre_bins": exp_ctx.tolist()},
                      caption=cap, privacy_review="synthetic_only; no patient data rendered")


# ======================================================================================
# V06 - Missingness and signal-quality audit
# ======================================================================================
def render_v06(ctx: FigureContext) -> dict:
    q = pd.read_parquet(ctx.path("row_quality")); idx = pd.read_parquet(ctx.path("row_index"), columns=["row", "patient_id", "partition"])
    ctx.hash_inputs("row_quality", "row_index", "cache_manifest")
    m = idx.merge(q, on="row", how="inner")
    parts = [p for p in PARTITIONS if (m.partition == p).any()]
    fig, axes = new_fig(12.5, 4.1, ncols=3, gridspec_kw={"width_ratios": [1, 1, 1.75]})
    stats = {}
    for p in parts:
        mm = m.partition == p
        stats[p] = {"rows": int(mm.sum()), "patients": int(m.patient_id[mm].nunique()), "mean_local_valid_fraction": float(m.local_valid_fraction[mm].mean()), "mean_context_valid_fraction": float(m.context_valid_fraction[mm].mean()),
                    "local_fully_valid_share": float((m.local_valid_fraction[mm] >= 1.0).mean()), "context_fully_valid_share": float((m.context_valid_fraction[mm] >= 1.0).mean()),
                    "local_all_invalid_rows": int((mm & (m.local_valid_fraction == 0)).sum()), "context_all_invalid_rows": int((mm & (m.context_valid_fraction == 0)).sum()),
                    "both_all_invalid_rows": int((mm & (m.local_valid_fraction == 0) & (m.context_valid_fraction == 0)).sum()), "local_lt_0.95_share": float((m.local_valid_fraction[mm] < 0.95).mean()), "context_lt_0.95_share": float((m.context_valid_fraction[mm] < 0.95).mean())}
    floor = 10.0 ** np.floor(np.log10(1.0 / len(m)))
    for ax, col, key, title in zip(axes[:2], ["local_valid_fraction", "context_valid_fraction"], ["local", "context"], ["(a) Local 50 s view: valid-cell fraction", "(b) Context 600 s view: valid-cell fraction"]):
        for p in parts:
            x = np.sort(m.loc[m.partition == p, col].to_numpy(dtype=np.float64)); y = np.arange(1, x.size + 1) / x.size
            ax.step(x, np.maximum(y, floor), where="post", color=PARTITION_COLORS[p], ls=PARTITION_STYLES[p], label=f"{PARTITION_LABELS[p]}: {100 * stats[p][f'{key}_fully_valid_share']:.1f}% fully valid")
            ctx.track(f"{col}_{p}", x)
        ax.set_yscale("log"); ax.set_ylim(floor, 1.0); ax.set_xlim(0, 1.0)
        ax.set_xlabel("Valid fraction of time-frequency cells"); ax.set_ylabel("Cumulative fraction of rows (ECDF, log scale)"); ax.set_title(title, loc="left"); ax.legend(loc="upper left", fontsize=7)
    ax = axes[2]; ax.set_axis_off(); ax.grid(False)
    cols = ["Partition", "Rows", "Patients", "Mean valid\n(local / context)", "Rows < 0.95 valid\n(local / context)", "All-invalid rows\n(local / ctx / both)"]
    cells = [[PARTITION_LABELS[p], fmt(stats[p]["rows"]), fmt(stats[p]["patients"]), f"{stats[p]['mean_local_valid_fraction']:.4f} / {stats[p]['mean_context_valid_fraction']:.4f}",
              f"{fmt(stats[p]['local_lt_0.95_share'] * stats[p]['rows'])} / {fmt(stats[p]['context_lt_0.95_share'] * stats[p]['rows'])}",
              f"{fmt(stats[p]['local_all_invalid_rows'])} / {fmt(stats[p]['context_all_invalid_rows'])} / {fmt(stats[p]['both_all_invalid_rows'])}"] for p in parts]
    tb = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center", colWidths=[0.15, 0.11, 0.11, 0.21, 0.21, 0.24])
    tb.auto_set_font_size(False); tb.set_fontsize(6.4); tb.scale(1, 2.0)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor(GRID)
        if i == 0: c.set_facecolor("#eef3fb"); c.set_text_props(fontweight="semibold")
    ax.set_title("(c) Quality counts by partition (deterministic)", loc="left")
    ctx.track("invalid_counts", [stats[p][k] for p in parts for k in ("local_all_invalid_rows", "context_all_invalid_rows", "both_all_invalid_rows")])
    fig.suptitle("V06  Missingness and signal quality by partition (cache validity masks; intake audit, no outcomes)", x=0.01, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    ctx.save(fig)
    companion = None
    if ctx.has("strata") and ctx.has("protocol_lock"):
        st = read_json(ctx.path("strata")); lock = read_json(ctx.path("protocol_lock")); ctx.hash_inputs("strata", "protocol_lock")
        bands = [k for k in ["valid_fraction_lt_0.95", "valid_fraction_ge_0.95"] if k in st and st[k].get("status") == "PASS"]
        if bands:
            fig, ax = new_fig(6.2, 3.6)
            xs = np.arange(len(bands)); w = 0.36
            for j, (who, key, col) in enumerate([("candidate", "kl_candidate", METHOD_COLORS["candidate"]), ("comparator", "kl_comparator", METHOD_COLORS["comparator"])]):
                vals = [st[b][key] for b in bands]
                bars = ax.bar(xs + (j - 0.5) * w, vals, width=w, color=col, label=f"{lock[who]} ({who})", edgecolor="white")
                bar_labels(ax, bars, [f"{v:.3f}" for v in vals])
                ctx.track(f"kl_{who}", vals)
            ax.set_xticks(xs); ax.set_xticklabels([f"{b.replace('valid_fraction_', 'min valid fraction ').replace('lt_', '< ').replace('ge_', '>= ')}\n{fmt(st[b]['n_rows'])} rows · {fmt(st[b]['n_groups'])} groups" for b in bands], fontsize=7.5)
            ax.set_ylabel("Component-mean KL (calibrated, mean over seeds)"); ax.set_ylim(0, None); ax.margins(y=0.25); ax.legend(loc="upper right")
            ax.set_title("V06c  Locked-test KL by quality band (post-lock companion; exploratory)", loc="left")
            for i, b in enumerate(bands):
                ax.text(i, 0.02, f"delta {st[b]['delta']:+.3f} [{st[b]['ci_low']:+.3f}, {st[b]['ci_high']:+.3f}]", ha="center", va="bottom", fontsize=7, color=INK, transform=ax.get_xaxis_transform())
            fig.tight_layout(); ctx.save(fig, "quality_vs_kl")
            companion = {b: st[b] for b in bands}
    hashes = hashes_from(ctx)
    n_pat = int(m.patient_id.nunique())
    cap = caption_text("Missingness and signal-quality audit", partition="all partitions (intake audit)" + ("; locked test for the quality-band companion" if companion else ""),
                       counts=f"rows {fmt(len(m))}, patients {fmt(n_pat)}; per-partition rows and patients in the table",
                       seeds="no model for (a)-(c)" + ("; three-seed mean for the companion" if companion else ""), metric="fraction of valid time-frequency cells per view (cache validity mask); rows with an entirely invalid view",
                       interval="none for intake counts; paired 2,000-replicate component bootstrap for the companion" if companion else "none (deterministic intake counts)", hashes=hashes,
                       extra="Quality/outcome associations on the locked test are never used to change exclusion rules; per-example quality data stay private" + ("" if companion else "; post-lock quality-vs-KL companion NOT_RUN (strata.json absent)"))
    return ctx.record(split="all", n_rows=len(m), n_patients=n_pat, n_components=None,
                      parameters={"per_partition": stats, "quality_vs_kl": companion}, caption=cap, privacy_review="aggregate_only; per-example quality private")


# ======================================================================================
# V07 - Byte-budget representation comparison
# ======================================================================================
def _measured_bytes(cache_dir: Path) -> dict:
    sh = cache_dir / "shards"; alt = cache_dir / "alt_uniform"
    def tot(pattern, base): return int(sum(p.stat().st_size for p in base.glob(pattern))) if base.exists() else 0
    out = {"local_values": tot("shard_*_local.npy", sh), "local_masks": tot("shard_*_local_mask.npy", sh), "context_values": tot("shard_*_context.npy", sh), "context_masks": tot("shard_*_context_mask.npy", sh),
           "uniform_local_values": tot("shard_*_local.npy", alt), "uniform_local_masks": tot("shard_*_local_mask.npy", alt)}
    meta = 0
    for n in ("row_index.parquet", "row_quality.parquet", "manifest.json"):
        if (cache_dir / n).exists(): meta += (cache_dir / n).stat().st_size
    out["metadata"] = int(meta)
    out["foveated_total"] = out["local_values"] + out["local_masks"] + out["context_values"] + out["context_masks"] + meta
    out["uniform_total"] = out["uniform_local_values"] + out["uniform_local_masks"] + out["context_values"] + out["context_masks"] + meta
    return out


def render_v07(ctx: FigureContext) -> dict:
    cm = read_json(ctx.path("cache_manifest")); ctx.hash_inputs("cache_manifest", "development_table")
    cache_dir = ctx.path("cache_manifest").parent; by = _measured_bytes(cache_dir)
    dev = pd.read_csv(ctx.path("development_table"))
    ok = dev[(dev.status == "PASS") & dev.tune_patient_kl.notna() & dev.config.isin(["B2", "B3", "A1", "A2", "P", "P_MSF"])].copy()
    if "lr_mult" not in ok.columns: ok["lr_mult"] = 1.0
    ok["lr_mult"] = ok.lr_mult.fillna(1.0)
    points = []
    for (cfg, lr), g in ok.groupby(["config", "lr_mult"]):
        enc = "uniform" if cfg == "B2" else "foveated"
        gb = (by["uniform_total"] if enc == "uniform" else by["foveated_total"]) / 1e9
        points.append({"config": cfg, "lr_mult": float(lr), "encoding": enc, "cache_gb": gb, "kl_mean": float(g.tune_patient_kl.mean()), "kl_min": float(g.tune_patient_kl.min()), "kl_max": float(g.tune_patient_kl.max()),
                       "n_seeds": int(g.seed.nunique()), "params": int(g.params.iloc[0]) if "params" in g else None, "run_ids": [str(r) for r in g.run_id] if "run_id" in g else []})
    if not points:
        raise FigureVerificationError("V07: development table has no PASS run with a tune patient-KL")
    ctx.run_ids = sorted({r for p in points for r in p["run_ids"]})
    fig, axes = new_fig(10.0, 4.0, ncols=2, gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    markers = {"B2": "s", "B3": "P", "A1": "^", "A2": "v", "P": "o", "P_MSF": "X"}
    gbs = [p["cache_gb"] for p in points]; xmax = max(6.0 * 1.12, max(gbs) * 1.15); kmax = max(p["kl_max"] for p in points)
    ax.set_xlim(0, xmax); ax.set_ylim(0, kmax * 1.25)
    ordered = sorted(points, key=lambda p: p["kl_mean"])
    for i, pt in enumerate(ordered):
        col = METHOD_COLORS["comparator"] if pt["encoding"] == "uniform" else METHOD_COLORS["candidate"]
        lab = pt["config"] + ("" if pt["lr_mult"] == 1.0 else f" (lr x{pt['lr_mult']:g})")
        yerr = [[pt["kl_mean"] - pt["kl_min"]], [pt["kl_max"] - pt["kl_mean"]]] if pt["n_seeds"] > 1 else None
        ax.errorbar([pt["cache_gb"]], [pt["kl_mean"]], yerr=yerr, fmt=markers.get(pt["config"], "o"), color=col, ms=7, capsize=3, mfc=("white" if pt["lr_mult"] != 1.0 else col))
        yfrac = 0.12 + 0.72 * (i / max(len(ordered) - 1, 1))
        ax.annotate(f"{lab}: KL {pt['kl_mean']:.4f} ({pt['n_seeds']} seed{'s' if pt['n_seeds'] != 1 else ''}, {pt['cache_gb']:.3f} GB)", (pt["cache_gb"], pt["kl_mean"]), xytext=(0.30, yfrac), textcoords="axes fraction",
                    fontsize=7, color=INK2, va="center", ha="left", arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8, shrinkA=0, shrinkB=3))
        ctx.track(f"point_{lab}", [pt["cache_gb"], pt["kl_mean"]])
    ax.axvline(6.0, color=INK2, lw=0.8, ls=":"); ax.text(6.0, 0.02, "6 GB cap ", rotation=90, ha="right", va="bottom", fontsize=7, color=INK2, transform=ax.get_xaxis_transform())
    ax.set_xlabel("Measured on-disk cache (GB): float16 values + packed validity masks + metadata"); ax.set_ylabel("Tune patient-mean KL (best epoch)")
    ax.set_title("(a) Development KL versus measured cache bytes", loc="left")
    fig.legend(handles=[Line2D([], [], marker="s", color=METHOD_COLORS["comparator"], ls="", label="uniform local encoding"), Line2D([], [], marker="o", color=METHOD_COLORS["candidate"], ls="", label="foveated local encoding"),
                        Line2D([], [], marker="o", color=INK2, mfc="white", ls="", label="non-default learning rate"), Line2D([], [], color=INK2, ls=":", label="6 GB cache cap")], loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    ax = axes[1]
    comps = [("local_values", "local values"), ("local_masks", "local masks"), ("context_values", "context values"), ("context_masks", "context masks"), ("metadata", "metadata")]
    ucomps = [("uniform_local_values", "local values"), ("uniform_local_masks", "local masks"), ("context_values", "context values"), ("context_masks", "context masks"), ("metadata", "metadata")]
    seq = ["#2a78d6", "#7fb0e8", "#eb6834", "#f3a98a", "#6b6b66"]
    for i, (name, cc) in enumerate([("Foveated", comps), ("Uniform", ucomps)]):
        bottom = 0.0
        for (k, lab), c in zip(cc, seq):
            v = by[k] / 1e9; ax.bar(i, v, bottom=bottom, color=c, width=0.6, edgecolor="white", label=(lab if i == 0 else None)); bottom += v
        ax.text(i, bottom, f"{bottom:.2f} GB", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Foveated\n(active cache)", "Uniform\n(alternate local + shared context)"], fontsize=7.5)
    ax.set_ylabel("Measured bytes (GB, on disk, uncompressed .npy)"); ax.set_title("(b) Identical payload composition", loc="left"); ax.set_ylim(0, max(by["foveated_total"], by["uniform_total"]) / 1e9 * 1.6); ax.legend(loc="upper right", fontsize=7, ncol=2)
    ctx.track("bytes", [by[k] for k, _ in comps])
    fig.suptitle("V07  Byte-budget representation comparison (development data; storage measured on disk, not RAM)", x=0.01, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    ctx.save(fig)
    hashes = hashes_from(ctx)
    cap = caption_text("Byte-budget representation comparison", partition="train (fit) / tune (selection); no test data",
                       counts=f"cache rows {fmt(cm.get('rows', 0))}; {len(points)} configuration points; seeds per point annotated",
                       seeds="mean over completed development seeds with min-max bars where more than one seed was run", metric="tune patient-mean KL at the selected epoch versus measured on-disk cache bytes (values + validity masks + metadata)",
                       interval="seed spread only (no bootstrap); higher-resolution pilots that were not run are absent (NOT_RUN), not predicted", hashes=hashes,
                       extra="Foveated and uniform local encodings share shape, dtype and mask so their payloads are identical by construction; compressed disk bytes and RAM bytes are not mixed")
    return ctx.record(split="train/tune", n_rows=int(cm.get("rows", 0)), n_patients=None, n_components=None,
                      parameters={"points": [{k: v for k, v in p.items() if k != "run_ids"} for p in points], "measured_bytes": by, "cap_bytes": 6_000_000_000,
                                  "manifest_active_cache_bytes": cm.get("active_cache_bytes"), "manifest_alt_uniform_bytes": cm.get("alt_uniform_bytes"), "unrun_alternatives": ["higher-resolution pilot: NOT_RUN"]},
                      caption=cap, privacy_review="aggregate_only; engineering data, no cached tensors")


# ======================================================================================
# V08 - Learning curves and completed training exposure
# ======================================================================================
def render_v08(ctx: FigureContext) -> dict:
    runs = []
    for d in run_dirs_with_events(ctx.ws):
        cfg = read_json(d / "config.resolved.json"); st = read_json(d / "status.json") or {"status": "RUNNING"}
        ev = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines() if l.strip()]
        if not ev or cfg.get("phase") == "smoke":
            continue
        runs.append({"run_id": d.name, "config": cfg["config_id"], "seed": int(cfg["seed"]), "phase": cfg.get("phase", "dev"), "status": st.get("status", "?"),
                     "complete": bool(st.get("complete_schedule", False)), "events": ev, "epochs_planned": cfg.get("epochs"), "lr_mult": round(cfg.get("training", {}).get("compact_lr", 1e-3) / 1e-3, 3)})
        ctx.data_hashes[f"events_{d.name}"] = file_sha256(d / "events.jsonl")[:16]
    if not runs:
        raise FigureVerificationError("V08: no non-smoke run with events")
    ctx.run_ids = [r["run_id"] for r in runs]
    cfg_colors = {"B2": METHOD_COLORS["comparator"], "B3": "#4a3aa7", "A1": "#1baf7a", "A2": "#eda100", "P": METHOD_COLORS["candidate"], "P_MSF": "#e87ba4"}
    seed_styles = {}
    fig, axes = new_fig(11.0, 4.0, ncols=2)
    total_min = 0.0; total_examples = 0; incomplete = []; still_running = []
    for r in runs:
        ev = r["events"]; x = np.array([e["examples_seen"] for e in ev], dtype=np.float64) / 1e3
        col = cfg_colors.get(r["config"], INK2); ls = seed_styles.setdefault(r["seed"], ["-", "--", ":", "-."][len(seed_styles) % 4])
        alpha = 1.0 if r["lr_mult"] == 1.0 else 0.45
        y = np.array([e["train_soft_ce"] for e in ev], dtype=np.float64)
        axes[0].plot(x, y, color=col, ls=ls, alpha=alpha, lw=1.5 if r["phase"] == "dev" else 2.2, marker=("o" if r["phase"] == "dev" else "s"), ms=2.5)
        ctx.track(f"train_ce_{r['run_id']}", y)
        if r["phase"] == "dev" and "tune_patient_kl" in ev[-1]:
            yk = np.array([e.get("tune_patient_kl", np.nan) for e in ev], dtype=np.float64)
            axes[1].plot(x, yk, color=col, ls=ls, alpha=alpha, marker="o", ms=2.5)
            ctx.track(f"tune_kl_{r['run_id']}", yk, allow_nan=True)
        if not r["complete"]:
            running = r["status"] == "RUNNING"
            (still_running if running else incomplete).append(r["run_id"])
            for ax in axes: ax.plot([x[-1]], [y[-1] if ax is axes[0] else ev[-1].get("tune_patient_kl", np.nan)], marker=("o" if running else "x"), color=(INK2 if running else "#e34948"), mfc=("white" if running else "#e34948"), ms=8, mew=1.8, ls="")
        total_min += float(ev[-1].get("elapsed_min", 0.0)); total_examples += int(ev[-1]["examples_seen"])
    axes[0].set_xlabel("Training examples seen (thousands)"); axes[0].set_ylabel("Training soft cross-entropy (nats; includes target entropy)"); axes[0].set_title("(a) Training loss", loc="left")
    axes[1].set_xlabel("Training examples seen (thousands)"); axes[1].set_ylabel("Tune patient-mean KL (nats; development runs only)"); axes[1].set_title("(b) Tune selection metric", loc="left")
    for ax in axes: ax.set_ylim(0, None)
    handles = [Line2D([], [], color=c, label=k) for k, c in cfg_colors.items() if any(r["config"] == k for r in runs)]
    handles += [Line2D([], [], color=INK2, ls=ls, label=f"seed {s}") for s, ls in sorted(seed_styles.items())]
    handles += [Line2D([], [], color=INK2, marker="s", ls="", label="final refit (train+tune, no tune metric)"), Line2D([], [], color="#e34948", marker="x", ls="", label="INCOMPLETE (schedule not finished)"),
                Line2D([], [], color=INK2, marker="o", mfc="white", ls="", label="RUNNING (job in progress at render time)"), Line2D([], [], color=INK2, alpha=0.45, label="non-default learning rate")]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"V08  Learning curves for {len(runs)} runs (train soft-CE vs tune patient-KL are different quantities; nothing smoothed)", x=0.01, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    ctx.save(fig)
    ledger = None
    if ctx.has("gpu_ledger"):
        led = [json.loads(l) for l in ctx.path("gpu_ledger").read_text().splitlines() if l.strip()]
        ledger = {"gpu_hours_total": float(sum(l["hours"] for l in led)), "n_jobs": len(led), "non_pass_jobs": int(sum(l.get("status") != "PASS" for l in led))}
        ctx.hash_inputs("gpu_ledger")
    hashes = hashes_from(ctx)
    cap = caption_text("Learning curves and completed training exposure", partition="train (loss) and tune (selection metric); no test data",
                       counts=f"{len(runs)} runs ({sum(r['phase'] == 'dev' for r in runs)} development, {sum(r['phase'] == 'final' for r in runs)} final); examples seen in total {fmt(total_examples)}; summed elapsed training time {total_min:.1f} min" + (f"; GPU ledger {ledger['gpu_hours_total']:.2f} h over {ledger['n_jobs']} jobs" if ledger else ""),
                       seeds="every completed seed drawn separately (line style); no synthetic error band", metric="training soft cross-entropy (mean over epoch batches) and tune patient-mean KL after each epoch",
                       interval="none (per-run curves)", hashes=hashes,
                       extra=(f"INCOMPLETE runs marked with a red cross at their last epoch: {len(incomplete)}" if incomplete else "no INCOMPLETE run") + (f"; runs still in progress at render time (open marker): {len(still_running)}" if still_running else "; every drawn run completed its fixed schedule"))
    return ctx.record(split="train/tune", n_rows=None, n_patients=None, n_components=None,
                      parameters={"runs": [{k: r[k] for k in ("run_id", "config", "seed", "phase", "status", "complete", "epochs_planned", "lr_mult")} | {"epochs_logged": len(r["events"]), "examples_seen": r["events"][-1]["examples_seen"], "elapsed_min": r["events"][-1].get("elapsed_min")} for r in runs],
                                  "gpu_ledger": ledger, "incomplete": incomplete, "running_at_render": still_running},
                      caption=cap, privacy_review="aggregate_only; sanitized curves, run ids are study identifiers not patient identifiers")


# ======================================================================================
# V09 - Primary paired patient-KL improvement
# ======================================================================================
def _paired_component_losses(ctx, lock, calibrated: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    cand, comp, seeds = lock["candidate"], lock["comparator"], lock["final_seeds"]
    A, B, groups_ref, meta = [], [], None, {}
    for s in seeds:
        a = load_prediction(ctx, "test_predictions_all_seeds", cand, s); b = load_prediction(ctx, "test_predictions_all_seeds", comp, s)
        align_pair(a, b)
        qa, qb = targets(a), targets(b)
        if not np.allclose(qa, qb):
            raise FigureVerificationError("V09: vote targets differ between methods for the same keys")
        ga, ma = M.patient_mean(M.kl_rows(qa, probs(a, calibrated)), a.component.to_numpy()); gb, mb = M.patient_mean(M.kl_rows(qb, probs(b, calibrated)), b.component.to_numpy())
        if not np.array_equal(ga, gb) or (groups_ref is not None and not np.array_equal(groups_ref, ga)):
            raise FigureVerificationError("V09: component keys differ between methods/seeds")
        groups_ref = ga; A.append(ma); B.append(mb)
        meta = {"n_rows": int(len(a)), "n_patients": int(a.patient_id.nunique()), "n_components": int(ga.size)}
    return np.stack(A, 1), np.stack(B, 1), groups_ref, meta


def _delta_panels(ctx, delta, bs, lock, calibrated, suffix):
    fig, axes = new_fig(9.6, 3.9, ncols=2, gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    lim = float(np.max(np.abs(delta))) * 1.05
    bins = np.linspace(-lim, lim, 41)
    ax.hist(delta, bins=bins, color="#2a78d6", alpha=0.75, edgecolor="white")
    ax.axvline(0, color=INK, lw=1.2, ls="--"); ax.axvspan(bs["ci_low"], bs["ci_high"], color="#eb6834", alpha=0.25, lw=0)
    ax.axvline(bs["point_estimate"], color="#eb6834", lw=1.8)
    ax.set_xlabel(f"Paired delta KL per component = {lock['candidate']} - {lock['comparator']} (nats; mean over {bs['n_seeds']} seeds; negative favours {lock['candidate']})"); ax.set_ylabel("Independent components (count)")
    ax.set_title("(a) Distribution of paired component differences", loc="left")
    ax.text(0.02, 0.97, f"delta = {bs['point_estimate']:+.4f}\n95% CI [{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}]\n{fmt(bs['n_groups'])} components, {bs['n_replicates']} resamples\ncomponents with delta < 0: {100 * float((delta < 0).mean()):.1f}%\ndecision: {bs['decision']}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5, bbox=dict(fc="white", ec=GRID))
    ax = axes[1]
    vp = ax.violinplot([delta], positions=[0], widths=0.7, showmedians=False, showextrema=False)
    for b in vp["bodies"]: b.set_facecolor("#2a78d6"); b.set_alpha(0.4)
    ax.boxplot([delta], positions=[0], widths=0.18, showfliers=False, medianprops=dict(color=INK), boxprops=dict(color=INK2), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    ax.errorbar([0.42], [bs["point_estimate"]], yerr=[[bs["point_estimate"] - bs["ci_low"]], [bs["ci_high"] - bs["point_estimate"]]], fmt="o", color="#eb6834", capsize=4, ms=5)
    ax.axhline(0, color=INK, lw=1.2, ls="--"); ax.set_xticks([0, 0.42]); ax.set_xticklabels(["per-component\ndistribution", "mean with\n95% bootstrap CI"], fontsize=7.5); ax.set_xlim(-0.5, 0.8)
    ax.set_ylabel("Paired delta KL (nats)"); ax.set_title("(b) Summary", loc="left"); ax.set_ylim(-lim, lim)
    fig.suptitle(f"V09  Primary paired patient-KL improvement on the locked test ({'temperature-calibrated, primary' if calibrated else 'raw probabilities, secondary'})", x=0.01, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    ctx.track(f"delta_{suffix or 'primary'}", delta); ctx.track(f"ci_{suffix or 'primary'}", [bs["point_estimate"], bs["ci_low"], bs["ci_high"]])
    ctx.save(fig, suffix)


def render_v09(ctx: FigureContext) -> dict:
    lock = load_lock(ctx)
    ref = read_json(ctx.path("bootstrap_primary")) if ctx.has("bootstrap_primary") else None
    if ref: ctx.hash_inputs("bootstrap_primary")
    results = {}
    for calibrated, suffix, refkey in [(True, None, "primary_calibrated"), (False, "raw", "secondary_raw")]:
        A, B, groups, meta = _paired_component_losses(ctx, lock, calibrated)
        bs = paired_cluster_bootstrap(A, B, n_replicates=BOOT_REPLICATES, seed=BOOT_SEED)
        bs["decision"] = "superiority_supported" if bs["ci_high"] < 0 else ("inferior" if bs["ci_low"] > 0 else "inconclusive")
        if ref and refkey in ref:
            for k in ("point_estimate", "ci_low", "ci_high"):
                if abs(float(ref[refkey][k]) - bs[k]) > 1e-8:
                    raise FigureVerificationError(f"V09: recomputed {k} ({bs[k]:.6f}) differs from bootstrap_primary.json ({ref[refkey][k]:.6f})")
        delta = A.mean(1) - B.mean(1)
        _delta_panels(ctx, delta, bs, lock, calibrated, suffix)
        results[refkey] = {k: bs[k] for k in ("point_estimate", "mean_a", "mean_b", "ci_low", "ci_high", "n_groups", "n_seeds", "n_replicates", "seed", "decision")} | meta
        if calibrated:
            ctx.save_private("v09_per_component_delta_calibrated.csv", pd.DataFrame({"component": groups, "delta_kl": delta, "kl_candidate": A.mean(1), "kl_comparator": B.mean(1)}))
    m = results["primary_calibrated"]
    ctx.run_ids = [f"final_{c}_s{s}" for c in (lock["candidate"], lock["comparator"]) for s in lock["final_seeds"]]
    hashes = hashes_from(ctx)
    cap = caption_text("Primary paired patient-KL improvement", partition="locked test",
                       counts=f"rows {fmt(m['n_rows'])}, patients {fmt(m['n_patients'])}, independent components {fmt(m['n_components'])}",
                       seeds=f"seeds {lock['final_seeds']}: per-component mean KL averaged over seeds for each method (losses averaged, never probabilities)",
                       metric=f"delta = {lock['candidate']} - {lock['comparator']} of component-mean KL(q || p) (natural log, eps 1e-7, temperature-calibrated on calibration-T); negative favours {lock['candidate']}",
                       interval=f"percentile 95% interval from {BOOT_REPLICATES} paired component bootstrap resamples (seed {BOOT_SEED}); point estimate {m['point_estimate']:+.4f} [{m['ci_low']:+.4f}, {m['ci_high']:+.4f}], {m['decision']}",
                       hashes=hashes, extra="Raw-probability companion is secondary; no seed selection and no method-specific patient dropping; per-component values stay private")
    return ctx.record(split="test", n_rows=m["n_rows"], n_patients=m["n_patients"], n_components=m["n_components"],
                      parameters={"results": results, "candidate": lock["candidate"], "comparator": lock["comparator"], "cross_checked_against_pipeline": bool(ref)},
                      caption=cap, privacy_review="aggregate_only; distribution and summary, no per-patient values in public outputs")


# ======================================================================================
# V10 / V11 - One-vs-rest ROC and precision-recall curves
# ======================================================================================
def _class_curves(ctx, lock, kind: str):
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    seed = lock["deployment_seed"]; methods = [("candidate", lock["candidate"]), ("comparator", lock["comparator"])]
    out = {}
    for calibrated in (True, False):
        for role, cid in methods:
            df = load_prediction(ctx, "test_predictions_deploy", cid, seed)
            v = df[V_COLS].to_numpy(dtype=np.int64); labels, ties = M.unique_majority_labels(v); keep = ~ties
            P = M.clip_and_renormalize(probs(df, calibrated))[keep]; y = labels[keep]; g = df.component.to_numpy()[keep]; pat = df.patient_id.to_numpy()[keep]
            def stat(rows, P=P, y=y):
                return np.array([(fast_auroc if kind == "roc" else fast_average_precision)(y[rows] == k, P[rows, k]) for k in range(N_CLASSES)])
            bs = cluster_bootstrap_statistic(stat, g, n_replicates=BOOT_REPLICATES, seed=BOOT_SEED)
            classes = []
            for k in range(N_CLASSES):
                pos = y == k; n_pos, n_neg = int(pos.sum()), int((~pos).sum())
                rec = {"class": CLASS_NAMES[k], "n_pos_rows": n_pos, "n_neg_rows": n_neg, "n_pos_patients": int(np.unique(pat[pos]).size), "n_neg_patients": int(np.unique(pat[~pos]).size), "prevalence": n_pos / y.size}
                if n_pos == 0 or n_neg == 0:
                    rec.update(status="NOT_ESTIMABLE", reason="class absent or universal among unique-majority rows"); classes.append(rec); continue
                if kind == "roc":
                    fpr, tpr, _ = roc_curve(pos.astype(int), P[:, k]); rec.update(x=fpr, y=tpr, point=float(roc_auc_score(pos.astype(int), P[:, k])))
                else:
                    pr, rc, _ = precision_recall_curve(pos.astype(int), P[:, k]); rec.update(x=rc, y=pr, point=float(average_precision_score(pos.astype(int), P[:, k])))
                rec.update(ci_low=bs["ci_low"][k] if bs["ci_low"] else None, ci_high=bs["ci_high"][k] if bs["ci_high"] else None, status="PASS",
                           low_support=(rec["n_pos_patients"] < 30 or rec["n_neg_patients"] < 30))
                classes.append(rec)
            out[(calibrated, role)] = {"cid": cid, "classes": classes, "n_rows": int(keep.sum()), "n_ties": int(ties.sum()), "n_patients": int(np.unique(pat).size), "n_groups": int(np.unique(g).size),
                                      "macro": float(np.nanmean([c.get("point", np.nan) for c in classes]))}
    return out


def _curve_figure(ctx, res, lock, kind, calibrated, suffix):
    fig, axes = new_fig(10.5, 6.6, nrows=2, ncols=3)
    metric = "AUROC" if kind == "roc" else "AP"
    for k, ax in enumerate(axes.ravel()):
        for role, col, ls in [("candidate", METHOD_COLORS["candidate"], "-"), ("comparator", METHOD_COLORS["comparator"], "--")]:
            r = res[(calibrated, role)]; c = r["classes"][k]
            if c["status"] != "PASS":
                ax.text(0.5, 0.5, "NOT_ESTIMABLE", ha="center", va="center", transform=ax.transAxes, color=INK2); continue
            ci = f" [{c['ci_low']:.3f}, {c['ci_high']:.3f}]" if c["ci_low"] is not None else ""
            ax.plot(c["x"], c["y"], color=col, ls=ls, lw=1.6, label=f"{r['cid']}: {metric} {c['point']:.3f}{ci}")
            ctx.track(f"{kind}_{role}_{CLASS_NAMES[k]}_{suffix or 'cal'}", np.concatenate([c["x"], c["y"]]))
        c = res[(calibrated, "candidate")]["classes"][k]
        if kind == "roc":
            ax.plot([0, 1], [0, 1], color=GRID, lw=1.0, ls=":")
            ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
        else:
            ax.axhline(c["prevalence"], color=INK2, lw=0.9, ls=":"); ax.text(0.99, c["prevalence"] + 0.01, f"prevalence {c['prevalence']:.3f}", ha="right", va="bottom", fontsize=6.5, color=INK2)
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        sup = " (low support)" if c.get("low_support") else ""
        ax.set_title(f"{CLASS_NAMES[k]}{sup}\npos {fmt(c['n_pos_rows'])} rows/{fmt(c['n_pos_patients'])} pts · neg {fmt(c['n_neg_rows'])} rows/{fmt(c['n_neg_patients'])} pts", fontsize=8, color=CLASS_COLORS[CLASS_NAMES[k]])
        ax.legend(loc="lower right" if kind == "roc" else "lower left", fontsize=6.5)
    rc, rb = res[(calibrated, "candidate")], res[(calibrated, "comparator")]
    fig.suptitle(f"V{'10' if kind == 'roc' else '11'}  One-vs-rest {'ROC' if kind == 'roc' else 'precision-recall'} curves on unique-majority test rows "
                 f"({'temperature-calibrated' if calibrated else 'raw'} probabilities; macro {metric}: {rc['cid']} {rc['macro']:.3f}, {rb['cid']} {rb['macro']:.3f})", x=0.01, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    ctx.save(fig, suffix)


def _render_curves(ctx, kind):
    lock = load_lock(ctx)
    res = _class_curves(ctx, lock, kind)
    _curve_figure(ctx, res, lock, kind, True, None); _curve_figure(ctx, res, lock, kind, False, "raw")
    r = res[(True, "candidate")]
    ctx.run_ids = [f"final_{c}_s{lock['deployment_seed']}" for c in (lock["candidate"], lock["comparator"])]
    hashes = hashes_from(ctx)
    per_class = {f"{'calibrated' if cal else 'raw'}_{role}": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items() if k not in ("x", "y")} for c in rr["classes"]] for (cal, role), rr in res.items()}
    metric = ("one-vs-rest AUROC (rank-based; fast rank estimator inside the bootstrap)" if kind == "roc" else "one-vs-rest average precision (step-wise AP, not trapezoidal PR area; fast step estimator inside the bootstrap)")
    cap = caption_text("One-vs-rest ROC curves" if kind == "roc" else "One-vs-rest precision-recall curves", partition="locked test, unique-majority rows only",
                       counts=f"rows {fmt(r['n_rows'])} (tied maxima excluded: {fmt(r['n_ties'])}), patients {fmt(r['n_patients'])}, components {fmt(r['n_groups'])}; per-class positives/negatives in panel titles",
                       seeds=f"deployment seed {lock['deployment_seed']} for both methods ({lock['candidate']} solid, {lock['comparator']} dashed)", metric=metric + "; predicted score = class probability",
                       interval=f"percentile 95% component-cluster bootstrap with {BOOT_REPLICATES} resamples (seed {BOOT_SEED}); classes with fewer than 30 positive or negative patients are marked low support", hashes=hashes,
                       extra="Calibrated curves are the main output and raw-probability curves the companion; absent classes would be reported NOT_ESTIMABLE; the actual test class mix is used, no resampling of prevalence")
    return ctx.record(split="test", n_rows=r["n_rows"], n_patients=r["n_patients"], n_components=r["n_groups"],
                      parameters={"per_class": per_class, "macro": {f"{'calibrated' if cal else 'raw'}_{role}": rr["macro"] for (cal, role), rr in res.items()}, "bootstrap": {"replicates": BOOT_REPLICATES, "seed": BOOT_SEED, "unit": "component"}},
                      caption=cap, privacy_review="aggregate_only; curves without point-level identifiers")


def render_v10(ctx: FigureContext) -> dict:
    return _render_curves(ctx, "roc")


def render_v11(ctx: FigureContext) -> dict:
    return _render_curves(ctx, "pr")


# ======================================================================================
# V12 - Confusion matrix with support
# ======================================================================================
def _confusion_figure(ctx, cm, cm_norm, cell_patients, recall, cid, normalized, suffix, support_rows, support_pat, n_eval):
    fig, ax = new_fig(7.4, 6.2)
    Z = cm_norm if normalized else cm.astype(float)
    ax.imshow(Z, cmap="Blues", vmin=0, vmax=(1.0 if normalized else max(Z.max(), 1)), interpolation="nearest"); ax.grid(False)
    n_supp = 0
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if cm[i, j] > 0 and suppress(cell_patients[i, j]):
                n_supp += 1; ax.add_patch(FancyBboxPatch((j - 0.5, i - 0.5), 1, 1, boxstyle="square,pad=0", fc="white", ec=GRID, hatch="////"))
                txt = "s"
            else:
                txt = f"{cm_norm[i, j]:.2f}\n({fmt(cm[i, j])})" if normalized else f"{fmt(cm[i, j])}\n{fmt(cell_patients[i, j])} pts"
            dark = Z[i, j] > 0.55 * (1.0 if normalized else max(Z.max(), 1))
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=("white" if dark else INK))
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES)); ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted class (argmax p)"); ax.set_ylabel("True class (unique majority vote)")
    for i in range(N_CLASSES):
        ax.text(N_CLASSES - 0.4, i, f"support {fmt(support_rows[i])} rows / {fmt(support_pat[i])} pts\nrecall {recall['point'][i]:.3f} [{recall['ci_low'][i]:.3f}, {recall['ci_high'][i]:.3f}]", ha="left", va="center", fontsize=6.8, color=INK2)
    ax.set_xlim(-0.5, N_CLASSES - 0.5)
    for s in ax.spines.values(): s.set_visible(True); s.set_color(GRID)
    ax.set_title(f"V12  Confusion matrix, {cid} ({'row-normalized' if normalized else 'raw counts'}; {fmt(n_eval)} unique-majority test rows; suppressed cells: {n_supp})", loc="left", fontsize=9.5)
    fig.tight_layout()
    ctx.track(f"cm_{suffix}", Z)
    ctx.save(fig, suffix)
    return n_supp


def render_v12(ctx: FigureContext) -> dict:
    lock = load_lock(ctx); seed = lock["deployment_seed"]
    results = {}
    for role, cid in [("candidate", lock["candidate"]), ("comparator", lock["comparator"])]:
        df = load_prediction(ctx, "test_predictions_deploy", cid, seed)
        v = df[V_COLS].to_numpy(dtype=np.int64); P = probs(df, True); hm = M.hard_metrics(v, P)
        labels, ties = M.unique_majority_labels(v); keep = ~ties; y = labels[keep]; yp = M.clip_and_renormalize(P)[keep].argmax(1); pat = df.patient_id.to_numpy()[keep]; g = df.component.to_numpy()[keep]
        cm = np.asarray(hm["confusion_matrix"], dtype=np.int64); cm_norm = np.asarray([[x if x is not None else 0.0 for x in row] for row in hm["confusion_matrix_row_normalized"]], dtype=float)
        if cm.sum() != hm["n_rows_evaluated"] or cm.sum() != keep.sum():
            raise FigureVerificationError("V12: confusion cells do not sum to the unique-majority row count")
        cell_pat = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        for i in range(N_CLASSES):
            for j in range(N_CLASSES): cell_pat[i, j] = np.unique(pat[(y == i) & (yp == j)]).size
        def stat(rows, y=y, yp=yp):
            yy, pp = y[rows], yp[rows]; tp = np.bincount(yy[yy == pp], minlength=N_CLASSES); sup = np.bincount(yy, minlength=N_CLASSES)
            return np.where(sup > 0, tp / np.maximum(sup, 1), np.nan)
        bs = cluster_bootstrap_statistic(stat, g, n_replicates=BOOT_REPLICATES, seed=BOOT_SEED)
        recall = {"point": [float(x) for x in bs["point_estimate"]], "ci_low": bs["ci_low"], "ci_high": bs["ci_high"]}
        support_rows = cm.sum(1); support_pat = np.array([np.unique(pat[y == i]).size for i in range(N_CLASSES)])
        n_supp = 0
        if role == "candidate":
            n_supp += _confusion_figure(ctx, cm, cm_norm, cell_pat, recall, cid, False, "counts", support_rows, support_pat, hm["n_rows_evaluated"])
            n_supp += _confusion_figure(ctx, cm, cm_norm, cell_pat, recall, cid, True, "row_normalized", support_rows, support_pat, hm["n_rows_evaluated"])
        else:
            n_supp += _confusion_figure(ctx, cm, cm_norm, cell_pat, recall, cid, True, "row_normalized_comparator", support_rows, support_pat, hm["n_rows_evaluated"])
        results[role] = {"method": cid, "n_rows_evaluated": hm["n_rows_evaluated"], "n_ties_excluded": hm["n_ties_excluded"], "n_patients": int(np.unique(pat).size), "n_components": int(np.unique(g).size),
                         "confusion_matrix": cm.tolist(), "row_normalized": cm_norm.tolist(), "cell_patients": cell_pat.tolist(), "recall": recall, "support_rows": support_rows.tolist(), "support_patients": support_pat.tolist(),
                         "balanced_accuracy": hm["balanced_accuracy"], "accuracy": hm["accuracy"], "macro_f1": hm["macro_f1"], "suppressed_cells": n_supp}
    r = results["candidate"]
    ctx.run_ids = [f"final_{c}_s{seed}" for c in (lock["candidate"], lock["comparator"])]
    hashes = hashes_from(ctx)
    cap = caption_text("Confusion matrix with support", partition="locked test, unique-majority rows",
                       counts=f"rows {fmt(r['n_rows_evaluated'])} evaluated, tied maxima excluded {fmt(r['n_ties_excluded'])}, patients {fmt(r['n_patients'])}, components {fmt(r['n_components'])}; per-class row and patient support at the right margin",
                       seeds=f"deployment seed {seed}; {lock['candidate']} (counts and row-normalized) with {lock['comparator']} row-normalized companion", metric="argmax of temperature-calibrated probabilities versus the unique-majority vote label; rows = true class, columns = predicted class, fixed six-class order",
                       interval=f"class recall with percentile 95% component-cluster bootstrap ({BOOT_REPLICATES} resamples, seed {BOOT_SEED}); cell colours are descriptive only", hashes=hashes,
                       extra=f"Balanced accuracy {r['balanced_accuracy']:.3f}, accuracy {r['accuracy']:.3f}; cells with fewer than {SMALL_CELL} contributing patients are hatched and marked s")
    return ctx.record(split="test", n_rows=r["n_rows_evaluated"], n_patients=r["n_patients"], n_components=r["n_components"],
                      parameters={"results": results, "small_cell_threshold": SMALL_CELL}, caption=cap, privacy_review="aggregate_only; small-cell suppression applied")


# ======================================================================================
# V13 - Calibration reliability
# ======================================================================================
def _reliability(ctx, df, calibrated):
    q = targets(df); P = M.clip_and_renormalize(probs(df, calibrated)); g = df.component.to_numpy()
    ece = M.soft_top_label_ece(q, P, groups=g)
    n = q.shape[0]; pred = P.argmax(1); conf = P[np.arange(n), pred]; obs = q[np.arange(n), pred]
    upper = np.asarray(ece["bin_upper"]); B = len(upper)
    order = np.argsort(conf, kind="stable"); ranks = np.empty(n, dtype=np.int64); ranks[order] = np.arange(n)
    edges = np.cumsum(ece["bin_count"])
    bin_of = np.searchsorted(edges, ranks, side="right")
    def stat(rows):
        cnt = np.bincount(bin_of[rows], minlength=B); s = np.bincount(bin_of[rows], weights=obs[rows], minlength=B)
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    bs = cluster_bootstrap_statistic(stat, g, n_replicates=BOOT_REPLICATES, seed=BOOT_SEED)
    brier = float(M.expected_brier_rows(q, P).mean()); kl = float(M.patient_weighted_mean(M.kl_rows(q, P), df.patient_id.to_numpy()))
    return {"ece": ece, "ci_low": bs["ci_low"], "ci_high": bs["ci_high"], "brier": brier, "patient_kl": kl}


def render_v13(ctx: FigureContext) -> dict:
    lock = load_lock(ctx); seed = lock["deployment_seed"]; summ = read_json(ctx.path("eval_summary")); ctx.hash_inputs("eval_summary")
    dfc = load_prediction(ctx, "test_predictions_deploy", lock["candidate"], seed); dfb = load_prediction(ctx, "test_predictions_deploy", lock["comparator"], seed)
    series = {"cand_raw": _reliability(ctx, dfc, False), "cand_cal": _reliability(ctx, dfc, True), "comp_cal": _reliability(ctx, dfb, True)}
    def temp(cid):
        s = summ["methods"][cid]["seeds"]; s = s.get(str(seed), s.get(seed)); return float(s["temperature"]["T"])
    T_c, T_b = temp(lock["candidate"]), temp(lock["comparator"])
    fig = plt.figure(figsize=(10.0, 5.6)); plt.rcParams.update(RC)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.35, wspace=0.25)
    panels = [("(a) %s: raw versus temperature-scaled (T = %.3f)" % (lock["candidate"], T_c), [("cand_raw", METHOD_COLORS["alt1"], "raw", "s"), ("cand_cal", METHOD_COLORS["candidate"], "temperature-scaled", "o")]),
              ("(b) Calibrated: %s versus %s (T = %.3f)" % (lock["candidate"], lock["comparator"], T_b), [("cand_cal", METHOD_COLORS["candidate"], lock["candidate"], "o"), ("comp_cal", METHOD_COLORS["comparator"], lock["comparator"], "^")])]
    for col, (title, items) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col]); axh = fig.add_subplot(gs[1, col], sharex=ax)
        ax.plot([0, 1], [0, 1], color=GRID, ls=":", lw=1.0)
        for key, colr, lab, mk in items:
            r = series[key]; e = r["ece"]; x = np.asarray(e["bin_mean_confidence"]); y = np.asarray(e["bin_mean_observed"])
            lo = np.asarray(r["ci_low"], dtype=float); hi = np.asarray(r["ci_high"], dtype=float)
            ax.errorbar(x, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)], fmt=mk + "-", color=colr, ms=5, capsize=2.5, lw=1.4, label=f"{lab}: ECE {e['ece']:.3f}, Brier {r['brier']:.3f}")
            axh.bar(x, e["bin_count"], width=0.02, color=colr, alpha=0.6)
            ctx.track(f"rel_{key}_{col}", np.concatenate([x, y, lo, hi]))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_ylabel("Mean vote mass on the predicted class (observed)"); ax.set_title(title, loc="left", fontsize=9); ax.legend(loc="upper left", fontsize=7)
        axh.set_ylabel("rows per bin", fontsize=7); axh.set_xlabel("Mean predicted confidence max_k p_k per bin")
    fig.suptitle("V13  Calibration reliability on the locked test (soft top-label agreement with expert votes; ten equal-count bins, sparse bins merged)", x=0.01, ha="left", fontsize=10, fontweight="semibold")
    ctx.save(fig)
    ctx.run_ids = [f"final_{c}_s{seed}" for c in (lock["candidate"], lock["comparator"])]
    e = series["cand_cal"]["ece"]
    hashes = hashes_from(ctx)
    cap = caption_text("Calibration reliability", partition="locked test (temperature fitted on calibration-T only)",
                       counts=f"rows {fmt(e['n_rows'])}, components {fmt(e['n_groups_total'])}, patients {fmt(dfc.patient_id.nunique())}; bins after merging: {e['n_bins_final']} (min {e['min_groups_per_bin']} groups per bin, {e['n_merges']} merges)",
                       seeds=f"deployment seed {seed}", metric="soft top-label reliability: mean max_k p_k versus mean vote mass q on the predicted class per equal-count confidence bin; ECE = count-weighted absolute gap; expected categorical Brier and patient-KL in legend/caption",
                       interval=f"per-bin 95% percentile component-cluster bootstrap of the observed mean with bins fixed ({BOOT_REPLICATES} resamples, seed {BOOT_SEED})", hashes=hashes,
                       extra=f"Proper scoring rules: {lock['candidate']} patient-KL raw {series['cand_raw']['patient_kl']:.4f} -> calibrated {series['cand_cal']['patient_kl']:.4f}, Brier raw {series['cand_raw']['brier']:.4f} -> {series['cand_cal']['brier']:.4f}; nothing is fitted to the plotted test points")
    return ctx.record(split="test", n_rows=e["n_rows"], n_patients=int(dfc.patient_id.nunique()), n_components=e["n_groups_total"],
                      parameters={k: {"ece": r["ece"]["ece"], "mce": r["ece"]["mce"], "bins": {kk: r["ece"][kk] for kk in ("bin_lower", "bin_upper", "bin_mean_confidence", "bin_mean_observed", "bin_count", "bin_n_groups")}, "ci_low": r["ci_low"], "ci_high": r["ci_high"], "brier": r["brier"], "patient_kl": r["patient_kl"]} for k, r in series.items()}
                      | {"temperatures": {lock["candidate"]: T_c, lock["comparator"]: T_b}, "merge_rule": e["merge_rule"]},
                      caption=cap, privacy_review="aggregate_only; binned aggregates")


# ======================================================================================
# V14 - Predicted versus observed disagreement
# ======================================================================================
def render_v14(ctx: FigureContext) -> dict:
    lock = load_lock(ctx); seed = lock["deployment_seed"]
    df = load_prediction(ctx, "test_predictions_deploy", lock["candidate"], seed)
    v = df[V_COLS].to_numpy(dtype=np.int64); n = v.sum(1); d_obs = M.pairwise_disagreement(v); d_hat = df.d_hat.to_numpy(dtype=np.float64); g = df.component.to_numpy(); pat = df.patient_id.to_numpy()
    if not np.all(np.isfinite(d_hat)):
        raise FigureVerificationError("V14: d_hat contains non-finite values (auxiliary head absent?)")
    dm = M.disagreement_metrics(d_hat, v)
    bands = [s for s in VOTE_STRATA if s[1] >= 2]
    fig, axes = new_fig(11.0, 3.9, ncols=len(bands) + 1, gridspec_kw={"width_ratios": [0.55] + [1] * len(bands)})
    ax = axes[0]; m1 = n == 1
    ax.set_axis_off(); ax.text(0.5, 0.5, f"Vote count 1\n{fmt(m1.sum())} rows\n{fmt(np.unique(pat[m1]).size)} patients\n\nobserved disagreement\nNOT_ESTIMABLE\n(single vote)\n\nmean predicted d_hat\n{d_hat[m1].mean():.3f}" if m1.any() else "no single-vote rows", ha="center", va="center", fontsize=7.5, color=INK2,
                                bbox=dict(fc="#f4f4f1", ec=GRID, hatch="////"))
    results = {"1": {"status": "NOT_ESTIMABLE", "n_rows": int(m1.sum()), "mean_predicted": float(d_hat[m1].mean()) if m1.any() else None}}
    n_bins = 8
    for ax, (name, lo, hi) in zip(axes[1:], bands):
        m = (n >= lo) & (n <= hi) & np.isfinite(d_obs)
        ax.plot([0, 1], [0, 1], color=GRID, ls=":", lw=1.0)
        if m.sum() < 20:
            ax.text(0.5, 0.5, "NOT_ESTIMABLE\n(<20 rows)", ha="center", va="center", transform=ax.transAxes, color=INK2); results[name] = {"status": "NOT_ESTIMABLE", "n_rows": int(m.sum())}; continue
        dh, do, gg = d_hat[m], d_obs[m], g[m]
        order = np.argsort(dh, kind="stable"); chunks = np.array_split(order, n_bins); bin_of = np.empty(dh.size, dtype=np.int64)
        for b, c in enumerate(chunks): bin_of[c] = b
        def stat(rows, bin_of=bin_of, do=do):
            cnt = np.bincount(bin_of[rows], minlength=n_bins); s = np.bincount(bin_of[rows], weights=do[rows], minlength=n_bins); return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
        bs = cluster_bootstrap_statistic(stat, gg, n_replicates=BOOT_REPLICATES, seed=BOOT_SEED)
        x = np.array([dh[c].mean() for c in chunks]); y = np.asarray(bs["point_estimate"]); lo_, hi_ = np.asarray(bs["ci_low"]), np.asarray(bs["ci_high"])
        ax.errorbar(x, y, yerr=[np.clip(y - lo_, 0, None), np.clip(hi_ - y, 0, None)], fmt="o-", color=METHOD_COLORS["candidate"], ms=4.5, capsize=2.5, lw=1.3)
        blk = next(b for b in dm["strata"] if int(b["n_min"]) == lo and (b["n_max"] is None or int(b["n_max"]) == hi))
        ax.text(0.03, 0.97, f"{fmt(m.sum())} rows · {fmt(np.unique(pat[m]).size)} pts · {fmt(np.unique(gg).size)} comps\nSpearman {blk['spearman']:.3f}, MSE {blk['mse']:.4f}" if blk["spearman"] is not None else f"{fmt(m.sum())} rows", transform=ax.transAxes, ha="left", va="top", fontsize=7, bbox=dict(fc="white", ec=GRID))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_xlabel("Predicted d_hat (bin mean, 8 equal-count bins)"); ax.set_title(f"Vote count {name}", loc="left")
        results[name] = {"status": "PASS", "n_rows": int(m.sum()), "n_patients": int(np.unique(pat[m]).size), "n_groups": int(np.unique(gg).size), "bin_mean_predicted": x.tolist(), "bin_mean_observed": y.tolist(), "ci_low": lo_.tolist(), "ci_high": hi_.tolist(), "spearman": blk["spearman"], "mse": blk["mse"]}
        ctx.track(f"disagreement_{name}", np.concatenate([x, y, lo_, hi_]))
    axes[1].set_ylabel("Observed pairwise disagreement d (bin mean)")
    fig.suptitle(f"V14  Predicted versus observed expert disagreement, {lock['candidate']} auxiliary head (locked test; overall Spearman {dm['overall']['spearman']:.3f}, MSE {dm['overall']['mse']:.4f}, n>1 rows {fmt(dm['overall']['n_rows_estimable'])})", x=0.01, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    ctx.save(fig)
    ctx.run_ids = [f"final_{lock['candidate']}_s{seed}"]
    hashes = hashes_from(ctx)
    cap = caption_text("Predicted versus observed disagreement", partition="locked test",
                       counts=f"rows {fmt(len(df))} ({fmt(dm['n_rows_not_estimable'])} single-vote rows not estimable), patients {fmt(df.patient_id.nunique())}, components {fmt(np.unique(g).size)}; per-band counts in panels",
                       seeds=f"deployment seed {seed}, {lock['candidate']} only (comparator has no auxiliary head)", metric="observed d = 1 - sum v_k(v_k-1)/(n(n-1)) (may reach 1, never capped at 5/6) versus predicted d_hat, eight equal-count bins per prespecified vote-count band",
                       interval=f"per-bin 95% percentile component-cluster bootstrap of the observed mean ({BOOT_REPLICATES} resamples, seed {BOOT_SEED}); Spearman and MSE per band", hashes=hashes,
                       extra="Single-vote rows are shown as NOT_ESTIMABLE, never as zero; no Bayesian uncertainty decomposition is claimed")
    return ctx.record(split="test", n_rows=len(df), n_patients=int(df.patient_id.nunique()), n_components=int(np.unique(g).size), parameters={"bands": results, "overall": dm["overall"]},
                      caption=cap, privacy_review="aggregate_only; binned aggregates, raw vote rows private")


# ======================================================================================
# V15 - Review risk-coverage curves
# ======================================================================================
def _pw_risk_curve(score, risk, groups, grid):
    """Patient-weighted accepted risk at every coverage on the grid (rows with the lowest score accepted first)."""
    order = np.argsort(score, kind="stable"); r = risk[order]; _, gi = np.unique(groups[order], return_inverse=True); n = score.size
    ks = np.clip(np.round(grid * n), 1, n).astype(int); out = np.empty(len(grid))
    for i, k in enumerate(ks):
        cnt = np.bincount(gi[:k], minlength=gi.max() + 1); s = np.bincount(gi[:k], weights=r[:k], minlength=gi.max() + 1); out[i] = float((s[cnt > 0] / cnt[cnt > 0]).mean())
    return out


def _risk_curve_boot(score, risk, groups, grid, n_rep=BOOT_REPLICATES, seed=BOOT_SEED):
    uniq, inv = np.unique(groups, return_inverse=True); G = uniq.size; rng = np.random.default_rng(seed)
    rows_by_g = [np.where(inv == g)[0] for g in range(G)]
    point = _pw_risk_curve(score, risk, inv, grid); reps = np.empty((n_rep, len(grid)))
    for r in range(n_rep):
        draw = rng.integers(0, G, G); rows = np.concatenate([rows_by_g[g] for g in draw]); gid = np.repeat(np.arange(G), [rows_by_g[g].size for g in draw])
        reps[r] = _pw_risk_curve(score[rows], risk[rows], gid, grid)
    lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)
    return point, lo, hi


def render_v15(ctx: FigureContext) -> dict:
    lock = load_lock(ctx); seed = lock["deployment_seed"]; summ = read_json(ctx.path("eval_summary")); ctx.hash_inputs("eval_summary")
    grid = M.DEFAULT_COVERAGE_GRID
    fig, axes = new_fig(10.5, 4.2, ncols=2)
    results = {}; n_rows = n_pat = n_comp = None
    for role, cid, col in [("candidate", lock["candidate"], METHOD_COLORS["candidate"]), ("comparator", lock["comparator"], METHOD_COLORS["comparator"])]:
        df = load_prediction(ctx, "test_predictions_deploy", cid, seed); q = targets(df); P = M.clip_and_renormalize(probs(df, True)); g = df.component.to_numpy()
        kl = M.kl_rows(q, P); labels, ties = M.unique_majority_labels(df[V_COLS].to_numpy(dtype=np.int64)); err = np.where(labels >= 0, (P.argmax(1) != labels).astype(float), np.nan)
        s = df.referral_score.to_numpy(dtype=np.float64)
        ms = summ["methods"][cid]["seeds"]; ms = ms.get(str(seed), ms.get(seed))
        res = {"kl": {}, "error": {}}
        for ax, (risk, keep, key) in zip(axes, [(kl, np.ones(len(df), bool), "kl"), (err, labels >= 0, "error")]):
            pt, lo, hi = _risk_curve_boot(s[keep], risk[keep], g[keep], grid)
            ax.fill_between(grid, lo, hi, color=col, alpha=0.18, lw=0); ax.plot(grid, pt, color=col, lw=1.8, label=f"{cid}: locked score '{lock['referral_score']}'")
            res[key] = {"coverage": grid.tolist(), "risk": pt.tolist(), "ci_low": lo.tolist(), "ci_high": hi.tolist(), "full_cohort": float(pt[-1])}
            ctx.track(f"rc_{role}_{key}", np.concatenate([pt, lo, hi]))
            if role == "candidate":
                alts = {"entropy": M.predictive_entropy(P), "one_minus_max": M.one_minus_max_prob(P)}
                if np.isfinite(df.d_hat.to_numpy()).all(): alts["d_hat"] = df.d_hat.to_numpy(dtype=np.float64)
                for (aname, asc), ls in zip(alts.items(), [":", "--", "-."]):
                    if aname == lock["referral_score"]: continue
                    apt = _pw_risk_curve(asc[keep], risk[keep], g[keep], grid); ax.plot(grid, apt, color=METHOD_COLORS["alt1"], lw=1.1, ls=ls, label=f"{cid}: alternative score '{aname}' (descriptive)")
                    res[key][f"alt_{aname}"] = apt.tolist(); ctx.track(f"rc_alt_{aname}_{key}", apt)
            # frozen calibration-P operating points
            for ci, (cov, ref) in enumerate(sorted(ms["referral"].items(), key=lambda kv: float(kv[0]))):
                ac = ref["actual_coverage"]; y = np.interp(ac, grid, pt) if key == "error" else ref.get("accepted_risk_patient_weighted", ref["accepted_risk"])
                ax.plot([ac], [y], marker="D", color=col, ms=6, mec=INK, ls="", zorder=6)
                if role == "candidate": ax.annotate(f"planned {float(cov):.0%}\nactual {ac:.1%}\nreferred {fmt(ref['n_referred'])}", (ac, y), textcoords="offset points", xytext=(-18 + 18 * ci, 12 + 22 * (ci % 2)), fontsize=6.3, ha="center", color=INK2, arrowprops=dict(arrowstyle="-", color=GRID, lw=0.6))
            res[key]["operating_points"] = {cov: {k: ref[k] for k in ("threshold", "actual_coverage", "n_referred", "fraction_referred", "accepted_risk", "accepted_risk_patient_weighted", "whole_cohort_risk", "whole_cohort_risk_patient_weighted")} for cov, ref in ms["referral"].items()}
        results[role] = res | {"method": cid, "n_rows": int(len(df)), "n_patients": int(df.patient_id.nunique()), "n_components": int(np.unique(g).size), "n_unique_majority": int((labels >= 0).sum())}
        n_rows, n_pat, n_comp = int(len(df)), int(df.patient_id.nunique()), int(np.unique(g).size)
    axes[0].set_ylabel("Patient-weighted KL of accepted rows (nats)"); axes[0].set_title("(a) Accepted-case KL versus acceptance coverage", loc="left")
    axes[1].set_ylabel("Patient-weighted majority-label error of accepted rows"); axes[1].set_title("(b) Accepted-case majority error (unique-majority rows)", loc="left")
    for ax in axes:
        ax.set_xlim(0.5, 1.0); ax.set_ylim(0, None); ax.margins(y=0.15); ax.set_xlabel("Actual acceptance coverage (fraction accepted; remainder referred)")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=min(4, len(h)), fontsize=7, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("V15  Review risk-coverage curves on the locked test (thresholds frozen on calibration-P; diamonds = frozen operating points)", x=0.01, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    ctx.save(fig)
    ctx.run_ids = [f"final_{c}_s{seed}" for c in (lock["candidate"], lock["comparator"])]
    hashes = hashes_from(ctx)
    op = results["candidate"]["kl"]["operating_points"]
    cap = caption_text("Review risk-coverage curves", partition="locked test",
                       counts=f"rows {fmt(n_rows)}, patients {fmt(n_pat)}, components {fmt(n_comp)}; majority-error panel uses {fmt(results['candidate']['n_unique_majority'])} unique-majority rows",
                       seeds=f"deployment seed {seed} for both methods", metric=f"patient-weighted mean KL (a) and majority-label error (b) over accepted rows as a function of actual acceptance coverage on the common grid [0.5, 1.0]; locked input-only referral score '{lock['referral_score']}' (higher = refer)",
                       interval=f"95% percentile component-cluster bootstrap bands ({BOOT_REPLICATES} resamples, seed {BOOT_SEED})", hashes=hashes,
                       extra="Operating points use thresholds frozen on calibration-P (no threshold chosen on test); " + "; ".join(f"planned {float(c):.0%}: actual coverage {o['actual_coverage']:.3f}, referred {fmt(o['n_referred'])}, whole-cohort KL {o['whole_cohort_risk_patient_weighted']:.4f}" for c, o in op.items()) + "; the 100% point is the whole-cohort risk")
    return ctx.record(split="test", n_rows=n_rows, n_patients=n_pat, n_components=n_comp, parameters={"results": results, "referral_score": lock["referral_score"], "grid": grid.tolist()},
                      caption=cap, privacy_review="aggregate_only; policy curves")


# ======================================================================================
# V16 - Subgroup performance forest
# ======================================================================================
def render_v16(ctx: FigureContext) -> dict:
    lock = load_lock(ctx); st = read_json(ctx.path("strata")); bp = read_json(ctx.path("bootstrap_primary")); ctx.hash_inputs("strata", "bootstrap_primary")
    pr = bp["primary_calibrated"]
    groups = [("Overall (confirmatory primary)", ["__overall__"]), ("Vote-count band", ["votes_1", "votes_2-4", "votes_5-9", "votes_10+"]), ("Target-entropy tertile (frozen on development)", ["entropy_low", "entropy_mid", "entropy_high"]),
              ("Minimum valid fraction", ["valid_fraction_lt_0.95", "valid_fraction_ge_0.95"]), ("Unique-majority class", [f"majority_{c}" for c in CLASS_NAMES] + ["tied_majority"])]
    rows = []
    for gname, keys in groups:
        rows.append(("__header__", gname))
        for k in keys:
            rows.append((k, k))
    fig, ax = new_fig(8.6, 0.32 * len(rows) + 1.4)
    y = 0; labels = []; yticks = []; plotted = {}; n_est = 0
    for key, name in rows:
        y -= 1
        if key == "__header__":
            ax.text(-0.01, y, name, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8, fontweight="semibold", color=INK); continue
        if key == "__overall__":
            rec = {"delta": pr["point_estimate"], "ci_low": pr["ci_low"], "ci_high": pr["ci_high"], "n_groups": pr["n_groups"], "n_rows": None, "status": "PASS", "low_support": False}
        else:
            rec = st.get(key, {"status": "MISSING"})
        pretty = key.replace("votes_", "votes ").replace("entropy_", "entropy ").replace("valid_fraction_lt_0.95", "valid fraction < 0.95").replace("valid_fraction_ge_0.95", "valid fraction >= 0.95").replace("majority_", "majority ").replace("tied_majority", "tied maxima (soft metrics only)").replace("__overall__", "all test components")
        if rec.get("status") != "PASS":
            ax.text(0.02, y, f"{'NOT_ESTIMABLE' if rec.get('status') != 'MISSING' else 'NOT_RUN'} (n rows {rec.get('n_rows', '?')})", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7, color=INK2)
            labels.append(pretty); yticks.append(y); plotted[key] = rec; continue
        col = "#e34948" if key == "__overall__" else (METHOD_COLORS["candidate"] if not rec.get("low_support") else INK2)
        ax.errorbar([rec["delta"]], [y], xerr=[[rec["delta"] - rec["ci_low"]], [rec["ci_high"] - rec["delta"]]], fmt=("D" if key == "__overall__" else "o"), color=col, ms=(6 if key == "__overall__" else 4.5), mfc=("white" if rec.get("low_support") else col), capsize=3, lw=1.2)
        note = f"n groups {fmt(rec['n_groups'])}" + (f", rows {fmt(rec['n_rows'])}" if rec.get("n_rows") else "") + ("  low support" if rec.get("low_support") else "")
        ax.text(1.01, y, note, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7, color=INK2)
        labels.append(pretty); yticks.append(y); plotted[key] = rec; n_est += 1
        ctx.track(f"forest_{key}", [rec["delta"], rec["ci_low"], rec["ci_high"]])
    ax.axvline(0, color=INK, lw=1.0, ls="--"); ax.set_yticks(yticks); ax.set_yticklabels(labels, fontsize=8); ax.set_ylim(y - 0.8, 0)
    lim = max(abs(v) for r in plotted.values() if r.get("status") == "PASS" for v in (r["ci_low"], r["ci_high"])) * 1.1
    ax.set_xlim(-lim, lim); ax.set_xlabel(f"Delta component-mean KL = {lock['candidate']} - {lock['comparator']} (nats; negative favours {lock['candidate']}); calibrated, mean over seeds {lock['final_seeds']}")
    ax.set_title("V16  Subgroup forest on the locked test (prespecified strata; subgroup intervals are exploratory)", loc="left")
    fig.legend(handles=[Line2D([], [], marker="D", color="#e34948", ls="", label="primary comparison"), Line2D([], [], marker="o", color=METHOD_COLORS["candidate"], ls="", label="prespecified subgroup (exploratory)"), Line2D([], [], marker="o", color=INK2, mfc="white", ls="", label="low support (< 30 groups)")], loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    ctx.save(fig)
    # complementary-count check
    def tot(keys): return sum(st[k]["n_rows"] for k in keys if k in st)
    checks = {"votes": tot(["votes_1", "votes_2-4", "votes_5-9", "votes_10+"]), "entropy": tot(["entropy_low", "entropy_mid", "entropy_high"]), "valid_fraction": tot(["valid_fraction_lt_0.95", "valid_fraction_ge_0.95"]), "majority_plus_tied": tot([f"majority_{c}" for c in CLASS_NAMES] + ["tied_majority"])}
    vals = set(checks.values())
    if len(vals) != 1:
        raise FigureVerificationError(f"V16: complementary strata do not sum to the same row count: {checks}")
    n_rows = checks["votes"]
    ctx.run_ids = [f"final_{c}_s{s}" for c in (lock["candidate"], lock["comparator"]) for s in lock["final_seeds"]]
    hashes = hashes_from(ctx)
    cap = caption_text("Subgroup performance forest", partition="locked test", counts=f"rows {fmt(n_rows)} (complementary strata each sum to this count), components {fmt(pr['n_groups'])}; per-subgroup groups at the right margin",
                       seeds=f"seeds {lock['final_seeds']} averaged as losses", metric=f"paired difference of component-mean calibrated KL, {lock['candidate']} minus {lock['comparator']}",
                       interval=f"percentile 95% paired component bootstrap ({pr['n_replicates']} resamples, seed {pr['seed']}); only the overall comparison is confirmatory, subgroup intervals are exploratory without multiplicity adjustment", hashes=hashes,
                       extra="Strata are the prespecified vote-count bands, development-frozen entropy tertiles, minimum valid fraction bands and unique-majority class; no age, sex, site or disease metadata exist in the source; strata with fewer than 10 rows are NOT_ESTIMABLE; open markers flag fewer than 30 groups")
    return ctx.record(split="test", n_rows=n_rows, n_patients=None, n_components=pr["n_groups"], parameters={"strata": plotted, "complementary_counts": checks, "n_estimable": n_est},
                      caption=cap, privacy_review="aggregate_only; suppression via NOT_ESTIMABLE for tiny strata")


# ======================================================================================
# V17 - Preprocessing and corruption sensitivity
# ======================================================================================
def render_v17(ctx: FigureContext) -> dict:
    lock = load_lock(ctx); re_ = read_json(ctx.path("robustness_evidence")); ctx.hash_inputs("robustness_evidence")
    sub = read_json(ctx.path("robustness_subset")) if ctx.has("robustness_subset") else None
    if sub: ctx.hash_inputs("robustness_subset")
    conds = [c for c in lock.get("corruption_conditions", []) if c != "clean"]
    if not conds:
        conds = [k for k in re_[lock["candidate"]]["robustness"] if k != "clean"]
    fig, axes = new_fig(11.0, 4.3, ncols=3)
    xs = np.arange(len(conds)); w = 0.36; table = {}
    for j, (role, cid, col) in enumerate([("candidate", lock["candidate"], METHOD_COLORS["candidate"]), ("comparator", lock["comparator"], METHOD_COLORS["comparator"])]):
        rob = re_[cid]["robustness"]; table[cid] = {}
        for i, c in enumerate(conds):
            r = rob.get(c, {"status": "NOT_RUN"}); x = xs[i] + (j - 0.5) * w
            if r.get("status") != "PASS":
                for ax in axes: ax.text(x, 0.01, "NOT_RUN", rotation=90, ha="center", va="bottom", fontsize=6.5, color=INK2, transform=ax.get_xaxis_transform())
                table[cid][c] = {"status": r.get("status"), "reason": r.get("reason")}; continue
            axes[0].bar(x, r["delta_kl"], width=w, color=col, edgecolor="white", label=(cid if i == 0 else None))
            axes[0].errorbar([x], [r["delta_kl"]], yerr=[[r["delta_kl"] - r["ci_low"]], [r["ci_high"] - r["delta_kl"]]], fmt="none", ecolor=INK, capsize=2.5, lw=1)
            axes[1].bar(x, r["js_distance"], width=w, color=col, edgecolor="white", label=(cid if i == 0 else None))
            axes[2].bar(x, 100 * r["label_flip_fraction"], width=w, color=col, edgecolor="white", label=(cid if i == 0 else None))
            table[cid][c] = {k: r.get(k) for k in ("delta_kl", "ci_low", "ci_high", "kl", "js_distance", "label_flip_fraction", "majority_error", "severity", "units", "description", "status")}
            ctx.track(f"rob_{cid}_{c}", [r["delta_kl"], r["ci_low"], r["ci_high"], r["js_distance"], r["label_flip_fraction"]])
    def tick(c):
        r = re_[lock["candidate"]]["robustness"].get(c, {}); return f"{c.replace('_', ' ')}\n({r.get('severity', '?')} {r.get('units', '')})"
    for ax, ylab, title in zip(axes, ["Delta patient-KL versus clean (nats)", "Jensen-Shannon distance clean vs perturbed (mean)", "Predicted-label flips (%)"], ["(a) Loss change", "(b) Prediction shift", "(c) Label flips"]):
        ax.set_xticks(xs); ax.set_xticklabels([tick(c) for c in conds], fontsize=6.5, rotation=25, ha="right"); ax.set_ylabel(ylab); ax.set_title(title, loc="left"); ax.axhline(0, color=INK, lw=0.8)
    axes[0].legend(loc="upper left", fontsize=7)
    for ax in axes[1:]: ax.set_ylim(0, None)
    n_rows = re_[lock["candidate"]].get("n_rows"); n_pat = re_[lock["candidate"]].get("n_patients")
    fig.suptitle(f"V17  Sensitivity to the six prespecified perturbations on the fixed label-blind test subset ({fmt(n_rows)} rows, {fmt(n_pat)} patients; representation-level stress tests, not validated artifact simulation)", x=0.01, ha="left", fontsize=9.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    ctx.save(fig)
    ctx.run_ids = [f"final_{c}_s{lock['deployment_seed']}" for c in (lock["candidate"], lock["comparator"])]
    hashes = hashes_from(ctx)
    cap = caption_text("Preprocessing and corruption sensitivity", partition="locked test, fixed label-blind subset" + (f" (seed {sub['seed']}, subset hash {sub['hash']})" if sub else ""),
                       counts=f"rows {fmt(n_rows)}, patients {fmt(n_pat)}; identical rows for clean and perturbed inputs", seeds=f"deployment seed {lock['deployment_seed']} for both methods",
                       metric="change in patient-weighted calibrated KL relative to the clean matched input, mean row-wise Jensen-Shannon distance between clean and perturbed predictions, fraction of argmax label flips; severity in the stated units on the x-axis",
                       interval="paired 95% percentile component-cluster bootstrap for delta KL (2,000 resamples); JS and flips descriptive", hashes=hashes,
                       extra="Conditions not applicable to a method are marked NOT_RUN; artificial masking and gain shifts are representation-level perturbations")
    return ctx.record(split="test_subset", n_rows=n_rows, n_patients=n_pat, n_components=None, parameters={"conditions": conds, "results": table, "subset": sub}, caption=cap, privacy_review="aggregate_only; perturbed examples private")


# ======================================================================================
# V18 - Evidence deletion and region dependence
# ======================================================================================
def render_v18(ctx: FigureContext) -> dict:
    lock = load_lock(ctx); re_ = read_json(ctx.path("robustness_evidence")); ctx.hash_inputs("robustness_evidence")
    ev = re_.get("evidence")
    if not ev:
        raise FigureVerificationError("V18: robustness_evidence.json has no 'evidence' block")
    vr = ev["view_region_deletion"]; cur = ev["ranked_vs_random_deletion"]
    keys = ["local_removed", "context_removed"] + [f"region_removed_{r}" for r in REGIONS]
    pretty = {"local_removed": "local 50 s view removed", "context_removed": "context 600 s view removed"} | {f"region_removed_{r}": f"region {r} removed (both views)" for r in REGIONS}
    fig, axes = new_fig(12.0, 4.6, ncols=3, gridspec_kw={"width_ratios": [1.3, 1, 1], "wspace": 0.45})
    ax = axes[0]; ys = np.arange(len(keys))[::-1]
    for y, k in zip(ys, keys):
        r = vr[k]; ax.errorbar([r["delta_kl"]], [y], xerr=[[r["delta_kl"] - r["ci_low"]], [r["ci_high"] - r["delta_kl"]]], fmt="o", color=METHOD_COLORS["candidate"], capsize=3, ms=5)
        ctx.track(f"del_{k}", [r["delta_kl"], r["ci_low"], r["ci_high"]])
    ax.axvline(0, color=INK, lw=1.0, ls="--"); ax.set_yticks(ys); ax.set_yticklabels([f"{pretty[k]}\nJS {vr[k]['js_distance']:.3f} · flips {100 * vr[k]['label_flip_fraction']:.1f}%" for k in keys], fontsize=7.5)
    ax.set_xlabel("Delta patient-KL vs intact input (nats; + = worse)"); ax.set_title(f"(a) View/region deletion on the full test\n({fmt(ev['n_rows'])} rows, {fmt(ev['n_patients'])} patients)", loc="left", fontsize=8.5)
    fr = np.asarray(cur["fractions"])
    for ax, key, ylab in zip(axes[1:], ["kl", "js"], ["Mean KL of the selected cases (nats)", "Mean JS distance from the intact prediction"]):
        for name, col, ls in [("ranked", METHOD_COLORS["candidate"], "-"), ("random", METHOD_COLORS["comparator"], "--")]:
            yv = np.asarray(cur[name][key]); ax.plot(fr, yv, marker="o", ms=4, color=col, ls=ls, label=f"{name} deletion order"); ctx.track(f"curve_{name}_{key}", yv)
        ax.set_xlabel("Fraction of local (region, time-bin) blocks removed"); ax.set_ylabel(ylab); ax.set_xlim(0, 1); ax.set_ylim(0, None); ax.legend(loc="upper left", fontsize=7)
    axes[1].set_title(f"(b) Ranked (|grad x input|) vs random deletion\n({fmt(cur['n_cases'])} prespecified cases; same masked area)", loc="left", fontsize=8.5); axes[2].set_title("(c) Prediction shift under deletion\n(Jensen-Shannon distance from intact)", loc="left", fontsize=8.5)
    fig.suptitle(f"V18  Evidence-use audit for the frozen {lock['candidate']} model (seed {lock['deployment_seed']}); aggregate only, no saliency image is treated as causal evidence", x=0.01, y=1.0, ha="left", fontsize=9.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    ctx.save(fig)
    ctx.run_ids = [f"final_{lock['candidate']}_s{lock['deployment_seed']}"]
    hashes = hashes_from(ctx)
    cap = caption_text("Evidence deletion and region dependence", partition="locked test (view/region deletion on every test row; ranked-vs-random curves on the prespecified case set)",
                       counts=f"rows {fmt(ev['n_rows'])}, patients {fmt(ev['n_patients'])}; {fmt(cur['n_cases'])} cases selected by class and entropy rules (no outcome-based selection)",
                       seeds=f"deployment seed {lock['deployment_seed']}, {lock['candidate']} only", metric="change in patient-weighted calibrated KL, Jensen-Shannon distance and label-flip fraction after deleting a view or a region; KL/JS versus the fraction of local blocks deleted in saliency-ranked versus random order with the same masked area",
                       interval="paired 95% percentile component-cluster bootstrap for panel (a); panels (b)-(c) report case means without an interval (the pipeline stores means only; exploratory, case count limited)", hashes=hashes,
                       extra="A ranked curve above the random curve indicates the ranking predicts deletion effects; this does not establish a biological cause; individual case maps require separate approval")
    return ctx.record(split="test", n_rows=ev["n_rows"], n_patients=ev["n_patients"], n_components=None, parameters={"view_region_deletion": vr, "ranked_vs_random": cur}, caption=cap, privacy_review="aggregate_only; no case-level maps")


# ======================================================================================
# V19 - Mechanism-ablation effect plot
# ======================================================================================
def render_v19(ctx: FigureContext) -> dict:
    dev = pd.read_csv(ctx.path("development_table")); ctx.hash_inputs("development_table")
    lock = read_json(ctx.path("protocol_lock")) if ctx.has("protocol_lock") else None
    bp = read_json(ctx.path("bootstrap_primary")) if ctx.has("bootstrap_primary") else None
    if lock: ctx.hash_inputs("protocol_lock")
    if bp: ctx.hash_inputs("bootstrap_primary")
    ok = dev[(dev.status == "PASS") & dev.tune_patient_kl.notna()].copy()
    if "lr_mult" not in ok.columns: ok["lr_mult"] = 1.0
    ok["lr_mult"] = ok.lr_mult.fillna(1.0); ok = ok[ok.lr_mult == 1.0]
    steps = [("Foveated local bins (A1 - B2)", "A1", "B2"), ("Learned view gate (A2 - A1)", "A2", "A1"), ("Disagreement auxiliary loss (P - A2)", "P", "A2"), ("All three combined (P - B2)", "P", "B2"),
             ("Pretrained MobileNetV3 reference (B3 - B2)", "B3", "B2"), ("Exploratory multi-scale fusion (P_MSF - P)", "P_MSF", "P")]
    rows = []
    for name, a, b in steps:
        ga = ok[ok.config == a].set_index("seed").tune_patient_kl; gb = ok[ok.config == b].set_index("seed").tune_patient_kl
        common = sorted(set(ga.index) & set(gb.index))
        if not common:
            rows.append({"name": name, "status": "NOT_RUN", "reason": f"no seed with completed runs for both {a} and {b}"}); continue
        d = np.array([ga[s] - gb[s] for s in common])
        rows.append({"name": name, "status": "PASS", "delta": float(d.mean()), "min": float(d.min()), "max": float(d.max()), "n_seeds": len(common), "seeds": common, "a": a, "b": b, "scope": "development (tune)"})
    if lock and bp:
        pr = bp["primary_calibrated"]
        rows.append({"name": f"Frozen final comparison ({lock['candidate']} - {lock['comparator']}), locked test", "status": "PASS", "delta": pr["point_estimate"], "ci_low": pr["ci_low"], "ci_high": pr["ci_high"], "n_seeds": pr["n_seeds"], "scope": "confirmatory (test)", "n_groups": pr["n_groups"]})
    fig, ax = new_fig(8.8, 0.5 * len(rows) + 1.6)
    ys = np.arange(len(rows))[::-1]; vals = []
    for y, r in zip(ys, rows):
        if r["status"] != "PASS":
            ax.text(0.0, y, "NOT_RUN: " + r["reason"], ha="left", va="center", fontsize=7, color=INK2); continue
        if r.get("scope", "").startswith("confirmatory"):
            ax.errorbar([r["delta"]], [y], xerr=[[r["delta"] - r["ci_low"]], [r["ci_high"] - r["delta"]]], fmt="D", color="#e34948", capsize=3, ms=6)
            note = f"95% paired bootstrap CI, {fmt(r['n_groups'])} components, {r['n_seeds']} seeds"; vals += [r["ci_low"], r["ci_high"]]
        else:
            if r["n_seeds"] > 1:
                ax.errorbar([r["delta"]], [y], xerr=[[r["delta"] - r["min"]], [r["max"] - r["delta"]]], fmt="o", color=METHOD_COLORS["candidate"], capsize=3, ms=5); note = f"seed-matched delta, {r['n_seeds']} seeds (bars = min-max); development only"
                vals += [r["min"], r["max"]]
            else:
                ax.plot([r["delta"]], [y], marker="o", color=METHOD_COLORS["candidate"], mfc="white", ms=6, ls=""); note = "single-seed pilot: no seed interval; development only"
            vals.append(r["delta"])
        ax.text(1.01, y, note, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7, color=INK2)
        ctx.track(f"abl_{r['name']}", [r["delta"]])
    ax.axvline(0, color=INK, lw=1.0, ls="--"); ax.set_yticks(ys); ax.set_yticklabels([r["name"] for r in rows], fontsize=8)
    lim = (max(abs(v) for v in vals) if vals else 0.1) * 1.15; ax.set_xlim(-lim, lim)
    ax.set_xlabel("Paired delta patient-mean KL (nats; negative = intervention helps)")
    ax.set_title("V19  Mechanism ablations at fixed bytes and exposure\n(development deltas on tune; only the frozen final comparison is confirmatory, on the locked test)", loc="left")
    fig.legend(handles=[Line2D([], [], marker="o", color=METHOD_COLORS["candidate"], ls="", label="development (tune), seed-matched"), Line2D([], [], marker="o", color=METHOD_COLORS["candidate"], mfc="white", ls="", label="development, single seed"), Line2D([], [], marker="D", color="#e34948", ls="", label="confirmatory frozen comparison (test)")], loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    ctx.save(fig)
    ctx.run_ids = sorted(str(r) for r in ok.run_id.dropna().unique()) if "run_id" in ok.columns else []
    hashes = hashes_from(ctx)
    n_pass = sum(r["status"] == "PASS" for r in rows)
    cap = caption_text("Mechanism-ablation effect plot", partition="tune for development ablations" + ("; locked test for the frozen final comparison" if lock and bp else ""),
                       counts=f"{n_pass} estimable comparisons of {len(rows)}; seeds per comparison at the right margin", seeds="seed-matched differences at the default learning rate; single-seed pilots carry no invented interval",
                       metric="difference of tune patient-mean KL between configurations sharing cache bytes and training exposure (B2 uniform control -> A1 foveation -> A2 gate -> P auxiliary loss)",
                       interval="development: min-max over matched seeds only (no bootstrap; exploratory); final comparison: percentile 95% paired component bootstrap", hashes=hashes,
                       extra="Ablations that made results worse are shown with the same rule as improvements; pilot effects are not called confirmatory")
    return ctx.record(split="tune" + ("/test" if lock and bp else ""), n_rows=None, n_patients=None, n_components=None, parameters={"comparisons": rows, "configs_available": sorted(ok.config.unique().tolist())},
                      caption=cap, privacy_review="aggregate_only; exact config ids only")


# ======================================================================================
# V20 - Accuracy-resource Pareto and budget ledger
# ======================================================================================
def render_v20(ctx: FigureContext) -> dict:
    lock = load_lock(ctx); lat = read_json(ctx.path("latency")); res = read_json(ctx.path("resources")); summ = read_json(ctx.path("eval_summary")); ctx.hash_inputs("latency", "resources", "eval_summary")
    pts = {}
    for role, cid, col in [("candidate", lock["candidate"], METHOD_COLORS["candidate"]), ("comparator", lock["comparator"], METHOD_COLORS["comparator"])]:
        A = []
        for s in lock["final_seeds"]:
            df = load_prediction(ctx, "test_predictions_all_seeds", cid, s); _, m = M.patient_mean(M.kl_rows(targets(df), probs(df, True)), df.component.to_numpy()); A.append(m)
        A = np.stack(A, 1); per = A.mean(1); rng = np.random.default_rng(BOOT_SEED); idx = rng.integers(0, per.size, (BOOT_REPLICATES, per.size)); reps = per[idx].mean(1)
        lo, hi = np.percentile(reps, [2.5, 97.5]); L = lat[cid]
        pts[role] = {"method": cid, "kl": float(per.mean()), "ci_low": float(lo), "ci_high": float(hi), "n_groups": int(per.size), "batch_32_median_ms": L["batch_32"]["median_ms"], "batch_32_p95_ms": L["batch_32"]["p95_ms"],
                     "batch_1_median_ms": L["batch_1"]["median_ms"], "batch_1_p95_ms": L["batch_1"]["p95_ms"], "raw_to_prediction_median_ms": L.get("raw_to_prediction_batch_1", {}).get("median_ms"), "examples_per_s": L["batch_32"]["examples_per_s"],
                     "params": L["parameters"]["total"] if isinstance(L.get("parameters"), dict) else L.get("parameters"), "device": L.get("device"), "dtype": L.get("dtype"), "color": col}
    fig, ax = new_fig(6.4, 4.4)
    for role, p in pts.items():
        ax.errorbar([p["batch_32_median_ms"]], [p["kl"]], xerr=[[0.0], [max(p["batch_32_p95_ms"] - p["batch_32_median_ms"], 0.0)]], yerr=[[p["kl"] - p["ci_low"]], [p["ci_high"] - p["kl"]]], fmt="o", color=p["color"], ms=7, capsize=3, label=f"{p['method']}: {fmt(p['params'])} params")
        ax.annotate(f"{p['method']}\n{p['batch_32_median_ms']:.2f} ms (p95 {p['batch_32_p95_ms']:.2f})\nKL {p['kl']:.4f}", (p["batch_32_median_ms"], p["kl"]), textcoords="offset points", xytext=(10, 6), fontsize=7, color=INK2)
        ctx.track(f"pareto_{role}", [p["batch_32_median_ms"], p["batch_32_p95_ms"], p["kl"], p["ci_low"], p["ci_high"]])
    xs = [p["batch_32_median_ms"] for p in pts.values()]; ax.set_xlim(0, max(p["batch_32_p95_ms"] for p in pts.values()) * 1.35)
    ys = [p["ci_low"] for p in pts.values()] + [p["ci_high"] for p in pts.values()]; ax.set_ylim(0, max(ys) * 1.15)
    ax.set_xlabel(f"Cached-tensor inference latency, batch 32, median ms (bar to p95); {pts['candidate']['device']}, {pts['candidate']['dtype']}"); ax.set_ylabel("Test patient-KL (calibrated; mean over seeds)")
    ax.set_title("V20  Accuracy versus measured latency on the locked test", loc="left"); ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout(); ctx.save(fig)
    # companion ledger table
    fits = res.get("final_fits", {}); cache = res.get("cache", {})
    rows = [["GPU hours total (all jobs incl. failures)", f"{res['gpu_hours_total']:.2f} of {res.get('ceiling_hours', 12.0):.0f} h ceiling"], ["GPU jobs / failed or incomplete", f"{res['n_gpu_jobs']} / {res['failed_or_incomplete_jobs']}"]]
    rows += [[f"GPU hours: {k}", f"{v:.2f} h"] for k, v in res.get("gpu_hours_by_stage", {}).items()]
    rows += [["Active cache (values + masks + metadata)", f"{cache.get('active_cache_bytes', 0) / 1e9:.3f} GB (cap 6.0; within cap: {cache.get('within_cap')})"], ["Alternate uniform local cache", f"{cache.get('alt_uniform_bytes', 0) / 1e9:.3f} GB"],
             ["Source bytes (read-only)", f"{res.get('source_bytes', 0) / 1e9:.2f} GB"], ["Cache conversion wall time", f"{(res.get('conversion') or {}).get('elapsed_seconds', float('nan')) / 60:.1f} min"]]
    for k, f in fits.items():
        rows.append([f"Final fit {k}: wall / peak CUDA / peak RSS / params", f"{f['wall_min']:.1f} min / {f['peak'].get('cuda_alloc_gib', float('nan')):.2f} GiB / {f['peak'].get('rss_gib', float('nan')):.2f} GiB / {fmt(f['params'])}"])
    for role, p in pts.items():
        rows.append([f"Latency {p['method']}: batch 1 / batch 32 / raw-to-prediction", f"{p['batch_1_median_ms']:.2f} ms (p95 {p['batch_1_p95_ms']:.2f}) / {p['batch_32_median_ms']:.2f} ms ({p['examples_per_s']:.0f} ex/s) / {p['raw_to_prediction_median_ms']:.0f} ms" if p["raw_to_prediction_median_ms"] is not None else "n/a"])
    rows.append(["Energy", str(lat.get("energy", "NOT_MEASURED"))])
    fig, ax = new_fig(9.0, 0.28 * len(rows) + 0.9); ax.set_axis_off(); ax.grid(False)
    tb = ax.table(cellText=rows, colLabels=["Resource item", "Measured value"], loc="center", cellLoc="left", colLoc="left", colWidths=[0.5, 0.5])
    tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.scale(1, 1.25)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor(GRID)
        if i == 0: c.set_facecolor("#eef3fb"); c.set_text_props(fontweight="semibold")
    ax.set_title("V20 companion  Resource ledger (measured; private paths redacted; energy not fabricated)", loc="left", fontsize=9)
    ctx.save(fig, "ledger_table")
    ctx.run_ids = [f"final_{c}_s{s}" for c in (lock["candidate"], lock["comparator"]) for s in lock["final_seeds"]]
    hashes = hashes_from(ctx)
    p = pts["candidate"]
    cap = caption_text("Accuracy-resource Pareto and budget ledger", partition="locked test for KL; latency benchmark on cached test tensors",
                       counts=f"components {fmt(p['n_groups'])}; latency: 100 timed batches after 20 warm-up batches per setting; final fits {len(fits)}", seeds=f"KL averaged over seeds {lock['final_seeds']}; latency for the deployment seed {lock['deployment_seed']} weights",
                       metric="patient-KL (calibrated) versus median batch-32 cached-tensor inference latency in ms (bar to p95) at fixed hardware, batch and dtype; companion table lists memory, storage, GPU hours including failed jobs and end-to-end raw-to-prediction latency separately",
                       interval=f"KL: percentile 95% component bootstrap ({BOOT_REPLICATES} resamples, seed {BOOT_SEED}); latency: median and p95 of timed batches", hashes=hashes,
                       extra=f"Measurement boundary: forward pass on cached normalized tensors with CUDA synchronization; preprocessing is excluded from the plotted latency and reported separately; energy {lat.get('energy', 'NOT_MEASURED')}")
    return ctx.record(split="test", n_rows=None, n_patients=None, n_components=p["n_groups"], parameters={"points": {k: {kk: vv for kk, vv in v.items() if kk != "color"} for k, v in pts.items()}, "ledger_rows": rows}, caption=cap,
                      privacy_review="aggregate_only; resource data with private paths redacted")


# ======================================================================================
# Dispatcher
# ======================================================================================
RENDERERS = {"V01": render_v01, "V02": render_v02, "V03": render_v03, "V04": render_v04, "V05": render_v05, "V06": render_v06, "V07": render_v07, "V08": render_v08, "V09": render_v09, "V10": render_v10,
             "V11": render_v11, "V12": render_v12, "V13": render_v13, "V14": render_v14, "V15": render_v15, "V16": render_v16, "V17": render_v17, "V18": render_v18, "V19": render_v19, "V20": render_v20}
assert set(RENDERERS) == set(FIGURES_BY_ID)


def not_run_record(spec: FigureSpec, ws: Workspace, reason: str) -> dict:
    return {"figure_id": spec.figure_id, "title": spec.title, "question": spec.question, "data_hashes": {}, "run_ids": [], "split": None, "n_rows": None, "n_patients": None, "n_components": None,
            "generation_code_commit": git_commit(ws.repo), "parameters": {"required_inputs": list(spec.required), "policy": spec.policy}, "output_paths": [], "private_output_paths": [],
            "status": "NOT_RUN", "reason": reason, "privacy_review": "not_applicable", "caption": f"{spec.title}: NOT_RUN ({reason}).", "generated": utc_now()}


def _verify(ctx: FigureContext, rec: dict):
    if not ctx.output_paths:
        raise FigureVerificationError(f"{ctx.spec.figure_id}: renderer produced no output")
    for rel in ctx.output_paths:
        p = ctx.ws.repo / rel
        if not p.exists() or p.stat().st_size < 1000:
            raise FigureVerificationError(f"{ctx.spec.figure_id}: output {p.name} missing or empty")
    if not ctx.tracked:
        raise FigureVerificationError(f"{ctx.spec.figure_id}: no plotted values were tracked for the finite check")
    for name, arr in ctx.tracked.items():
        if arr.size and not np.all(np.isfinite(arr)):
            raise FigureVerificationError(f"{ctx.spec.figure_id}: non-finite plotted values in '{name}'")
    text = rec["caption"] + " " + json.dumps(rec["parameters"], default=str) + " " + " ".join(rec["output_paths"])
    if ID_PATTERN.search(text):
        raise FigureVerificationError(f"{ctx.spec.figure_id}: identifier field name leaked into public caption/parameters")
    for bad in (str(ctx.ws.root), str(Path.home())):
        if bad and bad in text:
            raise FigureVerificationError(f"{ctx.spec.figure_id}: filesystem path leaked into public caption/parameters")


def render_figure(figure_id: str, ws: Workspace, out_dir: Path | None = None, private_dir: Path | None = None, resolved: dict | None = None) -> dict:
    """Render one figure family. Missing required inputs -> NOT_RUN record and no files. Verification failures raise."""
    spec = FIGURES_BY_ID[figure_id]
    resolved = resolved if resolved is not None else resolve_artefacts(ws)
    missing = missing_inputs(spec, resolved)
    if missing:
        return not_run_record(spec, ws, "missing inputs: " + ", ".join(missing))
    out_dir = Path(out_dir or ws.figures_public); private_dir = Path(private_dir or ws.figures_private)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(f"{spec.stem}*"):
        if stale.suffix in (".png", ".svg"):
            stale.unlink()
    ctx = FigureContext(ws=ws, spec=spec, paths=resolved, out_dir=out_dir, private_dir=private_dir)
    rec = RENDERERS[figure_id](ctx)
    _verify(ctx, rec)
    rec["parameters"] = json.loads(json.dumps(rec["parameters"], default=_json_default))
    return rec


def _json_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, Path): return str(o)
    return str(o)
