"""Figure registry for the twenty planned visualization families (docs/05).

The registry is the single source of truth for figure identity, the artefact keys each
figure needs, and its export policy. The resolver maps artefact keys to concrete paths in
the workspace and reports which are missing; nothing here reads data or renders anything.
A figure whose required inputs are missing is never rendered: the manifest receives a
``NOT_RUN`` record with the missing keys as the reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..paths import Workspace
from ..status import read_json

__all__ = ["FigureSpec", "FIGURES", "FIGURES_BY_ID", "ARTEFACT_DESCRIPTIONS", "resolve_artefacts",
           "missing_inputs", "latest_cache_dir", "protocol_eval_dir", "run_dirs_with_events"]


@dataclass(frozen=True)
class FigureSpec:
    figure_id: str            # V01 .. V20
    slug: str                 # file stem after the id, e.g. "cohort_flow" -> figures/v01_cohort_flow.png
    title: str
    question: str
    required: tuple[str, ...]  # artefact keys that must resolve for the figure to be rendered
    optional: tuple[str, ...] = ()  # artefact keys used when present (companions, extra annotations)
    stage: str = "intake"     # intake | development | post_lock
    policy: str = "public_aggregate_only"
    companions: tuple[str, ...] = ()  # named companion outputs (suffixes) the renderer may emit

    @property
    def stem(self) -> str:
        return f"{self.figure_id.lower()}_{self.slug}"


ARTEFACT_DESCRIPTIONS = {
    "source_manifest": "private/provenance/source_manifest.json (file inventory, byte totals, intake exclusions)",
    "schema_report": "private/provenance/schema_report.json (train.csv schema gate: rows, patients, votes)",
    "preflight": "private/provenance/preflight.json (hardware/software snapshot)",
    "split_summary": "private/manifests/split_summary.json (per-partition rows/patients/components, split_hash)",
    "leakage_audit": "private/manifests/leakage_audit.json (5x5 partition intersection matrices)",
    "split_manifest": "private/manifests/split_manifest.parquet (label -> partition/component map)",
    "train_csv": "<data root>/train.csv (source label table; read-only)",
    "raw_votes": "expert vote counts per label: cache row_index.parquet when the cache exists, else train.csv",
    "cache_manifest": "private/cache/<preprocess_hash>/manifest.json",
    "row_index": "private/cache/<preprocess_hash>/row_index.parquet",
    "row_quality": "private/cache/<preprocess_hash>/row_quality.parquet",
    "conversion_resources": "private/cache/<preprocess_hash>/conversion_resources.json",
    "cache_shards": "private/cache/<preprocess_hash>/shards (finalized .npy shards)",
    "run_events": "private/runs/<run_id>/events.jsonl for at least one run with config.resolved.json",
    "gpu_ledger": "private/runs/gpu_ledger.jsonl",
    "development_table": "private/evaluation/development/development_table.csv",
    "protocol_candidate": "private/evaluation/development/protocol_candidate.json",
    "protocol_lock": "private/manifests/protocol_lock.json",
    "eval_summary": "private/evaluation/<protocol_hash>/summary.json",
    "bootstrap_primary": "private/evaluation/<protocol_hash>/bootstrap_primary.json",
    "strata": "private/evaluation/<protocol_hash>/strata.json",
    "robustness_evidence": "private/evaluation/<protocol_hash>/robustness_evidence.json",
    "robustness_subset": "private/evaluation/<protocol_hash>/robustness_subset.json",
    "latency": "private/evaluation/<protocol_hash>/latency.json",
    "resources": "private/evaluation/<protocol_hash>/resources.json",
    "test_predictions_deploy": "private/evaluation/<protocol_hash>/test_predictions/{candidate,comparator}_s<deployment_seed>.parquet",
    "test_predictions_all_seeds": "private/evaluation/<protocol_hash>/test_predictions/{candidate,comparator}_s<seed>.parquet for every final seed",
    "public_summary": "results/aggregate/final_evaluation_summary.json",
}


FIGURES: list[FigureSpec] = [
    FigureSpec("V01", "cohort_flow", "Cohort and exclusion flow",
               "Who entered the analysis, and who was excluded?",
               required=("source_manifest", "schema_report", "split_summary"),
               optional=("row_index", "row_quality", "cache_manifest", "eval_summary", "protocol_lock"),
               stage="intake", policy="public_aggregate_after_small_cell_suppression"),
    FigureSpec("V02", "partition_independence", "Partition independence matrix",
               "Are train, tune, calibration and test genuinely separated?",
               required=("leakage_audit", "split_summary"), stage="intake",
               policy="public_counts_only"),
    FigureSpec("V03", "label_distribution", "Six-class label distribution",
               "How much evidence supports each label and partition?",
               required=("raw_votes", "split_manifest"), optional=("split_summary",), stage="intake",
               policy="public_aggregates_with_suppressed_low_count_cells",
               companions=("vote_mass", "majority_counts")),
    FigureSpec("V04", "vote_ambiguity", "Vote count and ambiguity distribution",
               "How are annotation effort and disagreement related?",
               required=("raw_votes", "split_manifest"), optional=("split_summary",), stage="intake",
               policy="public_aggregate_no_label_ids"),
    FigureSpec("V05", "temporal_alignment", "Temporal alignment and foveated allocation",
               "Does the cache preserve the correct central ten seconds?",
               required=(), optional=("cache_manifest",), stage="intake",
               policy="public_synthetic_only_real_traces_require_rights_review"),
    FigureSpec("V06", "missingness_quality", "Missingness and signal-quality audit",
               "Does data quality differ across partitions or dominate errors?",
               required=("row_quality", "row_index", "cache_manifest"), optional=("strata", "protocol_lock", "split_summary"),
               stage="intake", policy="public_aggregate_per_example_quality_private",
               companions=("quality_vs_kl",)),
    FigureSpec("V07", "byte_budget", "Byte-budget representation comparison",
               "What information allocation is affordable without hiding storage costs?",
               required=("cache_manifest", "cache_shards", "development_table"), optional=("split_summary",),
               stage="development", policy="public_engineering_aggregate_no_cached_tensors",
               companions=("byte_breakdown",)),
    FigureSpec("V08", "learning_curves", "Learning curves and completed training exposure",
               "Did models converge under a comparable schedule?",
               required=("run_events",), optional=("gpu_ledger",), stage="development",
               policy="public_sanitized_curves_no_paths_or_sample_ids"),
    FigureSpec("V09", "paired_delta_kl", "Primary paired patient-KL improvement",
               "Is the proposed method better for the same held-out patients?",
               required=("protocol_lock", "test_predictions_all_seeds"), optional=("bootstrap_primary", "split_summary"),
               stage="post_lock", policy="public_distribution_summary_only_per_patient_values_private",
               companions=("raw",)),
    FigureSpec("V10", "roc_curves", "One-vs-rest ROC curves",
               "Which patterns are discriminable under unique-majority labels?",
               required=("protocol_lock", "test_predictions_deploy"), optional=("eval_summary",),
               stage="post_lock", policy="public_aggregate_curves", companions=("raw",)),
    FigureSpec("V11", "pr_curves", "One-vs-rest precision-recall curves",
               "Does minority-class discrimination survive prevalence effects?",
               required=("protocol_lock", "test_predictions_deploy"), optional=("eval_summary",),
               stage="post_lock", policy="public_aggregate", companions=("raw",)),
    FigureSpec("V12", "confusion_matrix", "Confusion matrix with support",
               "Which harmful-activity patterns are confused?",
               required=("protocol_lock", "test_predictions_deploy"), optional=("eval_summary",),
               stage="post_lock", policy="public_aggregate_with_small_cell_suppression",
               companions=("counts", "row_normalized", "row_normalized_comparator")),
    FigureSpec("V13", "calibration_reliability", "Calibration reliability",
               "Are stated probabilities consistent with expert vote frequencies?",
               required=("protocol_lock", "test_predictions_deploy", "eval_summary"),
               stage="post_lock", policy="public_binned_aggregate_only"),
    FigureSpec("V14", "disagreement_calibration", "Predicted versus observed disagreement",
               "Does the auxiliary head learn annotation ambiguity?",
               required=("protocol_lock", "test_predictions_deploy"), optional=("eval_summary",),
               stage="post_lock", policy="public_binned_aggregates_raw_votes_private"),
    FigureSpec("V15", "risk_coverage", "Review risk-coverage curves",
               "Can the model prioritize difficult cases without concealing referral volume?",
               required=("protocol_lock", "test_predictions_deploy", "eval_summary"),
               stage="post_lock", policy="public_aggregate_policy_curves"),
    FigureSpec("V16", "subgroup_forest", "Subgroup performance forest",
               "Are gains consistent across class, annotation support and missingness?",
               required=("protocol_lock", "strata", "bootstrap_primary"), optional=("split_summary",),
               stage="post_lock", policy="public_aggregate_with_suppression"),
    FigureSpec("V17", "corruption_sensitivity", "Preprocessing and corruption sensitivity",
               "Does performance remain stable under the fixed stress suite?",
               required=("protocol_lock", "robustness_evidence"), optional=("robustness_subset",),
               stage="post_lock", policy="public_aggregate_raw_perturbed_examples_private"),
    FigureSpec("V18", "evidence_deletion", "Evidence deletion and region dependence",
               "Does claimed important evidence actually affect the prediction?",
               required=("protocol_lock", "robustness_evidence"),
               stage="post_lock", policy="aggregate_public_case_maps_require_approval"),
    FigureSpec("V19", "mechanism_ablation", "Mechanism-ablation effect plot",
               "Which change contributes beyond extra parameters or favorable preprocessing?",
               required=("development_table",), optional=("protocol_lock", "bootstrap_primary"),
               stage="development", policy="public_sanitized_aggregates_and_exact_configs"),
    FigureSpec("V20", "accuracy_resource_pareto", "Accuracy-resource Pareto and budget ledger",
               "Is predictive improvement worth the real DGX cost?",
               required=("protocol_lock", "latency", "resources", "eval_summary", "test_predictions_all_seeds"),
               optional=("cache_manifest",), stage="post_lock",
               policy="public_aggregate_resource_data_private_paths_redacted", companions=("ledger_table",)),
]

FIGURES_BY_ID: dict[str, FigureSpec] = {f.figure_id: f for f in FIGURES}
assert len(FIGURES) == 20 and len(FIGURES_BY_ID) == 20, "the plan specifies exactly twenty figure families"


# --------------------------------------------------------------------------------------
# Resolver
# --------------------------------------------------------------------------------------
def _exists(p: Path | None) -> Path | None:
    return p if (p is not None and p.exists()) else None


def latest_cache_dir(ws: Workspace) -> Path | None:
    """The cache directory named by the protocol lock, else the newest finalized cache (has manifest.json)."""
    lock = read_json(ws.manifests / "protocol_lock.json")
    if lock and lock.get("preprocess_hash") and (ws.cache / lock["preprocess_hash"] / "manifest.json").exists():
        return ws.cache / lock["preprocess_hash"]
    cands = sorted(ws.cache.glob("*/manifest.json"))
    return cands[-1].parent if cands else None


def protocol_eval_dir(ws: Workspace) -> tuple[dict | None, Path | None]:
    lock = read_json(ws.manifests / "protocol_lock.json")
    if not lock or "protocol_hash" not in lock:
        return None, None
    d = ws.evaluation / lock["protocol_hash"]
    return lock, (d if d.exists() else None)


def run_dirs_with_events(ws: Workspace) -> list[Path]:
    out = []
    for d in sorted(ws.runs.glob("*")):
        if d.is_dir() and (d / "events.jsonl").exists() and (d / "config.resolved.json").exists():
            out.append(d)
    return out


def _predictions_complete(pred_dir: Path | None, lock: dict | None, seeds) -> Path | None:
    if pred_dir is None or lock is None or not pred_dir.exists():
        return None
    for cid in (lock.get("candidate"), lock.get("comparator")):
        if cid is None:
            return None
        for s in seeds:
            if not (pred_dir / f"{cid}_s{s}.parquet").exists():
                return None
    return pred_dir


def resolve_artefacts(ws: Workspace) -> dict[str, Path | None]:
    """Map every artefact key to an existing path (or None when missing)."""
    r: dict[str, Path | None] = {}
    r["source_manifest"] = _exists(ws.provenance / "source_manifest.json")
    r["schema_report"] = _exists(ws.provenance / "schema_report.json")
    r["preflight"] = _exists(ws.provenance / "preflight.json")
    r["split_summary"] = _exists(ws.manifests / "split_summary.json")
    r["leakage_audit"] = _exists(ws.manifests / "leakage_audit.json")
    r["split_manifest"] = _exists(ws.manifests / "split_manifest.parquet")
    r["train_csv"] = _exists(ws.data / "train.csv")
    cache = latest_cache_dir(ws)
    r["cache_manifest"] = _exists(cache / "manifest.json") if cache else None
    r["row_index"] = _exists(cache / "row_index.parquet") if cache else None
    r["row_quality"] = _exists(cache / "row_quality.parquet") if cache else None
    r["conversion_resources"] = _exists(cache / "conversion_resources.json") if cache else None
    shards = (cache / "shards") if cache else None
    r["cache_shards"] = shards if (shards and shards.exists() and any(shards.glob("shard_*_local.npy"))) else None
    r["raw_votes"] = r["row_index"] or r["train_csv"]
    runs = run_dirs_with_events(ws)
    r["run_events"] = ws.runs if runs else None
    r["gpu_ledger"] = _exists(ws.runs / "gpu_ledger.jsonl")
    r["development_table"] = _exists(ws.evaluation / "development" / "development_table.csv")
    r["protocol_candidate"] = _exists(ws.evaluation / "development" / "protocol_candidate.json")
    r["protocol_lock"] = _exists(ws.manifests / "protocol_lock.json")
    lock, ev = protocol_eval_dir(ws)
    for key, name in [("eval_summary", "summary.json"), ("bootstrap_primary", "bootstrap_primary.json"), ("strata", "strata.json"),
                      ("robustness_evidence", "robustness_evidence.json"), ("robustness_subset", "robustness_subset.json"),
                      ("latency", "latency.json"), ("resources", "resources.json")]:
        r[key] = _exists(ev / name) if ev else None
    pred_dir = (ev / "test_predictions") if ev else None
    r["test_predictions_deploy"] = _predictions_complete(pred_dir, lock, [lock["deployment_seed"]] if lock and "deployment_seed" in lock else [])
    r["test_predictions_all_seeds"] = _predictions_complete(pred_dir, lock, lock.get("final_seeds", []) if lock else [])
    r["public_summary"] = _exists(ws.results_public / "final_evaluation_summary.json")
    return r


def missing_inputs(spec: FigureSpec, resolved: dict[str, Path | None]) -> list[str]:
    return [k for k in spec.required if resolved.get(k) is None]
