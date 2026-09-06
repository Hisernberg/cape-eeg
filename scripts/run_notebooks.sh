#!/usr/bin/env bash
# Execute the seven notebooks in order with papermill. Executed copies go to private/notebooks_executed/ first;
# a copy whose outputs passed the identifier/secret scan is placed in notebooks/executed/ for the public repo.
set -euo pipefail
cd "$(dirname "$0")/.."
export CAPE_ROOT="${CAPE_ROOT:-$(cd .. && pwd)}"
export HMS_DATA_ROOT="${HMS_DATA_ROOT:-$CAPE_ROOT}" PYTHONWARNINGS=ignore
NBS="${NBS:-00_environment_and_intake 01_cohort_splits_and_cache 02_baselines_and_budget 03_ablation_and_freeze 04_final_training_and_calibration 05_locked_evaluation_and_figures 06_release_and_reproducibility}"
mkdir -p "$CAPE_ROOT/private/notebooks_executed" notebooks/executed
for nb in $NBS; do
  echo "== executing $nb"
  papermill "notebooks/$nb.ipynb" "$CAPE_ROOT/private/notebooks_executed/$nb.ipynb" --cwd "$(pwd)" -k python3 --log-output 2>&1 | grep -vE "Warning|warn|Found GPU|Minimum and|\(8.0\)|^\s*$" | tail -40
  python3 - "$CAPE_ROOT/private/notebooks_executed/$nb.ipynb" "notebooks/executed/$nb.ipynb" <<'PY'
import sys, json, re
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
nb = json.loads(src.read_text())
pat = re.compile(r"(ghp_[A-Za-z0-9]{30,}|KGAT_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{30,}|\b(patient_id|eeg_id|spectrogram_id|label_id)\b\s*[:=]\s*\d{5,})")
bad = 0
for c in nb.get("cells", []):
    if c.get("cell_type") == "code":
        c.get("metadata", {}).pop("papermill", None)
        for o in c.get("outputs", []):
            if pat.search(json.dumps(o)): bad += 1
nb.get("metadata", {}).pop("papermill", None)
if bad: print(f"executed copy withheld from the public repo: {bad} outputs matched identifier/secret patterns"); sys.exit(0)
dst.write_text(json.dumps(nb, indent=1)); print("public executed copy written:", dst.name)
PY
done
