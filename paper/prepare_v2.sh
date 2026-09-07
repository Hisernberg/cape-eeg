#!/usr/bin/env bash
# After the post-lock development runs: last-epoch tune predictions for the head comparison, then analysis, figures and PDFs.
set -euo pipefail
cd "$(dirname "$0")/.."
export CAPE_ROOT="${CAPE_ROOT:-$(cd .. && pwd)}"; export HMS_DATA_ROOT="${HMS_DATA_ROOT:-$CAPE_ROOT}" PYTHONWARNINGS=ignore
for rd in "$CAPE_ROOT"/private/runs/dev_{A2,P,P_DM}_s*_*; do
  ep=$(python3 -c "import json;c=json.load(open('$rd/config.resolved.json'));print(c['epochs'], c.get('patient_fraction',1.0))")
  [ "$ep" = "3 1.0" ] || continue
  [ -f "$rd/predictions/tune_full_last.parquet" ] || python3 scripts/predict.py --run "$(basename "$rd")" --partitions tune --views full --weights last 2>&1 | grep -vE "Warning|warn|Found GPU|Minimum and|\(8.0\)|^\s*$"
done
cd paper && python3 v2_analysis.py && bash build.sh
