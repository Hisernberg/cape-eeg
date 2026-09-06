#!/usr/bin/env bash
# Final phase driver (after protocol lock): refit candidate and comparator on train+tune for the frozen epoch count,
# seeds 101/202/303 sequentially; optionally the exploratory P_MSF; then the one-time locked evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."
export CAPE_ROOT="${CAPE_ROOT:-$(cd .. && pwd)}"
export HMS_DATA_ROOT="${HMS_DATA_ROOT:-$CAPE_ROOT}" PYTHONWARNINGS=ignore
LOCK="$CAPE_ROOT/private/manifests/protocol_lock.json"; [ -f "$LOCK" ] || { echo "BLOCKED: no protocol lock"; exit 1; }
COMP=$(python3 -c "import json;print(json.load(open('$LOCK'))['comparator'])"); EP=$(python3 -c "import json;print(json.load(open('$LOCK'))['final_epochs'])")
CLR=$(python3 -c "import json;print(json.load(open('$LOCK'))['candidate_lr_mult'])"); BLR=$(python3 -c "import json;print(json.load(open('$LOCK'))['comparator_lr_mult'])")
CONFIGS="${CONFIGS:-P $COMP ${EXPLORATORY:-}}"
for cfg in $CONFIGS; do
  lr=$CLR; [ "$cfg" = "$COMP" ] && lr=$BLR
  for seed in 101 202 303; do
    echo "== final fit $cfg seed $seed epochs $EP lr x$lr"
    python3 scripts/train.py --config "$cfg" --phase final --seed "$seed" --epochs "$EP" --lr-mult "$lr" --stage final --max-minutes 70 2>&1 | grep -vE "Warning|warn|Found GPU|Minimum and|\(8.0\)|^\s*$"
  done
done
if [ "${SKIP_EVALUATION:-0}" = "1" ]; then echo "== locked evaluation skipped (SKIP_EVALUATION=1); run scripts/final_evaluate.py once via notebook 05"; exit 0; fi
PH=$(python3 -c "import json;print(json.load(open('$LOCK'))['protocol_hash'])")
if [ -f "$CAPE_ROOT/private/evaluation/$PH/summary.json" ]; then echo "== locked evaluation already exists for protocol $PH; not re-run"; exit 0; fi
echo "== locked evaluation"; python3 scripts/final_evaluate.py 2>&1 | grep -vE "Warning|warn|Found GPU|Minimum and|\(8.0\)|^\s*$"
