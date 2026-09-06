#!/usr/bin/env bash
# Development phase driver: CPU baselines, then the named GPU configurations sequentially (one GPU job at a time),
# tune predictions for each, then the development table. Test partition is never touched here.
set -euo pipefail
cd "$(dirname "$0")/.."
export CAPE_ROOT="${CAPE_ROOT:-$(cd .. && pwd)}"
export HMS_DATA_ROOT="${HMS_DATA_ROOT:-$CAPE_ROOT}"
export PYTHONWARNINGS=ignore
CONFIGS="${CONFIGS:-B2 B3 A1 A2 P P_MSF}"
SEED="${SEED:-101}"; EPOCHS="${EPOCHS:-12}"; LRMULT="${LRMULT:-1.0}"; TAG="${TAG:-}"
echo "== CPU baselines"; python3 scripts/run_baselines_cpu.py 2>&1 | grep -vE "Warning|warn|Found GPU|Minimum and|\(8.0\)|^\s*$" || true
for cfg in $CONFIGS; do
  case $cfg in B2|B3) stage=baselines;; *) stage=ablations;; esac
  echo "== train $cfg (seed $SEED, epochs $EPOCHS, lr x$LRMULT)"
  python3 scripts/train.py --config "$cfg" --phase dev --seed "$SEED" --epochs "$EPOCHS" --lr-mult "$LRMULT" --stage "$stage" --tag "$TAG" 2>&1 | grep -vE "Warning|warn|Found GPU|Minimum and|\(8.0\)|^\s*$"
  run=$(ls -td "$CAPE_ROOT"/private/runs/dev_${cfg}_s${SEED}_* | head -1); run=$(basename "$run")
  python3 scripts/predict.py --run "$run" --partitions tune --views full 2>&1 | grep -vE "Warning|warn|Found GPU|Minimum and|\(8.0\)|^\s*$"
done
echo "== development table"; python3 scripts/evaluate_dev.py 2>&1 | grep -vE "Warning|warn|Found GPU|Minimum and|\(8.0\)|^\s*$"
