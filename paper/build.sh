#!/usr/bin/env bash
# Builds the double-blind submission PDF and the author version with tectonic (self-contained LaTeX engine).
set -euo pipefail
cd "$(dirname "$0")"
TEC="${TECTONIC:-$HOME/.conda/envs/tex/bin/tectonic}"
python3 multiseed_ablation.py >/dev/null 2>&1 || { echo "multiseed_ablation.py failed"; exit 1; }
python3 make_paper_figures.py >/dev/null 2>&1 || { echo "make_paper_figures.py failed"; exit 1; }
python3 v2_analysis.py >/dev/null 2>&1 || { echo "v2_analysis.py failed"; exit 1; }
for v in submission camera supplement; do
  "$TEC" --keep-intermediates --outdir build "$v.tex" 2>&1 | grep -vE "Requested font|^note:|^\s*$" || true
done
cp build/submission.pdf CAPE-EEG_NSysS2026_submission_anonymous.pdf
cp build/camera.pdf CAPE-EEG_NSysS2026_author_version.pdf
cp build/supplement.pdf CAPE-EEG_NSysS2026_supplement_anonymous.pdf
python3 - <<'PY'
from pypdf import PdfReader
for f in ["CAPE-EEG_NSysS2026_submission_anonymous.pdf", "CAPE-EEG_NSysS2026_author_version.pdf", "CAPE-EEG_NSysS2026_supplement_anonymous.pdf"]:
    print(f, "pages:", len(PdfReader(f).pages))
PY
