#!/usr/bin/env bash
# Builds the double-blind submission PDF and the author version with tectonic (self-contained LaTeX engine).
set -euo pipefail
cd "$(dirname "$0")"
TEC="${TECTONIC:-$HOME/.conda/envs/tex/bin/tectonic}"
for v in submission camera; do
  "$TEC" --keep-intermediates --outdir build "$v.tex" 2>&1 | grep -vE "Requested font|^note:|^\s*$" || true
done
cp build/submission.pdf CAPE-EEG_NSysS2026_submission_anonymous.pdf
cp build/camera.pdf CAPE-EEG_NSysS2026_author_version.pdf
python3 - <<'PY'
from pypdf import PdfReader
for f in ["CAPE-EEG_NSysS2026_submission_anonymous.pdf", "CAPE-EEG_NSysS2026_author_version.pdf"]:
    print(f, "pages:", len(PdfReader(f).pages))
PY
