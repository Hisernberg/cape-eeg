# Paper: NSysS 2026 manuscript

*When Compact Is Not Enough: A Pre-Registered, Byte-Budgeted Evaluation of Foveated Two-View EEG Models for Harmful Brain Activity Classification*

Target venue: 13th International Conference on Next Generation Computing, Communication, Systems and Security (NSysS 2026, Dhaka, 17–19 December 2026). Format: ACM primary article template (`acmart`, `sigconf`), full-paper limit 9 pages including references and appendix, double-blind submission.

| File | Purpose |
|---|---|
| `CAPE-EEG_NSysS2026_submission_anonymous.pdf` | double-blind version for the CMT submission (no author, anonymised repository link) |
| `CAPE-EEG_NSysS2026_author_version.pdf` | author version with name and repository link (camera-ready basis) |
| `CAPE-EEG_NSysS2026_latex_source.zip` | complete LaTeX source (this folder without `build/`) |
| `main.tex` | manuscript body; the author block and repository link switch on `\ANONYMOUS` |
| `submission.tex` / `camera.tex` | wrappers that build the two versions |
| `refs.bib` | bibliography (source register S01–S08 plus methods references) |
| `figures/` | the study's audited figures used in the paper (copied from `../figures/`) |
| `build.sh` | builds both PDFs with tectonic and prints page counts |

Build (no system LaTeX needed): `conda create -n tex -c conda-forge tectonic` then `bash build.sh`. The two TikZ diagrams (architecture, protocol) are drawn in `main.tex`. Every number in the manuscript comes from `../results/aggregate/` and `../EVALUATION_REPORT.md`.

Before submission: replace the affiliation placeholder in `main.tex` with the real institution and add an author e-mail; NSysS requires the anonymous PDF for review.
