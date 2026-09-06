"""Workspace path resolution. Data stays read-only outside repo/; artifacts go to private/."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Workspace:
    root: Path
    data: Path
    private: Path
    repo: Path

    @property
    def provenance(self): return self.private / "provenance"
    @property
    def manifests(self): return self.private / "manifests"
    @property
    def cache(self): return self.private / "cache"
    @property
    def normalization(self): return self.private / "normalization"
    @property
    def runs(self): return self.private / "runs"
    @property
    def evaluation(self): return self.private / "evaluation"
    @property
    def figures_private(self): return self.private / "figures_private"
    @property
    def approvals(self): return self.private / "approvals"
    @property
    def executed_notebooks(self): return self.private / "notebooks_executed"
    @property
    def results_public(self): return self.repo / "results" / "aggregate"
    @property
    def figures_public(self): return self.repo / "figures"

    def ensure(self):
        for p in [self.provenance, self.manifests, self.cache, self.normalization, self.runs,
                  self.evaluation, self.figures_private, self.approvals, self.executed_notebooks,
                  self.results_public, self.figures_public]:
            p.mkdir(parents=True, exist_ok=True)
        return self


def resolve_workspace() -> Workspace:
    """CAPE_ROOT holds private/; HMS_DATA_ROOT holds train.csv etc. Defaults: repo parent."""
    root = Path(os.environ.get("CAPE_ROOT", REPO_DIR.parent)).resolve()
    data = Path(os.environ.get("HMS_DATA_ROOT", root)).resolve()
    ws = Workspace(root=root, data=data, private=root / "private", repo=REPO_DIR)
    return ws.ensure()


def redact(path: Path | str, ws: Workspace | None = None) -> str:
    """Render a path relative to the workspace for logs so personal absolute paths are not printed."""
    p = str(path)
    ws = ws or resolve_workspace()
    return p.replace(str(ws.root), "$CAPE_ROOT").replace(str(Path.home()), "~")
