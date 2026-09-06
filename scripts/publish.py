#!/usr/bin/env python3
"""Gated GitHub publication (docs/07 state machine). Credentials come only from the GH_TOKEN environment variable
of this process; the token is never written to disk, config, remote URL or logs.

Steps: precheck -> commit allowlisted files (explicit list, never `git add .`) -> history scan -> recheck hash
-> dry run -> push (only with --push) -> verify remote commit -> record publication_record.json.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import urllib.request
from cape_eeg.paths import resolve_workspace
from cape_eeg.status import write_json, read_json, Ledger, utc_now
from cape_eeg.release.export import precheck, git_history_scan, iter_allowed

ap = argparse.ArgumentParser()
ap.add_argument("--owner", required=True); ap.add_argument("--repo", required=True); ap.add_argument("--branch", default="main")
ap.add_argument("--visibility", choices=["private", "public"], default="private"); ap.add_argument("--push", action="store_true"); ap.add_argument("--message", default="CAPE-EEG: reproducible experiment release")
ap.add_argument("--author-name", default=None); ap.add_argument("--author-email", default=None)
a = ap.parse_args()
ws = resolve_workspace(); led = Ledger(ws.provenance / "ledger.jsonl"); repo = ws.repo
token = os.environ.get("GH_TOKEN")
def git(*args, check=True, capture=True):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=capture, text=True)
    if check and r.returncode != 0: raise RuntimeError(f"git {' '.join(args[:2])} failed: {r.stderr.strip()[:300]}")
    return r.stdout.strip()
def api(path, method="GET", data=None):
    req = urllib.request.Request(f"https://api.github.com{path}", method=method, data=json.dumps(data).encode() if data else None,
                                 headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "Content-Type": "application/json", "User-Agent": "cape-eeg-publisher"})
    try:
        with urllib.request.urlopen(req) as r: return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read() or b"{}")

rep = precheck(repo)
if rep["status"] != "PASS": sys.exit("BLOCKED: content precheck failed: " + "; ".join(rep["problems"][:5]))
if not (repo / ".git").exists():
    git("init", "-q", "-b", a.branch)
name = a.author_name or git("config", "user.name", check=False) or "CAPE-EEG"; email = a.author_email or git("config", "user.email", check=False) or "cape-eeg@users.noreply.github.com"
git("config", "user.name", name); git("config", "user.email", email)
files = [str(f.relative_to(repo)) for f in iter_allowed(repo) if not f.is_symlink()]
git("add", "--", *files)
if git("status", "--porcelain"):
    git("-c", f"user.name={name}", "-c", f"user.email={email}", "commit", "-q", "-m", a.message + "\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01PZGUMBsPB9SRBScXGhfKC9")
commit = git("rev-parse", "HEAD")
hist = git_history_scan(repo)
if hist["status"] != "PASS": sys.exit(f"BLOCKED: history scan {hist['status']} ({len(hist.get('hits', []))} hits)")
rep2 = precheck(repo)
if rep2["export_sha256"] != rep["export_sha256"]: sys.exit("BLOCKED: export hash changed between precheck and commit")
record = {"owner": a.owner, "repository": a.repo, "branch": a.branch, "visibility": a.visibility, "export_sha256": rep["export_sha256"], "content_scan": "PASS", "history_scan": "PASS",
          "commit": commit, "n_files": rep["n_files"], "total_bytes": rep["total_bytes"], "timestamp": utc_now(), "push": "DRY_RUN", "remote_commit_verified": False}
print(f"dry run: {rep['n_files']} files, {rep['total_bytes']/1e6:.1f} MB, commit {commit[:12]}, destination {a.owner}/{a.repo}@{a.branch} ({a.visibility})")
if a.push:
    if not token: sys.exit("BLOCKED: GH_TOKEN not present in the process environment")
    code, me = api("/user")
    if code != 200 or me.get("login", "").lower() != a.owner.lower(): sys.exit(f"BLOCKED: token identity {me.get('login')} != destination owner {a.owner}")
    code, existing = api(f"/repos/{a.owner}/{a.repo}")
    if code == 404:
        code, created = api("/user/repos", "POST", {"name": a.repo, "private": a.visibility == "private", "description": "CAPE-EEG: byte-budgeted foveated spectrogram learning with expert-disagreement audits on HMS harmful brain activity EEG", "has_wiki": False})
        if code not in (200, 201): sys.exit(f"BLOCKED: repository creation failed {code} {created.get('message')}")
        record["repository_created"] = True
    elif code == 200:
        if existing.get("private") != (a.visibility == "private"): sys.exit("BLOCKED: existing repository visibility differs from the approved visibility; no automatic visibility change")
        record["repository_created"] = False
    else: sys.exit(f"BLOCKED: repository lookup failed {code}")
    remote = f"https://github.com/{a.owner}/{a.repo}.git"
    if "origin" in git("remote").split(): git("remote", "set-url", "origin", remote)
    else: git("remote", "add", "origin", remote)
    # credential passed via an ephemeral askpass helper; never embedded in the remote URL
    helper = ws.approvals / ".askpass.sh"; helper.write_text("#!/bin/sh\ncase \"$1\" in *sername*) echo x-access-token;; *) echo \"$GH_TOKEN\";; esac\n"); helper.chmod(0o700)
    env = dict(os.environ, GIT_ASKPASS=str(helper), GIT_TERMINAL_PROMPT="0")
    r = subprocess.run(["git", "-C", str(repo), "push", "-u", "origin", f"HEAD:{a.branch}"], capture_output=True, text=True, env=env); helper.unlink(missing_ok=True)
    if r.returncode != 0: sys.exit("BLOCKED: push failed: " + r.stderr.strip()[-300:].replace(token, "***"))
    code, ref = api(f"/repos/{a.owner}/{a.repo}/commits/{a.branch}")
    record.update(push="PASS" if code == 200 and ref.get("sha") == commit else "FAIL", remote_commit_verified=(code == 200 and ref.get("sha") == commit), remote_url=f"https://github.com/{a.owner}/{a.repo}")
    print("push:", record["push"], "remote commit verified:", record["remote_commit_verified"], record.get("remote_url"))
write_json(ws.approvals / "publication_record.json", record)
led.append("I7_publication", "PASS" if record["push"] == "PASS" else ("NOT_RUN" if record["push"] == "DRY_RUN" else "FAIL"), f"push={record['push']}", commit=commit, export_sha256=rep["export_sha256"])
