"""CPU tests for the training engine, the resource supervisor, GPU lock/ledger and the status ledger."""

import json
import math
import os

import numpy as np
import pytest
import torch

from cape_eeg import model as M
from cape_eeg.contracts import vote_targets, pairwise_disagreement
from cape_eeg.status import Ledger, write_json, read_json
from cape_eeg.training import engine as E
from cape_eeg.training.supervisor import Supervisor, BudgetStop, GpuLock, GpuLedger

DEVICE = torch.device("cpu")


# ------------------------------------------------------------------ schedule
def test_lr_lambda_warmup_then_cosine_to_zero():
    total, warm = 100, 10
    vals = [E.lr_lambda(s, total, warm) for s in range(total + 1)]
    assert vals[0] == pytest.approx(1 / warm)
    assert vals[warm - 1] == pytest.approx(1.0)
    assert all(vals[i] < vals[i + 1] for i in range(warm - 1))          # strictly increasing warmup
    assert vals[warm] == pytest.approx(1.0)
    assert all(vals[i] >= vals[i + 1] for i in range(warm, total))      # non-increasing cosine
    assert vals[(total + warm) // 2] == pytest.approx(0.5, abs=1e-6)
    assert vals[total] == pytest.approx(0.0, abs=1e-12)
    assert E.lr_lambda(total + 50, total, warm) == pytest.approx(0.0, abs=1e-12)   # clamped past the end
    assert E.lr_lambda(0, 1, 1) == 1.0


# ------------------------------------------------------------------ augmentation
def _aug_batch(seed=0, n=6):
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    lm = torch.rand(n, 4, 64, 32, generator=g) < 0.7
    cm = torch.rand(n, 4, 64, 64, generator=g) < 0.7
    local = torch.randn(n, 4, 64, 32, generator=g) * lm
    context = torch.randn(n, 4, 64, 64, generator=g) * cm
    return {"local": local, "local_mask": lm, "context": context, "context_mask": cm}


def test_augment_keeps_invalid_cells_zero_and_only_touches_valid_ones():
    cfg = dict(E.DEFAULT_TRAINING)
    b = _aug_batch()
    orig = {k: v.clone() for k, v in b.items()}
    gen = torch.Generator(device="cpu"); gen.manual_seed(1)
    out = E.augment(b, cfg, gen)
    for view, mk in [("local", "local_mask"), ("context", "context_mask")]:
        x, m = out[view], out[mk]
        assert x.shape == orig[view].shape and x.device.type == "cpu"
        assert torch.all(x[~m] == 0.0)                          # invalid cells stay exactly zero
        assert torch.equal(m, orig[mk])                          # masks are untouched
        assert not torch.equal(x[m], orig[view][m])              # valid cells were modified
        assert torch.isfinite(x).all()
    # gain jitter alone shifts every valid cell of a sample by one scalar
    cfg_gain = {**cfg, "freq_mask_max": 0, "region_dropout_p": 0, "gain_jitter": 0.1}
    b2 = _aug_batch(seed=2); orig2 = {k: v.clone() for k, v in b2.items()}
    gen.manual_seed(2)
    out2 = E.augment(b2, cfg_gain, gen)
    for i in range(out2["local"].shape[0]):
        m = orig2["local_mask"][i]
        delta = (out2["local"][i] - orig2["local"][i])[m]
        assert torch.allclose(delta, delta[0].expand_as(delta), atol=1e-6)
        assert torch.all(out2["local"][i][~m] == 0.0)
    # no augmentation configured: exact identity
    cfg_id = {**cfg, "freq_mask_max": 0, "region_dropout_p": 0, "gain_jitter": 0.0}
    b3 = _aug_batch(seed=3); orig3 = {k: v.clone() for k, v in b3.items()}
    out3 = E.augment(b3, cfg_id, gen)
    assert torch.equal(out3["local"], orig3["local"]) and torch.equal(out3["context"], orig3["context"])
    # determinism under the generator
    ba, bb = _aug_batch(seed=4), _aug_batch(seed=4)
    ga = torch.Generator(device="cpu"); ga.manual_seed(9); gb = torch.Generator(device="cpu"); gb.manual_seed(9)
    assert torch.equal(E.augment(ba, cfg, ga)["local"], E.augment(bb, cfg, gb)["local"])


# ------------------------------------------------------------------ supervisor
def _relaxed(**kw):
    return Supervisor(cuda_stop_gib=1e6, rss_stop_gib=1e6, sys_min_available_gib=0.0, **kw)


def test_supervisor_check_passes_and_records_peaks():
    sup = _relaxed()
    s = sup.check()
    for key in ["elapsed_min", "rss_gib", "sys_available_gib"]:
        assert key in s and math.isfinite(s[key])
    assert s["rss_gib"] > 0 and s["elapsed_min"] >= 0
    assert sup.peak["rss_gib"] >= s["rss_gib"] and sup.peak["sys_available_min_gib"] <= s["sys_available_gib"]


def test_supervisor_stops_on_time_budget():
    sup = _relaxed(max_minutes=0.001)
    sup.t0 -= 60.0                                   # pretend a minute has elapsed
    with pytest.raises(BudgetStop, match="elapsed"):
        sup.check()
    ok = _relaxed(max_minutes=1e6)
    ok.check()


def test_supervisor_stops_on_rss_and_system_memory():
    with pytest.raises(BudgetStop, match="RSS"):
        Supervisor(cuda_stop_gib=1e6, rss_stop_gib=-1.0, sys_min_available_gib=0.0).check()
    with pytest.raises(BudgetStop, match="available memory"):
        Supervisor(cuda_stop_gib=1e6, rss_stop_gib=1e6, sys_min_available_gib=1e9).check()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not present; cuda budget branch not exercised")
def test_supervisor_stops_on_negative_cuda_budget():
    with pytest.raises(BudgetStop, match="CUDA"):
        Supervisor(cuda_stop_gib=-1.0, rss_stop_gib=1e6, sys_min_available_gib=0.0).check()


def test_gpu_lock_blocks_second_acquisition(tmp_path):
    path = tmp_path / "locks" / "gpu.lock"
    with GpuLock(path) as first:
        assert first.fh is not None and path.exists()
        assert path.read_text().split()[0] == str(os.getpid())
        with pytest.raises(RuntimeError, match="another GPU job"):
            with GpuLock(path):
                pass
    # released: a fresh acquisition succeeds
    with GpuLock(path):
        pass


def test_gpu_ledger_totals(tmp_path):
    led = GpuLedger(tmp_path / "gpu_ledger.jsonl", ceiling_hours=12.0)
    assert led.total_hours() == 0.0 and led.remaining_hours() == 12.0
    r = led.record("run_a", "train", 3600.0, "PASS", note="synthetic")
    assert r["hours"] == pytest.approx(1.0) and r["note"] == "synthetic" and "timestamp" in r
    led.record("run_b", "train", 1800.0, "INCOMPLETE")
    assert led.total_hours() == pytest.approx(1.5)
    assert led.remaining_hours() == pytest.approx(10.5)
    lines = [json.loads(l) for l in led.path.read_text().splitlines() if l.strip()]
    assert [l["run_id"] for l in lines] == ["run_a", "run_b"]


# ------------------------------------------------------------------ status ledger
def test_ledger_append_and_latest(tmp_path):
    led = Ledger(tmp_path / "ledger" / "status.jsonl")
    assert led.latest() == {}
    led.append("stage_a", "RUNNING")
    led.append("stage_b", "PASS", reason="done", rows=np.int64(5))
    led.append("stage_a", "FAIL", reason="boom")
    latest = led.latest()
    assert latest["stage_a"]["status"] == "FAIL" and latest["stage_a"]["reason"] == "boom"
    assert latest["stage_b"]["status"] == "PASS" and latest["stage_b"]["rows"] == 5
    assert len(led.path.read_text().splitlines()) == 3
    with pytest.raises(AssertionError):
        led.append("stage_c", "NOT_A_STATUS")


def test_write_json_is_atomic_and_roundtrips(tmp_path):
    path = tmp_path / "deep" / "dir" / "obj.json"
    obj = {"a": np.int64(3), "b": np.float32(0.5), "arr": np.arange(3), "p": tmp_path, "s": {2, 1}}
    write_json(path, obj)
    assert path.exists()
    assert not path.with_suffix(".json.partial").exists()
    assert not any(p.name.endswith(".partial") for p in path.parent.iterdir())
    back = read_json(path)
    assert back == {"a": 3, "b": 0.5, "arr": [0, 1, 2], "p": str(tmp_path), "s": [1, 2]}
    assert read_json(tmp_path / "missing.json", default="dflt") == "dflt"
    write_json(path, {"overwritten": True})
    assert read_json(path) == {"overwritten": True}
    with pytest.raises(TypeError):
        write_json(tmp_path / "bad.json", {"x": object()})
    assert not (tmp_path / "bad.json").exists()


# ------------------------------------------------------------------ tiny CPU fit
class SyntheticDataset:
    """In-memory stand-in exposing exactly what fit()/predict() use from CacheDataset."""

    def __init__(self, n=8, seed=0):
        rng = np.random.default_rng(seed)
        self.local = rng.normal(size=(n, 4, 64, 32)).astype(np.float32)
        self.context = rng.normal(size=(n, 4, 64, 64)).astype(np.float32)
        self.local_mask = rng.random((n, 4, 64, 32)) < 0.9
        self.context_mask = rng.random((n, 4, 64, 64)) < 0.9
        self.local *= self.local_mask; self.context *= self.context_mask
        votes = rng.integers(0, 3, size=(n, 6)); votes[:, 1] += 1
        votes[0] = [1, 0, 0, 0, 0, 0]                                # one n=1 row (d unavailable)
        self.votes = votes.astype(np.int64)
        self.q = vote_targets(self.votes).astype(np.float32)
        self.d = pairwise_disagreement(self.votes).astype(np.float32)
        self.n_votes = self.votes.sum(1)
        self.patient = np.repeat(np.arange(n // 2), 2)[:n]
        self.rows = np.arange(n)

    def __len__(self):
        return len(self.rows)

    def gather(self, sel):
        return {"local": self.local[sel], "local_mask": self.local_mask[sel], "context": self.context[sel],
                "context_mask": self.context_mask[sel],
                "valid_l": self.local_mask[sel].reshape(len(sel), -1).mean(1).astype(np.float32),
                "valid_c": self.context_mask[sel].reshape(len(sel), -1).mean(1).astype(np.float32),
                "q": self.q[sel], "votes": self.votes[sel], "d": self.d[sel], "n_votes": self.n_votes[sel], "pos": sel}

    def batches(self, batch_size, shuffle, seed=0, **kw):
        n = len(self)
        order = np.random.default_rng(seed).permutation(n) if shuffle else np.arange(n)
        for i in range(0, n, batch_size):
            yield self.gather(np.sort(order[i:i + batch_size]))


CFG = {"physical_batch": 4, "effective_batch": 4, "amp": "off", "augment": False, "warmup_steps": 1}


def test_fit_one_epoch_on_cpu_and_predict_order(tmp_path):
    torch.manual_seed(0)
    model = M.build_model("P")
    train_ds, tune_ds = SyntheticDataset(8, seed=1), SyntheticDataset(6, seed=2)
    run_dir = tmp_path / "run"
    logs = []
    sup = _relaxed()
    status = E.fit(model, train_ds, tune_ds, run_dir, CFG, seed=3, device=DEVICE, supervisor=sup, log=logs.append, epochs=1)
    assert status["status"] == "PASS" and status["complete_schedule"] is True
    assert status["epochs_completed"] == 1 and status["examples_seen"] == 8
    assert status["best"]["epoch"] == 1 and math.isfinite(status["best"]["tune_patient_kl"])
    assert status["config"]["physical_batch"] == 4 and status["config"]["amp"] == "off"
    on_disk = json.loads((run_dir / "status.json").read_text())
    assert on_disk["status"] == "PASS"
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines() if l.strip()]
    assert len(events) == 1
    ev = events[0]
    assert ev["epoch"] == 1 and ev["examples_seen"] == 8 and math.isfinite(ev["train_loss"])
    assert math.isfinite(ev["tune_patient_kl"]) and ev["n_rows"] == 6 and 0 <= ev["tune_gate_mean"] <= 1
    usage = (run_dir / "resource_usage.csv").read_text().splitlines()
    assert usage[0].startswith("epoch,elapsed_min,examples_seen") and len(usage) == 2
    ck = run_dir / "checkpoints"
    assert (ck / "last.safetensors").exists() or (ck / "last.pt").exists()
    assert (ck / "best.safetensors").exists() or (ck / "best.pt").exists()
    assert (ck / "last.pt").exists()
    assert len(logs) == 1 and logs[0].startswith("epoch 1/1")
    assert all(next(model.parameters()).device.type == "cpu" for _ in [0])

    # predict returns rows in original order and equals a direct forward pass
    res = E.predict(model, tune_ds, DEVICE, batch_size=4, amp="off")
    np.testing.assert_array_equal(res["pos"], np.arange(6))
    assert res["p"].shape == (6, 6) and res["d_hat"].shape == (6,)
    np.testing.assert_allclose(res["p"].sum(1), 1.0, atol=1e-5)
    b = tune_ds.gather(np.arange(6))
    with torch.no_grad():
        direct = model.eval()(torch.from_numpy(b["local"]), torch.from_numpy(b["context"]),
                              torch.from_numpy(b["valid_l"]), torch.from_numpy(b["valid_c"]))
    np.testing.assert_allclose(res["p"], direct["p"].numpy(), atol=1e-5)
    np.testing.assert_allclose(res["g"], direct["g"].numpy(), atol=1e-5)
    # a batch size that does not divide the length still restores the order
    res3 = E.predict(model, tune_ds, DEVICE, batch_size=4, amp="off", force_view="local")
    np.testing.assert_array_equal(res3["pos"], np.arange(6))
    assert np.all(res3["g"] == 0.0)

    # the saved weights reload into a fresh model and reproduce the predictions
    fresh = E.load_weights(M.build_model("P"), ck / "last.safetensors")
    res_fresh = E.predict(fresh, tune_ds, DEVICE, batch_size=4, amp="off")
    np.testing.assert_allclose(res_fresh["p"], res["p"], atol=1e-6)


def test_fit_reports_budget_stop_as_incomplete(tmp_path):
    torch.manual_seed(0)
    model = M.build_model("A1")
    ds = SyntheticDataset(8, seed=4)
    sup = _relaxed(max_minutes=0.001); sup.t0 -= 60.0
    status = E.fit(model, ds, None, tmp_path / "run_stop", CFG, seed=1, device=DEVICE, supervisor=sup, log=lambda *a: None, epochs=1)
    assert status["status"] == "INCOMPLETE" and status["complete_schedule"] is False
    assert "elapsed" in status["reason"]
    assert json.loads((tmp_path / "run_stop" / "status.json").read_text())["status"] == "INCOMPLETE"
    assert not (tmp_path / "run_stop" / "checkpoints" / "last.safetensors").exists()


def test_fit_without_tune_keeps_last_epoch(tmp_path):
    torch.manual_seed(0)
    model = M.build_model("B2")
    ds = SyntheticDataset(8, seed=5)
    status = E.fit(model, ds, None, tmp_path / "run_b2", CFG, seed=2, device=DEVICE, supervisor=_relaxed(), log=lambda *a: None, epochs=1)
    assert status["status"] == "PASS" and status["best"]["epoch"] is None
    assert (tmp_path / "run_b2" / "checkpoints" / "last.safetensors").exists()
    assert not (tmp_path / "run_b2" / "checkpoints" / "best.safetensors").exists()
    ev = json.loads((tmp_path / "run_b2" / "events.jsonl").read_text().strip())
    assert "tune_patient_kl" not in ev and ev["train_aux_mse"] == 0.0
