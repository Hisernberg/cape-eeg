"""Raw-EEG STFT, regional averaging, foveated/uniform binning and supplied-spectrogram binning.

All aggregation is overlap-weighted integration of *linear* power; the logarithm is applied
once, after integration. Validity is carried alongside every value.
"""
from __future__ import annotations

import numpy as np
from scipy.signal.windows import hann

from ..contracts import (STFT, EEG_SAMPLE_RATE_HZ, RAW_WINDOW_SAMPLES, RAW_FREQUENCY_RANGE_HZ, FREQUENCY_BINS,
                         FOCAL_TIME_BINS, CONTEXT_TIME_BINS, N_REGIONS, MAX_INTERPOLATION_GAP_SECONDS,
                         VALID_WINDOW_FRACTION, MINIMUM_VALID_BIPOLAR_LEADS, MIN_BIN_VALID_WEIGHT_FRACTION,
                         foveated_time_edges, uniform_time_edges, context_time_edges, linear_frequency_edges,
                         overlap_matrix)
from .montage import bipolar_index_pairs

LOG_FLOOR = 1e-8


class RawSpectralEncoder:
    """Deterministic 50 s raw window -> (foveated [4,64,32], uniform [4,64,32]) log-power + validity."""

    def __init__(self, columns, nperseg=STFT["nperseg"], hop=STFT["hop"], nfft=STFT["nfft"]):
        self.a, self.b = bipolar_index_pairs(columns)
        self.nperseg, self.hop, self.nfft = int(nperseg), int(hop), int(nfft)
        self.win = hann(self.nperseg, sym=False).astype(np.float32)
        self.n_frames = 1 + (RAW_WINDOW_SAMPLES - self.nperseg) // self.hop
        centers = (self.nperseg / 2 + self.hop * np.arange(self.n_frames)) / EEG_SAMPLE_RATE_HZ
        half = self.nperseg / 2 / EEG_SAMPLE_RATE_HZ
        self.frame_edges = np.concatenate([[centers[0] - half], centers + half])  # overlapping frames handled by overlap weights
        # frame i spans [centers_i - half, centers_i + half); overlap_matrix expects contiguous edges, so build explicitly
        self.frame_lo, self.frame_hi = centers - half, centers + half
        freqs = np.fft.rfftfreq(self.nfft, d=1.0 / EEG_SAMPLE_RATE_HZ)
        df = freqs[1] - freqs[0]
        src_f_edges = np.concatenate([[freqs[0] - df / 2], freqs + df / 2])
        dst_f_edges = linear_frequency_edges(FREQUENCY_BINS, *RAW_FREQUENCY_RANGE_HZ)
        Wf = overlap_matrix(src_f_edges, dst_f_edges)  # [64, n_freq]
        self.Wf = (Wf / Wf.sum(1, keepdims=True)).astype(np.float32)
        self.Wt_fov = self._time_weights(foveated_time_edges())
        self.Wt_uni = self._time_weights(uniform_time_edges())
        self.frequency_edges = dst_f_edges
        self.gap_samples = int(round(MAX_INTERPOLATION_GAP_SECONDS * EEG_SAMPLE_RATE_HZ))

    def _time_weights(self, edges: np.ndarray) -> np.ndarray:
        lo, hi = edges[:-1][:, None], edges[1:][:, None]
        W = np.clip(np.minimum(self.frame_hi[None, :], hi) - np.maximum(self.frame_lo[None, :], lo), 0.0, None)
        return W.astype(np.float32)  # [bins, frames], unnormalised

    # ---------------------------------------------------------------- preprocessing
    def _interpolate_small_gaps(self, x: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """Linear interpolation of interior gaps <= 0.25 s; longer gaps stay zero. Mask is untouched."""
        if obs.all():
            return x
        out = x.copy()
        n = x.shape[0]
        for c in range(x.shape[1]):
            m = obs[:, c]
            if m.all() or not m.any():
                continue
            idx = np.where(~m)[0]
            # split into runs
            runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            good = np.where(m)[0]
            for r in runs:
                if r[0] == 0 or r[-1] == n - 1 or r.size > self.gap_samples:
                    continue  # edge or too long: remain zero and unobserved
                out[r, c] = np.interp(r, good, x[good, c])
        return out

    def power(self, window: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Linear power [16 leads, n_freq, n_frames] and lead-frame validity [16, n_frames]."""
        x = self._interpolate_small_gaps(window, observed)
        obs_pair = observed[:, self.a] & observed[:, self.b]              # [T, 16]
        leads = (x[:, self.a] - x[:, self.b]).astype(np.float32)            # [T, 16]
        # median removal per valid lead (window-wise)
        for j in range(leads.shape[1]):
            m = obs_pair[:, j]
            if m.any():
                leads[m, j] -= np.median(leads[m, j])
            leads[~m, j] = 0.0
        idx = self.hop * np.arange(self.n_frames)[:, None] + np.arange(self.nperseg)[None, :]  # [F, nperseg]
        frames = leads.T[:, idx] * self.win[None, None, :]                   # [16, F, nperseg]
        spec = np.fft.rfft(frames, n=self.nfft, axis=-1)
        pw = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)           # [16, F, n_freq]
        pw = np.transpose(pw, (0, 2, 1))                                     # [16, n_freq, F]
        valid_frac = obs_pair.T[:, idx].mean(-1)                             # [16, F]
        lead_valid = valid_frac >= VALID_WINDOW_FRACTION
        return pw, lead_valid

    def regional(self, pw: np.ndarray, lead_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Average linear power across valid leads per region -> [4, n_freq, F]; region-frame validity [4, F]."""
        pw = pw.reshape(N_REGIONS, 4, pw.shape[1], pw.shape[2])
        lv = lead_valid.reshape(N_REGIONS, 4, -1).astype(np.float32)        # [4, 4, F]
        cnt = lv.sum(1)                                                       # [4, F]
        num = (pw * lv[:, :, None, :]).sum(1)                                 # [4, n_freq, F]
        reg = num / np.maximum(cnt, 1.0)[:, None, :]
        rvalid = cnt >= MINIMUM_VALID_BIPOLAR_LEADS
        return reg, rvalid

    def bin(self, reg: np.ndarray, rvalid: np.ndarray, Wt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Integrate [4, n_freq, F] linear power into [4, 64, bins]; log after integration; validity [4, 64, bins]."""
        fv = rvalid.astype(np.float32)                                        # [4, F]
        w = Wt[None, :, :] * fv[:, None, :]                                   # [4, bins, F] valid-only weights
        wsum = w.sum(-1)                                                      # [4, bins]
        total = Wt.sum(-1)[None, :]                                           # [1, bins]
        reg_f = np.einsum("kf,rft->rkt", self.Wf, reg)                        # [4, 64, F] frequency binning (mean)
        num = np.einsum("rkt,rbt->rkb", reg_f, w)                              # [4, 64, bins]
        val = num / np.maximum(wsum, 1e-12)[:, None, :]
        valid = (wsum >= MIN_BIN_VALID_WEIGHT_FRACTION * total)               # [4, bins]
        logp = np.log(np.maximum(val, LOG_FLOOR)).astype(np.float32)
        mask = np.repeat(valid[:, None, :], FREQUENCY_BINS, axis=1)
        logp[~mask] = 0.0
        return logp, mask

    def encode(self, window: np.ndarray, observed: np.ndarray) -> dict:
        pw, lead_valid = self.power(window, observed)
        reg, rvalid = self.regional(pw, lead_valid)
        fov, fov_m = self.bin(reg, rvalid, self.Wt_fov)
        uni, uni_m = self.bin(reg, rvalid, self.Wt_uni)
        return {"foveated": fov, "foveated_mask": fov_m, "uniform": uni, "uniform_mask": uni_m,
                "lead_valid_fraction": float(lead_valid.mean()), "region_valid_fraction": float(rvalid.mean())}


class ContextSpectralEncoder:
    """Supplied 600 s spectrogram rows -> [4, 64, 64] log-power + validity."""

    def __init__(self, columns):
        cols = [c for c in columns if c != "time"]
        self.region_cols, self.freqs = {}, None
        for region in ["LL", "RL", "LP", "RP"]:
            rc = [(float(c.split("_", 1)[1]), c) for c in cols if c.split("_", 1)[0] == region]
            rc.sort(key=lambda z: z[0])
            self.region_cols[region] = [c for _, c in rc]
            f = np.asarray([z for z, _ in rc])
            if self.freqs is None:
                self.freqs = f
            elif not np.allclose(self.freqs, f):
                raise ValueError(f"frequency axis differs for region {region}")
        if self.freqs is None or self.freqs.size == 0:
            raise ValueError("no regional columns found")
        self.col_index = {c: i for i, c in enumerate(columns)}
        self.region_idx = np.asarray([[self.col_index[c] for c in self.region_cols[r]] for r in ["LL", "RL", "LP", "RP"]])
        df = np.median(np.diff(self.freqs))
        src_edges = np.concatenate([[self.freqs[0] - df / 2], (self.freqs[1:] + self.freqs[:-1]) / 2, [self.freqs[-1] + df / 2]])
        self.frequency_range = (float(src_edges[0]), float(src_edges[-1]))
        dst_edges = linear_frequency_edges(FREQUENCY_BINS, *self.frequency_range)
        Wf = overlap_matrix(src_edges, dst_edges)
        self.Wf = (Wf / Wf.sum(1, keepdims=True)).astype(np.float32)
        self.frequency_edges = dst_edges
        self.time_edges = context_time_edges()

    def encode(self, values: np.ndarray, row_edges: np.ndarray) -> dict:
        """values: [n_rows, n_columns] of the selected rows (full source column order); row_edges: [n_rows+1] s rel. to s."""
        Wt = overlap_matrix(row_edges, self.time_edges).astype(np.float32)   # [64, n_rows]
        total = self.time_edges[1:] - self.time_edges[:-1]                    # [64]
        out = np.zeros((N_REGIONS, FREQUENCY_BINS, CONTEXT_TIME_BINS), dtype=np.float32)
        mask = np.zeros_like(out, dtype=bool)
        for r in range(N_REGIONS):
            x = values[:, self.region_idx[r]].astype(np.float32)             # [n_rows, n_src_freq]
            fin = np.isfinite(x) & (x >= 0)
            xf = np.where(fin, x, 0.0)
            xf = np.maximum(xf, 0.0)
            # frequency binning with validity-aware normalisation
            wf = fin.astype(np.float32) @ self.Wf.T                            # [n_rows, 64] valid weight
            num_f = xf @ self.Wf.T                                            # [n_rows, 64]
            val_f = num_f / np.maximum(wf, 1e-12)
            ok_f = wf >= MIN_BIN_VALID_WEIGHT_FRACTION                        # rows fully weighted =1
            # time binning
            wt = Wt @ ok_f.astype(np.float32)                                 # [64 t, 64 f] valid weight per cell
            num_t = Wt @ (val_f * ok_f)                                       # [64 t, 64 f]
            val = num_t / np.maximum(wt, 1e-12)
            ok = wt >= MIN_BIN_VALID_WEIGHT_FRACTION * total[:, None]
            logp = np.log(np.maximum(val, LOG_FLOOR))
            out[r] = np.where(ok, logp, 0.0).T
            mask[r] = ok.T
        return {"context": out, "context_mask": mask, "valid_fraction": float(mask.mean())}


def pack_mask(mask: np.ndarray) -> np.ndarray:
    return np.packbits(mask.reshape(mask.shape[0], -1), axis=-1, bitorder="little")


def unpack_mask(packed: np.ndarray, shape) -> np.ndarray:
    n = int(np.prod(shape[1:]))
    return np.unpackbits(packed, axis=-1, bitorder="little", count=n).reshape(packed.shape[0], *shape[1:]).astype(bool)
