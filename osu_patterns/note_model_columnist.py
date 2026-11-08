"""Numpy inference for the causal Columnist per-note model (osu_patterns.deep.columnist.LensColumnist): the Lens note model.

No torch at serving time. `NoteModelColumnist.scores(m)` returns, for every note and pattern task, the probability the
model gives from the rows up to that note (rows <= the note's row; nothing after it), as note_model_v3 does. Chart
characteristics (TB, Stamina) and the scroll-speed tasks stay chart-level / rule-based, so they are not served per note.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import mania
from .deep import columnist_rows as C

LN_EPS = 1e-5
BLOCK = 128


def _ln(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + LN_EPS) * g + b


def _erf_gelu(x):
    from scipy.special import erf
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))


def _tanh_gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def row_features(tok: np.ndarray, gap: np.ndarray, sv: np.ndarray):
    """numpy twin of deep.columnist.row_features for one chart: (L,14) features and the time of every row in s."""
    tok = tok.astype(np.int64)
    states = np.stack([(tok >> (2 * c)) & 3 for c in range(4)], -1)
    delta = (states == 2).astype(np.int64) - (states == 3)
    held = np.clip(delta.cumsum(0) - delta, 0, 1).astype(np.float32)
    gap = np.nan_to_num(gap.astype(np.float32), nan=0.0, posinf=60000.0)
    g = np.log1p(np.clip(gap, 0, None)) / 6.9
    four = [f(g * (2.0 ** k) * math.pi) for k in range(4) for f in (np.sin, np.cos)]
    svf = np.log(np.clip(np.nan_to_num(sv.astype(np.float32), nan=1.0, posinf=10.0, neginf=0.1), 0.1, 10.0))
    feats = np.concatenate([g[:, None], np.stack(four, -1), svf[:, None], held], 1).astype(np.float32)
    return feats, np.cumsum(np.clip(gap, 0, None)) / 1000.0


class NoteModelColumnist:
    def __init__(self, path: str | Path):
        path = Path(path)
        z = np.load(path, allow_pickle=False)
        self.p = {k: z[k] for k in z.files}
        self.heads, self.layers = int(z["heads"]), int(z["layers"])
        meta = json.loads(path.with_suffix(".json").read_text("utf-8"))
        self.tasks: list[str] = meta["tasks"]
        self.served: list[str] = meta.get("served", self.tasks)
        self.thresholds: dict[str, float] = meta.get("thresholds", {})
        self.report: dict = meta.get("report", {})
        try:
            _erf_gelu(np.zeros(1))
            self.gelu = _erf_gelu
        except Exception:  # noqa: BLE001 - no scipy: the tanh form (max error ~1e-3)
            self.gelu = _tanh_gelu

    def _attn(self, i, x, t):
        p, H = self.p, self.heads
        k = f"enc.blocks.{i}."
        L, D = x.shape
        dh = D // H
        qkv = _ln(x, p[k + "n1.weight"], p[k + "n1.bias"]) @ p[k + "qkv.weight"].T + p[k + "qkv.bias"]
        q, kk, vv = (qkv[:, j * D:(j + 1) * D] for j in range(3))
        slopes = p["enc.slopes"]
        o = np.zeros_like(x)
        for b0 in range(0, L, BLOCK):
            qs = np.arange(b0, min(b0 + BLOCK, L))
            ks = np.arange(max(0, b0 - C.WINDOW), qs[-1] + 1)           # causal: back up to WINDOW rows
            ok = (ks[None] <= qs[:, None]) & (qs[:, None] - ks[None] <= C.WINDOW)
            dt = np.abs(t[ks][None] - t[qs][:, None])
            for h in range(H):
                sl = slice(h * dh, (h + 1) * dh)
                sc = q[qs][:, sl] @ kk[ks][:, sl].T / math.sqrt(dh) - slopes[h] * dt
                sc = np.where(ok, sc, -np.inf)
                sc -= sc.max(1, keepdims=True)
                w = np.exp(sc)
                w /= w.sum(1, keepdims=True)
                o[qs, sl] = w @ vv[ks][:, sl]
        x = x + (o @ p[k + "o.weight"].T + p[k + "o.bias"])
        f = _ln(x, p[k + "n2.weight"], p[k + "n2.bias"])
        f = self.gelu(f @ p[k + "ff.0.weight"].T + p[k + "ff.0.bias"]) @ p[k + "ff.2.weight"].T + p[k + "ff.2.bias"]
        return x + f

    def encode(self, tok, gap, sv):
        p = self.p
        feats, t = row_features(tok, gap, sv)
        x = p["enc.tok.weight"][tok.astype(np.int64)] + feats @ p["enc.num.weight"].T + p["enc.num.bias"]
        for j in range(2):
            w, b = p[f"enc.stem.{j}.weight"], p[f"enc.stem.{j}.bias"]       # (2d, d, 5)
            xp = np.concatenate([np.zeros((4, x.shape[1]), x.dtype), x])
            L = len(x)
            y = sum(xp[k:k + L] @ w[:, :, k].T for k in range(5)) + b
            a, g = np.split(y, 2, axis=1)
            x = x + a * _sigmoid(g)
        for i in range(self.layers):
            x = self._attn(i, x, t)
        return _ln(x, p["enc.norm.weight"], p["enc.norm.bias"])

    def row_outputs(self, tok, gap, sv):
        """(L,T) per-row task logits s and gate logits g."""
        p = self.p
        h = self.encode(tok, gap, sv)
        s = h @ p["cls_w"].T + p["cls_b"]
        T = s.shape[1]
        a = np.tanh(h @ p["V.weight"].T + p["V.bias"]) * _sigmoid(h @ p["U.weight"].T + p["U.bias"])
        g = np.einsum("ltk,tk->lt", a.reshape(len(h), T, -1), p["gw"])
        return s, g

    def scores(self, m: mania.ManiaMap) -> dict:
        tok, gap, sv, times, note_row = C.map_rows(m)
        s, g = self.row_outputs(tok, gap, sv)
        keep = [self.tasks.index(t) for t in self.served]
        w = np.exp(g - g.max(0))
        w /= w.sum(0)
        chart = _sigmoid((w * s).sum(0))
        t0 = float(times[0])
        return {"tasks": self.served, "note": _sigmoid(s[note_row][:, keep]),
                "chart": {t: float(chart[self.tasks.index(t)]) for t in self.served},
                "event_t": times.astype(np.float64), "event": _sigmoid(s[:, keep]), "t0": t0}
