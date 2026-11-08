"""Numpy inference for the causal v3 network (osu_patterns.deep.v3.EventNet, causal=True): the Pattern Lens note model.

No torch at serving time. `NoteModelV3.scores(m)` returns a probability per note and per pattern task. Only pattern
tasks are served per note: chart characteristics (TB, Stamina) and the SV tasks stay chart-level / rule-based.
The per-event score of task t is s_it = w_t . h_i + b_t, the term the chart logit is made of
(chart logit = sum_i a_it s_it), and h_i depends only on events up to i.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import mania
from .deep import tokens, v3

LN_EPS = 1e-5
BLOCK = v3.BLOCK


def _ln(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + LN_EPS) * g + b


def _gelu(x):
    from math import sqrt
    return 0.5 * x * (1 + np.tanh(sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))   # tanh form; torch default is erf


def _erf_gelu(x):
    from scipy.special import erf  # noqa: PLC0415 - optional, exact
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


class NoteModelV3:
    def __init__(self, path: str | Path):
        path = Path(path)
        z = np.load(path, allow_pickle=False)
        self.p = {k: z[k] for k in z.files}
        self.n_conv, self.n_attn, self.heads = int(z["n_conv"]), int(z["n_attn"]), int(z["heads"])
        meta = json.loads(path.with_suffix(".json").read_text("utf-8"))
        self.tasks: list[str] = meta["tasks"]
        self.served: list[str] = meta.get("served", self.tasks)
        self.thresholds: dict[str, float] = meta.get("thresholds", {})
        self.report: dict = meta.get("report", {})
        try:
            self.gelu = _erf_gelu
            self.gelu(np.zeros(1))
        except Exception:  # noqa: BLE001 - no scipy: tanh approximation (max error ~1e-3)
            self.gelu = _gelu

    def _conv(self, x, w, b, d):
        L = len(x)
        pad = np.zeros((2 * d, x.shape[1]), x.dtype)
        xp = np.concatenate([pad, x])
        return xp[0:L] @ w[:, :, 0].T + xp[d:d + L] @ w[:, :, 1].T + xp[2 * d:2 * d + L] @ w[:, :, 2].T + b

    def _attn(self, i, h, t):
        p, H = self.p, self.heads
        k = f"attn.{i}."
        L, C = h.shape
        d = C // H
        qkv = _ln(h, p[k + "norm.weight"], p[k + "norm.bias"]) @ p[k + "qkv.weight"].T + p[k + "qkv.bias"]
        q, kk, vv = np.split(qkv, 3, axis=1)
        slope = _softplus(p[k + "slope"])                              # (H,)
        o = np.zeros_like(h)
        for b0 in range(0, L, BLOCK):
            qs = np.arange(b0, min(b0 + BLOCK, L))
            ks = np.arange(max(b0 - BLOCK, 0), qs[-1] + 1)               # previous block + own block, causal
            dt = t[qs][:, None] - t[ks][None]
            ok = (ks[None] <= qs[:, None]) & (np.abs(dt) <= v3.LOCAL_S)
            for hh in range(H):
                sl = slice(hh * d, (hh + 1) * d)
                sc = q[qs][:, sl] @ kk[ks][:, sl].T / np.sqrt(d) - slope[hh] * np.abs(dt)
                sc = np.where(ok, sc, -np.inf)
                sc -= sc.max(1, keepdims=True)
                w = np.exp(sc)
                w /= w.sum(1, keepdims=True)
                o[qs, sl] = w @ vv[ks][:, sl]
        h = h + (o @ p[k + "proj.weight"].T + p[k + "proj.bias"])
        f = _ln(h, p[k + "ff.0.weight"], p[k + "ff.0.bias"])
        f = self.gelu(f @ p[k + "ff.1.weight"].T + p[k + "ff.1.bias"]) @ p[k + "ff.4.weight"].T + p[k + "ff.4.bias"]
        return h + f

    def event_logits(self, ids: np.ndarray, tf: np.ndarray, t: np.ndarray) -> np.ndarray:
        """(L, T) per-event task logits; the next-gap features are zeroed as in training (causal)."""
        p = self.p
        tf = tf.copy()
        tf[:, v3.NEXT_COLS] = 0
        tm = self.gelu(tf @ p["time.0.weight"].T + p["time.0.bias"]) @ p["time.2.weight"].T + p["time.2.bias"]
        h = p["row.weight"][ids] + tm
        for i in range(self.n_conv):
            k = f"convs.{i}."
            y = self._conv(_ln(h, p[k + "norm.weight"], p[k + "norm.bias"]), p[k + "conv.weight"], p[k + "conv.bias"], 2 ** i)
            a, g = np.split(y, 2, axis=1)
            h = h + a * _sigmoid(g)
        for i in range(self.n_attn):
            h = self._attn(i, h, t)
        h = _ln(h, p["norm.weight"], p["norm.bias"])
        return h @ p["cls_w"].T + p["cls_b"]

    def scores(self, m: mania.ManiaMap) -> dict:
        ev = tokens.encode(m, visual=True)
        pr = v3.prepare(ev)
        s = self.event_logits(pr["ids"], pr["tf"], pr["t"].astype(np.float64))
        keep = [self.tasks.index(t) for t in self.served]
        t0 = float(min(m.start.min(), m.end[m.is_ln].min() if m.is_ln.any() else m.start.min()))
        ev_ms = ev.t.astype(np.float64) * 1000 + t0
        idx = np.clip(np.searchsorted(ev_ms, m.start + 2.0, side="right") - 1, 0, len(ev_ms) - 1)
        return {"tasks": self.served, "note": _sigmoid(s[idx][:, keep]), "event_t": ev_ms, "event": _sigmoid(s[:, keep])}
