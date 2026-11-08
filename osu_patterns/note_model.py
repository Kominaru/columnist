"""Numpy inference for the note-level pattern model (osu_patterns.deep.local.LocalNet).

No torch at serving time: the NAS container stays small and its CPU (no x86-64-v2) can't run
recent wheels anyway. `NoteModel.scores(m)` returns, for every note of the chart, a probability per
task, plus the chart-level probabilities the same weights pool to.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import mania
from .deep import tokens


def _ln(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def _conv(x, w, b, d, causal=False):
    """Dilated conv1d, kernel 3, 'same' padding (causal: left padding only). x (L, C), w (2C, C, 3) -> (L, 2C)."""
    L = len(x)
    pad = np.zeros((d, x.shape[1]), x.dtype)
    xp = np.concatenate([pad, pad, x]) if causal else np.concatenate([pad, x, pad])
    return xp[0:L] @ w[:, :, 0].T + xp[d:d + L] @ w[:, :, 1].T + xp[2 * d:2 * d + L] @ w[:, :, 2].T + b


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NoteModel:
    def __init__(self, path: str | Path):
        path = Path(path)
        z = np.load(path, allow_pickle=False)
        self.p = {k: z[k].astype(np.float32) for k in z.files if k not in ("dilations", "causal")}
        self.dil = [int(d) for d in z["dilations"]]
        self.causal = bool(z["causal"]) if "causal" in z.files else False
        meta = json.loads(path.with_suffix(".json").read_text("utf-8"))
        self.tasks: list[str] = meta["tasks"]
        self.thresholds: dict[str, float] = meta["thresholds"]
        self.report: dict = meta.get("report", {})
        self.visual: bool = meta.get("visual", True)
        self.grid: bool = meta.get("grid", True)      # models trained before the grid was dropped have it

    def event_logits(self, x: np.ndarray) -> np.ndarray:
        p = self.p
        h = x @ p["inp.weight"].T + p["inp.bias"]
        for i, d in enumerate(self.dil):
            k = f"blocks.{i}."
            y = _conv(_ln(h, p[k + "norm.weight"], p[k + "norm.bias"]), p[k + "conv.weight"], p[k + "conv.bias"], d, self.causal)
            a, g = np.split(y, 2, axis=1)
            h = h + a * _sigmoid(g)
        return _ln(h, p["norm.weight"], p["norm.bias"]) @ p["out.weight"].T + p["out.bias"]

    def scores(self, m: mania.ManiaMap) -> dict:
        ev = tokens.encode(m, visual=self.visual)
        cont = ev.cont if self.grid else tokens.drop_grid(ev.cont)
        x = np.concatenate([tokens.pad_columns(ev.cols, ev.keys), cont.astype(np.float32)], 1)
        s = self.event_logits(x.astype(np.float32))                      # (L, T)
        # chart probabilities by the same auto-pooling the model was trained with
        a = np.clip(self.p["alpha"], 0, 8)
        w = np.exp(s * a - (s * a).max(0))
        w /= w.sum(0)
        chart = _sigmoid((w * s).sum(0))
        # events -> notes: a note takes the score of the event where it starts
        t0 = float(min(m.start.min(), m.end[m.is_ln].min() if m.is_ln.any() else m.start.min()))
        ev_ms = ev.t.astype(np.float64) * 1000 + t0
        idx = np.clip(np.searchsorted(ev_ms, m.start + 2.0, side="right") - 1, 0, len(ev_ms) - 1)
        return {"tasks": self.tasks, "note": _sigmoid(s[idx]), "chart": dict(zip(self.tasks, chart.tolist())),
                "event_t": ev_ms, "event": _sigmoid(s)}
