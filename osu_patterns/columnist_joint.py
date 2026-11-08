"""Numpy inference for the shipped joint Columnist model (tags, categories, difficulty), no torch at serving time.

Mirrors joint_infer.JointModel.predict of the release package: windows of 512 rows run through the encoder
(ColumnistEmbedder.encode), then a gated-attention head per task, a chart-level length MLP and a row-level difficulty score.
Weights come from `joint.npz` (made by scripts/export_joint_npz.py) plus `joint.json` (tasks, thresholds, calibration).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .columnist_embed import ColumnistEmbedder, L_CROP
from .note_model_columnist import _sigmoid


class JointColumnist:
    def __init__(self, path: str | Path):
        path = Path(path)
        z = np.load(path, allow_pickle=False)
        self.enc = ColumnistEmbedder(path)                 # encoder weights are stored as enc.* keys
        self.h = {k[5:]: z[k].astype(np.float32) for k in z.files if k.startswith("head.")}
        meta = json.loads(path.with_suffix(".json").read_text("utf-8"))
        self.tasks: list[str] = meta["tasks"]
        self.thresholds: dict[str, float] = meta.get("thresholds", {})
        self.len_stats = meta["len_feat_stats"]
        self.topk = int(meta.get("topk", 16))
        self.cal = meta.get("calibration", {})
        self.n = len(self.tasks)

    def _len_feat(self, n_rows: int, gap: np.ndarray) -> np.ndarray:
        raw = np.array([math.log(max(1.0, math.ceil(n_rows / L_CROP))), math.log1p(float(np.sum(gap)) / 1000.0)])
        return ((raw - np.array(self.len_stats["mean"])) / np.array(self.len_stats["std"])).astype(np.float32)

    def stars(self, score: float) -> float | None:
        c = self.cal
        if "isotonic" in c:
            return float(np.interp(score, c["isotonic"]["x"], c["isotonic"]["y"]))
        if "intercept" in c:
            return float(c["intercept"] + c["slope"] * score)
        return None

    def predict(self, tok, gap, sv) -> dict:
        h, n = self.h, self.n
        tok = np.asarray(tok).astype(np.int64)
        gap = np.clip(np.nan_to_num(np.asarray(gap, np.float32)), 0, 60000)
        sv = np.asarray(sv, np.float32)
        N = len(tok)
        logits, wins, pats, rows = [], [], [], []
        for s in range(0, max(1, N), L_CROP):
            e = slice(s, min(s + L_CROP, N))
            x = self.enc.encode(tok[e], gap[e], sv[e])                       # (l, d)
            g = np.tanh(x @ h["v.weight"].T + h["v.bias"]) * _sigmoid(x @ h["u.weight"].T + h["u.bias"])
            g = g.reshape(len(x), n, 32)
            a = np.einsum("ltd,td->tl", g, h["w"])
            a = np.exp(a - a.max(1, keepdims=True))
            a /= a.sum(1, keepdims=True)                                      # (n, l)
            z = a @ x                                                          # (n, d)
            logits.append((z * h["cls"]).sum(-1) + h["b"])
            r = (x @ h["rank.weight"].T + h["rank.bias"])[:, 0]
            top = np.sort(r)[::-1][:self.topk]
            wins.append(float(top.mean()))
            pats.append((a * r[None]).sum(-1))
            rows.append(r)
        lf = self._len_feat(N, gap)
        wide = self.enc.gelu(lf @ h["wide.1.weight"].T + h["wide.1.bias"]) @ h["wide.4.weight"].T + h["wide.4.bias"]
        logit = np.mean(logits, 0) + wide
        prob = _sigmoid(logit)
        diff = max(wins)
        return {"tasks": self.tasks, "tag_prob": prob, "tag_logit": logit, "difficulty": diff, "stars": self.stars(diff),
                "window_difficulty": np.array(wins), "row_score": np.concatenate(rows),
                "pattern_difficulty": np.max(pats, 0),
                "tags": [t for t, p in zip(self.tasks, prob) if p >= self.thresholds.get(t, 0.5)]}
