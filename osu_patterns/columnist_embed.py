"""Numpy chart embeddings from the pretrained Columnist (no torch at serving time).

Same computation as embed_charts in scripts/foundation/columnist.py: the chart is cut into non-overlapping windows of
512 rows, every window runs through the bidirectional encoder, and the chart embedding is the mean of the per-row
outputs. The index additionally z-scores every dimension (mean/std of the whole index) and L2-normalises, so cosine
similarity is a dot product (the retrieval metric the Columnist was evaluated with).

Only the plain network is supported (no look-ahead or old-network inputs). Weights: `model.npz` (torch state_dict names,
float32, plus `heads`, `layers`, `d`).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .deep import columnist_rows as C
from .note_model_columnist import _ln, _sigmoid, _erf_gelu, _tanh_gelu, row_features

L_CROP = 512


class ColumnistEmbedder:
    def __init__(self, path: str | Path):
        z = np.load(path, allow_pickle=False)
        self.p = {(k[4:] if k.startswith("enc.") else k): z[k].astype(np.float32) for k in z.files}
        self.layers = int(z["layers"]) if "layers" in z.files else 1 + max(
            int(k.split(".")[1]) for k in self.p if k.startswith("blocks."))
        self.slopes = self.p["slopes"]
        self.heads = int(z["heads"]) if "heads" in z.files else len(self.slopes)
        self.d = self.p["tok.weight"].shape[1]
        try:
            _erf_gelu(np.zeros(1))
            self.gelu = _erf_gelu
        except Exception:  # noqa: BLE001
            self.gelu = _tanh_gelu

    def _block(self, i, x, t):
        p, H = self.p, self.heads
        k = f"blocks.{i}."
        L, D = x.shape
        dh = D // H
        qkv = _ln(x, p[k + "n1.weight"], p[k + "n1.bias"]) @ p[k + "qkv.weight"].T + p[k + "qkv.bias"]
        q, kk, vv = (qkv[:, j * D:(j + 1) * D] for j in range(3))
        idx = np.arange(L)
        ok = np.abs(idx[None] - idx[:, None]) <= C.WINDOW
        dt = np.abs(t[None, :] - t[:, None])
        o = np.empty_like(x)
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh)
            sc = q[:, sl] @ kk[:, sl].T / math.sqrt(dh) - self.slopes[h] * dt
            sc = np.where(ok, sc, -np.inf)
            sc -= sc.max(1, keepdims=True)
            w = np.exp(sc)
            w /= w.sum(1, keepdims=True)
            o[:, sl] = w @ vv[:, sl]
        x = x + (o @ p[k + "o.weight"].T + p[k + "o.bias"])
        f = _ln(x, p[k + "n2.weight"], p[k + "n2.bias"])
        f = self.gelu(f @ p[k + "ff.0.weight"].T + p[k + "ff.0.bias"]) @ p[k + "ff.2.weight"].T + p[k + "ff.2.bias"]
        return x + f

    def encode(self, tok, gap, sv):
        """One window of rows -> (L, d)."""
        p = self.p
        feats, t = row_features(tok, gap, sv)
        x = p["tok.weight"][tok.astype(np.int64)] + feats @ p["num.weight"].T + p["num.bias"]
        L = len(x)
        for j in range(2):
            w, b = p[f"stem.{j}.weight"], p[f"stem.{j}.bias"]
            xp = np.concatenate([np.zeros((4, x.shape[1]), x.dtype), x])
            y = sum(xp[k:k + L] @ w[:, :, k].T for k in range(5)) + b
            a, g = np.split(y, 2, axis=1)
            x = x + a * _sigmoid(g)
        for i in range(self.layers):
            x = self._block(i, x, t)
        return _ln(x, p["norm.weight"], p["norm.bias"])

    def embed_rows(self, tok, gap, sv) -> np.ndarray:
        """Chart embedding (unnormalised mean over windows, weighted by rows) from rows."""
        tok, gap, sv = np.asarray(tok), np.asarray(gap, np.float32), np.asarray(sv, np.float32)
        n = len(tok)
        acc = np.zeros(self.d, np.float64)
        for s in range(0, max(1, n), L_CROP):
            e = slice(s, min(s + L_CROP, n))
            acc += self.encode(tok[e], gap[e], sv[e]).sum(0)
        return (acc / max(1, n)).astype(np.float32)

    def embed_map(self, m) -> np.ndarray:
        tok, gap, sv, _, _ = C.map_rows(m)
        return self.embed_rows(tok, gap, sv)


class SimilarIndex:
    """Embeddings of the indexed charts plus their metadata; `query` returns the nearest ones."""

    def __init__(self, folder: str | Path):
        import pandas as pd
        f = Path(folder)
        if (f / "embeddings_q8.npy").exists():       # int8 copy (scale in embeddings_q8_scale.npy), 4x smaller to ship
            self.emb = np.load(f / "embeddings_q8.npy").astype(np.float32) * float(np.load(f / "embeddings_q8_scale.npy"))
        else:
            self.emb = np.load(f / "embeddings.npy").astype(np.float32)
        self.mean = np.load(f / "mean.npy").astype(np.float32)
        self.std = np.load(f / "std.npy").astype(np.float32)
        self.meta = pd.read_csv(f / "charts.csv")
        assert len(self.meta) == len(self.emb), "charts.csv and embeddings.npy differ in length"
        self.row_of = {int(b): i for i, b in enumerate(self.meta.beatmap_id)}
        self.embedder = ColumnistEmbedder(f / "model.npz")

    def normalise(self, e: np.ndarray) -> np.ndarray:
        z = (e - self.mean) / (self.std + 1e-9)
        return z / (np.linalg.norm(z) + 1e-9)

    def query(self, z: np.ndarray, k: int = 20, exclude: set[int] | None = None, skip_set: int | None = None,
              hide_copies: bool = True) -> list[dict]:
        sim = self.emb @ z
        ok = np.ones(len(sim), bool)
        if hide_copies and "dup_of" in self.meta:
            ok &= self.meta.dup_of.isna().to_numpy()
        if skip_set:
            ok &= (self.meta.beatmapset_id != skip_set).to_numpy()
        if exclude:
            ok &= ~self.meta.beatmap_id.isin(exclude).to_numpy()
        sim = np.where(ok, sim, -9)
        top = np.argsort(-sim)[:k]
        out = []
        for i in top:
            r = self.meta.iloc[int(i)]
            out.append({**{c: (None if (isinstance(v, float) and math.isnan(v)) else (v.item() if hasattr(v, "item") else v))
                           for c, v in r.items()}, "similarity": round(float(sim[i]), 4)})
        return out
