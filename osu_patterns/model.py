"""Learned tagger: predicts osu! user tags from map-level pattern features.

Framing
-------
* Multi-label classification, one binary head per user tag.
* Labels come from osu!(lazer) user-tag votes on ranked maps. Votes are positive-only
  and sparse: a missing tag on a barely-played map means "nobody voted", not "absent".
  So each (map, tag) pair is one of
      positive  votes >= min_votes
      negative  votes == 0 on a map with >= min_total votes in total (well annotated)
      unknown   everything else (masked out of that tag's loss)
* Features: the statistics from tagger.analyse() plus the heuristic scores (stacking),
  so the model can learn when the hand rules are wrong.
* Splits are grouped by beatmapset, so difficulties of one set never sit on both sides.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .parser import Beatmap, parse
from .tagger import TAGS, tag


def feature_vector(bm: Beatmap) -> dict[str, float]:
    res = tag(bm)
    feats = {k: float(v) for k, v in res.stats.items()}
    feats.update({f"h:{k}": v for k, v in res.scores.items()})
    return feats


def build_features(osu_paths: dict[int, Path]) -> pd.DataFrame:
    rows = {}
    for bid, p in osu_paths.items():
        try:
            bm = parse(p)
        except Exception as e:  # noqa: BLE001 - one broken file shouldn't stop a dataset build
            print(f"skip {bid}: {e}")
            continue
        if bm.mode != 0 or len(bm.objects) < 20:
            continue
        rows[bid] = feature_vector(bm)
    return pd.DataFrame.from_dict(rows, orient="index")


def make_labels(meta: pd.DataFrame, tags: list[str], min_votes: int = 3, min_total: int = 15) -> pd.DataFrame:
    """meta has a 'tags' column of {tag_name: votes}. Returns {1, 0, NaN} per (map, tag)."""
    out = pd.DataFrame(index=meta.index, columns=tags, dtype=float)
    for bid, tv in meta["tags"].items():
        total = sum(tv.values())
        for t in tags:
            v = tv.get(t, 0)
            out.at[bid, t] = 1.0 if v >= min_votes else (0.0 if v == 0 and total >= min_total else np.nan)
    return out


@dataclass
class TagModel:
    heads: dict[str, HistGradientBoostingClassifier] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)
    report: dict[str, dict] = field(default_factory=dict)

    # ------------------------------------------------------------------ training
    @classmethod
    def fit(cls, X: pd.DataFrame, Y: pd.DataFrame, groups: pd.Series, *, min_pos: int = 15,
            n_splits: int = 5, seed: int = 0) -> "TagModel":
        m = cls(columns=list(X.columns))
        for t in Y.columns:
            mask = Y[t].notna().to_numpy()
            y = Y[t].to_numpy()[mask].astype(int)
            if y.sum() < min_pos or (len(y) - y.sum()) < min_pos:
                continue
            Xt, gt = X.to_numpy()[mask], groups.to_numpy()[mask]
            oof = np.zeros(len(y))
            k = min(n_splits, len(np.unique(gt)))
            for tr_idx, te_idx in GroupKFold(n_splits=k).split(Xt, y, gt):
                clf = _make_clf(seed)
                if len(np.unique(y[tr_idx])) < 2:
                    oof[te_idx] = y[tr_idx].mean()
                    continue
                clf.fit(Xt[tr_idx], y[tr_idx])
                oof[te_idx] = clf.predict_proba(Xt[te_idx])[:, 1]
            thr = _best_threshold(y, oof)
            h_col = f"h:{t}"
            base_ap = average_precision_score(y, X.loc[mask, h_col]) if h_col in X else float("nan")
            m.report[t] = {
                "n_labeled": int(len(y)), "n_pos": int(y.sum()), "prevalence": round(float(y.mean()), 3),
                "ap": round(float(average_precision_score(y, oof)), 3),
                "auc": round(float(roc_auc_score(y, oof)), 3),
                "f1": round(float(f1_score(y, oof >= thr)), 3),
                "heuristic_ap": round(float(base_ap), 3),
                "threshold": round(float(thr), 3),
            }
            clf = _make_clf(seed)
            clf.fit(Xt, y)
            m.heads[t], m.thresholds[t] = clf, float(thr)
        return m

    # ------------------------------------------------------------------ inference
    def predict(self, bm: Beatmap) -> dict[str, float]:
        f = feature_vector(bm)
        x = np.array([[f.get(c, np.nan) for c in self.columns]])
        out = {}
        for t in TAGS:
            if t in self.heads:
                out[t] = float(self.heads[t].predict_proba(x)[0, 1])
            else:  # no trained head (too few labels): fall back to the heuristic score
                out[t] = f.get(f"h:{t}", 0.0)
        return out

    def decide(self, probs: dict[str, float]) -> list[str]:
        return [t for t, p in sorted(probs.items(), key=lambda kv: -kv[1]) if p >= self.thresholds.get(t, 0.5)]

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps(self))
        Path(path).with_suffix(".report.json").write_text(json.dumps(self.report, indent=2))

    @staticmethod
    def load(path: str | Path) -> "TagModel":
        return pickle.loads(Path(path).read_bytes())


def _make_clf(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=10, l2_regularization=1.0,
                                         class_weight="balanced", random_state=seed)


def _best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.unique(np.quantile(p, np.linspace(0.05, 0.95, 37)))
    if len(grid) == 0:
        return 0.5
    scores = [f1_score(y, p >= g) for g in grid]
    return float(grid[int(np.argmax(scores))])
