"""Tournament-slot classifier for osu!mania.

Labels
------
Official mappools on the osu! wiki give every map a slot. Raw slot names vary by tournament
("LN", "Long Note", "Long Notes"; JHC's "Dense Chordjack" is a rice sub-slot; everything in
o!mLN is LN). `normalise_slot` maps them onto the classic four:

    RC  rice (no or few long notes)          LN  long notes
    HB  hybrid (rice + LN mix)               SV  scroll-speed gimmicks

and keeps the raw name as a sub-slot so finer categories can be added later. FreeMod,
NoMod and Tiebreaker say nothing about patterns, so those maps are unlabelled.

Model
-----
Gradient boosting on mania.stats(), class-balanced, cross-validated with groups = song
(artist + title), so rate edits and re-uploads of one song never sit on both sides.
The rule-based `rule_slot` is the baseline it has to beat.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

SLOTS = ["RC", "LN", "HB", "SV"]

_MAIN = {
    "rice": "RC", "rc": "RC", "mixed rice": "RC", "accuracy": "RC",
    "ln": "LN", "long note": "LN", "long notes": "LN",
    "hybrid": "HB", "hb": "HB",
    "sv": "SV", "slider velocity": "SV",
}
# tournaments whose whole pool is one category
_TOURNAMENT_SLOT = {"o!mLN": "LN", "SVSE": "SV", "JHC": "RC"}
_SKIP_TOURNAMENTS = ("JOKE/",)  # 1K / joke tournaments
_NO_INFO = {"freemod", "free mod", "nomod", "tiebreaker", "tiebreaker:", "wildcard", "hidden"}


def normalise_slot(tournament: str, slot: str) -> str | None:
    s = slot.strip().lower()
    if tournament.startswith(_SKIP_TOURNAMENTS):
        return None
    series = tournament.split("/")[0]
    if series in _TOURNAMENT_SLOT:
        return _TOURNAMENT_SLOT[series]
    if s in _NO_INFO:
        return None
    return _MAIN.get(s)


def song_key(title: str) -> str:
    """'Artist - Title (Mapper) [Diff]' -> 'artist - title' for grouping."""
    t = re.sub(r"\s*\[[^\]]*\]\s*$", "", title or "")
    t = re.sub(r"\s*\([^)]*\)\s*$", "", t)
    return re.sub(r"\W+", " ", t.lower()).strip()


def label_table(pools: pd.DataFrame) -> pd.DataFrame:
    """One row per beatmap_id: slot (majority over all pools it appeared in), sub-slot, group.

    Maps whose pools disagree (e.g. HB in one tournament, LN in another) keep the majority;
    exact ties are dropped.
    """
    p = pools.copy()
    p["label"] = [normalise_slot(t, s) for t, s in zip(p.tournament, p.slot)]
    p = p[~p.tournament.str.startswith(_SKIP_TOURNAMENTS)]
    rows = []
    for bid, g in p.groupby("beatmap_id"):
        votes = g.label.dropna().value_counts()
        label = None
        if len(votes) and (len(votes) == 1 or votes.iloc[0] > votes.iloc[1]):
            label = votes.index[0]
        rows.append({"beatmap_id": bid, "slot": label,
                     "subslot": "; ".join(sorted(set(g.slot))),
                     "tournaments": "; ".join(sorted(set(g.tournament))),
                     "group": song_key(g.title.iloc[0]), "title": g.title.iloc[0]})
    return pd.DataFrame(rows).set_index("beatmap_id")


# ---------------------------------------------------------------- rule baseline

def rule_scores(s: dict) -> dict[str, float]:
    """Hand-written slotting rules, the way a pooler would eyeball a map."""
    ln = s["ln_ratio"]
    sv = min(1.0, s["sv_log_std"] / 0.3) * min(1.0, s["sv_changes_near_notes_ps"] / 0.5)
    return {
        "SV": sv,
        "LN": float(np.clip((ln - 0.3) / 0.3, 0, 1)),
        "HB": float(np.clip(1 - abs(ln - 0.25) / 0.15, 0, 1)),
        "RC": float(np.clip(1 - ln / 0.12, 0, 1)),
    }


def rule_slot(s: dict) -> str:
    sc = rule_scores(s)
    if sc["SV"] >= 0.5:
        return "SV"
    return max(("RC", "HB", "LN"), key=sc.get)


# ---------------------------------------------------------------- learned model

@dataclass
class SlotModel:
    features: list[str]
    clf: object
    classes: list[str]

    @classmethod
    def fit(cls, X: pd.DataFrame, y: pd.Series, **kw) -> "SlotModel":
        clf = make_classifier(**kw)
        clf.fit(X.values, y.values)
        return cls(list(X.columns), clf, list(clf.classes_))

    def predict_proba(self, stats: dict) -> dict[str, float]:
        x = np.array([[stats.get(f, np.nan) for f in self.features]])
        return dict(zip(self.classes, self.clf.predict_proba(x)[0].round(3)))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "SlotModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def make_classifier(max_iter: int = 300, learning_rate: float = 0.05, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=learning_rate,
                                          max_leaf_nodes=15, l2_regularization=1.0,
                                          class_weight="balanced", random_state=seed)


def cross_validate(X: pd.DataFrame, y: pd.Series, groups: pd.Series, folds: int = 5, seed: int = 0):
    """Out-of-fold probabilities, grouped by song."""
    from sklearn.model_selection import StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    classes = sorted(y.unique())
    oof = pd.DataFrame(0.0, index=X.index, columns=classes)
    for tr, te in cv.split(X, y, groups):
        clf = make_classifier(seed=seed)
        clf.fit(X.values[tr], y.values[tr])
        oof.iloc[te] = pd.DataFrame(clf.predict_proba(X.values[te]), columns=clf.classes_)[classes].values
    return oof
