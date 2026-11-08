"""Komi's tournament-style categories (2026-10-30) and where their labels come from.

    Rice     Chordjack · Streams · Jumpstream/Handstream · Technical · Mixed Rice · Stamina
             (Mixed Rice: rice skills from two or more of jacks / streams / tech)
    LN       Release · Coordination · Density/Inverse
    SV       Reading · Memo
    Hybrid   (no sub-categories)
    TB       a map that would fit a tiebreaker slot

The main slot (Rice / LN / Hybrid / SV) comes from slots.SlotModel. Everything else is a
multi-label head: a map can be Jumpstream/Handstream *and* Stamina *and* Technical at once.

Labels for each head come from two sources, OR-ed together:
* osu!(lazer) user tags on ranked maps (positive at >= 2 votes, negative only on diffs with
  >= 8 votes in total and none for the category's tags);
* tournament sub-slots on the osu! wiki (JHC chordjack slots, o!mLN Coordination / Density,
  SVSE Reading / Memorisation, every pool's Tiebreaker). A map in one sub-slot counts as a
  negative for the sibling sub-slots of the same tournament, since poolers slot by the dominant
  skill; Tiebreaker uses every other labelled pool map as a negative.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import mania_tags


@dataclass(frozen=True)
class Category:
    name: str
    parent: str           # RC / LN / SV / TB
    user_tags: tuple      # any of these voted -> positive
    pool_slots: tuple = ()  # (tournament prefix, raw slot name) pairs -> positive
    sv_only: bool = False   # user-tag positives only count on maps that really move scroll speed


CATEGORIES = [
    # Komi (2026-11-02): speedjacks and anchors are Chordjack, minijacks are Technical
    Category("Chordjack", "RC", ("style/chordjack", "skillset/speedjack"),
             tuple(("JHC/", s) for s in ("Dense Chordjack", "Middle Chordjack", "Light Chordjack", "Dense Jack",
                                         "Middle Jack", "Light Jack", "Full Jack", "Heavy Jack", "Difficult Jack",
                                         "Speed Jack", "Anchor Jack"))),
    Category("Streams", "RC", ("skillset/streams",)),
    Category("Jumpstream/Handstream", "RC",
             ("style/jumpstream", "style/handstream", "style/chordstream", "style/quadstream"),
             (("JHC/", "Quadstream"),)),
    Category("Technical", "RC", ("skillset/tech", "tech/complex snap"), (("JHC/", "Minijack"),)),
    # derived, no head: the "style/mixed rice" tag (97 votes) was too noisy to learn (AP 0.09)
    Category("Mixed Rice", "RC", ()),
    Category("Stamina", "RC", ("streams/stamina",), (("JHC/", "Anchor Stamina"),)),
    Category("Release", "LN", ("style/LN release",)),
    Category("Coordination", "LN", ("style/LN coordination",), (("o!mLN/", "Coordination"),)),
    Category("Density/Inverse", "LN", ("style/LN density", "gimmick/LN inverse"), (("o!mLN/", "Density"),)),
    # mania's "skillset/reading" tag is about dense patterns, not SV, so SV reading uses the SV tags
    Category("Reading", "SV", ("sliders/complex sv", "sliders/high sv", "sliders/low sv"), (("SVSE/", "Reading"),)),
    Category("Memo", "SV", ("gimmick/memory",), (("SVSE/", "Memorisation"),), sv_only=True),
    Category("TB", "TB", ("style/tiebreaker",),
             (("", "Tiebreaker"), ("", "TieBreaker"), ("", "Tiebreaker:"))),
]
NAMES = [c.name for c in CATEGORIES if c.user_tags or c.pool_slots]  # categories with a learned head
# Mixed Rice = skills from two or more families (jumpstream maps are nearly always "streams" too,
# so those two alone don't make a map mixed)
MIXED_FAMILIES = ({"Chordjack"}, {"Streams", "Jumpstream/Handstream"}, {"Technical"})
BY_NAME = {c.name: c for c in CATEGORIES}
PARENT_NAMES = {"RC": "Rice", "LN": "LN", "HB": "Hybrid", "SV": "SV"}


def heuristic_scores(s: dict) -> dict[str, float]:
    """Rule baseline per category, built from the per-tag rules in mania_tags."""
    h = mania_tags.heuristic_scores(s)
    rice = {
        "Chordjack": h["style/chordjack"],
        "Streams": h["skillset/streams"],
        "Jumpstream/Handstream": max(h["style/jumpstream"], h["style/handstream"], h["style/chordstream"]),
        "Technical": h["skillset/tech"],
    }
    return {
        **rice,
        "Stamina": h["streams/stamina"],
        "Release": h["style/LN release"],
        "Coordination": h["style/LN coordination"],
        "Density/Inverse": max(h["style/LN density"], h["gimmick/LN inverse"]),
        "Reading": h["sliders/complex sv"],
        "Memo": max(h["sliders/low sv"], h["sliders/high sv"]) * h["sliders/complex sv"],
        "TB": mania_tags._sq(s["duration_s"], 200, 25) * mania_tags._sq(s["nps"], 12, 3),
    }


def labels_from_votes(votes: pd.Series, min_votes: int = 2, min_total: int = 8) -> pd.DataFrame:
    """votes: beatmap_id -> {tag: count}. Returns {1, 0, NaN} per (map, category)."""
    out = pd.DataFrame(np.nan, index=votes.index, columns=NAMES)
    for bid, tv in votes.items():
        total = sum(tv.values())
        for c in map(BY_NAME.get, NAMES):
            v = max((tv.get(t, 0) for t in c.user_tags), default=0)
            if v >= min_votes:
                out.at[bid, c.name] = 1.0
            elif v == 0 and total >= min_total:
                out.at[bid, c.name] = 0.0
    return out


def labels_from_pools(pools: pd.DataFrame, labelled_slot: pd.Series) -> pd.DataFrame:
    """pools: rows of mappools.csv. labelled_slot: beatmap_id -> RC/LN/HB/SV (or NaN)."""
    ids = pools.beatmap_id.unique()
    out = pd.DataFrame(np.nan, index=ids, columns=NAMES)
    for c in map(BY_NAME.get, NAMES):
        if not c.pool_slots:
            continue
        hit = np.zeros(len(pools), bool)
        for prefix, slot in c.pool_slots:
            hit |= pools.tournament.str.startswith(prefix).values & (pools.slot == slot).values
        pos = set(pools.beatmap_id[hit])
        if c.name == "TB":
            neg = set(labelled_slot.dropna().index) - pos
        else:  # sibling sub-slots of the same tournaments
            tourneys = set(pools.tournament[hit])
            sib = pools[pools.tournament.isin(tourneys) & ~hit
                        & ~pools.slot.isin(["Tiebreaker", "TieBreaker", "FreeMod", "Wildcard"])]
            neg = set(sib.beatmap_id) - pos
        out.loc[list(neg), c.name] = 0.0
        out.loc[list(pos), c.name] = 1.0
    return out


def combine(*frames: pd.DataFrame) -> pd.DataFrame:
    """Positive anywhere wins; otherwise negative anywhere; otherwise unknown."""
    allf = pd.concat(frames)
    return allf.groupby(level=0).max()


# SV is a layer on top of the patterns, not a slot of its own for the categories (Komi, 2026-11-02): an
# SVLN burst is SV *and* LN, so SV categories never hide or suppress Rice / LN ones and vice versa.
# And SV never fires on a chart whose scroll speed never visibly changes (reading.scroll_effect).
SV_CATEGORIES = tuple(c.name for c in CATEGORIES if c.parent == "SV")
SV_TAGS = ("sliders/complex sv", "sliders/low sv", "sliders/high sv")
SV_KEYS = SV_CATEGORIES + SV_TAGS + tuple(f"tag:{t}" for t in SV_TAGS)
MEMO_KEYS = ("Memo",)  # per-note / per-window Memo also needs a visible speed step (reading.memo_note_gate)


def gate_sv(probs: dict[str, float], effective: bool) -> dict[str, float]:
    """Scores with every SV category / tag set to 0 when the chart has no effective scroll change."""
    if effective:
        return dict(probs)
    return {k: (0.0 if k in SV_KEYS else v) for k, v in probs.items()}


def gate_slot_probs(slot_probs: dict[str, float] | None, effective: bool) -> dict[str, float] | None:
    """Slot probabilities with SV removed (and the rest renormalised) when the scroll never changes."""
    if slot_probs is None or effective or "SV" not in slot_probs:
        return slot_probs
    rest = {k: float(v) for k, v in slot_probs.items() if k != "SV"}
    tot = sum(rest.values())
    out = {k: (v / tot if tot > 0 else 1 / len(rest)) for k, v in rest.items()}
    out["SV"] = 0.0
    return out


def base_slot(slot: str, slot_probs: dict[str, float] | None = None) -> str:
    """The pattern slot under the SV layer: RC / LN / HB. An SV-slot map takes its best other slot."""
    if slot != "SV":
        return slot
    if slot_probs:
        rest = {k: v for k, v in slot_probs.items() if k in ("RC", "LN", "HB")}
        if rest:
            return max(rest, key=rest.get)
    return "HB"  # unknown: Rice and LN sub-categories both at their own cuts


def slot_label(slot: str, slot_probs: dict[str, float] | None, shown: list[str]) -> str:
    """'LN + SV' style name: the pattern slot, plus SV when an SV category is shown."""
    base = PARENT_NAMES[base_slot(slot, slot_probs)]
    sv = slot == "SV" or any(n in SV_CATEGORIES for n in shown)
    return f"{base} + SV" if sv else base


def pick(cat_probs: dict[str, float], slot: str, thresholds: dict[str, float], tb_threshold: float | None = None,
         slot_probs: dict[str, float] | None = None, sv_effective: bool = True):
    """Categories to show for a map whose main slot is `slot`.

    SV categories (Reading, Memo) show whenever they pass their own cut and the chart has an effective
    scroll change, whatever the slot. Rice / LN sub-categories of the pattern slot (the slot itself, or
    for an SV-slot map its best Rice / LN / Hybrid slot) show at their cut, and sub-categories of the
    other family only when they fire strongly (a hybrid with real jumpstream should say so).
    TB is reported separately.
    """
    base = base_slot(slot, slot_probs)
    shown = []
    for c in CATEGORIES:
        if c.parent == "TB":
            continue
        p, thr = cat_probs.get(c.name, 0.0), thresholds.get(c.name, 0.5)
        if c.parent == "SV":
            if sv_effective and p >= thr:
                shown.append(c.name)
        elif c.parent == base or base == "HB" and c.parent in ("RC", "LN"):
            if p >= thr:
                shown.append(c.name)
        elif p >= max(thr, 0.8):
            shown.append(c.name)
    if base in ("RC", "HB") and sum(bool(fam & set(shown)) for fam in MIXED_FAMILIES) >= 2:
        shown.append("Mixed Rice")
    shown.sort(key=lambda n: -cat_probs.get(n, 1.0))
    tb = cat_probs.get("TB", 0.0) >= (tb_threshold if tb_threshold is not None else thresholds.get("TB", 0.5))
    return shown, tb


def predict(stats: dict, slot_probs: dict | None, model=None) -> dict[str, float]:
    """Probability per category head; the rule score where no head was kept."""
    h = heuristic_scores(stats)
    f = {**mania_tags.feature_vector(stats, slot_probs), **{f"h:{k}": v for k, v in h.items()}}
    if model is None:
        return {n: h[n] for n in NAMES}
    x = np.array([[f.get(c, np.nan) for c in model.columns]])
    return {n: float(model.heads[n].predict_proba(x)[0, 1]) if n in model.heads else h[n] for n in NAMES}
