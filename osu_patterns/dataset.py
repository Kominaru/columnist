"""The labelled category dataset shared by the boosted heads and the deep models.

`category_dataset` returns the cached chart statistics, slot probabilities, category labels
({1, 0, NaN} per map and category), the song groups used for cross-validation, and the boosted
heads' feature table, exactly as scripts/train_categories.py trains on them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import categories, mania_tags, slots


@dataclass
class CategoryData:
    stats: pd.DataFrame   # every map with cached statistics
    probs: pd.DataFrame   # slot probabilities (out-of-fold for labelled pool maps)
    Y: pd.DataFrame       # category labels for maps with at least one label
    groups: pd.Series     # song key per labelled map
    X: pd.DataFrame       # boosted-head features per labelled map
    slot: pd.Series       # tournament slot label (RC / LN / HB / SV) where known
    all_groups: pd.Series  # song key for every ranked chart and pool map
    votes: pd.Series      # beatmap_id -> {user tag: votes}


def category_dataset(data: str | Path = "data", slot_model: str | Path = "models/slots.pkl") -> CategoryData:
    d = Path(data)
    pools = pd.read_csv(d / "mappools.csv")
    table = slots.label_table(pools)
    if (d / "all_features.csv.gz").exists():  # every downloaded chart (scripts/build_all_features.py)
        allf = pd.read_csv(d / "all_features.csv.gz", index_col=0)
        stats = allf[[c for c in allf.columns if not c.startswith(("wmax:", "wp75:", "wstd:"))]]
    else:  # older checkouts: the pool and ranked tables
        stats = pd.concat([pd.read_csv(d / "mania_features.csv", index_col=0),
                           pd.read_csv(d / "ranked_features.csv", index_col=0)])
    stats = stats[~stats.index.duplicated()]

    # slot probabilities: out-of-fold for labelled pool maps, model output for everything else
    sm = slots.SlotModel.load(slot_model)
    probs = pd.DataFrame(sm.clf.predict_proba(stats[sm.features].values), index=stats.index, columns=sm.classes)
    oof = pd.read_csv(d / "slot_predictions.csv", index_col=0)[[f"p_{c}" for c in sm.classes]]
    oof.columns = sm.classes
    probs.update(oof)

    votes, group = {}, {}
    tag_rows = [json.loads(line) for line in open(d / "user_tags.jsonl")]
    ranked = [json.loads(line) for line in open(d / "ranked_mania.jsonl")]
    titles = {r["beatmapset_id"]: r["title"] for r in ranked}
    # every ranked chart is grouped by song, so pool rate edits and other sets of a song can't leak
    for r in ranked:
        for b in r["beatmaps"]:
            group[int(b["id"])] = slots.song_key(r["title"] + " [x]") or f"set{r['beatmapset_id']}"
    for j in tag_rows:
        for bid, tv in j["beatmaps"].items():
            votes[int(bid)] = tv
            group.setdefault(int(bid), slots.song_key(titles.get(j["beatmapset_id"], "") + " [x]")
                             or f"set{j['beatmapset_id']}")
    group.update(table.group.to_dict())

    Y = categories.combine(categories.labels_from_votes(pd.Series(votes)),
                           categories.labels_from_pools(pools, table.slot))
    Y = Y.loc[Y.index.intersection(stats.index)]
    moving = stats.loc[Y.index, "sv_log_std"] >= 0.1
    for c in categories.CATEGORIES:  # SV-only tags on maps without SV are probably storyboard gimmicks
        if c.sv_only and c.name in Y:
            Y.loc[(Y[c.name] == 1) & ~moving & ~Y.index.isin(pools.beatmap_id), c.name] = float("nan")
    Y = Y[Y.notna().any(axis=1)]
    X = pd.DataFrame.from_dict(
        {b: {**mania_tags.feature_vector(stats.loc[b].to_dict(), probs.loc[b].to_dict()),
             **{f"h:{k}": v for k, v in categories.heuristic_scores(stats.loc[b].to_dict()).items()}}
         for b in Y.index}, orient="index")
    groups = pd.Series({b: group.get(b, str(b)) for b in Y.index})
    return CategoryData(stats, probs, Y, groups, X, table.slot.dropna(), pd.Series(group), pd.Series(votes))


def song_folds(groups: pd.Series, n_splits: int = 5) -> pd.Series:
    """Fold number per map, grouped by song, so every model is scored on the same splits."""
    fold = pd.Series(-1, index=groups.index)
    for k, (_, te) in enumerate(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups)):
        fold.iloc[te] = k
    return fold


def chart_folds(groups: pd.Series, ids, n_splits: int = 5) -> pd.Series:
    """Song-grouped folds over a whole chart corpus (labelled or not), so a model trained or
    pretrained for fold k never sees a chart of a song in fold k's test set."""
    ids = pd.Index(ids)
    g = groups.reindex(ids)
    g = g.fillna(pd.Series(ids.astype(str), index=ids))
    return song_folds(g, n_splits)
