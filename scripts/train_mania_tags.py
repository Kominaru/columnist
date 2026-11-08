"""Train the osu!mania pattern-tag heads on user-tag votes from ranked maps.

    python scripts/list_ranked_mania.py data/ranked_mania.jsonl
    python scripts/fetch_user_tags.py data/ranked_mania.jsonl data/user_tags.jsonl 3000
    python scripts/fetch_osu.py data/ranked_top.csv data/osu
    python scripts/train_mania_tags.py --tags data/user_tags.jsonl --osu data/osu \
        --slot-model models/slots.pkl --out models/mania_tags.pkl

Prints, per tag, the grouped-CV average precision of the learned head next to the
heuristic score's AP on the same maps (the rule baseline).
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # small data: sklearn's thread pools only add overhead here

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osu_patterns import mania, mania_tags, slots  # noqa: E402
from osu_patterns.model import TagModel, make_labels  # noqa: E402


def load_votes(path):
    rows = {}
    for line in open(path):
        j = json.loads(line)
        for bid, tags in j["beatmaps"].items():
            rows[int(bid)] = {"beatmapset_id": j["beatmapset_id"], "tags": tags}
    return pd.DataFrame.from_dict(rows, orient="index")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="data/user_tags.jsonl")
    ap.add_argument("--osu", default="data/osu")
    ap.add_argument("--features", default="data/ranked_features.csv")
    ap.add_argument("--slot-model", default="models/slots.pkl")
    ap.add_argument("--out", default="models/mania_tags.pkl")
    ap.add_argument("--min-votes", type=int, default=2)
    ap.add_argument("--min-total", type=int, default=8)
    a = ap.parse_args()

    meta = load_votes(a.tags)
    cache = Path(a.features)
    stats = pd.read_csv(cache, index_col=0) if cache.exists() else pd.DataFrame()
    new = {}
    for bid in meta.index.difference(stats.index):
        p = Path(a.osu) / f"{bid}.osu.gz"
        if p.exists() and p.stat().st_size > 40:
            m = mania.load(p)
            if m is not None and len(m.start) >= 50 and m.keys >= 4:
                new[bid] = mania.stats(m)
    if new:
        stats = pd.concat([stats, pd.DataFrame.from_dict(new, orient="index")])
        stats.index.name = "beatmap_id"
        cache.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(cache)
    meta = meta.loc[meta.index.intersection(stats.index)]
    slot_model = slots.SlotModel.load(a.slot_model)

    stats = stats.loc[meta.index]
    probs = pd.DataFrame(slot_model.clf.predict_proba(stats[slot_model.features].values),
                         index=stats.index, columns=slot_model.classes)
    X = pd.DataFrame.from_dict({b: mania_tags.feature_vector(stats.loc[b].to_dict(), probs.loc[b].to_dict())
                                for b in meta.index}, orient="index")
    Y = make_labels(meta, mania_tags.TAGS, a.min_votes, a.min_total)
    print(f"{len(X)} ranked mania maps with features from {meta.beatmapset_id.nunique()} sets")
    print(pd.DataFrame({"pos": (Y == 1).sum(), "neg": (Y == 0).sum()}).to_string())

    model = TagModel.fit(X, Y, meta.beatmapset_id, min_pos=12)
    rep = pd.DataFrame(model.report).T
    if rep.empty:
        sys.exit("no tag has enough labelled maps yet; fetch more user tags")
    print(rep[["n_labeled", "n_pos", "prevalence", "ap", "heuristic_ap", "auc", "f1", "threshold"]].to_string())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(a.out)


if __name__ == "__main__":
    main()
