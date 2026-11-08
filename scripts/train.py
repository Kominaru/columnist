"""Train and evaluate the learned tagger on data/ from build_dataset.py.

    python scripts/train.py --data data --model models/tagger.pkl

Prints a per-tag report (grouped-CV AP / AUC / F1 vs. the heuristic baseline) and saves the model.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osu_patterns.model import TagModel, build_features, make_labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="models/tagger.pkl")
    ap.add_argument("--min-votes", type=int, default=3)
    ap.add_argument("--min-total", type=int, default=15)
    a = ap.parse_args()

    d = Path(a.data)
    meta = pd.DataFrame([json.loads(l) for l in (d / "meta.jsonl").read_text().splitlines() if l]).set_index("beatmap_id")
    feat_cache = d / "features.pkl"
    if feat_cache.exists():
        X = pd.read_pickle(feat_cache)
    else:
        X = build_features({b: d / "osu" / f"{b}.osu" for b in meta.index})
        X.to_pickle(feat_cache)
    meta = meta.loc[meta.index.intersection(X.index)]
    X = X.loc[meta.index]

    tags = sorted({t for tv in meta["tags"] for t in tv})
    Y = make_labels(meta, tags, a.min_votes, a.min_total)
    model = TagModel.fit(X, Y, meta["beatmapset_id"])
    Path(a.model).parent.mkdir(parents=True, exist_ok=True)
    model.save(a.model)

    rep = pd.DataFrame(model.report).T.sort_values("ap", ascending=False)
    print(rep.to_string())
    print(f"\n{len(model.heads)} heads trained on {len(X)} difficulties from {meta['beatmapset_id'].nunique()} sets")


if __name__ == "__main__":
    main()
