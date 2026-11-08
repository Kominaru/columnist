"""Train Komi's category heads (Chordjack, Streams, ..., Reading, Memo, TB).

    python scripts/train_categories.py --data data --out models/categories.pkl

Uses the feature caches written by train_slots.py (pool maps) and train_mania_tags.py (ranked
maps), assembled by osu_patterns.dataset. Pool maps get their out-of-fold slot probabilities from data/slot_predictions.csv, so the
heads never see a slot probability the slot model produced for its own training map.
Cross-validation is grouped by song. Prints each head's average precision next to the rule's.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # small data: sklearn's thread pools only add overhead here

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osu_patterns import categories, dataset  # noqa: E402
from osu_patterns.model import TagModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--slot-model", default="models/slots.pkl")
    ap.add_argument("--out", default="models/categories.pkl")
    a = ap.parse_args()

    data = dataset.category_dataset(a.data, a.slot_model)
    X, Y, groups = data.X, data.Y, data.groups
    print(f"{len(X)} maps with at least one category label, {groups.nunique()} songs")

    model = TagModel.fit(X, Y, groups, min_pos=12)
    rep = pd.DataFrame(model.report).T.reindex(categories.NAMES)
    print(rep[["n_labeled", "n_pos", "prevalence", "ap", "heuristic_ap", "auc", "f1", "threshold"]].to_string())
    for name, r in model.report.items():  # keep a head only where it beats the rule
        if r["heuristic_ap"] > r["ap"]:
            print(f"{name}: rule beats the model ({r['heuristic_ap']} vs {r['ap']} AP), using the rule")
            del model.heads[name]
            model.thresholds[name] = 0.5
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(a.out)


if __name__ == "__main__":
    main()
