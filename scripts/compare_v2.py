"""AP tables for the v2 experiments: Komi's categories and the 20 user tags, on the v2 folds.

    python scripts/compare_v2.py --deep scratch ssl ssl_stats full

Methods: rule, gbm, gbm+win (from teacher_cv.py), deep:<name> (train_deep_v2.py, all five folds),
and avg:<name>, the mean of gbm+win's and the deep model's probabilities (an ensemble that needs
no further fitting, so it can't leak).
"""
import argparse
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osu_patterns import categories, evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results/v2")
    ap.add_argument("--deep", nargs="*", default=[])
    a = ap.parse_args()
    res = Path(a.results)
    Y = pd.read_csv(Path(a.data) / "v2_labels.csv.gz", index_col=0).drop(columns="slot")
    oof = pd.read_csv(res / "oof.csv", index_col=0, header=[0, 1])
    preds = {m: oof[m] for m in oof.columns.levels[0] if not m.startswith("Unnamed")}
    for n in a.deep:
        parts = sorted(res.glob(f"deep_{n}_oof_f*.csv"))
        P = pd.concat([pd.read_csv(p, index_col=0) for p in parts])
        if len(parts) < 5:
            print(f"warning: deep:{n} has {len(parts)} of 5 folds; its column only covers those charts")
        preds[f"deep:{n}"] = P.reindex(Y.index)[Y.columns]
        preds[f"avg:{n}"] = (preds["gbm+win"] + preds[f"deep:{n}"]) / 2
    # score on charts every method covers
    keep = Y.index
    for P in preds.values():
        keep = keep.intersection(P.dropna(how="all").index)
    for name, cols in (("categories", [c for c in Y.columns if c in categories.NAMES]),
                       ("tags", [c for c in Y.columns if c.startswith("tag:")])):
        t = evaluate.ap_table(Y.loc[keep, cols], {m: P.loc[keep, cols] for m, P in preds.items()})
        print(f"\n== {name} ({len(keep)} charts) ==")
        print(t.to_string())
        t.to_csv(res / f"compare_{name}.csv")


if __name__ == "__main__":
    main()
