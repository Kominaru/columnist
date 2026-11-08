"""Mean AP on the 4K charts only, for deep runs (and their seed means) against the first spec.

    python scripts/compare_4k.py full5X_s0+full5X_s1+full5X_s2 full5U_s0+full5U_s1+full5U_s2 --boot 500

A spec is a deep run name (results/v2/deep_<name>_oof_f*.csv) or several joined by "+", whose
probabilities are averaged; "gbm+win" and the other oof.csv methods work too. --boot gives a 95%
song-resample interval for each spec's difference to the first one.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osu_patterns import categories, evaluate  # noqa: E402


def load(spec, res, oof, index):
    if spec in oof.columns.levels[0]:
        return oof[spec].reindex(index)
    runs = [pd.concat([pd.read_csv(p, index_col=0) for p in sorted(res.glob(f"deep_{n}_oof_f*.csv"))]) for n in spec.split("+")]
    return (sum(r.reindex(index) for r in runs) / len(runs))


def mean_ap(Y, P, cols):
    return evaluate.ap_table(Y[cols], {"m": P[cols]}).loc["mean", "m"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results/v2")
    ap.add_argument("--keys", type=int, default=4)
    ap.add_argument("--boot", type=int, default=0)
    a = ap.parse_args()
    d, res = Path(a.data), Path(a.results)
    Y = pd.read_csv(d / "v2_labels.csv.gz", index_col=0).drop(columns="slot")
    keys = pd.read_csv(d / "all_features.csv.gz", index_col=0, usecols=["beatmap_id", "keys"])["keys"]
    Y = Y[keys.reindex(Y.index) == a.keys]
    oof = pd.read_csv(res / "oof.csv", index_col=0, header=[0, 1])
    preds = {s: load(s, res, oof, Y.index) for s in a.specs}
    keep = Y.index
    for P in preds.values():
        keep = keep.intersection(P.dropna(how="all").index)
    Y = Y.loc[keep]
    groups = {"categories": [c for c in Y.columns if c in categories.NAMES],
              "tags": [c for c in Y.columns if c.startswith("tag:")]}
    print(f"{a.keys}K charts with a label and every prediction: {len(keep)}")
    for name, cols in groups.items():
        t = evaluate.ap_table(Y[cols], {s: P.loc[keep, cols] for s, P in preds.items()})
        t.to_csv(res / f"compare_{a.keys}k_{name}.csv")
        print(f"\n== {name} ==\n{t.to_string()}")
    if a.boot:
        folds = pd.read_csv(d / "v2_folds.csv", index_col=0)["group"].reindex(keep).fillna("?")
        songs = folds.unique()
        rows_of = {g: np.where(folds.to_numpy() == g)[0] for g in songs}
        rng = np.random.default_rng(0)
        print(f"\n== difference to {a.specs[0]}, 95% interval over {a.boot} song resamples ==")
        for name, cols in groups.items():
            base = mean_ap(Y, preds[a.specs[0]].loc[keep], cols)
            for s in a.specs[1:]:
                diff, ds = mean_ap(Y, preds[s].loc[keep], cols) - base, []
                for _ in range(a.boot):
                    pick = rng.choice(songs, len(songs))
                    idx = pd.Index(np.concatenate([rows_of[g] for g in pick]))
                    yy = Y.iloc[idx].reset_index(drop=True)
                    pa = preds[a.specs[0]].loc[keep].iloc[idx].reset_index(drop=True)
                    pb = preds[s].loc[keep].iloc[idx].reset_index(drop=True)
                    ds.append(mean_ap(yy, pb, cols) - mean_ap(yy, pa, cols))
                lo, hi = np.percentile(ds, [2.5, 97.5])
                print(f"{name:10s} {s}: {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
