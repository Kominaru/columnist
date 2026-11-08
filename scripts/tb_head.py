"""Rule-like heads (TB, SV tasks): a boosted tree on the chart-statistics table (data/all_features.csv.gz, no snaps) for the one task that
is defined by duration, density and LN share, which a sequence net can't learn from ~450 positives.
Fold-safe: the fold-k model sees no chart of a fold-k test song.

    python scripts/tb_head.py [--small]          # writes results/v3/tbhead_oof.csv (beatmap_id, TB)
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.metrics import average_precision_score as ap

warnings.filterwarnings("ignore")
SMALL = ["duration_s", "nps", "ln_ratio", "wstd:ln_ratio", "wmax:nps_p90", "notes"]


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--small", action="store_true", help="six hand-picked statistics instead of the whole table")
    a.add_argument("--out", default="results/v3/tbhead_oof.csv")
    a = a.parse_args()
    Y = pd.read_csv("data/v6_labels.csv.gz", index_col=0)["TB"]
    F = pd.read_csv("data/all_features.csv.gz", index_col=0)
    fo = pd.read_csv("data/v2_folds.csv", index_col=0)["fold"]
    cols = SMALL if a.small else list(F.columns)
    ids = F.index.intersection(fo.index)
    p = pd.Series(np.nan, index=ids)
    for k in range(5):
        tr = ids[(fo.reindex(ids) != k).to_numpy()]
        te = ids[(fo.reindex(ids) == k).to_numpy()]
        y = Y.reindex(tr)
        m = y.notna().to_numpy()
        mdl = H(max_depth=3, max_iter=100, learning_rate=0.08).fit(F.reindex(tr)[cols][m], y[m].astype(int))
        p[te] = mdl.predict_proba(F.reindex(te)[cols])[:, 1]
    p.rename("TB").rename_axis("beatmap_id").to_csv(a.out, float_format="%.5g")
    y = Y.reindex(ids)
    m = y.notna()
    print(f"TB head ({len(cols)} features): out-of-fold AP {ap(y[m], p[m]):.3f} on {int(m.sum())} labelled charts")


if __name__ == "__main__":
    main()
