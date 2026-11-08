"""Per-category average precision of every model on the same song-grouped folds.

    python scripts/compare_models.py --deep scratch pretrained --out results/compare.csv

Methods:
  rule        the hand-written rule score
  gbm         boosted heads on the 71 chart statistics (+ rule scores, slot probabilities)
  gbm+win     the same plus pooled sliding-window statistics (data/window_features.csv)
  deep:<n>    the deep event model's out-of-fold probability (results/deep_<n>_oof.csv)
  fuse:<n>    gbm+win plus the deep model's out-of-fold task and slot probabilities
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osu_patterns import dataset, evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--slot-model", default="models/slots.pkl")
    ap.add_argument("--windows", default="data/window_features.csv")
    ap.add_argument("--deep", nargs="*", default=[])
    ap.add_argument("--out", default="results/compare.csv")
    a = ap.parse_args()

    d = dataset.category_dataset(a.data, a.slot_model)
    Y = d.Y
    deep = {n: pd.read_csv(f"results/deep_{n}_oof.csv", index_col=0) for n in a.deep}
    # score every method on the maps every method has (charts that failed to parse are dropped)
    keep = Y.index
    for P in deep.values():
        keep = keep.intersection(P.index)
    win = pd.read_csv(a.windows, index_col=0) if Path(a.windows).exists() else None
    if win is not None:
        keep = keep.intersection(win.index)
    Y = Y.loc[keep]
    # the same folds train_deep.py uses: grouped over every labelled chart, then restricted
    groups = d.all_groups.reindex(Y.index).fillna(pd.Series(Y.index.astype(str), index=Y.index))
    fold = dataset.song_folds(groups)
    if deep:
        ref = pd.read_csv(f"results/deep_{a.deep[0]}_oof.csv", index_col=0).index
        g_all = d.all_groups.reindex(ref).fillna(pd.Series(ref.astype(str), index=ref))
        fold = dataset.song_folds(g_all).loc[Y.index]
    X = d.X.loc[Y.index]
    preds = {"rule": X[[f"h:{c}" for c in Y.columns]].set_axis(Y.columns, axis=1)}
    preds["gbm"] = evaluate.gbm_oof(X, Y, fold)
    if win is not None:
        Xw = X.join(win.loc[Y.index])
        preds["gbm+win"] = evaluate.gbm_oof(Xw, Y, fold)
    for n, P in deep.items():
        preds[f"deep:{n}"] = P.loc[Y.index].reindex(columns=Y.columns)
        base = Xw if win is not None else X
        preds[f"fuse:{n}"] = evaluate.gbm_oof(base.join(P.loc[Y.index].add_prefix("deep:")), Y, fold)
    t = evaluate.ap_table(Y, preds)
    print(t.to_string())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out)
    oof = pd.concat({k: v for k, v in preds.items()}, axis=1)
    oof.to_csv(Path(a.out).with_suffix(".oof.csv"))


if __name__ == "__main__":
    main()
