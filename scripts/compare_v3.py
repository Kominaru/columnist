"""Per-task AP of v3 against a deep v2 run on the same 4K charts, labels and folds, with a paired bootstrap over songs.

    python scripts/compare_v3.py v3a deep_fullS_s0 [--b2 deep_fullS_s1]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

SV = ("Reading", "Memo", "tag:sliders/complex sv", "tag:sliders/low sv", "tag:sliders/high sv")


def load(name, root, cols):
    d = Path(root)
    parts = sorted(d.glob(f"{name}_oof_f*.csv"))
    return pd.concat([pd.read_csv(p, index_col=0) for p in parts])[cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b", nargs="+", help="v2 run(s) in results/v2, averaged")
    ap.add_argument("--tag", default="v6")
    ap.add_argument("--a-dir", default="results/v3", help="where the first run's OOF files are (results/v2 for train_deep_v2 runs)")
    ap.add_argument("--only-4k-of", default="v3a", help="score only the 4K charts that this v3 run predicted (all-key runs)")
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    Y = pd.read_csv(f"data/{a.tag}_labels.csv.gz", index_col=0).drop(columns="slot")
    A = load(a.a, a.a_dir, Y.columns)
    k4 = pd.concat([pd.read_csv(p, index_col=0, usecols=[0]) for p in sorted(Path("results/v3").glob(f"{a.only_4k_of}_oof_f*.csv"))]).index
    A = A[A.index.isin(k4)]
    B = sum(load(n, "results/v2", Y.columns).reindex(A.index) for n in a.b) / len(a.b)
    Y = Y.reindex(A.index)
    g = pd.read_csv("data/sv_gate.csv.gz", index_col=0)["sve_effective"].reindex(Y.index).fillna(1.0).to_numpy()
    sv = [c for c in Y.columns if c in SV]
    for P in (A, B):
        P.loc[g == 0, sv] = 0.0
    Y.loc[g == 0, sv] = Y.loc[g == 0, sv].where(Y.loc[g == 0, sv].isna(), 0.0)
    grp = pd.read_csv("data/v2_folds.csv", index_col=0)["group"].reindex(Y.index).fillna(pd.Series(Y.index.astype(str), index=Y.index))
    codes, ug = pd.factorize(grp)
    rng = np.random.default_rng(0)
    rows = []
    for t in Y.columns:
        m = Y[t].notna().to_numpy()
        y, pa, pb, c = Y[t].to_numpy()[m], A[t].to_numpy()[m], B[t].to_numpy()[m], codes[m]
        d = average_precision_score(y, pa) - average_precision_score(y, pb)
        bs = []
        by = {k: np.where(c == k)[0] for k in np.unique(c)}
        keys = np.array(list(by))
        for _ in range(a.boot):
            ix = np.concatenate([by[k] for k in rng.choice(keys, len(keys))])
            if 0 < y[ix].sum() < len(ix):
                bs.append(average_precision_score(y[ix], pa[ix]) - average_precision_score(y[ix], pb[ix]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append((t, int(y.sum()), average_precision_score(y, pa), average_precision_score(y, pb), d, lo, hi))
    tb = pd.DataFrame(rows, columns=["task", "pos", "v3", "ref", "diff", "lo", "hi"]).set_index("task")
    tb["flag"] = np.where(tb.hi < 0, "WORSE", np.where(tb.lo > 0, "better", ""))
    print(tb.round(3).to_string())
    cat = [c for c in tb.index if not c.startswith("tag:")]
    tag = [c for c in tb.index if c.startswith("tag:")]
    print(f"\nmean AP categories v3 {tb.loc[cat,'v3'].mean():.3f} ref {tb.loc[cat,'ref'].mean():.3f}; tags v3 {tb.loc[tag,'v3'].mean():.3f} ref {tb.loc[tag,'ref'].mean():.3f}")


if __name__ == "__main__":
    main()
