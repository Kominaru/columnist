"""Paired bootstrap over songs: how much the plain-mean ensemble beats the baseline, per benchmark.

    python scripts/bootstrap_v2.py "boosted|oof:gbm+win+pat+x+relax" deep:<runs> deep:<runs> mlp:v5 \
        --oof results/v2/oof.csv results/v5_x/oof.csv

Components use the same specs as scripts/ensemble_v2.py. Songs (the fold groups) are resampled
with replacement; each resample recomputes every task's AP for the ensemble and for the baseline
on the same charts, and the benchmark's mean AP difference. Prints the full-data difference, its
95% interval and the share of resamples at or above +0.05.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as aps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osu_patterns import categories  # noqa: E402

spec = importlib.util.spec_from_file_location("ens", Path(__file__).with_name("ensemble_v2.py"))
ens = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("components", nargs="+")
    ap.add_argument("--data", default="data")
    ap.add_argument("--labels", default="v2_labels.csv.gz", help="strict labels to score on (v6_labels.csv.gz: "
                    "Komi's 2026-11-02 jack mapping)")
    ap.add_argument("--results", default="results/v2")
    ap.add_argument("--oof", nargs="*", default=["results/v2/oof.csv"])
    ap.add_argument("--boosted", nargs="+", default=["gbm", "gbm+win", "gbm+win+pat", "gbm+win+relax", "gbm+win+pat+relax"])
    ap.add_argument("--baseline", default="gbm+win")
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    d = Path(a.data)
    Y = pd.read_csv(d / a.labels, index_col=0).drop(columns="slot")
    folds = pd.read_csv(d / "v2_folds.csv", index_col=0).reindex(Y.index)
    grp = folds["group"].fillna(pd.Series(Y.index.astype(str), index=Y.index)).to_numpy()
    oof = pd.concat([pd.read_csv(p, index_col=0, header=[0, 1]) for p in a.oof], axis=1)
    oof = oof.loc[:, ~oof.columns.duplicated()].reindex(Y.index)
    comps = [ens.load(s, Y, Path(a.results), oof, a.boosted) for s in a.components]
    E = sum(comps) / len(comps)
    base = oof[a.baseline][Y.columns]
    rng = np.random.default_rng(0)
    ug = np.unique(grp)
    gi = {g: np.where(grp == g)[0] for g in ug}
    for name, cs in (("categories", [c for c in Y.columns if c in categories.NAMES]),
                     ("tags", [c for c in Y.columns if c.startswith("tag:")])):
        yv = {t: Y[t].to_numpy() for t in cs}
        ev = {t: E[t].to_numpy() for t in cs}
        bv = {t: base[t].to_numpy() for t in cs}

        def diff(idx):
            out = []
            for t in cs:
                y, e, b = yv[t][idx], ev[t][idx], bv[t][idx]
                m = ~np.isnan(y) & ~np.isnan(e) & ~np.isnan(b)
                if y[m].sum() == 0:
                    continue
                out.append(aps(y[m], e[m]) - aps(y[m], b[m]))
            return float(np.mean(out))

        full = diff(np.arange(len(Y)))
        boot = np.array([diff(np.concatenate([gi[g] for g in rng.choice(ug, len(ug))])) for _ in range(a.n)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"{name}: +{full:.4f}  95% interval [{lo:+.4f}, {hi:+.4f}]  share of resamples >= +0.05: {np.mean(boot >= 0.05):.2f}",
              flush=True)


if __name__ == "__main__":
    main()
