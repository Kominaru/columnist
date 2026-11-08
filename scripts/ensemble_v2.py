"""The v2/v3 ensemble: boosted variants, deep runs and MLPs, averaged and blended per task.

    python scripts/ensemble_v2.py boosted deep:full3_s0+full3_s1+full3_s2 mlp:v3

Each argument is one component, the mean of its members' out-of-fold probabilities:
  boosted              the --boosted variants in results/v2/oof.csv (or the --oof files)
  deep:<a>+<b>...      train_deep_v2.py runs, results/v2/deep_<name>_oof_f*.csv
  mlp:<a>+<b>...       train_mlp_v3.py runs, results/v2/mlp_<name>_oof.csv
  oof:<a>,<b>...       named boosted variants from the --oof files (comma-separated)
  A|B                  A for most tasks, B for the SV tasks (Reading, Memo, tag:sliders/*): the
                       reading features only describe scroll speed, so models that use them
                       stand in for the SV heads only

Combinations:
  mean       the plain mean of the components (nothing fitted)
  cv_blend   per task and fold k, convex weights on a 0.1 grid chosen by AP on the other four folds'
             labelled charts, so fold k's labels never choose its own weights
  half_blend halfway between cv_blend's weights and equal weights (less sensitive to noisy tasks)

Prints AP tables (categories, user tags) and writes results/v2/ensemble_{categories,tags}.csv and
the blend weights.
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osu_patterns import categories, evaluate  # noqa: E402


def cv_blend(Y, fold, comps):
    """Per task and fold, the convex blend of the components whose weights maximise AP on the other folds."""
    grid = [np.array(w) / 10 for w in itertools.product(range(11), repeat=len(comps)) if sum(w) == 10]
    out = pd.DataFrame(np.nan, index=Y.index, columns=Y.columns)
    half = out.copy()
    chosen = {}
    for t in Y.columns:
        y = Y[t].to_numpy()
        A = np.column_stack([c[t].to_numpy() for c in comps])
        lab = ~np.isnan(y) & ~np.isnan(A).any(1)
        for k in np.unique(fold):
            tr, te = lab & (fold != k), lab & (fold == k)
            if not te.any() or y[tr].sum() == 0:
                continue
            w = max(grid, key=lambda w: average_precision_score(y[tr], A[tr] @ w))
            out.loc[Y.index[te], t] = A[te] @ w
            half.loc[Y.index[te], t] = A[te] @ (w + 1 / len(comps)) / 2
            chosen[(t, int(k))] = w
    return out, half, chosen


SV_TASKS = ("Reading", "Memo", "tag:sliders/complex sv", "tag:sliders/low sv", "tag:sliders/high sv")


def load(spec, Y, res, oof, boosted):
    if "|" in spec:
        main, sv = (load(s, Y, res, oof, boosted) for s in spec.split("|"))
        cols = [c for c in Y.columns if c in SV_TASKS]
        main[cols] = sv[cols]
        return main
    if spec == "boosted":
        return sum(oof[m][Y.columns] for m in boosted) / len(boosted)
    kind, names = spec.split(":", 1)
    runs = []
    for n in names.split("," if kind == "oof" else "+"):  # boosted variant names contain "+"
        if kind == "deep":
            parts = sorted(res.glob(f"deep_{n}_oof_f*.csv"))
            if len(parts) != 5:
                sys.exit(f"deep_{n}: {len(parts)} of 5 folds")
            runs.append(pd.concat([pd.read_csv(p, index_col=0) for p in parts]))
        elif kind == "oof":
            runs.append(oof[n])
        elif kind == "mlp":
            runs.append(pd.read_csv(res / f"mlp_{n}_oof.csv", index_col=0))
        else:
            sys.exit(f"unknown component {spec}")
    return sum(r.reindex(Y.index)[Y.columns] for r in runs) / len(runs)


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
    ap.add_argument("--sv-gate", default="", help="data/sv_gate.csv.gz: SV tasks score 0 on charts with no "
                    "effective scroll change (Komi, 2026-11-02), and their labels there count as negatives")
    a = ap.parse_args()
    res = Path(a.results)
    Y = pd.read_csv(Path(a.data) / a.labels, index_col=0).drop(columns="slot")
    fold = pd.read_csv(Path(a.data) / "v2_folds.csv", index_col=0)["fold"].reindex(Y.index).to_numpy()
    oof = pd.concat([pd.read_csv(p, index_col=0, header=[0, 1]) for p in a.oof], axis=1)
    oof = oof.loc[:, ~oof.columns.duplicated()].reindex(Y.index)
    comps = {s: load(s, Y, res, oof, a.boosted) for s in a.components}
    preds = {"baseline": oof[a.baseline][Y.columns], **comps}
    if a.sv_gate:
        gate = pd.read_csv(a.sv_gate, index_col=0)["sve_effective"].reindex(Y.index).fillna(1.0).to_numpy()
        sv = [c for c in Y.columns if c in SV_TASKS]
        Y.loc[gate == 0, sv] = Y.loc[gate == 0, sv].where(Y.loc[gate == 0, sv].isna(), 0.0)
        for P in preds.values():
            P.loc[gate == 0, sv] = 0.0
    if len(comps) > 1:
        blend, half, chosen = cv_blend(Y, fold, list(comps.values()))
        preds.update({"mean": sum(comps.values()) / len(comps), "cv_blend": blend, "half_blend": half})
        w = pd.DataFrame(chosen, index=list(comps)).T.rename_axis(["task", "fold"])
        w.to_csv(res / "ensemble_weights.csv")
    for name, cols in (("categories", [c for c in Y.columns if c in categories.NAMES]),
                       ("tags", [c for c in Y.columns if c.startswith("tag:")])):
        t = evaluate.ap_table(Y[cols], {m: P[cols] for m, P in preds.items()})
        print(f"\n== {name} ==\n{t.to_string()}")
        t.to_csv(res / f"ensemble_{name}.csv")


if __name__ == "__main__":
    main()
