"""Score a v3 run (out-of-fold) on the 4K charts: mean AP over Komi's 11 categories and the 20 user tags.

    python scripts/eval_v3.py v3a [--sv-gate data/sv_gate.csv.gz]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osu_patterns import categories  # noqa: E402

SV_TASKS = ("Reading", "Memo", "tag:sliders/complex sv", "tag:sliders/low sv", "tag:sliders/high sv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--data", default="data")
    ap.add_argument("--tag", default="v6")
    ap.add_argument("--results", default="results/v3")
    ap.add_argument("--sv-gate", default="data/sv_gate.csv.gz", help="SV tasks score 0 where the chart has no effective scroll change ('' = off)")
    a = ap.parse_args()
    Y = pd.read_csv(Path(a.data) / f"{a.tag}_labels.csv.gz", index_col=0).drop(columns="slot")
    parts = sorted(Path(a.results).glob(f"{a.name}_oof_f*.csv"))
    P = pd.concat([pd.read_csv(p, index_col=0) for p in parts])
    Y = Y.reindex(P.index)
    P = P[Y.columns]
    if a.sv_gate:
        g = pd.read_csv(a.sv_gate, index_col=0)["sve_effective"].reindex(Y.index).fillna(1.0).to_numpy()
        sv = [c for c in Y.columns if c in SV_TASKS]
        Y.loc[g == 0, sv] = Y.loc[g == 0, sv].where(Y.loc[g == 0, sv].isna(), 0.0)
        P.loc[g == 0, sv] = 0.0
    print(f"{len(parts)} folds, {len(P)} 4K charts")
    rows = []
    for t in Y.columns:
        m = Y[t].notna().to_numpy()
        y = Y[t].to_numpy()[m]
        rows.append((t, int(y.sum()), average_precision_score(y, P[t].to_numpy()[m]) if 0 < y.sum() < len(y) else np.nan))
    tb = pd.DataFrame(rows, columns=["task", "positives", "AP"]).set_index("task")
    print(tb.round(3).to_string())
    cat = tb.loc[[c for c in tb.index if c in categories.NAMES], "AP"].mean()
    tag = tb.loc[[c for c in tb.index if c.startswith("tag:")], "AP"].mean()
    print(f"\nmean AP: categories {cat:.3f}  user tags {tag:.3f}")


if __name__ == "__main__":
    main()
