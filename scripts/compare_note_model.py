"""Chart-level AP of the note model's fold run vs the boosted heads on the same 4K test charts.

    python scripts/compare_note_model.py results/note/f0.oof.csv
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

P = pd.read_csv(sys.argv[1], index_col=0)
Y = pd.read_csv("data/v2_labels.csv.gz", index_col=0)
oof = pd.read_csv("results/v2/oof.csv", header=[0, 1], index_col=0)
rows = []
for t in P.columns:
    y = Y[t].reindex(P.index)
    m = y.notna()
    if y[m].sum() < 3:
        continue
    g = oof[("gbm+win+relax", t)].reindex(P.index)
    rows.append({"task": t, "n": int(m.sum()), "pos": int(y[m].sum()),
                 "note_model": average_precision_score(y[m], P[t][m]),
                 "gbm+win+relax": average_precision_score(y[m], g[m])})
df = pd.DataFrame(rows).set_index("task")
cat = df[~df.index.str.startswith("tag:")]
tag = df[df.index.str.startswith("tag:")]
print(df.round(3).to_string())
print("mean categories", cat[["note_model", "gbm+win+relax"]].mean().round(3).to_dict())
print("mean tags", tag[["note_model", "gbm+win+relax"]].mean().round(3).to_dict())
