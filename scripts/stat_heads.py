"""Hand-built heads for the tasks that are defined by chart statistics: TB and Stamina (duration, density, LN share) and the
SV tasks (Reading, Memo, tag:sliders/*; scroll behaviour the reading features describe). One boosted tree per task
on data/all_features.csv.gz (+ data/reading_features.csv.gz), fold-safe, no snaps. These are the exemptions to
"one network"; everything else comes from the sequence net.

    python scripts/stat_heads.py            # writes results/v3/stat_heads_oof.csv (beatmap_id, one column per task)
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.metrics import average_precision_score as ap

warnings.filterwarnings("ignore")
TASKS = ["TB", "Stamina", "tag:streams/stamina", "Reading", "Memo", "tag:sliders/complex sv", "tag:sliders/low sv", "tag:sliders/high sv"]


def main():
    Y = pd.read_csv("data/v6_labels.csv.gz", index_col=0)[TASKS]
    F = pd.read_csv("data/all_features.csv.gz", index_col=0)
    R = pd.read_csv("data/reading_features.csv.gz", index_col=0)
    F = F.join(R.add_prefix("rd:"), how="left")
    fo = pd.read_csv("data/v2_folds.csv", index_col=0)["fold"]
    ids = F.index.intersection(fo.index)
    out = pd.DataFrame(np.nan, index=ids, columns=TASKS)
    for t in TASKS:
        for k in range(5):
            tr = ids[(fo.reindex(ids) != k).to_numpy()]
            te = ids[(fo.reindex(ids) == k).to_numpy()]
            y = Y[t].reindex(tr)
            m = y.notna().to_numpy()
            mdl = H(max_depth=3, max_iter=100, learning_rate=0.08).fit(F.reindex(tr)[m], y[m].astype(int))
            out.loc[te, t] = mdl.predict_proba(F.reindex(te))[:, 1]
        y = Y[t].reindex(ids)
        m = y.notna()
        print(f"{t}: out-of-fold AP {ap(y[m], out.loc[m, t]):.3f} ({int(y[m].sum())} positives)", flush=True)
    out.rename_axis("beatmap_id").to_csv("results/v3/stat_heads_oof.csv", float_format="%.5g")


if __name__ == "__main__":
    main()
