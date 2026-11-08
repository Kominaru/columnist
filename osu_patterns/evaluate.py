"""Out-of-fold scoring on shared song-grouped folds, so boosted and deep models compare fairly."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .model import _make_clf


def gbm_oof(X: pd.DataFrame, Y: pd.DataFrame, fold: pd.Series, min_pos: int = 12, seed: int = 0) -> pd.DataFrame:
    """One boosted head per category, trained on that category's labelled maps of the other folds."""
    P = pd.DataFrame(np.nan, index=Y.index, columns=Y.columns)
    Xa = X.loc[Y.index].to_numpy()
    for c in Y.columns:
        lab = Y[c].notna().to_numpy()
        y = Y[c].to_numpy()
        if np.nansum(y) < min_pos:
            continue
        for k in sorted(fold.unique()):
            tr = lab & (fold.to_numpy() != k)
            te = lab & (fold.to_numpy() == k)
            clf = _make_clf(seed)
            clf.fit(Xa[tr], y[tr].astype(int))
            P.loc[Y.index[te], c] = clf.predict_proba(Xa[te])[:, 1]
    return P


def ap_table(Y: pd.DataFrame, preds: dict[str, pd.DataFrame], min_pos: int = 12) -> pd.DataFrame:
    """Average precision per category (rows) and method (columns) on the labelled maps."""
    rows = {}
    for c in Y.columns:
        lab = Y[c].notna()
        y = Y.loc[lab, c].astype(int)
        if y.sum() < min_pos:
            continue
        r = {"n": int(lab.sum()), "pos": int(y.sum())}
        for name, P in preds.items():
            p = P.loc[lab[lab].index, c] if c in P else None
            r[name] = round(float(average_precision_score(y, p)), 3) if p is not None and p.notna().all() else np.nan
        rows[c] = r
    t = pd.DataFrame(rows).T
    t.loc["mean"] = t.drop(columns=["n", "pos"]).mean().round(3)
    return t
