"""Coverage and quality of the text-mined labels (osu_patterns.pool_annotations), 4K charts only.

    python scripts/audit_pool_labels.py --set-text data/set_text.jsonl

Quality is checked against a model that never saw these labels: the fold-k gbm+win teacher in
data/v2_teacher.csv.gz, read out of fold (each chart from its own test fold). For each source and
category: how many 4K charts get a label the v2 labels don't have, how often it contradicts an
existing v2 label, and the AP of the out-of-fold teacher on the new labels (against their
positive rate, which is a random ranking's AP).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osu_patterns import pool_annotations as pa  # noqa: E402

CATS = list(pa.FAMILY)


def oof_teacher(data: Path) -> pd.DataFrame:
    T = pd.read_csv(data / "v2_teacher.csv.gz", index_col=0)
    fold = pd.read_csv(data / "v2_folds.csv", index_col=0)["fold"].reindex(T.index)
    out = pd.DataFrame(np.nan, index=T.index, columns=CATS)
    for k in range(5):
        m = (fold == k).to_numpy()
        for c in CATS:
            if f"{k}|{c}" in T:
                out.loc[m, c] = T.loc[m, f"{k}|{c}"]
    return out


def audit(name, S, L, P, rows, k4):
    Lr = L.reindex(S.index)[CATS]
    for c in CATS:
        s, l, p = S[c], Lr[c], P.reindex(S.index)[c]
        new = s.notna() & l.isna()
        both = s.notna() & l.notna()
        y, q = s[new & p.notna()], p[new & p.notna()]
        ap = average_precision_score(y, q) if 0 < y.sum() < len(y) else np.nan
        rows.append({"source": name, "category": c, "pos": int((s == 1).sum()), "neg": int((s == 0).sum()),
                     "new_pos": int(((s == 1) & new).sum()), "new_neg": int(((s == 0) & new).sum()),
                     "overlap": int(both.sum()), "contradict": int((both & (s != l)).sum()),
                     "teacher_ap_new": ap, "base_rate_new": y.mean() if len(y) else np.nan,
                     "teacher_mean_new_pos": q[y == 1].mean() if (y == 1).any() else np.nan,
                     # positives against every other 4K chart: 0.5 = the label carries no signal
                     "auc_pos_vs_4k": _auc(q[y == 1], P.loc[P.index.isin(k4) & ~P.index.isin(S.index[s.notna()]), c])})


def _auc(pos, rest):
    rest = rest.dropna()
    if len(pos) < 3 or not len(rest):
        return np.nan
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(rest))], np.r_[pos.to_numpy(), rest.to_numpy()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", type=Path)
    ap.add_argument("--set-text", nargs="*", default=["data/set_text.jsonl.gz"])
    ap.add_argument("--out", default="results/pool_labels/audit.csv")
    a = ap.parse_args()
    keys = pd.read_csv(a.data / "all_features.csv.gz", usecols=["beatmap_id", "keys"]).drop_duplicates("beatmap_id")
    k4 = set(keys.beatmap_id[keys["keys"] == 4])
    L = pd.read_csv(a.data / "v2_labels.csv.gz", index_col=0)
    P = oof_teacher(a.data)
    diffs = pa.set_diffs(*a.set_text)
    pools = pd.read_csv(a.data / "mappools.csv")
    sources = {
        "sheet": pa.labels_from_sheets(a.data / "pool_sheets.jsonl.gz", False, diffs)[0],
        "sheet+loose": pa.labels_from_sheets(a.data / "pool_sheets.jsonl.gz", True, diffs)[0],
        "wiki": pa.labels_from_wiki(pools)[0],
        "wiki+loose": pa.labels_from_wiki(pools, True)[0],
    }
    if any(Path(p).exists() for p in a.set_text):
        ex = [p for p in a.set_text if Path(p).exists()]
        sources["diffname"] = pd.concat([pa.labels_from_set_text(p, ("diff",))[0] for p in ex]).groupby(level=0).max()
        sources["mapper tags"] = pd.concat([pa.labels_from_set_text(p, ("tags",))[0] for p in ex]).groupby(level=0).max()
        sources["description"] = pd.concat([pa.labels_from_set_text(p, ("description",))[0] for p in ex]).groupby(level=0).max()
    rows = []
    for name, S in sources.items():
        S = S[S.index.isin(k4)]
        print(f"{name}: {len(S)} 4K charts with a label, {int(S.index.isin(L.index[L[CATS].isna().all(axis=1)]).sum())} "
              f"of them without any v2 category label")
        audit(name, S, L, P, rows, k4)
    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False, float_format="%.3f")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(out[(out.pos + out.neg) > 0].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
