"""Train and evaluate the osu!mania slot classifier on tournament mappools.

    python scripts/fetch_osu.py data/mappools.csv data/osu            # .osu files (gzipped)
    python scripts/train_slots.py --pools data/mappools.csv --osu data/osu --out models/slots.pkl

Builds data/mania_features.csv (cached), prints grouped-CV metrics for the model and the
rule baseline, writes per-map out-of-fold predictions to data/slot_predictions.csv, and fits
the final model on all labelled maps.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # small data: sklearn's thread pools only add overhead here

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osu_patterns import mania, slots  # noqa: E402


def build_features(ids, osu_dir: Path, cache: Path) -> pd.DataFrame:
    have = pd.read_csv(cache, index_col=0) if cache.exists() else pd.DataFrame()
    todo = [b for b in ids if b not in have.index]
    rows = {}
    for i, b in enumerate(todo):
        p = osu_dir / f"{b}.osu.gz"
        if not p.exists() or p.stat().st_size < 40:
            continue
        try:
            m = mania.load(p)
            if m is None or len(m.start) < 50 or m.keys < 4:
                continue
            rows[b] = mania.stats(m)
        except Exception as e:  # noqa: BLE001 - a broken file shouldn't stop the build
            print(f"skip {b}: {e}")
        if i % 1000 == 0:
            print(f"features {i}/{len(todo)}", flush=True)
    if rows:
        have = pd.concat([have, pd.DataFrame.from_dict(rows, orient="index")])
        have.index.name = "beatmap_id"
        cache.parent.mkdir(parents=True, exist_ok=True)
        have.to_csv(cache)
    return have


def report(y, pred, name):
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    print(f"\n== {name}: accuracy {np.mean(y == pred):.3f}, macro-F1 {f1_score(y, pred, average='macro'):.3f}")
    print(classification_report(y, pred, digits=3, zero_division=0))
    labs = slots.SLOTS
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=labs), index=[f"true {l}" for l in labs], columns=labs)
    print(cm.to_string())
    return {"accuracy": float(np.mean(y == pred)), "macro_f1": float(f1_score(y, pred, average="macro")),
            "per_class_f1": dict(zip(labs, f1_score(y, pred, labels=labs, average=None).round(3).tolist())),
            "confusion": cm.values.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="data/mappools.csv")
    ap.add_argument("--osu", default="data/osu")
    ap.add_argument("--features", default="data/mania_features.csv")
    ap.add_argument("--out", default="models/slots.pkl")
    a = ap.parse_args()

    labels = slots.label_table(pd.read_csv(a.pools))
    feats = build_features(list(labels.index), Path(a.osu), Path(a.features))
    df = labels.join(feats, how="inner")
    lab = df[df.slot.notna()]
    print(f"{len(df)} mania maps with features, {len(lab)} labelled, {lab.group.nunique()} songs")
    print(lab.groupby(["slot", "keys"]).size().unstack(fill_value=0).to_string())

    X, y = lab[feats.columns], lab.slot
    oof = slots.cross_validate(X, y, lab.group)
    pred_m = oof.idxmax(axis=1)
    pred_r = pd.Series([slots.rule_slot(r) for r in X.to_dict("records")], index=X.index)
    res = {"n_labelled": len(lab), "n_songs": int(lab.group.nunique()),
           "rule": report(y, pred_r, "rule baseline"), "model": report(y, pred_m, "gradient boosting (grouped 5-fold CV)")}
    for k in sorted(lab["keys"].unique()):
        mk = lab["keys"] == k
        if mk.sum() >= 30:
            res[f"model_{k}K_acc"] = float(np.mean(y[mk] == pred_m[mk]))
            print(f"{k}K: model acc {res[f'model_{k}K_acc']:.3f}  rule acc {np.mean(y[mk] == pred_r[mk]):.3f}  (n={mk.sum()})")

    out = lab[["title", "tournaments", "subslot", "slot"]].join(oof.add_prefix("p_"))
    out["pred"], out["rule"] = pred_m, pred_r
    out.to_csv(Path(a.features).with_name("slot_predictions.csv"))

    model = slots.SlotModel.fit(X, y)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(a.out)
    # what the final model learned to lean on
    from sklearn.inspection import permutation_importance
    imp = permutation_importance(model.clf, X.values, y.values, n_repeats=3, random_state=0, scoring="f1_macro")
    top = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False).head(15)
    print("\nmost useful features:\n" + top.round(4).to_string())
    res["top_features"] = top.round(4).to_dict()
    Path(a.out).with_suffix(".metrics.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
