"""Experiments for the paper. Run from the pattern-classifier root.

Re-implements the CV loops of slots.cross_validate and model.TagModel.fit with the same
classifiers, splits and seeds, but keeps out-of-fold predictions so we can bootstrap CIs and
run ablations. Writes everything to paper_out/.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from osu_patterns import categories, mania_tags, slots  # noqa: E402
from osu_patterns.model import _best_threshold, _make_clf, make_labels  # noqa: E402

OUT = Path("paper_out")
OUT.mkdir(exist_ok=True)
NJ = int(os.environ.get("NJ", 3))
RNG = np.random.default_rng(0)
B = 1000

FAMILIES = {
    "meta": ["keys", "od", "hp", "notes", "duration_s"],
    "density": ["nps", "nps_per_key", "nps_p50", "nps_p90", "nps_max", "nps_peak_ratio", "stamina",
                "density_cv", "breaks", "rows_ps"],
    "chords/jacks/streams": ["notes_per_row", "row_single", "row_jump", "row_hand", "row_quad",
                             "row_chord_frac_of_keys", "row_full", "jack_row", "jack_note", "chordjack_row",
                             "full_jack_row", "stream_row", "jumpstream_row", "handstream_row", "jack_gap_ms",
                             "row_gap_ms_p10", "row_gap_ms_p50", "jack_chain3", "jack_longest", "trill_row",
                             "col_imbalance", "anchor"],
    "rhythm": ["snap_1/2", "snap_1/4", "snap_1/8", "snap_1/16", "snap_1/3", "snap_1/6", "snap_1/12",
               "snap_other", "snap_triplet", "snap_fine", "rhythm_variety", "gap_ratio_change", "bpm_main",
               "bpm_changes"],
    "long notes": ["ln_ratio", "ln_count_ps", "ln_len_ms_p50", "ln_len_beats_p50", "ln_short", "ln_long",
                   "ln_hold_cov", "ln_inverse", "ln_release_offgrid", "ln_held_mean", "has_ln"],
    "scroll speed": ["sv_points", "sv_points_ps", "sv_log_std", "sv_off_frac", "sv_far_frac", "sv_stop_frac",
                     "sv_fast_frac", "sv_changes_ps", "sv_changes_near_notes_ps"],
}


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=float))


def cluster_boot(groups, fn, b=B, seed=0):
    """Bootstrap over clusters (songs / sets): resample groups with replacement."""
    rng = np.random.default_rng(seed)
    g = pd.Series(groups).reset_index(drop=True)
    idx_by = g.groupby(g).indices
    keys = list(idx_by)
    vals = []
    for _ in range(b):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([idx_by[keys[i]] for i in pick])
        vals.append(fn(idx))
    return np.percentile(vals, [2.5, 97.5]).tolist()


# ================================================================ slots
def slot_data():
    pools = pd.read_csv("data/mappools.csv")
    labels = slots.label_table(pools)
    feats = pd.read_csv("data/mania_features.csv", index_col=0)
    df = labels.join(feats, how="inner")
    return pools, labels, feats, df[df.slot.notna()]


def slot_cv(X, y, groups, make, grouped=True, seed=0):
    classes = sorted(y.unique())
    cv = (StratifiedGroupKFold(5, shuffle=True, random_state=seed) if grouped
          else StratifiedKFold(5, shuffle=True, random_state=seed))
    splits = list(cv.split(X, y, groups) if grouped else cv.split(X, y))

    def run(tr, te):
        clf = make()
        clf.fit(X.values[tr], y.values[tr])
        return te, pd.DataFrame(clf.predict_proba(X.values[te]), columns=clf.classes_)[classes].values

    oof = np.zeros((len(X), len(classes)))
    for te, p in Parallel(n_jobs=NJ)(delayed(run)(tr, te) for tr, te in splits):
        oof[te] = p
    return pd.Series(np.array(classes)[oof.argmax(1)], index=X.index)


def slot_metrics(y, p):
    return {"accuracy": float(np.mean(y.values == p.values)),
            "macro_f1": float(f1_score(y, p, average="macro")),
            "f1": dict(zip(slots.SLOTS, f1_score(y, p, labels=slots.SLOTS, average=None).round(3).tolist()))}


def slots_experiments():
    pools, labels, feats, lab = slot_data()
    X, y, g = lab[feats.columns], lab.slot, lab.group
    res = {}
    # dataset description
    res["pool_rows"] = len(pools)
    res["tournaments"] = int(pools.tournament.nunique())
    res["series"] = int(pools.tournament.str.split("/").str[0].nunique())
    res["unique_maps"] = int(pools.beatmap_id.nunique())
    res["maps_with_features"] = int(len(labels.join(feats, how="inner")))
    res["labelled"] = len(lab)
    res["songs"] = int(g.nunique())
    res["by_slot_keys"] = lab.groupby(["slot", "keys"]).size().unstack(fill_value=0).to_dict()
    res["ln_ratio_median"] = lab.groupby("slot").ln_ratio.median().to_dict()
    res["ln_ratio_q"] = lab.groupby("slot").ln_ratio.quantile([0.1, 0.25, 0.75, 0.9]).unstack().to_dict()
    # main OOF from train_slots (saved predictions) + bootstrap CI
    sp = pd.read_csv("data/slot_predictions.csv", index_col=0).loc[lab.index]
    pm, pr = sp.pred, sp.rule
    for name, p in (("model", pm), ("rule", pr)):
        m = slot_metrics(y, p)
        yy, pp = y.values, p.values
        m["acc_ci"] = cluster_boot(g.values, lambda i: np.mean(yy[i] == pp[i]))
        m["mf1_ci"] = cluster_boot(g.values, lambda i: f1_score(yy[i], pp[i], average="macro"), b=300)
        res[name] = m
    yy = y.values
    res["acc_diff_ci"] = cluster_boot(g.values, lambda i: np.mean(yy[i] == pm.values[i]) - np.mean(yy[i] == pr.values[i]))
    print("slot main", res["model"], res["rule"], flush=True)

    hgb = lambda: slots.make_classifier()  # noqa: E731
    alt = {}
    alt["HGB (ours)"] = slot_metrics(y, pm)
    lr = lambda: make_pipeline(SimpleImputer(), StandardScaler(),  # noqa: E731
                               LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"))
    alt["Logistic regression"] = slot_metrics(y, slot_cv(X, y, g, lr))
    rf = lambda: make_pipeline(SimpleImputer(), RandomForestClassifier(  # noqa: E731
        500, min_samples_leaf=2, class_weight="balanced", n_jobs=1, random_state=0))
    alt["Random forest"] = slot_metrics(y, slot_cv(X, y, g, rf))
    alt["HGB, LN share only"] = slot_metrics(y, slot_cv(X[["ln_ratio"]], y, g, hgb))
    alt["HGB, LN share + SV"] = slot_metrics(y, slot_cv(X[["ln_ratio"] + FAMILIES["scroll speed"]], y, g, hgb))
    alt["HGB, ungrouped CV"] = slot_metrics(y, slot_cv(X, y, g, hgb, grouped=False))
    res["alternatives"] = alt
    print("alts", alt, flush=True)
    abl = {}
    for fam, cols in FAMILIES.items():
        keep = [c for c in X.columns if c not in cols]
        abl[f"- {fam}"] = slot_metrics(y, slot_cv(X[keep], y, g, hgb))
        abl[f"only {fam}"] = slot_metrics(y, slot_cv(X[cols], y, g, hgb))
        print(fam, abl[f"- {fam}"]["macro_f1"], abl[f"only {fam}"]["macro_f1"], flush=True)
    res["family_ablation"] = abl
    # seeds
    res["seed_acc"] = [float(np.mean(y.values == slot_cv(X, y, g, hgb, seed=s).values)) for s in (1, 2, 3)]
    # errors by tournament series
    err = lab.assign(series=lab.tournaments.str.split("/").str[0], wrong=(pm != y))
    res["errors_total"] = int(err.wrong.sum())
    top = err[err.wrong].series.value_counts().head(8)
    res["errors_by_series"] = top.to_dict()
    res["confusions"] = pd.crosstab(y[pm != y], pm[pm != y]).stack().loc[lambda s: s > 0].to_dict()
    res["confusions"] = {f"{a}->{b}": int(v) for (a, b), v in res["confusions"].items()}
    # ln_ratio for errors
    hb_ln = err[(y == "HB") & (pm == "LN") | (y == "LN") & (pm == "HB")]
    res["hb_ln_err_ln_ratio_q"] = hb_ln.ln_ratio.quantile([0.1, 0.5, 0.9]).tolist()
    dump("slots.json", res)
    lab[["slot", "ln_ratio", "keys"]].assign(pred=pm, rule=pr).to_csv(OUT / "slot_points.csv")


# ================================================================ pattern heads
def head_cv(X, y, groups, seed=0, train_labels=None):
    """Exactly TagModel.fit's loop for one head. train_labels: optional label vector used for
    training (e.g. missing-as-negative); y is always the evaluation label (masked)."""
    oof = np.zeros(len(y))
    k = min(5, len(np.unique(groups)))
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        ytr = y[tr] if train_labels is None else train_labels[tr]
        Xtr = X[tr] if train_labels is None else X[tr]
        if len(np.unique(ytr)) < 2:
            oof[te] = ytr.mean()
            continue
        clf = _make_clf(seed)
        clf.fit(Xtr, ytr)
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def head_cv_extra(Xall, yall, gall, eval_mask, seed=0):
    """Train on all rows (yall = 0/1 for everything), evaluate OOF on eval_mask rows only.
    Folds are built on all rows by group, so evaluation maps are never in training."""
    oof = np.full(len(yall), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(Xall, yall, gall):
        clf = _make_clf(seed)
        clf.fit(Xall[tr], yall[tr])
        oof[te] = clf.predict_proba(Xall[te])[:, 1]
    return oof[eval_mask]


def score(y, p, h, groups, boot=True):
    thr = _best_threshold(y, p)
    r = {"n": int(len(y)), "pos": int(y.sum()), "prev": float(y.mean()),
         "ap": float(average_precision_score(y, p)), "auc": float(roc_auc_score(y, p)),
         "f1": float(f1_score(y, p >= thr)), "rule_ap": float(average_precision_score(y, h)),
         "rule_auc": float(roc_auc_score(y, h))}
    if boot:
        def d(i):
            if y[i].sum() == 0 or y[i].sum() == len(i):
                return np.nan
            return average_precision_score(y[i], p[i]) - average_precision_score(y[i], h[i])
        vals = []
        rng = np.random.default_rng(0)
        gs = pd.Series(groups).reset_index(drop=True)
        idx_by = gs.groupby(gs).indices
        keys = list(idx_by)
        for _ in range(300):
            pick = rng.integers(0, len(keys), len(keys))
            idx = np.concatenate([idx_by[keys[j]] for j in pick])
            vals.append(d(idx))
        r["diff_ci"] = np.nanpercentile(vals, [2.5, 97.5]).tolist()
    return r


def tag_data():
    rows = {}
    for line in open("data/user_tags.jsonl"):
        j = json.loads(line)
        for bid, tags in j["beatmaps"].items():
            rows[int(bid)] = {"beatmapset_id": j["beatmapset_id"], "tags": tags}
    meta = pd.DataFrame.from_dict(rows, orient="index")
    stats = pd.read_csv("data/ranked_features.csv", index_col=0)
    meta = meta.loc[meta.index.intersection(stats.index)]
    stats = stats.loc[meta.index]
    sm = slots.SlotModel.load("models/slots.pkl")
    probs = pd.DataFrame(sm.clf.predict_proba(stats[sm.features].values), index=stats.index, columns=sm.classes)
    X = pd.DataFrame.from_dict({b: mania_tags.feature_vector(stats.loc[b].to_dict(), probs.loc[b].to_dict())
                                for b in meta.index}, orient="index")
    return meta, stats, X


def tags_experiments():
    meta, stats, X = tag_data()
    Y = make_labels(meta, mania_tags.TAGS, 2, 8)
    sets = meta.beatmapset_id
    totals = meta.tags.apply(lambda t: sum(t.values()))
    res = {"diffs": len(X), "sets": int(sets.nunique()), "diffs_with_any_vote": int((totals > 0).sum()),
           "diffs_well_annotated": int((totals >= 8).sum()),
           "votes_total_median_tagged": float(totals[totals > 0].median()),
           "keys": stats["keys"].value_counts().to_dict(), "tags": {}}
    stat_cols = [c for c in X.columns if not c.startswith(("h:", "slot:"))]
    no_slot = [c for c in X.columns if not c.startswith("slot:")]
    Xa = X.values
    sets_a = sets.to_numpy()
    votes_of = {t: meta.tags.apply(lambda tv, t=t: tv.get(t, 0)).to_numpy() for t in mania_tags.TAGS}

    def one(t):
        mask = Y[t].notna().to_numpy()
        y = Y[t].to_numpy()[mask].astype(int)
        if y.sum() < 12 or len(y) - y.sum() < 12:
            return t, None
        gm = sets_a[mask]
        h = X.loc[mask, f"h:{t}"].to_numpy()
        p = head_cv(Xa[mask], y, gm)
        r = score(y, p, h, gm)
        r["ap_stats_only"] = float(average_precision_score(y, head_cv(X.loc[mask, stat_cols].values, y, gm)))
        r["ap_no_slot"] = float(average_precision_score(y, head_cv(X.loc[mask, no_slot].values, y, gm)))
        yneg = (votes_of[t] >= 2).astype(int)
        r["ap_missing_as_neg"] = float(average_precision_score(y, head_cv_extra(Xa, yneg, sets_a, mask)))
        print(t, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
        return t, r

    for t, r in Parallel(n_jobs=NJ)(delayed(one)(t) for t in mania_tags.TAGS):
        if r is not None:
            res["tags"][t] = r
    dump("tags.json", res)


# ================================================================ categories
def category_data():
    pools = pd.read_csv("data/mappools.csv")
    table = slots.label_table(pools)
    stats = pd.concat([pd.read_csv("data/mania_features.csv", index_col=0),
                       pd.read_csv("data/ranked_features.csv", index_col=0)])
    stats = stats[~stats.index.duplicated()]
    sm = slots.SlotModel.load("models/slots.pkl")
    probs = pd.DataFrame(sm.clf.predict_proba(stats[sm.features].values), index=stats.index, columns=sm.classes)
    oof = pd.read_csv("data/slot_predictions.csv", index_col=0)[[f"p_{c}" for c in sm.classes]]
    oof.columns = sm.classes
    probs.update(oof)
    votes, group = {}, {}
    for line in open("data/user_tags.jsonl"):
        j = json.loads(line)
        for bid, tv in j["beatmaps"].items():
            votes[int(bid)] = tv
            group[int(bid)] = f"set{j['beatmapset_id']}"
    titles = {r["beatmapset_id"]: r["title"] for r in map(json.loads, open("data/ranked_mania.jsonl"))}
    for line in open("data/user_tags.jsonl"):
        j = json.loads(line)
        for bid in j["beatmaps"]:
            group[int(bid)] = slots.song_key(titles.get(j["beatmapset_id"], "") + " [x]") or group[int(bid)]
    group.update(table.group.to_dict())
    Yv = categories.labels_from_votes(pd.Series(votes))
    Yp = categories.labels_from_pools(pools, table.slot)
    Y = categories.combine(Yv, Yp)
    Y = Y.loc[Y.index.intersection(stats.index)]
    moving = stats.loc[Y.index, "sv_log_std"] >= 0.1
    for c in categories.CATEGORIES:
        if c.sv_only and c.name in Y:
            Y.loc[(Y[c.name] == 1) & ~moving & ~Y.index.isin(pools.beatmap_id), c.name] = np.nan
    Y = Y[Y.notna().any(axis=1)]
    X = pd.DataFrame.from_dict(
        {b: {**mania_tags.feature_vector(stats.loc[b].to_dict(), probs.loc[b].to_dict()),
             **{f"h:{k}": v for k, v in categories.heuristic_scores(stats.loc[b].to_dict()).items()}}
         for b in Y.index}, orient="index")
    groups = pd.Series({b: group.get(b, str(b)) for b in Y.index})
    src = pd.Series("vote", index=Y.index)
    src[Y.index.isin(pools.beatmap_id)] = "pool"
    return X, Y, groups, Yv, Yp, src


def categories_experiments():
    X, Y, groups, Yv, Yp, src = category_data()
    res = {"maps": len(X), "groups": int(groups.nunique()), "pool_maps": int((src == "pool").sum()), "cats": {}}
    curves = {}

    def one(name):
        mask = Y[name].notna().to_numpy()
        y = Y[name].to_numpy()[mask].astype(int)
        if y.sum() < 12:
            return name, None, None
        gm = groups.to_numpy()[mask]
        h = X.loc[mask, f"h:{name}"].to_numpy()
        p = head_cv(X.values[mask], y, gm)
        r = score(y, p, h, gm)
        s = src.to_numpy()[mask]
        r["n_from_pools"] = int((s == "pool").sum())
        r["pos_from_votes"] = int(((Yv.reindex(Y.index)[name] == 1).to_numpy()[mask]).sum())
        r["pos_from_pools"] = int(((Yp.reindex(Y.index)[name] == 1).to_numpy()[mask]).sum())
        for sname in ("vote", "pool"):
            sm = s == sname
            if y[sm].sum() >= 10 and (1 - y[sm]).sum() >= 10:
                r[f"ap_{sname}"] = float(average_precision_score(y[sm], p[sm]))
                r[f"rule_ap_{sname}"] = float(average_precision_score(y[sm], h[sm]))
        print(name, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
        return name, r, {"y": y.tolist(), "p": p.round(4).tolist(), "h": np.round(h, 4).tolist()}

    for name, r, cv in Parallel(n_jobs=NJ)(delayed(one)(n) for n in categories.NAMES):
        if r is not None:
            res["cats"][name], curves[name] = r, cv
    dump("categories.json", res)
    dump("category_curves.json", curves)


if __name__ == "__main__":
    which = sys.argv[1:] or ["slots", "tags", "categories"]
    for w in which:
        globals()[f"{w}_experiments"]()
