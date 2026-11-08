"""Deep v3 (4K only): per-fold masked-event pretraining, then fine-tuning on the labels.

    python scripts/train_v3.py --name v3a --extra data/unranked_4k_kept.csv --device cuda --amp
    python scripts/train_v3.py --name v3causal --causal --device cuda --amp      # the live Lens network

Nothing from a fold-k test song is used by the fold-k model (song-grouped folds in data/v2_folds.csv).
Stage 1, on all 4K charts outside the fold's test songs (+ the unranked 4K charts with --extra):
  masked rows (15% of events lose their row token; predict it from the timing and the other rows),
  a light regression of chart statistics from the pooled embedding, and the task BCE where a label exists.
Stage 2, on the labelled charts: task BCE (strict labels, plus relaxed labels at their weight), lower
learning rate, early stopping on mean AP of inner-validation songs.
Labels: --tag v6 (strict staff + user-tag labels, data/v6_labels.csv.gz, relaxed v6_labels_relaxed.csv.gz).
Writes results/v3/<name>_oof_f<k>.csv and models/v3/<name>_f<k>.pt. No teachers, no boosting, no ensembling.
"""
import argparse
import copy
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osu_patterns.deep import data as ddata, v3  # noqa: E402

SLOTS = ["RC", "LN", "HB", "SV"]


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def load_prepared(ids, a, visual_cache, osu_dir, cache):
    """beatmap_id -> v3.prepare() dict for the 4K charts among ids (cached; built from the event caches)."""
    cache = Path(cache)
    prep = pickle.loads(cache.read_bytes()) if cache.exists() else {}
    todo = [b for b in ids if b not in prep]
    if todo:
        ev = ddata.load_events(todo, osu_dir, visual_cache, visual=True)
        for b, e in ev.items():
            prep[b] = v3.prepare(e) if e.keys == 4 else None
        for b in todo:
            prep.setdefault(b, None)
        cache.write_bytes(pickle.dumps(prep, protocol=5))
    return {b: prep[b] for b in ids if prep.get(b) is not None}


WIDE = None   # (charts x features) hand-built chart statistics, z-scored per fold, with --wide
WIDE_DROP = 0.0


def make_batch(P, rows, train, crop, dev, mirror=None):
    b = v3.collate([P[i] for i in rows], train, crop, mirror)
    if WIDE is not None:
        w = torch.from_numpy(WIDE[rows])
        if train and WIDE_DROP:   # the wide inputs are zeroed for a share of charts so the encoder learns alone
            w = w * (torch.rand(len(w), 1) >= WIDE_DROP)
        b["wide"] = w
    return {k: v.to(dev) for k, v in b.items()}


def length_batches(lengths, max_tokens, shuffle):
    return ddata.length_batches(lengths, max_tokens, shuffle)


def task_loss(logits, target, pos_w, weight):
    known = (~torch.isnan(target)).float() * weight
    bce = F.binary_cross_entropy_with_logits(logits, torch.nan_to_num(target), pos_weight=pos_w, reduction="none")
    return (bce * known).sum() / known.sum().clamp(min=1)


@torch.no_grad()
def predict(net, P, idx, a, dev, tta):
    net.eval()
    lengths = np.array([min(len(P[i]["t"]), ddata.MAX_EVENTS) for i in idx])
    out, slot = {}, {}
    for bi in length_batches(lengths, a.max_tokens, False):
        rows = idx[bi]
        p = s = 0
        views = (False, True) if tta else (False,)
        for flip in views:
            with torch.autocast(dev.type, torch.float16, enabled=a.amp):
                _, lg, sl, *_ = net(make_batch(P, rows, False, ddata.MAX_EVENTS, dev, mirror=flip))
            p = p + torch.sigmoid(lg.float()).cpu().numpy() / len(views)
            s = s + sl.float().softmax(-1).cpu().numpy() / len(views)
        for i, r in enumerate(rows):
            out[r], slot[r] = p[i], s[i]
    return out, slot


def val_score(net, P, val, Yval, a, dev):
    out, _ = predict(net, P, val, a, dev, False)
    p = np.stack([out[r] for r in val])
    aps = []
    for j in range(Yval.shape[1]):
        m = ~np.isnan(Yval[:, j])
        if m.sum() and 5 <= Yval[m, j].sum() < m.sum():
            aps.append(average_precision_score(Yval[m, j], p[m, j]))
    return float(np.mean(aps)) if aps else 0.0


def run_stage(net, P, idx, Tt, Tw, S, Z, Zm, a, epochs, lr, ssl, use_stats, pos_w, val, Yval, log, dev):
    if epochs == 0:
        return
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.02)
    scaler = torch.amp.GradScaler(enabled=a.amp)
    lengths = np.array([min(len(P[i]["t"]), a.crop) for i in idx])
    steps = epochs * len(length_batches(lengths, a.max_tokens, False))
    warm = max(steps // 30, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1))))
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        net.train()
        t0, tot, n = time.time(), np.zeros(4), 0
        for bi in length_batches(lengths, a.max_tokens, True):
            rows = idx[bi]
            b = make_batch(P, rows, True, a.crop, dev)
            masked = ((torch.rand(b["mask"].shape, device=dev) < a.p_mask) & b["mask"]) if ssl else None
            with torch.autocast(dev.type, torch.float16, enabled=a.amp):
                h, lg, sl, st, _, _ = net(b, masked)
                l_ssl = torch.zeros((), device=dev)
                if ssl and masked.any():
                    # target = the true row id; reconstruct from the unmasked copy (mirror was applied in collate)
                    l_ssl = F.cross_entropy(net.row_logits(h[masked]).float(), b["ids"][masked])
            lg = lg.float()
            l_task = task_loss(lg, torch.tensor(Tt[rows], device=dev), pos_w, torch.tensor(Tw[rows], device=dev))
            s = torch.tensor(S[rows], device=dev)
            if (s >= 0).any():
                l_task = l_task + 0.5 * F.cross_entropy(sl.float()[s >= 0], s[s >= 0])
            l_st = torch.zeros((), device=dev)
            if use_stats:
                tm = torch.tensor(Zm[rows], device=dev)
                l_st = (F.smooth_l1_loss(st.float(), torch.tensor(Z[rows], device=dev), reduction="none") * tm).sum() \
                    / tm.sum().clamp(min=1)
            loss = a.ssl_weight * l_ssl + l_task + a.stats_weight * l_st
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += [loss.item(), l_ssl.item(), l_task.item(), l_st.item()]
            n += 1
        score = val_score(net, P, val, Yval, a, dev)
        print(f"{log} epoch {ep + 1}/{epochs} loss {tot[0] / n:.4f} (rows {tot[1] / n:.3f} task {tot[2] / n:.3f} "
              f"stats {tot[3] / n:.3f}) val mAP {score:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if score > best:
            best, best_state, bad = score, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if a.patience and bad >= a.patience:
                break
    net.load_state_dict(best_state)
    print(f"{log} best val mAP {best:.4f}", flush=True)


NOT_PER_NOTE = ("TB", "Stamina", "Reading", "Memo", "tag:streams/stamina", "tag:sliders/complex sv",
                "tag:sliders/low sv", "tag:sliders/high sv")   # characteristics and SV: chart-level / rules, not per-note


def export_final(net, charts, val, Yval, tasks, a, dev):
    """Numpy weights + metadata for osu_patterns.note_model_v3, with F1-optimal chart thresholds from the validation charts."""
    import json
    from sklearn.metrics import precision_recall_curve
    served = [t for t in tasks if t not in NOT_PER_NOTE]
    res, _ = predict(net, charts, val, a, dev, False)
    p = np.stack([res[r] for r in val])
    thr = {}
    for j, t in enumerate(tasks):
        m = ~np.isnan(Yval[:, j])
        if t in served and 5 <= Yval[m, j].sum() < m.sum():
            pr, rc, th = precision_recall_curve(Yval[m, j], p[m, j])
            f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
            thr[t] = float(th[int(np.argmax(f1[:-1]))])
    out = Path("models/v3")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{a.name}.npz", **v3.export(net.cpu(), tasks))
    (out / f"{a.name}.json").write_text(json.dumps({"tasks": tasks, "served": served, "thresholds": thr,
                                                    "report": {"val_charts": len(val), "stage1_epochs": a.stage1_epochs}}), "utf-8")
    print(f"exported {out / a.name}.npz: {len(served)} per-note tasks", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--osu", default="data/osu")
    ap.add_argument("--events", default="data/events_vis.pkl", help="visual event cache of train_deep_v2.py")
    ap.add_argument("--prep", default="data/v3b_prep.pkl")
    ap.add_argument("--name", required=True)
    ap.add_argument("--tag", default="v6")
    ap.add_argument("--no-relaxed", action="store_true")
    ap.add_argument("--extra", default="", help="unranked 4K charts for stage 1 (data/unranked_4k_kept.csv)")
    ap.add_argument("--extra-events", default="data/events_unranked_vis.pkl")
    ap.add_argument("--extra-prep", default="data/v3b_prep_unranked.pkl")
    ap.add_argument("--extra-features", default="data/unranked_features.csv.gz")
    ap.add_argument("--extra-limit", type=int, default=0)
    ap.add_argument("--stage1-epochs", type=int, default=8)
    ap.add_argument("--stage2-epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--stage2-lr", type=float, default=4e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--p-mask", type=float, default=0.15)
    ap.add_argument("--ssl-weight", type=float, default=1.0)
    ap.add_argument("--stats-weight", type=float, default=0.5)
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--n-conv", type=int, default=3)
    ap.add_argument("--n-attn", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no-glob", action="store_true", help="drop the chart-level path")
    ap.add_argument("--no-bins", action="store_true", help="drop the coarse bin path")
    ap.add_argument("--wide", action="store_true", help="hand-built chart statistics (all_features + row transitions) as chart-level inputs")
    ap.add_argument("--wide-drop", type=float, default=0.5)
    ap.add_argument("--patterns", default="data/pattern_features.csv.gz")
    ap.add_argument("--teacher", action="store_true", help="distil the fold-k teacher where a label is unknown")
    ap.add_argument("--teacher-file", default="data/v6_teacher.csv.gz")
    ap.add_argument("--teacher-weight", type=float, default=1.0)
    ap.add_argument("--win-mil", action="store_true", help="pool events into 4 s windows, MIL over windows (as deep v2); not causal")
    ap.add_argument("--final", action="store_true", help="train on every chart (no test fold) and export the causal Lens model to models/v3/<name>.npz (implies a numpy-servable net)")
    ap.add_argument("--multihot", action="store_true", help="old-style per-column head/tail/held bits instead of the 256-item row embedding")
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--crop", type=int, default=4000)
    ap.add_argument("--max-tokens", type=int, default=48000)
    ap.add_argument("--val-share", type=float, default=0.1)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--tta", action="store_true", help="average test predictions over the chart and its mirror")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke test on a random subset of charts")
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.set_num_threads(os.cpu_count())
    d = Path(a.data)

    folds = pd.read_csv(d / "v2_folds.csv", index_col=0)
    labels = pd.read_csv(d / f"{a.tag}_labels.csv.gz", index_col=0)
    tasks = [c for c in labels.columns if c != "slot"]
    targets = pd.read_csv(d / "all_features.csv.gz", index_col=0)
    P = load_prepared(list(folds.index), a, a.events, a.osu, a.prep)
    ids = folds.index[folds.index.isin(list(P))]
    if a.limit:
        ids = pd.Index(np.random.default_rng(0).choice(ids, a.limit, replace=False))
    print(f"{len(ids)} labelled-corpus 4K charts; tasks: {len(tasks)}", flush=True)
    charts = [P[b] for b in ids]
    fold = folds.loc[ids, "fold"].to_numpy()
    group = folds.loc[ids, "group"].fillna(pd.Series(ids.astype(str), index=ids)).to_numpy()
    Y = labels.reindex(ids)[tasks].to_numpy(np.float32)
    S = np.array([SLOTS.index(s) if isinstance(s, str) else -1 for s in labels.reindex(ids)["slot"]])
    T = targets.reindex(ids)
    nlab = len(ids)
    if a.extra:  # fold -1: only stage 1, never labelled, validated or tested
        xl = pd.read_csv(a.extra)
        if a.extra_limit:
            xl = xl.sort_values("playcount", ascending=False).head(a.extra_limit)
        XP = load_prepared(list(xl.beatmap_id), a, a.extra_events, a.osu, a.extra_prep)
        xids = pd.Index([b for b in xl.beatmap_id if b in XP])
        charts += [XP[b] for b in xids]
        T = pd.concat([T, pd.read_csv(a.extra_features, index_col=0).reindex(index=xids, columns=T.columns)])
        Y = np.vstack([Y, np.full((len(xids), len(tasks)), np.nan, np.float32)])
        S = np.concatenate([S, np.full(len(xids), -1)])
        fold = np.concatenate([fold, np.full(len(xids), -1)])
        group = np.concatenate([group, [f"x{b}" for b in xids]])
        print(f"+ {len(xids)} unranked 4K charts for stage 1", flush=True)
    T = T.to_numpy(np.float64)
    Yr = Wr = None
    if not a.no_relaxed:
        Yr = pd.read_csv(d / f"{a.tag}_labels_relaxed.csv.gz", index_col=0).reindex(ids)[tasks].to_numpy(np.float32)
        Wr = pd.read_csv(d / f"{a.tag}_weights_relaxed.csv.gz", index_col=0).reindex(ids)[tasks].to_numpy(np.float32)
        pad = len(charts) - nlab
        Yr, Wr = (np.vstack([x, np.full((pad, x.shape[1]), np.nan, np.float32)]) for x in (Yr, Wr))
    global WIDE, WIDE_DROP
    WIDE_DROP = a.wide_drop
    if a.wide:
        pat = pd.read_csv(a.patterns, index_col=0).reindex(ids).to_numpy(np.float64)
        pat = np.vstack([pat, np.full((len(charts) - nlab, pat.shape[1]), np.nan)])
        Wraw = np.hstack([slog(T), pat])
    teacher = pd.read_csv(a.teacher_file, index_col=0) if a.teacher else None
    out = Path("results/v3")
    out.mkdir(parents=True, exist_ok=True)
    Path("models/v3").mkdir(parents=True, exist_ok=True)
    allids = list(ids) + (list(xids) if a.extra else [])

    for k in [int(x) for x in a.folds.split(",")]:
        torch.manual_seed(a.seed + k)
        random.seed(a.seed + k)
        np.random.seed(a.seed + k)
        rest = np.arange(len(charts)) if a.final else np.where(fold != k)[0]
        test = np.array([], int) if a.final else np.where(fold == k)[0]
        hasy = ~np.isnan(Y).all(1)
        lab_groups = np.unique(group[rest][hasy[rest]])
        val_groups = set(np.random.default_rng(a.seed + k).choice(lab_groups, int(len(lab_groups) * a.val_share), replace=False))
        is_val = np.array([g in val_groups for g in group[rest]])
        val, train = rest[is_val & hasy[rest]], rest[~is_val]
        Tt, Tw = Y.copy(), np.ones_like(Y)
        if Yr is not None:
            fill = np.isnan(Tt) & ~np.isnan(Yr)
            Tt, Tw = np.where(fill, Yr, Tt), np.where(fill, Wr, Tw)
        labelled = train[~np.isnan(Tt[train]).all(1) | (S[train] >= 0)]
        if teacher is not None:   # stage 1 only: soft targets where the label is unknown
            tk = teacher.reindex(ids)[[f"{k}|{t}" for t in tasks]].to_numpy(np.float32)
            tk = np.vstack([tk, np.full((len(charts) - nlab, len(tasks)), np.nan, np.float32)])
            fill = np.isnan(Tt)
            Tt, Tw = np.where(fill, tk, Tt), np.where(fill, a.teacher_weight, Tw)
        Tw = Tw.astype(np.float32)
        if a.wide:
            wmu, wsd = np.nanmean(Wraw[train], 0), np.nanstd(Wraw[train], 0) + 1e-6
            WIDE = np.nan_to_num(np.clip((Wraw - wmu) / wsd, -5, 5)).astype(np.float32)
        Z = slog(T)
        mu, sd = np.nanmean(Z[train], 0), np.nanstd(Z[train], 0) + 1e-6
        Z = np.clip((Z - mu) / sd, -5, 5)
        Zm = (~np.isnan(Z)).astype(np.float32)
        Z = np.nan_to_num(Z).astype(np.float32)
        pos, neg = np.nansum(Y[train] == 1, 0), np.nansum(Y[train] == 0, 0)
        pos_w = torch.tensor(np.clip(np.sqrt(neg / np.maximum(pos, 1)), 1, 10), dtype=torch.float32, device=dev)
        net = v3.EventNet(len(tasks), a.channels, a.n_conv, a.n_attn, p_drop=a.dropout, causal=a.causal, n_stats=Z.shape[1], wide_dim=WIDE.shape[1] if a.wide else 0, use_glob=not a.no_glob, use_bins=not a.no_bins, win_mil=a.win_mil, multihot=a.multihot).to(dev)
        print(f"[fold {k}] params {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M; stage 1 on {len(train)}, "
              f"stage 2 on {len(labelled)}, val {len(val)}, test {len(test)}", flush=True)
        run_stage(net, charts, train, Tt, Tw, S, Z, Zm, a, a.stage1_epochs, a.lr, True, True, pos_w, val, Y[val], f"[fold {k} s1]", dev)
        run_stage(net, charts, labelled, Tt, Tw, S, Z, Zm, a, a.stage2_epochs, a.stage2_lr if a.stage1_epochs else a.lr,
                  False, False, pos_w, val, Y[val], f"[fold {k} s2]", dev)
        if a.final:
            export_final(net, charts, val, Y[val], tasks, a, dev)
            continue
        res, slot = predict(net, charts, test, a, dev, a.tta)
        oof = pd.DataFrame(np.stack([res[r] for r in test]), index=pd.Index([allids[r] for r in test]), columns=tasks)
        oof[[f"slot:{s}" for s in SLOTS]] = np.stack([slot[r] for r in test])
        oof.rename_axis("beatmap_id").to_csv(out / f"{a.name}_oof_f{k}.csv", float_format="%.5g")
        torch.save(net.state_dict(), f"models/v3/{a.name}_f{k}.pt")


if __name__ == "__main__":
    main()
