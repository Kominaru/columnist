"""Deep v2: per-fold pretraining that distils domain knowledge, then fine-tuning with early stopping.

Nothing from a fold-k test song is used to train the fold-k model, not even its notes: the folds
in data/v2_folds.csv are song-grouped over every chart (scripts/teacher_cv.py writes them).

Stage 1, on every chart outside the fold's test songs and inner-validation songs:
  --ssl      masked events + masked windows (osu_patterns.deep.objectives)
  --stats    regress the chart's statistics and pooled window statistics from the pooled embedding
             (the 71 hand-crafted features are free targets on unlabelled charts)
  labels     BCE on each task's true label where known; with --teacher, the fold-k gbm+win
             teacher's probability where the label is unknown (knowledge distillation)
Stage 2, on the labelled charts: task BCE (teacher where unknown with --teacher), lower learning
rate, early stopping on mean AP of inner-validation songs.
--wide adds a wide path: the chart's statistics, window statistics and row transitions (the boosted
heads' inputs) go through a small MLP whose logits are added to the attention heads' logits. The
whole wide path is dropped for a share of the charts in each batch (--wide-drop), so the event
encoder still has to learn to classify on its own.
--visual adds what the player sees to every event (on-screen time, on-screen gap vs time gap, speed
swings and changes while on screen; osu_patterns.deep.tokens), so scroll speed changes between
notes reach the encoder.

    python scripts/train_deep_v2.py --name full --ssl --stats --teacher --device cuda
    python scripts/train_deep_v2.py --name scratch --stage1-epochs 0 --device cuda

Writes results/v2/deep_<name>_oof_f<k>.csv per fold (task probabilities, slot probabilities).
"""
import argparse
import copy
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osu_patterns.deep import data as ddata, tokens  # noqa: E402
from osu_patterns.deep.model import PatternNet  # noqa: E402
from osu_patterns.deep.objectives import Pretrainer, add_col_valid, window_targets  # noqa: E402

SLOTS = ["RC", "LN", "HB", "SV"]
DEV = torch.device("cpu")


class Net(nn.Module):
    def __init__(self, n_tasks, n_targets, a, n_wide=0):
        super().__init__()
        d_in = tokens.KMAX * 3 + tokens.N_CONT + (tokens.N_VIS if a.visual else 0)
        self.net = PatternNet(n_tasks, c=a.channels, n_conv=a.n_conv, n_tf=a.n_tf, p_drop=a.dropout, d_in=d_in)
        self.pre = Pretrainer(self.net.enc, a.p_window)
        c = a.channels
        self.stats_head = nn.Sequential(nn.Linear(2 * c, 2 * c), nn.GELU(), nn.Linear(2 * c, n_targets))
        self.wide, self.wide_drop = None, a.wide_drop
        if n_wide:
            self.wide = nn.Sequential(nn.Dropout(0.1), nn.Linear(n_wide, 2 * c), nn.GELU(), nn.Dropout(a.dropout),
                                      nn.Linear(2 * c, c), nn.GELU(), nn.Linear(c, n_tasks + len(SLOTS)))

    def heads(self, h, wmask, wide=None):
        logits, slot_logits, _, pooled = self.net.heads(h, wmask)
        if self.wide is not None:
            wl = self.wide(wide)
            if self.training:
                wl = wl * (torch.rand(len(wl), 1, device=wl.device) >= self.wide_drop)
            logits, slot_logits = logits + wl[:, :logits.shape[1]], slot_logits + wl[:, logits.shape[1]:]
        mx = h.masked_fill(~wmask[..., None], -1e4).amax(1)
        return logits, slot_logits, pooled, self.stats_head(torch.cat([pooled, mx], -1))

    def forward(self, batch, ssl: bool):
        if ssl:
            losses, (h, wmask) = self.pre(batch, return_hidden=True)
        else:
            h, wmask = self.net.enc(batch["x"], batch["keys"], batch["mask"], batch["win"], batch["n_win"])
            losses = None
        return losses, self.heads(h, wmask, batch.get("wide"))


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def to_dev(batch):
    return {k: v.to(DEV) for k, v in batch.items()}


WIDE = None  # per-fold wide inputs (charts x features), set in main() with --wide


def make_batch(ev, rows, train, crop, w_norm, mirror=None):
    b = add_col_valid(ddata.collate([ev[i] for i in rows], train=train, crop=crop, mirror=mirror))
    b["w_mean"], b["w_std"] = w_norm
    if WIDE is not None:
        b["wide"] = torch.from_numpy(WIDE[rows])
    return to_dev(b)


def task_loss(logits, target, pos_w, weight=None):
    known = (~torch.isnan(target)).float()
    if weight is not None:
        known = known * weight
    bce = F.binary_cross_entropy_with_logits(logits, torch.nan_to_num(target), pos_weight=pos_w, reduction="none")
    return (bce * known).sum() / known.sum().clamp(min=1)


def run_stage(model, ev, idx, Ttarget, Tw, S, St, Sm, a, epochs, lr, ssl, use_stats, pos_w, w_norm, val, Yval, log):
    if epochs == 0:
        return
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    lengths = np.array([min(len(ev[i].t), a.crop) for i in idx])
    steps = epochs * len(ddata.length_batches(lengths, a.max_tokens, shuffle=False))
    warm = max(steps // 30, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1))))
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), np.zeros(4), 0
        for bi in ddata.length_batches(lengths, a.max_tokens, shuffle=True):
            rows = idx[bi]
            batch = make_batch(ev, rows, True, a.crop, w_norm)
            losses, (logits, slot_logits, _, stats_pred) = model(batch, ssl)
            l_ssl = sum(losses) if losses is not None else torch.zeros((), device=DEV)
            l_task = task_loss(logits, torch.tensor(Ttarget[rows], device=DEV), pos_w, torch.tensor(Tw[rows], device=DEV))
            s = torch.tensor(S[rows], device=DEV)
            if (s >= 0).any():
                l_task = l_task + 0.5 * F.cross_entropy(slot_logits[s >= 0], s[s >= 0])
            l_stats = torch.zeros((), device=DEV)
            if use_stats:
                tm = torch.tensor(Sm[rows], device=DEV)
                l_stats = (F.smooth_l1_loss(stats_pred, torch.tensor(St[rows], device=DEV), reduction="none") * tm).sum() \
                    / tm.sum().clamp(min=1)
            loss = l_ssl + l_task + a.stats_weight * l_stats
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += [loss.item(), l_ssl.item(), l_task.item(), l_stats.item()]
            n += 1
        score = val_score(model, ev, val, Yval, a, w_norm)
        print(f"{log} epoch {ep + 1}/{epochs} loss {tot[0] / n:.4f} (ssl {tot[1] / n:.3f} task {tot[2] / n:.3f} "
              f"stats {tot[3] / n:.3f}) val mAP {score:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if score > best:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if a.patience and bad >= a.patience:
                break
    model.load_state_dict(best_state)
    print(f"{log} best val mAP {best:.4f}", flush=True)


@torch.no_grad()
def predict(model, ev, idx, a, w_norm, tta=False):
    """Probabilities per chart; with tta, the mean over the chart and its mirror image."""
    model.eval()
    lengths = np.array([min(len(ev[i].t), ddata.MAX_EVENTS) for i in idx])
    P, Sl = {}, {}
    views = (False, True) if tta else (False,)
    for bi in ddata.length_batches(lengths, a.max_tokens, shuffle=False):
        rows = idx[bi]
        p = s = 0
        for flip in views:
            _, (logits, slot_logits, _, _) = model(make_batch(ev, rows, False, ddata.MAX_EVENTS, w_norm, mirror=flip), ssl=False)
            p = p + torch.sigmoid(logits).cpu().numpy() / len(views)
            s = s + slot_logits.softmax(-1).cpu().numpy() / len(views)
        for i, r in enumerate(rows):
            P[r], Sl[r] = p[i], s[i]
    return P, Sl


def val_score(model, ev, val, Yval, a, w_norm):
    P, _ = predict(model, ev, val, a, w_norm)
    p = np.stack([P[r] for r in val])
    aps = []
    for j in range(Yval.shape[1]):
        m = ~np.isnan(Yval[:, j])
        if m.sum() and 5 <= Yval[m, j].sum() < m.sum():
            aps.append(average_precision_score(Yval[m, j], p[m, j]))
    return float(np.mean(aps)) if aps else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--osu", default="data/osu")
    ap.add_argument("--events", default="data/events_all.pkl")
    ap.add_argument("--name", required=True)
    ap.add_argument("--ssl", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--teacher", action="store_true")
    ap.add_argument("--teacher-file", default="data/v2_teacher.csv.gz")
    ap.add_argument("--teacher-weight", type=float, default=1.0)
    ap.add_argument("--relaxed", action="store_true", help="train on relaxed labels too (v2_labels_relaxed), at their weight")
    ap.add_argument("--stage1-epochs", type=int, default=10)
    ap.add_argument("--stage2-epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--stage2-lr", type=float, default=4e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--stats-weight", type=float, default=1.0)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--n-conv", type=int, default=6)
    ap.add_argument("--n-tf", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--crop", type=int, default=4000)
    ap.add_argument("--max-tokens", type=int, default=64000)
    ap.add_argument("--val-share", type=float, default=0.1)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke test on a random subset of charts")
    ap.add_argument("--tta", action="store_true", help="average test predictions over the chart and its mirror")
    ap.add_argument("--visual", action="store_true", help="add what the player sees to every event (tokens.N_VIS "
                    "channels: on-screen time, on-screen gap, speed swings); events cached in data/events_vis.pkl")
    ap.add_argument("--wide", action="store_true", help="add the boosted heads' inputs through a wide MLP path")
    ap.add_argument("--wide-drop", type=float, default=0.5, help="share of charts per batch trained without the wide path")
    ap.add_argument("--patterns", default="data/pattern_features.csv.gz", help="row transitions for --wide")
    ap.add_argument("--wide-extra", default="", help="comma-separated extra feature tables for --wide (e.g. MSD ratings)")
    ap.add_argument("--relaxed-tag", default="v2", help="relaxed training labels data/<tag>_labels_relaxed.csv.gz "
                    "(v6: with the tournament-staff labels)")
    ap.add_argument("--p-window", type=float, default=0.15, help="share of window tokens masked in stage 1 (0 = no masked-window objective, as in v3)")
    ap.add_argument("--keys4", action="store_true", help="train and test on 4K charts only (as v3)")
    ap.add_argument("--extra", default="", help="unlabelled charts added to stage 1 of every fold and never tested "
                    "(scripts/prep_unranked.py output, e.g. data/unranked_4k_kept.csv)")
    ap.add_argument("--extra-events", default="data/events_unranked.pkl")
    ap.add_argument("--extra-features", default="data/unranked_features.csv.gz", help="their stats targets")
    ap.add_argument("--extra-limit", type=int, default=0, help="use only the N most-played extra charts")
    a = ap.parse_args()
    global DEV, WIDE
    DEV = torch.device(a.device)
    torch.set_num_threads(a.threads)
    d = Path(a.data)

    folds = pd.read_csv(d / "v2_folds.csv", index_col=0)
    labels = pd.read_csv(d / f"{a.relaxed_tag if a.relaxed_tag != 'v2' else 'v2'}_labels.csv.gz", index_col=0)
    tasks = [c for c in labels.columns if c != "slot"]
    teacher = pd.read_csv(a.teacher_file, index_col=0) if a.teacher else None
    if a.relaxed:
        Yr = pd.read_csv(d / f"{a.relaxed_tag}_labels_relaxed.csv.gz", index_col=0)
        Wr = pd.read_csv(d / f"{a.relaxed_tag}_weights_relaxed.csv.gz", index_col=0)
    targets = pd.read_csv(d / "all_features.csv.gz", index_col=0)  # chart + pooled window statistics
    if a.visual and a.events == "data/events_all.pkl":
        a.events = "data/events_vis.pkl"
    ev_map = ddata.load_events(list(folds.index), a.osu, a.events, visual=a.visual)
    ids = folds.index[folds.index.isin(list(ev_map))]
    if a.keys4:
        ids = ids[[ev_map[b].keys == 4 for b in ids]]
    if a.limit:
        ids = pd.Index(np.random.default_rng(0).choice(ids, a.limit, replace=False))
    print(f"{len(ids)} charts with events; tasks: {len(tasks)}", flush=True)
    ev = [ev_map[b] for b in ids]
    fold = folds.loc[ids, "fold"].to_numpy()
    group = folds.loc[ids, "group"].fillna(pd.Series(ids.astype(str), index=ids)).to_numpy()
    Y = labels.loc[ids, tasks].to_numpy(np.float32)
    S = np.array([SLOTS.index(s) if isinstance(s, str) else -1 for s in labels.loc[ids, "slot"]])
    if a.extra:  # fold -1: in every fold's stage-1 training set, never labelled, validated or tested
        xl = pd.read_csv(a.extra)
        if a.extra_limit:
            xl = xl.sort_values("playcount", ascending=False).head(a.extra_limit)
        xev = ddata.load_events(list(xl.beatmap_id), a.osu, a.extra_events, visual=a.visual)
        xids = pd.Index([b for b in xl.beatmap_id if b in xev and b not in set(ids)], name=ids.name)
        ev += [xev[b] for b in xids]
        fold = np.concatenate([fold, -np.ones(len(xids), int)])
        group = np.concatenate([group, xl.set_index("beatmap_id").group.reindex(xids).astype(str).to_numpy()])
        Y = np.vstack([Y, np.full((len(xids), Y.shape[1]), np.nan, np.float32)])
        S = np.concatenate([S, -np.ones(len(xids), int)])
        targets = pd.concat([targets, pd.read_csv(a.extra_features, index_col=0).reindex(columns=targets.columns)])
        ids = ids.append(xids)
        print(f"+ {len(xids)} extra unlabelled charts for stage 1", flush=True)
    T = targets.reindex(ids).to_numpy(np.float64)
    if a.wide:  # stats + window stats (signed log) and row transitions, z-scored per fold below
        Wraw = np.hstack([slog(T), pd.read_csv(a.patterns, index_col=0).reindex(ids).to_numpy(np.float64),
                          *[slog(pd.read_csv(f, index_col=0).reindex(ids).to_numpy(np.float64))
                            for f in a.wide_extra.split(",") if f]])
    out = Path("results/v2")
    out.mkdir(parents=True, exist_ok=True)

    for k in [int(x) for x in a.folds.split(",")]:
        torch.manual_seed(a.seed + k)
        random.seed(a.seed + k)
        np.random.seed(a.seed + k)
        rest = np.where(fold != k)[0]
        test = np.where(fold == k)[0]
        # inner validation: a share of the labelled non-test songs, used only for early stopping
        lab_groups = np.unique(group[rest][~np.isnan(Y[rest]).all(1)])
        val_groups = set(np.random.default_rng(a.seed + k).choice(lab_groups, int(len(lab_groups) * a.val_share), replace=False))
        is_val = np.array([g in val_groups for g in group[rest]])
        val, train = rest[is_val & ~np.isnan(Y[rest]).all(1)], rest[~is_val]

        # targets: true label where known, then the relaxed label (at its weight), then the fold-k teacher
        Tt, Tw = Y.copy(), np.ones_like(Y)
        if a.relaxed:
            yr = Yr.reindex(ids)[tasks].to_numpy(np.float32)
            wr = Wr.reindex(ids)[tasks].to_numpy(np.float32)
            fill = np.isnan(Tt) & ~np.isnan(yr)
            Tt, Tw = np.where(fill, yr, Tt), np.where(fill, wr, Tw)
        labelled = train[~np.isnan(Tt[train]).all(1) | (S[train] >= 0)]  # stage 2: charts with a (relaxed) label
        if teacher is not None:
            tk = teacher.reindex(ids)[[f"{k}|{t}" for t in tasks]].to_numpy(np.float32)
            fill = np.isnan(Tt)
            Tt, Tw = np.where(fill, tk, Tt), np.where(fill, a.teacher_weight, Tw)
        Tw = Tw.astype(np.float32)
        # stats targets, normalised on this fold's training charts
        Z = slog(T)
        mu, sd = np.nanmean(Z[train], 0), np.nanstd(Z[train], 0) + 1e-6
        Z = np.clip((Z - mu) / sd, -5, 5)
        Zm = (~np.isnan(Z)).astype(np.float32)
        Z = np.nan_to_num(Z).astype(np.float32)
        if a.wide:
            wmu, wsd = np.nanmean(Wraw[train], 0), np.nanstd(Wraw[train], 0) + 1e-6
            WIDE = np.nan_to_num(np.clip((Wraw - wmu) / wsd, -5, 5)).astype(np.float32)
        pos, neg = np.nansum(Y[train] == 1, 0), np.nansum(Y[train] == 0, 0)
        pos_w = torch.tensor(np.clip(np.sqrt(neg / np.maximum(pos, 1)), 1, 10), dtype=torch.float32, device=DEV)
        sample = add_col_valid(ddata.collate([ev[i] for i in np.random.choice(train, 256, replace=False)], train=False, crop=a.crop))
        tw = window_targets(sample)
        wm = torch.zeros(tw.shape[:2], dtype=torch.bool)
        for i, nw in enumerate(sample["n_win"]):
            wm[i, :nw] = True
        w_norm = (tw[wm].mean(0).to(DEV), tw[wm].std(0).clamp(min=1e-3).to(DEV))

        model = Net(len(tasks), Z.shape[1], a, WIDE.shape[1] if a.wide else 0).to(DEV)
        print(f"[fold {k}] stage 1 on {len(train)} charts, stage 2 on {len(labelled)}, val {len(val)}, test {len(test)}", flush=True)
        run_stage(model, ev, train, Tt, Tw, S, Z, Zm, a, a.stage1_epochs, a.lr, a.ssl, a.stats, pos_w, w_norm,
                  val, Y[val], f"[fold {k} s1]")
        run_stage(model, ev, labelled, Tt, Tw, S, Z, Zm, a, a.stage2_epochs,
                  a.stage2_lr if a.stage1_epochs else a.lr, False, False, pos_w, w_norm, val, Y[val], f"[fold {k} s2]")
        P, Sl = predict(model, ev, test, a, w_norm, tta=a.tta)
        oof = pd.DataFrame(np.stack([P[r] for r in test]), index=ids[test], columns=tasks)
        oof[[f"slot:{s}" for s in SLOTS]] = np.stack([Sl[r] for r in test])
        oof.rename_axis("beatmap_id").to_csv(out / f"deep_{a.name}_oof_f{k}.csv", float_format="%.5g")


if __name__ == "__main__":
    main()
