"""Train the note-level pattern model (osu_patterns.deep.local) on 4K charts.

    python scripts/train_note_model.py --fold 0 --out results/note/f0        # held-out check on v2 fold 0
    python scripts/train_note_model.py --fold -1 --out models/note_model      # production: every chart
    python scripts/train_note_model.py --fold 0 --causal --out results/note/f0_causal   # past notes only (add --grid for the BPM-grid inputs)

Labels: v2 relaxed labels (strict ones at weight 1, relaxed at 0.5) for the chart-level heads, plus
distillation to the v5 boosted teacher on every 4K chart, labelled or not (the fold-k teacher for a
fold-k run, so nothing leaks; the mean of the five for production). Length-only tasks (Stamina, TB,
streams/stamina) are left out: a local model can't see chart length.

Writes <out>.npz (weights for osu_patterns.note_model), <out>.json (tasks, thresholds, report) and,
for a fold run, <out>.oof.csv with the chart-level test predictions.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from osu_patterns.deep import data, tokens  # noqa: E402
from osu_patterns.deep.local import LocalNet  # noqa: E402

SKIP = {"Stamina", "TB", "tag:streams/stamina"}


def best_threshold(y, p):
    from sklearn.metrics import f1_score
    grid = np.unique(np.quantile(p, np.linspace(0.05, 0.95, 37)))
    return float(grid[int(np.argmax([f1_score(y, p >= g) for g in grid]))])


def make_batch(evs, idx, train, crop):
    b = data.collate([evs[i] for i in idx], train=train, crop=crop)
    return b["x"], b["mask"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0, help="-1 trains on every chart (production)")
    ap.add_argument("--out", default="results/note/f0")
    ap.add_argument("--cache", default="data/events4k_vis.pkl")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--c", type=int, default=64)
    ap.add_argument("--crop", type=int, default=3000)
    ap.add_argument("--max-tokens", type=int, default=48000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--kd", type=float, default=1.0, help="weight of the teacher term")
    ap.add_argument("--limit", type=int, default=0, help="smoke test on this many training charts")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", action="store_true",
                    help="keep the BPM-grid inputs (gap in beats, beat length, snap); default: ms timing only")
    ap.add_argument("--causal", action="store_true", help="convolutions look back only (no future notes)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    import pickle
    evs_all = pickle.loads(Path(a.cache).read_bytes())
    evs_all = {b: e for b, e in evs_all.items() if e is not None and e.keys == 4}
    Ys = pd.read_csv("data/v2_labels.csv.gz", index_col=0)
    Yr = pd.read_csv("data/v2_labels_relaxed.csv.gz", index_col=0)
    Wr = pd.read_csv("data/v2_weights_relaxed.csv.gz", index_col=0)
    teach = pd.read_csv("data/v5_teacher.csv.gz", index_col=0)
    folds = pd.read_csv("data/v2_folds.csv", index_col=0)["fold"]
    tasks = [t for t in Yr.columns if t not in SKIP]
    ids = np.array(sorted(set(evs_all) & set(folds.index) & set(teach.index)))
    f = folds.reindex(ids).to_numpy()
    if a.fold >= 0:
        T = teach[[f"{a.fold}|{t}" for t in tasks]].reindex(ids).to_numpy(np.float32)
    else:
        T = np.stack([teach[[f"{k}|{t}" for k in range(5)]].mean(1).reindex(ids).to_numpy(np.float32) for t in tasks], 1)
    Y = Yr[tasks].reindex(ids).to_numpy(np.float32)
    W = Wr[tasks].reindex(ids).fillna(0).to_numpy(np.float32)
    W[np.isnan(Y)] = 0
    Y = np.nan_to_num(Y)
    tr = np.where(f != a.fold)[0] if a.fold >= 0 else np.arange(len(ids))
    te = np.where(f == a.fold)[0] if a.fold >= 0 else np.array([], int)
    if a.limit:
        tr = np.random.permutation(tr)[: a.limit]
    evs = [evs_all[b] for b in ids]
    visual = evs[0].cont.shape[1] > tokens.N_CONT
    if not a.grid:  # ms timing only: no gap in beats, beat length or snap
        evs = [tokens.Events(e.keys, e.t, e.cols, tokens.drop_grid(e.cont)) for e in evs]
    lengths = np.array([len(e.t) for e in evs])
    print(f"{len(ids)} 4K charts, train {len(tr)}, test {len(te)}, {len(tasks)} tasks, device {a.device}", flush=True)

    d_in = tokens.KMAX * 3 + evs[0].cont.shape[1]
    net = LocalNet(d_in, len(tasks), c=a.c, causal=a.causal).to(a.device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-2)
    steps_per_epoch = len(data.length_batches(lengths[tr], a.max_tokens, False))
    total = a.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 200) * 0.5 * (1 + math.cos(math.pi * min(s / total, 1.0))))
    Yt, Wt, Tt = map(torch.from_numpy, (Y, W, T))

    def predict(rows):
        net.eval()
        out = np.zeros((len(rows), len(tasks)), np.float32)
        with torch.no_grad():
            for bi in data.length_batches(lengths[rows], a.max_tokens, False):
                idx = rows[bi]
                ps = 0
                for flip in (False, True):  # mirror test-time augmentation
                    b = data.collate([evs[i] for i in idx], train=False, crop=20000, mirror=flip)
                    logit, _ = net(b["x"].to(a.device), b["mask"].to(a.device))
                    ps = ps + torch.sigmoid(logit).cpu().numpy() / 2
                out[bi] = ps
        net.train()
        return out

    step = 0
    for ep in range(a.epochs):
        t0, tot = time.time(), 0.0
        for k, bi in enumerate(data.length_batches(lengths[tr], a.max_tokens, True)):
            idx = tr[bi]
            x, mask = make_batch(evs, idx, True, a.crop)
            logit, _ = net(x.to(a.device), mask.to(a.device))
            y, w, tch = Yt[idx].to(a.device), Wt[idx].to(a.device), Tt[idx].to(a.device)
            l_lab = (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * w).sum() / w.sum().clamp(min=1)
            tm = ~torch.isnan(tch)
            l_kd = F.binary_cross_entropy_with_logits(logit[tm], tch[tm])
            loss = l_lab + a.kd * l_kd
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            tot += loss.item()
            if k % 100 == 0:
                print(f"  ep {ep} step {k}/{steps_per_epoch} loss {tot / (k + 1):.4f} {time.time() - t0:.0f}s", flush=True)
        msg = f"epoch {ep} loss {tot / max(steps_per_epoch, 1):.4f} {time.time() - t0:.0f}s"
        if len(te) and (ep == a.epochs - 1 or ep % 2 == 1):
            from sklearn.metrics import average_precision_score
            P = predict(te)
            Ys_te = Ys[tasks].reindex(ids[te]).to_numpy()
            aps = {}
            for j, t in enumerate(tasks):
                m = ~np.isnan(Ys_te[:, j])
                if Ys_te[m, j].sum() >= 3:
                    aps[t] = average_precision_score(Ys_te[m, j], P[m, j])
            cats = [v for t, v in aps.items() if not t.startswith("tag:")]
            tg = [v for t, v in aps.items() if t.startswith("tag:")]
            msg += f" | AP cat {np.mean(cats):.3f} tags {np.mean(tg):.3f}"
        print(msg, flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), **net.cpu().export())
    net.to(a.device)
    meta = {"tasks": tasks, "visual": visual, "fold": a.fold, "causal": a.causal, "grid": a.grid,
            "args": vars(a), "thresholds": {}, "report": {}}
    if len(te):
        P = predict(te)
        pd.DataFrame(P, index=ids[te], columns=tasks).rename_axis("beatmap_id").to_csv(out.with_suffix(".oof.csv"), float_format="%.4g")
        from sklearn.metrics import average_precision_score
        Ys_te = Ys[tasks].reindex(ids[te]).to_numpy()
        for j, t in enumerate(tasks):
            m = ~np.isnan(Ys_te[:, j])
            if Ys_te[m, j].sum() >= 3:
                meta["thresholds"][t] = best_threshold(Ys_te[m, j], P[m, j])
                meta["report"][t] = {"n": int(m.sum()), "pos": int(Ys_te[m, j].sum()),
                                     "ap": round(float(average_precision_score(Ys_te[m, j], P[m, j])), 3)}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1), "utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
