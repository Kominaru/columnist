"""Cross-validated training of the deep event model (osu_patterns.deep) on the category, user-tag
and slot labels at once, on the same song-grouped folds as scripts/compare_models.py.

    python scripts/train_deep.py --data data --osu data/osu --name scratch --epochs 30
    python scripts/train_deep.py --init models/pretrained.pt --name pretrained

Writes results/deep_<name>_oof.csv (out-of-fold probabilities per task and slot) and
results/deep_<name>_emb.csv (out-of-fold pooled chart embeddings). With --final it also trains
one model on every labelled chart and saves models/deep_<name>.pt.
"""
import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osu_patterns import categories, dataset, mania_tags  # noqa: E402
from osu_patterns.deep import data as ddata  # noqa: E402
from osu_patterns.deep.model import PatternNet  # noqa: E402
from osu_patterns.model import make_labels  # noqa: E402

SLOTS = ["RC", "LN", "HB", "SV"]
DEV = torch.device("cpu")
TASKS = categories.NAMES + [f"tag:{t}" for t in mania_tags.TAGS]


def labels(d: dataset.CategoryData) -> tuple[pd.DataFrame, pd.Series]:
    tags = make_labels(pd.DataFrame({"tags": d.votes}), mania_tags.TAGS, 2, 8)
    tags.columns = [f"tag:{t}" for t in tags.columns]
    Y = pd.concat([d.Y, tags], axis=1).reindex(columns=TASKS)
    ids = Y.index[Y.notna().any(axis=1)].union(d.slot.index)
    return Y.reindex(ids), d.slot.reindex(ids)


def batch_targets(Y, S, idx):
    y = torch.tensor(Y[idx], dtype=torch.float32)
    s = torch.tensor(S[idx], dtype=torch.long)
    return y, s


def run_epoch(net, opt, sched, ev, Y, S, idx, pos_w, task_w, max_tokens):
    net.train()
    lengths = np.array([min(len(ev[i].t), ddata.MAX_EVENTS) for i in idx])
    tot, n = 0.0, 0
    for b in ddata.length_batches(lengths, max_tokens, shuffle=True):
        rows = idx[b]
        batch = {k: v.to(DEV) for k, v in ddata.collate([ev[i] for i in rows], train=True).items()}
        y, s = (t.to(DEV) for t in batch_targets(Y, S, rows))
        logits, slot_logits, _, _ = net(batch)
        known = ~torch.isnan(y)
        bce = F.binary_cross_entropy_with_logits(logits, torch.nan_to_num(y), pos_weight=pos_w, reduction="none")
        loss = (bce * known * task_w).sum() / known.sum().clamp(min=1)
        if (s >= 0).any():
            loss = loss + 0.5 * F.cross_entropy(slot_logits[s >= 0], s[s >= 0])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        tot, n = tot + loss.item(), n + 1
    return tot / max(n, 1)


@torch.no_grad()
def predict(net, ev, idx, max_tokens):
    net.eval()
    lengths = np.array([min(len(ev[i].t), ddata.MAX_EVENTS) for i in idx])
    P, Sl, E = {}, {}, {}
    for b in ddata.length_batches(lengths, max_tokens, shuffle=False):
        rows = idx[b]
        batch = {k: v.to(DEV) for k, v in ddata.collate([ev[i] for i in rows], train=False).items()}
        logits, slot_logits, _, pooled = (t.cpu() for t in net(batch))
        for i, r in enumerate(rows):
            P[r], Sl[r], E[r] = torch.sigmoid(logits[i]).numpy(), slot_logits[i].softmax(-1).numpy(), pooled[i].numpy()
    return P, Sl, E


def train(ev, Y, S, idx, a, log=""):
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    net = PatternNet(len(TASKS), c=a.channels, n_conv=a.n_conv, n_tf=a.n_tf, p_drop=a.dropout)
    if a.init:
        state = torch.load(a.init, map_location="cpu")
        missing = net.enc.load_state_dict(state, strict=False)
        print(f"loaded encoder from {a.init}; missing {len(missing.missing_keys)} keys", flush=True)
    net = net.to(DEV)
    yt = Y[idx]
    pos, neg = np.nansum(yt == 1, 0), np.nansum(yt == 0, 0)
    pos_w = torch.tensor(np.clip(np.sqrt(neg / np.maximum(pos, 1)), 1, 10), dtype=torch.float32, device=DEV)
    task_w = torch.tensor([1.0 if not t.startswith("tag:") else a.tag_weight for t in TASKS], device=DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.02)
    lengths = np.array([min(len(ev[i].t), ddata.MAX_EVENTS) for i in idx])
    steps = a.epochs * len(ddata.length_batches(lengths, a.max_tokens, shuffle=False))
    warm = max(steps // 20, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1))))
    for ep in range(a.epochs):
        t0 = time.time()
        loss = run_epoch(net, opt, sched, ev, Y, S, idx, pos_w, task_w, a.max_tokens)
        print(f"{log} epoch {ep + 1}/{a.epochs} loss {loss:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--osu", default="data/osu")
    ap.add_argument("--events", default="data/events.pkl")
    ap.add_argument("--slot-model", default="models/slots.pkl")
    ap.add_argument("--name", default="scratch")
    ap.add_argument("--init", default=None, help="pretrained encoder state_dict")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--n-conv", type=int, default=6)
    ap.add_argument("--n-tf", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--tag-weight", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=64000)
    ap.add_argument("--folds", default="", help="comma-separated folds to run (default all)")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    global DEV
    DEV = torch.device(a.device)

    d = dataset.category_dataset(a.data, a.slot_model)
    Yd, Sd = labels(d)
    ev_map = ddata.load_events(list(Yd.index), a.osu, a.events)
    ids = Yd.index[Yd.index.isin(list(ev_map))]
    groups = d.all_groups.reindex(ids).fillna(pd.Series(ids.astype(str), index=ids))
    fold = dataset.song_folds(groups)
    print(f"{len(ids)} charts, {groups.nunique()} songs; positives per task:", flush=True)
    print((Yd.loc[ids] == 1).sum().to_string(), flush=True)

    ev = [ev_map[b] for b in ids]
    Y = Yd.loc[ids].to_numpy(dtype=np.float32)
    S = np.array([SLOTS.index(s) if isinstance(s, str) else -1 for s in Sd.loc[ids]])
    out = Path("results")
    out.mkdir(exist_ok=True)
    run = [int(k) for k in a.folds.split(",")] if a.folds else sorted(fold.unique())
    P, Sl, E = {}, {}, {}
    for k in run:
        tr = np.where(fold.to_numpy() != k)[0]
        te = np.where(fold.to_numpy() == k)[0]
        net = train(ev, Y, S, tr, a, log=f"[fold {k}]")
        p, s, e = predict(net, ev, te, a.max_tokens)
        P.update(p), Sl.update(s), E.update(e)
    if P:
        rows = sorted(P)
        oof = pd.DataFrame(np.stack([P[r] for r in rows]), index=ids[rows], columns=TASKS)
        oof[[f"slot:{s}" for s in SLOTS]] = np.stack([Sl[r] for r in rows])
        oof.index.name = "beatmap_id"
        suffix = f"_f{a.folds.replace(',', '')}" if a.folds else ""
        oof.to_csv(out / f"deep_{a.name}_oof{suffix}.csv")
        emb = pd.DataFrame(np.stack([E[r] for r in rows]), index=ids[rows])
        emb.index.name = "beatmap_id"
        emb.to_csv(out / f"deep_{a.name}_emb{suffix}.csv")
    if a.final:
        net = train(ev, Y, S, np.arange(len(ids)), a, log="[final]")
        Path("models").mkdir(exist_ok=True)
        torch.save({"state": net.state_dict(), "tasks": TASKS, "slots": SLOTS, "args": vars(a)},
                   f"models/deep_{a.name}.pt")


if __name__ == "__main__":
    main()
