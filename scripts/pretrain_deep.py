"""Self-supervised pretraining of the event encoder on unlabelled mania charts.

Two masked objectives, no labels needed:

* masked events: 15% of events are replaced by a mask vector; the model predicts their head and
  tail columns and the gap to the previous event (pattern grammar, local scale);
* masked windows: 15% of the 4-second window tokens are replaced before the transformer; the
  model predicts that window's summary statistics (density, chord size, jacks, LN share,
  holds, snaps, scroll) from the surrounding windows (structure, long range).

    python scripts/pretrain_deep.py --ids all_ranked_ids.csv --osu data/osu --epochs 10 \
        --out models/pretrained.pt --device cuda

The saved file is the encoder's state_dict, loaded by train_deep.py --init.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osu_patterns.deep import data as ddata  # noqa: E402
from osu_patterns.deep.model import EventEncoder  # noqa: E402
from osu_patterns.deep.objectives import Pretrainer, add_col_valid, window_targets  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", default=["data/all_ranked_ids.csv", "data/labelled_ids.csv"])
    ap.add_argument("--osu", default="data/osu")
    ap.add_argument("--events", default="data/events_all.pkl")
    ap.add_argument("--out", default="models/pretrained.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--n-conv", type=int, default=6)
    ap.add_argument("--n-tf", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=64000)
    ap.add_argument("--crop", type=int, default=4000, help="random crop of events per chart")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    dev = torch.device(a.device)

    ids = sorted({int(b) for f in a.ids for b in pd.read_csv(f).beatmap_id})
    ev = list(ddata.load_events(ids, a.osu, a.events).values())
    print(f"{len(ev)} charts for pretraining", flush=True)

    # fixed normalisation of the window targets from a sample of charts
    sample = add_col_valid(ddata.collate(random.sample(ev, min(256, len(ev))), train=False, crop=a.crop))
    tw = window_targets(sample)
    wm = torch.zeros(tw.shape[:2], dtype=torch.bool)
    for i, n in enumerate(sample["n_win"]):
        wm[i, :n] = True
    w_mean, w_std = tw[wm].mean(0), tw[wm].std(0).clamp(min=1e-3)

    enc = EventEncoder(a.channels, n_conv=a.n_conv, n_tf=a.n_tf)
    model = Pretrainer(enc).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.02)
    lengths = np.array([min(len(e.t), a.crop) for e in ev])
    per_epoch = len(ddata.length_batches(lengths, a.max_tokens, shuffle=False))
    steps = a.epochs * per_epoch
    warm = max(steps // 50, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1))))
    for ep in range(a.epochs):
        t0, tot, n = time.time(), np.zeros(3), 0
        model.train()
        for b in ddata.length_batches(lengths, a.max_tokens, shuffle=True):
            batch = add_col_valid(ddata.collate([ev[i] for i in b], train=True, crop=a.crop))
            batch["w_mean"], batch["w_std"] = w_mean, w_std
            batch = {k: v.to(dev) for k, v in batch.items()}
            losses = model(batch)
            loss = losses[0] + losses[1] + losses[2]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += [x.item() for x in losses]
            n += 1
            if n % 200 == 0:
                print(f"  step {n}/{per_epoch} bits {tot[0] / n:.4f} gap {tot[1] / n:.4f} win {tot[2] / n:.4f}", flush=True)
        print(f"epoch {ep + 1}/{a.epochs} bits {tot[0] / n:.4f} gap {tot[1] / n:.4f} win {tot[2] / n:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in enc.state_dict().items()}, a.out)


if __name__ == "__main__":
    main()
