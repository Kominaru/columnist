"""Fine-tune the pretrained causal Columnist into the Lens per-note model (osu_patterns.deep.columnist.LensColumnist).

    python scripts/train_columnist_lens.py --name col_f0 --fold 0 --init columnist/runs/full/ckpt.pt   # held-out fold 0
    python scripts/train_columnist_lens.py --name col_prod --final --init ...                          # every chart; exports
                                                                                    models/columnist/<name>.npz + .json
Charts come from the Columnist row store (tok.npy / gap.npy / sv.npy / offsets.npy / meta.csv: snap-free rows) and are
matched to the labelled corpus by beatmap id. Labels as train_v3.py: --tag v6 strict labels plus the relaxed ones at their
weight; song-grouped folds from data/v2_folds.csv; 10% of the training songs for early stopping. Chart-level labels only:
attention-MIL over the rows of a random crop, mirrored half of the time. Nothing from a fold-k test song is used by fold k.
Writes results/columnist/<name>_oof_f<k>.csv (MIL readout) and ..._p90_f<k>.csv (90th percentile of the row scores).
"""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from osu_patterns.deep import columnist as C  # noqa: E402

NOT_PER_NOTE = ("TB", "Stamina", "Reading", "Memo", "tag:streams/stamina", "tag:sliders/complex sv",
                "tag:sliders/low sv", "tag:sliders/high sv")   # chart characteristics and SV: chart-level / rules (as train_v3)


class Store:
    def __init__(self, d):
        self.tok = np.load(f"{d}/tok.npy", mmap_mode="r")
        self.gap = np.load(f"{d}/gap.npy", mmap_mode="r")
        self.sv = np.load(f"{d}/sv.npy", mmap_mode="r")
        self.off = np.load(f"{d}/offsets.npy")

    def n(self, i):
        return int(self.off[i + 1] - self.off[i])

    def rows(self, i, a=0, b=None):
        lo = int(self.off[i]) + a
        hi = int(self.off[i + 1]) if b is None else int(self.off[i]) + b
        return np.asarray(self.tok[lo:hi]), np.asarray(self.gap[lo:hi]), np.asarray(self.sv[lo:hi])


def make_crop_batch(store, sidx, crop, rng, mirror, lut):
    n = [store.n(i) for i in sidx]
    L = min(crop, max(n))
    tok = np.zeros((len(sidx), L), np.int64)
    gap = np.zeros((len(sidx), L), np.float32)
    sv = np.ones((len(sidx), L), np.float32)
    valid = np.zeros((len(sidx), L), bool)
    for k, i in enumerate(sidx):
        a = int(rng.integers(0, store.n(i) - L + 1)) if store.n(i) > L else 0
        t, g, s = store.rows(i, a, min(a + L, store.n(i)))
        m = len(t)
        tok[k, :m], gap[k, :m], sv[k, :m], valid[k, :m] = t, g, s, True
    tok, gap, sv, valid = (torch.from_numpy(x) for x in (tok, gap, sv, valid))
    flip = torch.from_numpy(rng.random(len(sidx)) < 0.5) if mirror else torch.zeros(len(sidx), dtype=torch.bool)
    tok = torch.where(flip[:, None], lut[tok], tok)
    return tok, gap, sv, valid


def task_loss(logit, target, pos_w, weight):
    known = (~torch.isnan(target)).float() * weight
    bce = F.binary_cross_entropy_with_logits(logit, torch.nan_to_num(target), pos_weight=pos_w, reduction="none")
    return (bce * known).sum() / known.sum().clamp(min=1)


def predict(net, store, sidx, dev, amp):
    """Chart readouts per chart: MIL probability and the 90th percentile of the row probabilities, both (T,)."""
    mil, p90 = [], []
    with torch.no_grad():
        for i in sidx:
            t, g, s = store.rows(i)
            S, G = C.chart_scores(net, t, g, s, dev, amp=amp)
            w = np.exp(G - G.max(0))
            w /= w.sum(0)
            mil.append(1 / (1 + np.exp(-(w * S).sum(0))))
            p90.append(np.percentile(1 / (1 + np.exp(-S)), 90, axis=0))
    return np.stack(mil), np.stack(p90)


def mean_ap(Y, P):
    aps = []
    for j in range(Y.shape[1]):
        m = ~np.isnan(Y[:, j])
        if m.sum() and 5 <= Y[m, j].sum() < m.sum():
            aps.append(average_precision_score(Y[m, j], P[m, j]))
    return float(np.mean(aps)) if aps else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--store", default=str(ROOT.parent / "columnist/data/foundation/store"))
    ap.add_argument("--data", default=str(ROOT.parent / "om-bn-pipeline-auth/data"))
    ap.add_argument("--init", required=True, help="pretrained Columnist checkpoint (ckpt.pt), or 'random'")
    ap.add_argument("--tag", default="v6")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--final", action="store_true", help="train on every chart and export the numpy model")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--crop", type=int, default=2048)
    ap.add_argument("--lr-enc", type=float, default=2e-4)
    ap.add_argument("--lr-head", type=float, default=2e-3)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--val-charts", type=int, default=400)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--freeze-enc", type=int, default=0, help="steps with the encoder frozen (head warm-up)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test on this many charts")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    rng = np.random.default_rng(a.seed + max(a.fold, 0))
    torch.manual_seed(a.seed + max(a.fold, 0))
    d = Path(a.data)

    store = Store(a.store)
    meta = pd.read_csv(f"{a.store}/meta.csv")
    pos = {b: i for i, b in enumerate(meta.beatmap_id)}
    folds = pd.read_csv(d / "v2_folds.csv", index_col=0)
    labels = pd.read_csv(d / f"{a.tag}_labels.csv.gz", index_col=0)
    tasks = [c for c in labels.columns if c != "slot" and c not in NOT_PER_NOTE]
    ids = [b for b in folds.index if b in pos and b in labels.index and store.n(pos[b]) >= 32]
    if a.limit:
        ids = list(np.random.default_rng(0).choice(ids, a.limit, replace=False))
    ids = pd.Index(ids)
    sidx = np.array([pos[b] for b in ids])
    fold = folds.loc[ids, "fold"].to_numpy()
    group = folds.loc[ids, "group"].fillna(pd.Series(ids.astype(str), index=ids)).to_numpy()
    Y = labels.reindex(ids)[tasks].to_numpy(np.float32)
    Yr = pd.read_csv(d / f"{a.tag}_labels_relaxed.csv.gz", index_col=0).reindex(ids)[tasks].to_numpy(np.float32)
    Wr = pd.read_csv(d / f"{a.tag}_weights_relaxed.csv.gz", index_col=0).reindex(ids)[tasks].to_numpy(np.float32)
    fill = np.isnan(Y) & ~np.isnan(Yr)
    Tt, Tw = np.where(fill, Yr, Y), np.where(fill, Wr, 1.0).astype(np.float32)
    print(f"{len(ids)} labelled charts in the row store, {len(tasks)} per-note tasks", flush=True)

    k = -1 if a.final else a.fold
    rest = np.arange(len(ids)) if a.final else np.where(fold != k)[0]
    test = np.array([], int) if a.final else np.where(fold == k)[0]
    hasy = ~np.isnan(Y).all(1)
    lab_groups = np.unique(group[rest][hasy[rest]])
    val_groups = set(np.random.default_rng(a.seed + k + 1).choice(lab_groups, max(1, int(len(lab_groups) * 0.1)), replace=False))
    is_val = np.array([g in val_groups for g in group[rest]])
    val = rest[is_val & hasy[rest]]
    train = rest[~is_val]
    if len(val) > a.val_charts:
        val = np.random.default_rng(1).choice(val, a.val_charts, replace=False)
    print(f"train {len(train)}  val {len(val)}  test {len(test)}", flush=True)

    net = C.LensColumnist(len(tasks))
    if a.init != "random":
        print("pretrained step", net.load_pretrained(a.init), flush=True)
    net.to(dev)
    npos = np.nansum(Tt[train], 0)
    nneg = np.nansum(1 - Tt[train], 0)
    pos_w = torch.tensor(np.clip(np.sqrt(nneg / np.maximum(npos, 1)), 1, 4), dtype=torch.float32, device=dev)
    enc_params = list(net.enc.parameters())
    head_params = [p for n_, p in net.named_parameters() if not n_.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr_enc}, {"params": head_params, "lr": a.lr_head}], weight_decay=0.02)
    warm = max(a.steps // 20, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s / a.steps, 1))))
    scaler = torch.amp.GradScaler(enabled=a.amp and dev.type == "cuda")
    lut = C.mirror_lut()
    Ttt, Twt = torch.from_numpy(Tt), torch.from_numpy(Tw)
    best, best_state, bad, t0, run = -1.0, None, 0, time.time(), 0.0
    Yval = Y[val]
    for step in range(1, a.steps + 1):
        net.train()
        for p in enc_params:
            p.requires_grad_(step > a.freeze_enc)
        b = rng.choice(train, a.batch, replace=len(train) < a.batch)
        tok, gap, sv, valid = (x.to(dev) for x in make_crop_batch(store, sidx[b], a.crop, rng, True, lut))
        with torch.autocast(device_type=dev.type, enabled=a.amp and dev.type == "cuda"):
            s, g = net(tok, gap, sv, valid)
        logit = C.LensColumnist.pool(s, g, valid)
        loss = task_loss(logit, Ttt[b].to(dev), pos_w, Twt[b].to(dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        run = loss.item() if step == 1 else 0.98 * run + 0.02 * loss.item()
        if step % 50 == 0:
            print(f"step {step}/{a.steps} loss {run:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            mil, _ = predict(net, store, sidx[val], dev, a.amp)
            score = mean_ap(Yval, mil)
            print(f"step {step}: val mean AP {score:.4f}", flush=True)
            if score > best:
                best, best_state, bad = score, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if a.patience and bad >= a.patience:
                    break
    net.load_state_dict(best_state)
    print(f"best val mean AP {best:.4f}", flush=True)

    out = ROOT / "results/columnist"
    out.mkdir(parents=True, exist_ok=True)
    mdir = ROOT / "models/columnist"
    mdir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), mdir / f"{a.name}_f{k}.pt")
    if len(test):
        mil, p90 = predict(net, store, sidx[test], dev, a.amp)
        for tag, P in (("oof", mil), ("p90", p90)):
            pd.DataFrame(P, index=ids[test], columns=tasks).to_csv(out / f"{a.name}_{tag}_f{k}.csv")
        print(f"fold {k} test mean AP: MIL readout {mean_ap(Y[test], mil):.4f}, 90th-percentile readout {mean_ap(Y[test], p90):.4f}",
              flush=True)
    if True:   # every run exports the numpy model (fold runs too, so the per-note check can run the serving code)
        mil, p90 = predict(net, store, sidx[val], dev, a.amp)
        thr = {}
        for j, t in enumerate(tasks):
            m = ~np.isnan(Yval[:, j])
            if 5 <= Yval[m, j].sum() < m.sum():
                pr, rc, th = precision_recall_curve(Yval[m, j], mil[m, j])
                f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
                thr[t] = float(th[int(np.argmax(f1[:-1]))])
        net = net.cpu()
        np.savez(mdir / f"{a.name}.npz", **C.export(net))
        (mdir / f"{a.name}.json").write_text(json.dumps({"tasks": tasks, "served": tasks, "thresholds": thr,
                                                         "report": {"val_charts": int(len(val)), "val_mean_ap": best,
                                                                    "pretrained": str(a.init), "steps": a.steps}}), "utf-8")
        print(f"exported {mdir / a.name}.npz", flush=True)


if __name__ == "__main__":
    main()
