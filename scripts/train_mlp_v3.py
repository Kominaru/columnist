"""A multi-task MLP on the boosted heads' inputs, distilled from the per-fold teacher.

    python scripts/train_mlp_v3.py --name v3 --teacher data/v3_teacher.csv.gz --seeds 5

Inputs: chart statistics and pooled window statistics (signed log) plus the row transitions, z-scored
on the fold's training charts. Targets, like train_deep_v2.py: the true label where known, then the
relaxed label at its weight, then the fold-k teacher's probability. One network for all tasks,
early stopping on the mean AP of inner-validation songs; the seeds are averaged.

Same song-grouped folds as everything else in v2: the fold-k networks never see a fold-k chart.

Writes
  results/v2/mlp_<name>_oof.csv        out-of-fold probabilities per chart and task
  --teacher-out FILE                   fold-k probabilities for every chart, columns "<k>|<task>"
                                       (a teacher for the next student)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osu_patterns import categories, evaluate  # noqa: E402


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def task_aps(Y, P, min_pos=3):
    """Validation AP per task (NaN where the task has too few positives)."""
    out = np.full(Y.shape[1], np.nan)
    for j in range(Y.shape[1]):
        m = ~np.isnan(Y[:, j])
        if Y[m, j].sum() >= min_pos:
            out[j] = average_precision_score(Y[m, j], P[m, j])
    return out


def fit_one(X, Tt, Tw, Y, train, val, pos_w, a, rng):
    """One network. Returns its probabilities for every chart at the epoch with the best
    inner-validation mean AP, or with --per-task-es, each task's column from that task's best epoch."""
    net = nn.Sequential(nn.Dropout(0.1), nn.Linear(X.shape[1], 512), nn.GELU(), nn.Dropout(0.3),
                        nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, Tt.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    best, bad = -1.0, 0
    best_task = np.full(Tt.shape[1], -1.0)
    npos = np.nansum(Y[val] == 1, 0)
    P = None
    for _ in range(a.epochs):
        net.train()
        perm = rng.permutation(train)
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            bce = F.binary_cross_entropy_with_logits(net(X[b]), Tt[b], pos_weight=pos_w, reduction="none")
            loss = (bce * Tw[b]).sum() / Tw[b].sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            p_all = torch.sigmoid(net(X)).numpy()
        ta = task_aps(Y[val], p_all[val])
        score = float(np.nanmean(np.where(npos >= 5, ta, np.nan)))  # the mean AP over tasks with 5+ positives
        if P is None:
            P = p_all.copy()
        if a.per_task_es:
            up = np.nan_to_num(ta, nan=-1.0) > best_task
            P[:, up] = p_all[:, up]
            best_task = np.maximum(best_task, np.nan_to_num(ta, nan=-1.0))
        if score > best:
            best, bad = score, 0
            if not a.per_task_es:
                P = p_all.copy()
            else:  # tasks without enough validation positives follow the mean
                few = np.isnan(ta)
                P[:, few] = p_all[:, few]
        else:
            bad += 1
            if bad >= a.patience:
                break
    return P, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--name", required=True)
    ap.add_argument("--teacher", default="data/v3_teacher.csv.gz", help="comma-separated teacher files, averaged")
    ap.add_argument("--teacher-out", default="")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--val-share", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--per-task-es", action="store_true", help="keep each task's best epoch, not the best mean")
    ap.add_argument("--extra", default="", help="comma-separated extra feature tables (beatmap_id index), e.g. MSD ratings")
    ap.add_argument("--relaxed-tag", default="v2", help="relaxed training labels data/<tag>_labels_relaxed.csv.gz "
                    "(v6: with the tournament-staff labels)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    d = Path(a.data)

    allf = pd.read_csv(d / "all_features.csv.gz", index_col=0)
    allf = allf[~allf.index.duplicated()]
    ids = allf.index
    labels = pd.read_csv(d / f"{a.relaxed_tag if a.relaxed_tag != 'v2' else 'v2'}_labels.csv.gz", index_col=0).reindex(ids)
    tasks = [c for c in labels.columns if c != "slot"]
    Y = labels[tasks].to_numpy(np.float32)
    folds = pd.read_csv(d / "v2_folds.csv", index_col=0).reindex(ids)
    fold = folds["fold"].to_numpy()
    group = folds["group"].fillna(pd.Series(ids.astype(str), index=ids)).to_numpy()
    Yr = pd.read_csv(d / f"{a.relaxed_tag}_labels_relaxed.csv.gz", index_col=0).reindex(ids)[tasks].to_numpy(np.float32)
    Wr = pd.read_csv(d / f"{a.relaxed_tag}_weights_relaxed.csv.gz", index_col=0).reindex(ids)[tasks].to_numpy(np.float32)
    files = a.teacher.split(",")
    teacher = sum(pd.read_csv(f, index_col=0).reindex(ids) for f in files) / len(files)
    raw = np.hstack([slog(allf.to_numpy(np.float64)),
                     pd.read_csv(d / "pattern_features.csv.gz", index_col=0).reindex(ids).to_numpy(np.float64),
                     *[pd.read_csv(f, index_col=0).reindex(ids).to_numpy(np.float64) for f in a.extra.split(",") if f]])

    P = np.zeros(Y.shape, np.float32)
    every = {}
    for k in range(5):
        rest, test = np.where(fold != k)[0], np.where(fold == k)[0]
        tk = teacher[[f"{k}|{t}" for t in tasks]].to_numpy(np.float32)
        for seed in range(a.first_seed, a.first_seed + a.seeds):
            rng = np.random.default_rng(seed + k)
            torch.manual_seed(seed + k)
            lab_groups = np.unique(group[rest][~np.isnan(Y[rest]).all(1)])
            val_groups = set(rng.choice(lab_groups, int(len(lab_groups) * a.val_share), replace=False))
            is_val = np.array([g in val_groups for g in group[rest]])
            val, train = rest[is_val & ~np.isnan(Y[rest]).all(1)], rest[~is_val]
            mu, sd = np.nanmean(raw[train], 0), np.nanstd(raw[train], 0) + 1e-6
            X = torch.tensor(np.nan_to_num(np.clip((raw - mu) / sd, -5, 5)), dtype=torch.float32)
            # targets: true label, then the relaxed label at its weight, then the fold-k teacher
            Tt, Tw = Y.copy(), np.ones_like(Y)
            fill = np.isnan(Tt) & ~np.isnan(Yr)
            Tt, Tw = np.where(fill, Yr, Tt), np.where(fill, Wr, Tw)
            Tt = np.where(np.isnan(Tt), tk, Tt)
            Tw = np.where(np.isnan(Tt), 0, Tw)
            pos, neg = np.nansum(Y[train] == 1, 0), np.nansum(Y[train] == 0, 0)
            pos_w = torch.tensor(np.clip(np.sqrt(neg / np.maximum(pos, 1)), 1, 10), dtype=torch.float32)
            p, best = fit_one(X, torch.tensor(np.nan_to_num(Tt)), torch.tensor(Tw, dtype=torch.float32), Y,
                              train[~np.isnan(Tt[train]).all(1)], val, pos_w, a, rng)
            P[test] += p[test] / a.seeds
            for j, t in enumerate(tasks):
                every[f"{k}|{t}"] = every.get(f"{k}|{t}", 0) + p[:, j] / a.seeds
            print(f"[fold {k} seed {seed}] best val mAP {best:.4f}", flush=True)

    out = Path("results/v2")
    out.mkdir(parents=True, exist_ok=True)
    oof = pd.DataFrame(P, index=ids, columns=tasks).rename_axis("beatmap_id")
    oof.to_csv(out / f"mlp_{a.name}_oof.csv", float_format="%.5g")
    if a.teacher_out:
        pd.DataFrame(every, index=ids).rename_axis("beatmap_id").to_csv(a.teacher_out, float_format="%.4g")
    for name, cols in (("categories", [t for t in tasks if t in categories.NAMES]), ("tags", [t for t in tasks if t.startswith("tag:")])):
        t = evaluate.ap_table(labels[cols], {f"mlp:{a.name}": oof[cols]})
        print(f"{name}: mean AP {t.loc['mean', f'mlp:{a.name}']:.3f}")


if __name__ == "__main__":
    main()
