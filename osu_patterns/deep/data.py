"""Event caches, batching and the multi-task label matrix for the deep models."""
from __future__ import annotations

import pickle
import random
from pathlib import Path

import numpy as np
import torch

from .. import mania
from . import tokens
from .model import WINDOW_S

MAX_EVENTS = 12000


def load_events(ids, osu_dir: str | Path, cache: str | Path | None = None, jobs: int = 4, visual: bool = False) -> dict:
    """beatmap_id -> tokens.Events for every id whose .osu parses as a mania chart (>= 50 notes).
    visual: add the tokens.N_VIS visual channels (use a separate cache file)."""
    cache = Path(cache) if cache else None
    ev = pickle.loads(cache.read_bytes()) if cache and cache.exists() else {}
    todo = [b for b in ids if b not in ev]
    if todo:
        from multiprocessing import Pool
        with Pool(jobs) as pool:
            for bid, e, fetched in pool.imap_unordered(_encode_one, [(b, str(osu_dir), visual) for b in todo], chunksize=32):
                if fetched:  # not downloaded yet: try again next time
                    ev[bid] = e
        if cache:
            cache.write_bytes(pickle.dumps(ev, protocol=5))
    want = set(ids)
    return {b: e for b, e in ev.items() if e is not None and b in want}


def _encode_one(arg):
    bid, osu_dir, visual = arg
    p = Path(osu_dir) / f"{bid}.osu.gz"
    if not p.exists():
        return bid, None, False
    if p.stat().st_size < 40:
        return bid, None, True
    try:
        m = mania.load(p)
        if m is None or len(m.start) < 50 or not 4 <= m.keys <= tokens.KMAX:
            return bid, None, True
        return bid, tokens.encode(m, visual=visual), True
    except Exception:  # noqa: BLE001 - one broken chart shouldn't stop a dataset build
        return bid, None, True


def collate(items: list[tokens.Events], train: bool, crop: int = MAX_EVENTS, mirror: bool | None = None) -> dict:
    """mirror: None mirrors at random when training; True / False forces it (test-time augmentation)."""
    xs, ws = [], []
    for e in items:
        L = len(e.t)
        a = random.randint(0, L - crop) if train and L > crop else max(0, (L - crop) // 2)
        sl = slice(a, a + crop)
        flip = (train and random.random() < 0.5) if mirror is None else mirror
        x = np.concatenate([tokens.pad_columns(e.cols[sl], e.keys, flip), tokens.drop_grid(e.cont[sl]).astype(np.float32)], 1)
        t = e.t[sl] - e.t[sl][0]
        xs.append(x)
        ws.append((t // WINDOW_S).astype(np.int64))
    L = max(len(x) for x in xs)
    B = len(xs)
    X = np.zeros((B, L, xs[0].shape[1]), np.float32)
    M = np.zeros((B, L), bool)
    Wi = np.zeros((B, L), np.int64)
    for i, (x, w) in enumerate(zip(xs, ws)):
        X[i, :len(x)], M[i, :len(x)], Wi[i, :len(x)] = x, True, w
    return {"x": torch.from_numpy(X), "mask": torch.from_numpy(M), "win": torch.from_numpy(Wi),
            "n_win": torch.tensor([w.max() + 1 for w in ws]),
            "keys": torch.tensor([e.keys for e in items])}


def length_batches(lengths: np.ndarray, max_tokens: int, shuffle: bool) -> list[np.ndarray]:
    """Batches of similar length, each with at most max_tokens padded events."""
    order = np.argsort(lengths + (np.random.rand(len(lengths)) * 300 if shuffle else 0))
    batches, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * min(max(lengths[cur + [i]]), MAX_EVENTS) > max_tokens:
            batches.append(np.array(cur))
            cur = []
        cur.append(i)
    if cur:
        batches.append(np.array(cur))
    if shuffle:
        random.shuffle(batches)
    return batches
