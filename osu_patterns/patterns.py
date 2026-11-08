"""Bag of row transitions: how often each chord shape follows each other chord shape.

The chart statistics summarise patterns as a few ratios (jack rows, jumpstream rows...). Many
skillsets are finer than that: a jumptrill and a jumpstream have the same chord sizes but different
transitions. This counts consecutive-row pairs by their exact column masks, so a head can learn
these distinctions from data.

* 4K: full 4-column masks; a transition and its mirror image are counted together.
* 7K: per hand (columns 0-2 and 4-6, the left hand mirrored onto the right, so a roll towards the
  thumb reads the same on both hands), one shared 3-column vocabulary; the thumb column's share
  of rows is a separate feature.
* Each pair is bucketed by the gap between the rows: fast (< 150 ms) or slow (150-400 ms).
  Rows further apart than 400 ms are not a pattern.
Counts are normalised by the number of rows, so they are shares of the chart. Other key counts
get no pattern features (NaN).
"""
from __future__ import annotations

import numpy as np

from . import mania

FAST, SLOW = 150.0, 400.0


def _mirror(mask: int, k: int) -> int:
    return int(f"{mask:0{k}b}"[::-1], 2)


def _row_masks(m: mania.ManiaMap, tol: float = 2.0):
    times, cols = mania._rows(m, tol)
    masks = np.array([sum(1 << c for c in cs) for cs in cols], dtype=np.int64)
    return times, masks


def _bucket(gap: np.ndarray) -> np.ndarray:
    return np.where(gap < FAST, 0, np.where(gap < SLOW, 1, -1))


def _counts(prev: np.ndarray, nxt: np.ndarray, buck: np.ndarray, k: int, prefix: str,
            mirror: bool) -> dict[str, float]:
    out: dict[str, float] = {}
    for a, b, g in zip(prev, nxt, buck):
        if g < 0:
            continue
        key = (int(a), int(b))
        if mirror:
            key = min(key, (_mirror(key[0], k), _mirror(key[1], k)))
        name = f"{prefix}{key[0]}>{key[1]}:{'fs'[g]}"
        out[name] = out.get(name, 0.0) + 1.0
    return out


def vocabulary() -> list[str]:
    names = set()
    for k, prefix, mirror in ((4, "p4:", True), (3, "p7:", False)):
        for a in range(1, 1 << k):
            for b in range(1, 1 << k):
                key = min((a, b), (_mirror(a, k), _mirror(b, k))) if mirror else (a, b)
                for g in "fs":
                    names.add(f"{prefix}{key[0]}>{key[1]}:{g}")
    return sorted(names) + ["p7:thumb"]


VOCAB = vocabulary()


def pattern_features(m: mania.ManiaMap) -> dict[str, float]:
    t, masks = _row_masks(m)
    out = dict.fromkeys(VOCAB, 0.0)
    if len(t) < 3 or m.keys not in (4, 7):
        return {k: float("nan") for k in VOCAB}
    gap = np.diff(t)
    buck = _bucket(gap)
    if m.keys == 4:
        c = _counts(masks[:-1], masks[1:], buck, 4, "p4:", mirror=True)
    else:
        left = np.array([_mirror(int(x) & 0b111, 3) for x in masks])  # columns 0-2, mirrored onto the right hand
        right = (masks >> 4) & 0b111                                    # columns 4-6
        c = {}
        for hand in (left, right):
            # transitions between consecutive rows where this hand plays, gap measured between them
            idx = np.where(hand > 0)[0]
            if len(idx) > 1:
                hg = _bucket(np.diff(t[idx]))
                for kk, v in _counts(hand[idx][:-1], hand[idx][1:], hg, 3, "p7:", mirror=False).items():
                    c[kk] = c.get(kk, 0.0) + v
        out["p7:thumb"] = float(np.mean((masks >> 3) & 1))
    n = max(len(t) - 1, 1)
    out.update({k: v / n for k, v in c.items() if k in out})
    return out
