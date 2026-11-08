"""A chart as a sequence of events for the deep models.

An event is a moment where at least one note starts or a long note ends. Per event:

* per column: head (a note starts), tail (a long note is released), held (a long note body
  passes through without starting or ending here);
* timing: log gap to the previous and next event, in ms (no beat or snap information: live-timed
  maps have no reliable BPM grid);
* scroll: log effective scroll speed (BPM x SV relative to the main BPM).
* visual (optional, encode(..., visual=True)): what the player sees, from the scroll position
  x(t) = integral of the speed: how long the event is on screen, its on-screen gap to the previous
  event relative to the time gap, how far the speed swings and how often it changes while the
  event is on screen. Speed changes between events (teleports, stops) only show up here.

Columns are stored for the chart's own key count and padded into KMAX slots at batch time,
centred, so 4K and 7K charts share weights; the key count is a separate input.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import mania

KMAX = 10
N_CONT = 3               # log dt_prev, log dt_next, log scroll speed
OLD_N_CONT = 14          # caches built before 2026-11-02 also held beats_prev, log beat length and a snap one-hot
N_VIS = 4                # log visible time, log on-screen/time gap ratio, speed swing, speed changes on screen
H_MS = 600.0             # visible height in ms at the main BPM and 1.0x (as osu_patterns.reading)


@dataclass
class Events:
    keys: int
    t: np.ndarray        # (L,) event time in seconds from the first note
    cols: np.ndarray     # (L, keys, 3) uint8: head, tail, held
    cont: np.ndarray     # (L, N_CONT) float16


def _visual(m: mania.ManiaMap, ev: np.ndarray) -> np.ndarray:
    """(L, N_VIS) visual channels for event times ev (ms)."""
    st, du, sp = mania.scroll_profile(m, float(ev[0]) - 4 * H_MS, float(ev[-1]) + 1)
    kt = np.append(st, st[-1] + du[-1])
    kx = np.concatenate([[0.0], np.cumsum(du * sp)])
    x = np.interp(ev, kt, kx)
    onset = np.minimum(np.interp(x - H_MS, kx, kt), ev)
    vis = (ev - onset) / H_MS
    dt, dx = np.diff(ev, prepend=ev[0] - 1), np.diff(x, prepend=x[0] - 1)
    ratio = np.where(dt > 1, np.log(np.clip(dx / np.maximum(dt, 1e-3), 0.01, 100)), 0.0)
    a = np.clip(np.searchsorted(st, onset, side="right") - 1, 0, len(sp) - 1)
    b = np.clip(np.searchsorted(st, ev, side="right") - 1, 0, len(sp) - 1)
    lsp = np.log(np.clip(sp, 0.01, None))
    swing = np.zeros(len(ev))
    multi = b > a
    if multi.any():  # max - min of the log speed over segments a..b, with a sparse table
        n = len(lsp)
        lg = max(1, int(np.floor(np.log2(n))) + 1)
        mx = np.full((lg, n), -np.inf)
        mn = np.full((lg, n), np.inf)
        mx[0], mn[0] = lsp, lsp
        for j in range(1, lg):
            h = 1 << (j - 1)
            mx[j, :n - h] = np.maximum(mx[j - 1, :n - h], mx[j - 1, h:])
            mn[j, :n - h] = np.minimum(mn[j - 1, :n - h], mn[j - 1, h:])
        aa, bb = a[multi], b[multi]
        j = np.floor(np.log2(bb - aa + 1)).astype(int)
        e2 = bb - (1 << j) + 1
        swing[multi] = np.maximum(mx[j, aa], mx[j, e2]) - np.minimum(mn[j, aa], mn[j, e2])
    return np.column_stack([np.log(np.clip(vis, 0.05, 20)), ratio, np.minimum(swing, 6) / 2,
                            np.log1p(np.maximum(b - a, 0)) / 3]).astype(np.float32)


def encode(m: mania.ManiaMap, tol: float = 2.0, visual: bool = False) -> Events:
    K = m.keys
    times = np.concatenate([m.start, m.end[m.is_ln]])
    times.sort()
    # merge events closer than tol ms
    keep = np.concatenate([[True], np.diff(times) > tol])
    ev = times[keep]
    L = len(ev)
    idx_start = np.clip(np.searchsorted(ev, m.start + tol, side="right") - 1, 0, L - 1)
    cols = np.zeros((L, K, 3), np.uint8)
    cols[idx_start, m.col, 0] = 1
    ln = m.is_ln
    idx_end = np.clip(np.searchsorted(ev, m.end[ln] + tol, side="right") - 1, 0, L - 1)
    cols[idx_end, m.col[ln], 1] = 1
    # held: events strictly between a long note's head and tail
    held = np.zeros((L + 1, K), np.int16)
    np.add.at(held, (idx_start[ln] + 1, m.col[ln]), 1)
    np.add.at(held, (idx_end, m.col[ln]), -1)
    cols[:, :, 2] = (np.cumsum(held, 0)[:L] > 0)

    gap = np.diff(ev)
    dt_prev = np.concatenate([[2000.0], gap])
    dt_next = np.concatenate([gap, [2000.0]])
    st, _, sp = mania.scroll_profile(m, float(ev[0]), float(ev[-1]) + 1)
    speed = sp[np.clip(np.searchsorted(st, ev, side="right") - 1, 0, len(sp) - 1)]
    cont = np.column_stack([
        np.log1p(np.minimum(dt_prev, 2000)) / 4, np.log1p(np.minimum(dt_next, 2000)) / 4,
        np.log(np.clip(speed, 0.05, 20))]).astype(np.float16)
    if visual:
        cont = np.concatenate([cont, _visual(m, ev).astype(np.float16)], 1)
    return Events(K, ((ev - ev[0]) / 1000).astype(np.float32), cols, cont)


def drop_grid(cont: np.ndarray) -> np.ndarray:
    """Cached event features from before 2026-11-02 -> the grid-free layout (exactly what encode now
    returns): keep log dt_prev, log dt_next, log scroll speed, and any visual columns after them."""
    if cont.shape[1] < OLD_N_CONT:
        return cont
    return np.concatenate([cont[:, [0, 1, 3]], cont[:, OLD_N_CONT:]], 1)


def pad_columns(cols: np.ndarray, keys: int, mirror: bool = False) -> np.ndarray:
    """(L, keys, 3) -> (L, KMAX * 3) float32, centred, optionally mirrored left-right."""
    if mirror:
        cols = cols[:, ::-1]
    out = np.zeros((len(cols), KMAX, 3), np.float32)
    off = (KMAX - keys) // 2
    out[:, off:off + keys] = cols[:, :KMAX]
    return out.reshape(len(cols), -1)
