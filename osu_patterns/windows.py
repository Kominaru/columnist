"""Section-level statistics: mania.stats on sliding windows, pooled back to one row per chart.

Chart-level statistics dilute short passages (a 20-second LN release section in a rice chart).
Computing the same statistics on overlapping windows and pooling them with max / p75 / std
keeps "somewhere in this chart there is X" visible to a chart-level head. This is the
multi-instance baseline that needs no deep learning.
"""
from __future__ import annotations

import numpy as np

from . import mania

WIN_MS, HOP_MS, MIN_NOTES = 16000.0, 8000.0, 30
# whole-chart constants, or counts that only measure the window length
SKIP = {"keys", "od", "hp", "notes", "duration_s", "bpm_main", "bpm_changes", "sv_points"}


def submap(m: mania.ManiaMap, t0: float, t1: float) -> mania.ManiaMap:
    keep = (m.start >= t0) & (m.start < t1)
    return mania.ManiaMap(m.keys, m.od, m.hp, m.title, m.version, m.timing,
                          m.start[keep], m.end[keep], m.col[keep])


def window_stats(m: mania.ManiaMap, win: float = WIN_MS, hop: float = HOP_MS) -> list[dict]:
    t0, t1 = float(m.start.min()), float(m.start.max())
    out = []
    for a in np.arange(t0, max(t1 - win, t0) + 1, hop):
        w = submap(m, a, a + win)
        if len(w.start) < MIN_NOTES:
            continue
        try:
            s = mania.stats(w)
        except (ValueError, IndexError, ZeroDivisionError):
            continue
        out.append({"t": a, **{k: v for k, v in s.items() if k not in SKIP}})
    return out


def pool(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    keys = [k for k in rows[0] if k != "t"]
    a = np.array([[r[k] for k in keys] for r in rows], dtype=float)
    f = {}
    for name, v in (("max", np.nanmax(a, 0)), ("p75", np.nanpercentile(a, 75, 0)), ("std", np.nanstd(a, 0))):
        f.update({f"w{name}:{k}": float(x) for k, x in zip(keys, v)})
    return f
