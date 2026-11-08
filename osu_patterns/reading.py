"""Reading features: what scroll speed changes do to the notes on screen.

The SV statistics in mania.stats describe the speed values (how far from 1.0x, how often they
change). Reading and memorisation are about what the player sees: notes that appear late, crawl,
bunch up into a stack, or move while on screen. With x(t) the scroll position (the integral of the
effective speed), a note hit at T is at height x(T) - x(t) at time t and becomes visible when that
height drops below the screen height H. For constant speed every note is visible for H ms and the
on-screen gap between two rows equals their time gap; SV breaks both.
"""
from __future__ import annotations

import numpy as np

from .mania import ManiaMap, _rows, scroll_profile

H = 600.0  # visible height in ms at the main BPM and 1.0x SV, a typical mania scroll speed


def _position(st: np.ndarray, du: np.ndarray, sp: np.ndarray):
    """x(t) as a piecewise-linear function: knots (times, positions)."""
    kt = np.append(st, st[-1] + du[-1])
    kx = np.concatenate([[0.0], np.cumsum(du * sp)])
    return kt, kx


def reading_features(m: ManiaMap) -> dict[str, float]:
    rt, _ = _rows(m)
    if len(rt) < 20:
        return {}
    t0, t1 = float(rt[0]) - 4 * H, float(rt[-1]) + 1.0
    st, du, sp = scroll_profile(m, t0, t1)
    kt, kx = _position(st, du, sp)
    x = np.interp(rt, kt, kx)
    # time each row becomes visible, and how long it is on screen before it must be hit
    onset = np.interp(x - H, kx, kt)
    vis = (rt - onset) / H
    f = {}
    for q in (5, 25, 50, 95):
        f[f"rd_vis_p{q}"] = float(np.percentile(vis, q))
    f["rd_vis_short"] = float(np.mean(vis < 0.5))     # rows that pop up late
    f["rd_vis_sudden"] = float(np.mean(vis < 0.25))
    f["rd_vis_long"] = float(np.mean(vis > 2.0))      # rows that crawl
    f["rd_vis_logstd"] = float(np.std(np.log(np.clip(vis, 0.01, None))))
    # on-screen gap vs time gap between consecutive rows (1 = constant scroll)
    dt, dx = np.diff(rt), np.diff(x)
    ok = dt > 1
    ratio = np.log(np.clip(dx[ok] / dt[ok], 0.01, 100))
    if ok.sum() > 5:
        f["rd_gap_logstd"] = float(np.std(ratio))
        f["rd_gap_p5"], f["rd_gap_p95"] = (float(np.percentile(ratio, q)) for q in (5, 95))
        f["rd_stacked"] = float(np.mean(ratio < np.log(0.2)))   # rows drawn almost on top of each other
        f["rd_spread"] = float(np.mean(ratio > np.log(3.0)))
        # how much the gap ratio jumps from one row to the next: irregular, hard-to-read spacing
        f["rd_gap_jitter"] = float(np.mean(np.abs(np.diff(ratio)))) if len(ratio) > 2 else 0.0
    # speed at the moment each row is hit, and how much it varies locally (2 s neighbourhoods)
    seg = np.clip(np.searchsorted(st, rt, side="right") - 1, 0, len(sp) - 1)
    lsp = np.log(np.clip(sp[seg], 0.01, None))
    f["rd_hit_speed_std"] = float(np.std(lsp))
    blk = ((rt - rt[0]) // 2000).astype(int)
    loc = [np.std(lsp[blk == b]) for b in np.unique(blk) if (blk == b).sum() >= 4]
    f["rd_hit_speed_local_std"] = float(np.mean(loc)) if loc else 0.0
    # speed changes while rows are on screen (they make visible notes jump, stop or rush)
    ch = st[1:][np.abs(np.diff(np.log(np.clip(sp, 0.01, None)))) > 0.05]
    if len(ch):
        nxt = np.searchsorted(rt, ch)
        on = nxt < len(rt)
        on[on] &= onset[nxt[on]] <= ch[on]   # the next row is already visible when the speed changes
        dur_s = max((rt[-1] - rt[0]) / 1000, 1.0)
        f["rd_moving_changes_ps"] = float(on.sum() / dur_s)
        f["rd_moving_changes_frac"] = float(on.mean())
    else:
        f["rd_moving_changes_ps"] = f["rd_moving_changes_frac"] = 0.0
    # short extreme segments while rows are visible: teleports (very fast) and freezes (near stop)
    mid = st + du / 2
    nxt = np.clip(np.searchsorted(rt, mid), 0, len(rt) - 1)
    visible = onset[nxt] <= mid
    dur_m = max((rt[-1] - rt[0]) / 60000, 0.1)
    f["rd_teleports_pm"] = float(np.sum(visible & (sp > 4) & (du < 80)) / dur_m)
    f["rd_freezes_pm"] = float(np.sum(visible & (sp < 0.1) & (du > 150)) / dur_m)
    # notes on screen at once, relative to constant scroll
    crowd = np.searchsorted(x, x + H) - np.arange(len(x))
    base = np.searchsorted(rt, rt + H) - np.arange(len(rt))
    f["rd_crowd_ratio_p95"] = float(np.percentile(crowd / np.maximum(base, 1), 95))
    f["rd_crowd_ratio_p5"] = float(np.percentile(crowd / np.maximum(base, 1), 5))
    return f


# ---------------------------------------------------------------------- SV gate
# Komi (2026-11-02): SV Reading / Memo never fire on a chart whose scroll speed never changes in a
# way the player can see. "Effective" is what reaches the screen: BPM x SV relative to the main BPM
# (so normalising SV after a BPM change, or a chart played at one constant SV, is no change), measured
# on the notes themselves. Slowjams and fastjams (a burst drawn bunched up or stretched out by slow
# or fast scroll) are SV and pass; so do teleports, stops and BPM changes left un-normalised.
GATE_RATIO = float(np.log(1.25))   # a row counts as moved when its on-screen gap or its time on screen
                                   # is off the chart's usual value by more than 25%
# The gate only rules out charts with *no* visible scroll change; how much SV makes a chart Reading or
# Memo is the heads' call. So any one of these is enough:
GATE_MIN_ROWS = 4                  # rows drawn moved (bunched up, stretched, popping up late or early)
GATE_MIN_OFF_MS = 1000.0           # notes on screen while the speed is off by > 25% (stutter SV that lands
                                   # every row at its usual gap still moves the notes while they fall)
GATE_MIN_JUMPS = 1                 # a teleport or stop while notes are on screen


def _scroll_rows(m: ManiaMap):
    """Row times; rows drawn moved; those plus rows on screen during off-speed scroll or a teleport / stop
    (where SV may light up locally); off-speed on-screen ms; teleports / stops."""
    rt, _ = _rows(m)
    none = (rt, np.zeros(len(rt), bool), np.zeros(len(rt), bool), 0.0, 0)
    if len(rt) < 2:
        return none
    t0, t1 = float(rt[0]) - 4 * H, float(rt[-1]) + 1.0
    st, du, sp = scroll_profile(m, t0, t1)
    if len(sp) < 2 or np.ptp(np.log(np.clip(sp, 0.01, None))) < 0.05:
        return none   # one speed over the whole chart: nothing to read
    kt, kx = _position(st, du, sp)
    x = np.interp(rt, kt, kx)
    moved = np.zeros(len(rt), bool)
    # on-screen gap / time gap between consecutive rows, against the chart's median
    dt, dx = np.diff(rt), np.diff(x)
    ok = dt > 1
    if ok.sum() >= 2:
        g = np.log(np.clip(dx[ok] / dt[ok], 1e-3, 1e3))
        moved[1:][ok] |= np.abs(g - np.median(g)) > GATE_RATIO
    # time each row is on screen (scaled so the chart's median speed shows H ms)
    seg = np.clip(np.searchsorted(st, rt, side="right") - 1, 0, len(sp) - 1)
    med = float(np.exp(np.median(np.log(np.clip(sp[seg], 0.01, None)))))
    onset = np.interp(x - H * med, kx, kt)
    vis = np.log(np.clip((rt - onset) / H, 1e-3, None))
    moved |= np.abs(vis - np.median(vis)) > GATE_RATIO
    # off-speed time and teleports / stops while a row is on screen; they mark the next row to be hit
    mid = st + du / 2
    nxt = np.clip(np.searchsorted(rt, mid), 0, len(rt) - 1)
    visible = (onset[nxt] <= mid) & (mid <= rt[nxt])
    rel = sp / med
    off = visible & (np.abs(np.log(np.clip(rel, 1e-3, None))) > GATE_RATIO)
    jump = visible & (((rel > 4) & (du < 80)) | ((rel < 0.1) & (du > 30)))
    affected = moved.copy()
    affected[nxt[off | jump]] = True
    return rt, moved, affected, float(du[off].sum()), int(jump.sum())


def _passes(n_moved: int, off_ms: float, jumps: int) -> bool:
    return n_moved >= GATE_MIN_ROWS or off_ms >= GATE_MIN_OFF_MS or jumps >= GATE_MIN_JUMPS


def scroll_effect(m: ManiaMap) -> dict[str, float]:
    """How much the scroll speed visibly moves the notes, relative to the chart's own typical scroll.

    sve_rows      rows whose on-screen gap to the previous row, or whose time on screen, is off by > 25%
    sve_frac      the same as a share of the rows
    sve_off_frac  share of the chart spent at a speed off by > 25% while notes are on screen
    sve_jumps     teleports / stops while notes are on screen
    sve_effective 1.0 when the chart passes the SV gate, else 0.0
    """
    rt, moved, _, off_ms, jumps = _scroll_rows(m)
    n_moved = int(moved.sum())
    span = max(float(rt[-1] - rt[0]), 1.0) if len(rt) > 1 else 1.0
    return {"sve_rows": float(n_moved), "sve_frac": float(n_moved / max(len(rt), 1)),
            "sve_off_frac": off_ms / span, "sve_jumps": float(jumps),
            "sve_effective": float(_passes(n_moved, off_ms, jumps))}


def has_effective_sv(m: ManiaMap) -> bool:
    return scroll_effect(m)["sve_effective"] > 0


def sv_note_gate(m: ManiaMap, lookback_ms: float = 2000.0) -> np.ndarray:
    """Per note (index-aligned with m.start): 1.0 where SV may light up, else 0.0.

    All zeros when the chart fails the gate. Otherwise a note is open when a row the scroll visibly
    moves was hit in the `lookback_ms` before it, or is its own row. Only the past is used, so this
    fits the causal note model (a section lights up from its first moved row on, not before).
    """
    rt, moved, affected, off_ms, jumps = _scroll_rows(m)
    n_moved = int(moved.sum())
    out = np.zeros(len(m.start), np.float32)
    if not _passes(n_moved, off_ms, jumps):
        return out
    row = np.clip(np.searchsorted(rt, np.asarray(m.start, float) + 2.0, side="right") - 1, 0, len(rt) - 1)
    mt = rt[affected]
    if not len(mt):
        return out
    last = np.searchsorted(mt, rt, side="right") - 1          # latest moved row at or before each row
    ok = (last >= 0) & (rt - mt[np.clip(last, 0, None)] <= lookback_ms)
    return ok[row].astype(np.float32)


def sv_window_gate(m: ManiaMap, centres, win: float) -> np.ndarray:
    """Per window centre: 1.0 when the window [c - win/2, c + win/2] holds a moved row, else 0.0."""
    rt, moved, affected, off_ms, jumps = _scroll_rows(m)
    c = np.asarray(centres, float)
    if not _passes(int(moved.sum()), off_ms, jumps):
        return np.zeros(len(c), np.float32)
    mt = rt[affected]
    lo, hi = np.searchsorted(mt, c - win / 2), np.searchsorted(mt, c + win / 2, side="right")
    return (hi > lo).astype(np.float32)


# ---------------------------------------------------------------------- Memo gate
# Komi (2026-11-02): a constant slowjam is Reading, not Memo. The per-note Memo head had learned "far
# from the main speed" (its 43 positives are teleports, stops and fast flips), so a steady 0.5x slowjam
# scored 0.4 on it. Memo is about speed that *changes* under notes that are already on screen, so the
# per-note and window Memo scores are open only near a visible speed step (a jump of 1.5x or more at
# a segment boundary; the 1-2% steps of a smooth ramp don't count).
STEP_RATIO = 1.5


def _step_times(m: ManiaMap):
    rt, _ = _rows(m)
    if len(rt) < 2:
        return rt, np.zeros(0)
    st, du, sp = scroll_profile(m, float(rt[0]) - 4 * H, float(rt[-1]) + 1.0)
    if len(sp) < 2:
        return rt, np.zeros(0)
    kt, kx = _position(st, du, sp)
    x = np.interp(rt, kt, kx)
    ch = st[1:][np.abs(np.diff(np.log(np.clip(sp, 0.01, None)))) > np.log(STEP_RATIO)]
    nxt = np.clip(np.searchsorted(rt, ch), 0, len(rt) - 1)
    onset = np.interp(x - H, kx, kt)
    return rt, ch[(onset[nxt] <= ch) & (ch <= rt[nxt])]


def memo_note_gate(m: ManiaMap, lookback_ms: float = 2000.0) -> np.ndarray:
    """Per note: 1.0 from a visible speed step on, for `lookback_ms` (only the past), else 0.0."""
    _, ch = _step_times(m)
    t = np.asarray(m.start, float)
    if not len(ch):
        return np.zeros(len(t), np.float32)
    last = np.searchsorted(ch, t + 2.0, side="right") - 1
    return ((last >= 0) & (t - ch[np.clip(last, 0, None)] <= lookback_ms)).astype(np.float32)


def memo_window_gate(m: ManiaMap, centres, win: float) -> np.ndarray:
    """Per window centre: 1.0 when the window holds a visible speed step, else 0.0."""
    _, ch = _step_times(m)
    c = np.asarray(centres, float)
    if not len(ch):
        return np.zeros(len(c), np.float32)
    return (np.searchsorted(ch, c + win / 2, side="right") > np.searchsorted(ch, c - win / 2)).astype(np.float32)
