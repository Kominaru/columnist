"""osu!mania .osu parsing and pattern statistics.

A mania chart is reduced to *rows* (notes sharing a timestamp) over K columns. Every
statistic below is something a tournament mappooler looks at when slotting a map:

* density / speed       notes per second, peak sections, stamina (time spent near peak)
* chords                share of single / jump (2) / hand (3) / quad (4+) rows
* jacks                 notes repeating the previous row's column; chordjacks; jack speed
* streams               rows sharing no column with the previous row (jumpstream, handstream)
* rhythm / tech         interval variety, gap changes (ms only; no snap divisors or BPM grid)
* long notes            LN share, how much of the chart is held, LN length, inverse LN
* scroll speed (SV)     effective scroll = (bpm / main bpm) * sv; how much and how often it moves

Times are in ms. Everything is normalised by note count or duration so 4K and 7K maps are
comparable; `keys` is a feature on its own.
"""
from __future__ import annotations

import gzip
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class TimingPoint:
    time: float
    beat_length: float
    uninherited: bool


@dataclass
class ManiaMap:
    keys: int
    od: float
    hp: float
    title: str
    version: str
    timing: list[TimingPoint]
    # one entry per note: start, end (== start for a plain note), column
    start: np.ndarray = field(repr=False)
    end: np.ndarray = field(repr=False)
    col: np.ndarray = field(repr=False)

    @property
    def is_ln(self) -> np.ndarray:
        return self.end > self.start


def read_text(path: Path | str) -> str:
    p = Path(path)
    raw = p.read_bytes()
    if p.suffix == ".gz":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8-sig", errors="replace")


def parse_mania(text: str) -> ManiaMap | None:
    """Returns None for non-mania files."""
    section, kv = "", {}
    timing, objs = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section in ("General", "Metadata", "Difficulty"):
            if ":" in line:
                k, v = line.split(":", 1)
                kv[k.strip()] = v.strip()
        elif section == "TimingPoints":
            p = line.split(",")
            if len(p) < 2:
                continue
            unin = p[6].strip() == "1" if len(p) > 6 else True
            timing.append(TimingPoint(float(p[0]), float(p[1]), unin))
        elif section == "HitObjects":
            p = line.split(",")
            if len(p) < 5:
                continue
            x, t, typ = float(p[0]), float(p[2]), int(p[3])
            end = t
            if typ & 128 and len(p) > 5:
                end = float(p[5].split(":")[0])
            objs.append((t, max(end, t), x))
    if kv.get("Mode", "0") != "3" or not objs:
        return None
    keys = int(round(float(kv.get("CircleSize", 4))))
    keys = min(max(keys, 1), 18)
    objs.sort()
    a = np.array(objs, dtype=float)
    col = np.clip(np.floor(a[:, 2] * keys / 512), 0, keys - 1).astype(int)
    timing.sort(key=lambda tp: tp.time)
    return ManiaMap(keys, float(kv.get("OverallDifficulty", 8)), float(kv.get("HPDrainRate", 8)),
                    f"{kv.get('Artist', '')} - {kv.get('Title', '')}", kv.get("Version", ""),
                    timing, a[:, 0], a[:, 1], col)


def load(path: Path | str) -> ManiaMap | None:
    return parse_mania(read_text(path))


# ---------------------------------------------------------------- helpers

def _rows(m: ManiaMap, tol: float = 2.0):
    """Group notes into rows; returns (row_times, list of column sets)."""
    times, cols = [], []
    for t, c in zip(m.start, m.col):
        if times and t - times[-1] <= tol:
            cols[-1].add(int(c))
        else:
            times.append(float(t))
            cols.append({int(c)})
    return np.array(times), cols


def main_bpm(m: ManiaMap) -> float:
    """The BPM covering most of the chart (for display only; no model input uses it)."""
    red = [tp for tp in m.timing if tp.uninherited and tp.beat_length > 0]
    if not red:
        return 120.0
    t0, t1 = float(m.start.min()), float(m.end.max())
    dur = Counter()
    for i, tp in enumerate(red):
        nxt = red[i + 1].time if i + 1 < len(red) else t1
        dur[round(tp.beat_length, 3)] += max(min(nxt, t1) - max(tp.time, t0), 0)
    return 60000.0 / max(dur, key=dur.get)


def scroll_profile(m: ManiaMap, t0: float, t1: float):
    """Piecewise-constant effective scroll speed over [t0, t1].

    Returns (segment start times, durations, speeds) with speed 1 = the main BPM at 1.0x SV.
    The main BPM is the one covering most of the drain time, as osu! itself picks it.
    """
    red = [tp for tp in m.timing if tp.uninherited and tp.beat_length > 0]
    if not red or t1 <= t0:
        return np.array([t0]), np.array([max(t1 - t0, 1.0)]), np.array([1.0])
    # main beat length = longest-lasting one
    dur = Counter()
    for i, tp in enumerate(red):
        nxt = red[i + 1].time if i + 1 < len(red) else t1
        dur[round(tp.beat_length, 3)] += max(min(nxt, t1) - max(tp.time, t0), 0)
    main = max(dur, key=dur.get) if sum(dur.values()) > 0 else red[0].beat_length
    events = []  # (time, kind, value) kind 0 = red resets sv, 1 = green sets sv
    for tp in m.timing:
        if tp.uninherited and tp.beat_length > 0:
            events.append((tp.time, 0, tp.beat_length))
        elif not tp.uninherited and tp.beat_length < 0:
            events.append((tp.time, 1, min(max(-100.0 / tp.beat_length, 0.01), 10.0)))
    events.sort()
    bl, sv = red[0].beat_length, 1.0
    starts, speeds = [], []
    for t, kind, v in events:
        if kind == 0:
            bl, sv = v, 1.0
        else:
            sv = v
        speeds.append((main / bl) * sv)
        starts.append(t)
    starts = np.array(starts)
    speeds = np.array(speeds)
    # clip to [t0, t1]
    first = max(np.searchsorted(starts, t0, side="right") - 1, 0)
    starts, speeds = starts[first:].copy(), speeds[first:].copy()
    starts[0] = t0
    keep = starts < t1
    starts, speeds = starts[keep], speeds[keep]
    durs = np.diff(np.append(starts, t1))
    return starts, np.maximum(durs, 0), speeds


# ---------------------------------------------------------------- statistics
INVERSE_GAP_MS = 150.0  # an LN end closer than this to the next note in its column counts as inverse
# (was 0.3 beat, 150 ms at 120 BPM; models must not depend on the BPM grid)


def stats(m: ManiaMap) -> dict[str, float]:
    K = m.keys
    n = len(m.start)
    t0, t1 = float(m.start.min()), float(m.end.max())
    dur_s = max((t1 - t0) / 1000, 1.0)
    s: dict[str, float] = {"keys": K, "od": m.od, "hp": m.hp, "notes": n, "duration_s": dur_s}

    # ---- density / speed
    s["nps"] = n / dur_s
    s["nps_per_key"] = s["nps"] / K
    win = 2000.0
    edges = np.arange(t0, t1 + win, win)
    counts, _ = np.histogram(m.start, bins=edges) if len(edges) > 1 else (np.array([n]), None)
    rate = counts / (win / 1000)
    s["nps_p50"], s["nps_p90"], s["nps_max"] = (float(np.percentile(rate, q)) for q in (50, 90, 100))
    s["nps_peak_ratio"] = s["nps_max"] / max(s["nps_p50"], 0.1)
    s["stamina"] = float(np.mean(rate >= 0.8 * s["nps_p90"])) if s["nps_p90"] > 0 else 0.0
    s["density_cv"] = float(np.std(rate) / max(np.mean(rate), 1e-6))
    s["breaks"] = float(np.mean(counts == 0))

    # ---- rows, chords, jacks, streams
    rt, rc = _rows(m)
    R = len(rt)
    sizes = np.array([len(c) for c in rc])
    s["rows_ps"] = R / dur_s
    s["notes_per_row"] = n / max(R, 1)
    for k, name in ((1, "single"), (2, "jump"), (3, "hand")):
        s[f"row_{name}"] = float(np.mean(sizes == k))
    s["row_quad"] = float(np.mean(sizes >= 4))
    s["row_chord_frac_of_keys"] = float(np.mean(sizes / K))
    s["row_full"] = float(np.mean(sizes >= K))
    gaps = np.diff(rt) if R > 1 else np.array([1000.0])
    overlap = np.array([len(rc[i] & rc[i - 1]) for i in range(1, R)]) if R > 1 else np.zeros(1)
    fast = gaps < 220  # roughly 1/4 at 270 bpm .. 1/2 at 136 bpm: "in a pattern", not a pause
    jack_rows = (overlap > 0) & fast
    s["jack_row"] = float(np.mean(jack_rows))
    s["jack_note"] = float(overlap[fast].sum() / max(n, 1))
    s["chordjack_row"] = float(np.mean(jack_rows & (sizes[1:] >= 2) & (overlap >= 1)
                                       & (np.array([len(c) for c in rc[:-1]]) >= 2)))
    s["full_jack_row"] = float(np.mean(fast & (overlap == sizes[1:]) & (overlap > 0)))
    s["stream_row"] = float(np.mean((overlap == 0) & fast))
    s["jumpstream_row"] = float(np.mean((overlap == 0) & fast & (sizes[1:] == 2)))
    s["handstream_row"] = float(np.mean((overlap == 0) & fast & (sizes[1:] >= 3)))
    jg = gaps[jack_rows]
    s["jack_gap_ms"] = float(np.median(jg)) if len(jg) else 400.0
    s["row_gap_ms_p10"] = float(np.percentile(gaps, 10))
    s["row_gap_ms_p50"] = float(np.median(gaps))
    # jack chains (same column >= 3 consecutive fast rows)
    longest, cur = 0, {}
    chain_hits = 0
    for i in range(1, R):
        if not fast[i - 1]:
            cur = {}
            continue
        nxt = {c: cur.get(c, 1) + 1 for c in rc[i] if c in rc[i - 1]}
        chain_hits += sum(1 for v in nxt.values() if v >= 3)
        cur = nxt
        if nxt:
            longest = max(longest, max(nxt.values()))
    s["jack_chain3"] = chain_hits / max(n, 1)
    s["jack_longest"] = float(longest)
    # trills: column set alternates A,B,A,B
    trill = sum(1 for i in range(3, R) if fast[i - 1] and fast[i - 2] and fast[i - 3]
                and rc[i] == rc[i - 2] and rc[i - 1] == rc[i - 3] and not (rc[i] & rc[i - 1]))
    s["trill_row"] = trill / max(R, 1)
    # anchors: a column carrying far more than its share in 4-second windows
    colcount = np.bincount(m.col, minlength=K) / max(n, 1)
    s["col_imbalance"] = float(colcount.max() * K)
    anc = []
    for a in range(0, len(m.start), 32):
        c = np.bincount(m.col[a:a + 64], minlength=K)
        if c.sum() >= 32:
            anc.append(c.max() / c.sum() * K)
    s["anchor"] = float(np.percentile(anc, 90)) if anc else 1.0

    # ---- rhythm (ms only: no snap divisors or BPM grid, which live-timed maps don't have)
    # local rhythm variety: distinct rounded gaps per 16 rows
    var = [len({round(g / 5) for g in gaps[i:i + 16] if g < 400}) for i in range(0, len(gaps), 16)]
    s["rhythm_variety"] = float(np.mean(var)) if var else 0.0
    s["gap_ratio_change"] = float(np.mean(np.abs(np.diff(np.log(np.maximum(gaps, 5))))[fast[1:]] > 0.3)) \
        if len(gaps) > 2 and fast[1:].any() else 0.0

    # ---- long notes
    ln = m.is_ln
    lens = (m.end - m.start)[ln]
    s["ln_ratio"] = float(np.mean(ln))
    s["ln_count_ps"] = float(ln.sum() / dur_s)
    s["ln_len_ms_p50"] = float(np.median(lens)) if len(lens) else 0.0
    s["ln_short"] = float(np.mean(lens < 120)) if len(lens) else 0.0
    s["ln_long"] = float(np.mean(lens > 1000)) if len(lens) else 0.0
    s["ln_hold_cov"] = float(lens.sum() / ((t1 - t0) * K)) if t1 > t0 else 0.0
    # inverse LN: in each column, gap from LN end to the next note's start is tiny
    inv, tot_ln = 0, 0
    rel_gaps = []
    for c in range(K):
        idx = np.where(m.col == c)[0]
        for i, j in zip(idx[:-1], idx[1:]):
            if ln[i]:
                tot_ln += 1
                g = m.start[j] - m.end[i]
                rel_gaps.append(g)
                if g < INVERSE_GAP_MS:
                    inv += 1
    s["ln_inverse"] = inv / max(tot_ln, 1)
    # releases that don't land on another note's start: separate release timing
    if ln.any():
        starts_set = np.unique(np.round(m.start))
        ends = np.round(m.end[ln])
        pos = np.searchsorted(starts_set, ends)
        near = np.minimum(np.abs(starts_set[np.clip(pos, 0, len(starts_set) - 1)] - ends),
                          np.abs(starts_set[np.clip(pos - 1, 0, len(starts_set) - 1)] - ends))
        s["ln_release_offgrid"] = float(np.mean(near > 3))
        # how many other columns are held while a note is hit (LN coordination)
        held = []
        sl, el = m.start[ln], m.end[ln]
        for t in m.start[:: max(1, n // 2000)]:
            held.append(int(np.sum((sl < t - 1) & (el > t + 1))))
        s["ln_held_mean"] = float(np.mean(held)) / K
    else:
        s["ln_release_offgrid"] = 0.0
        s["ln_held_mean"] = 0.0
    s["has_ln"] = float(s["ln_ratio"] > 0.02)

    # ---- scroll speed
    st, du, sp = scroll_profile(m, t0, t1)
    w = du / max(du.sum(), 1)
    logsp = np.log(np.maximum(sp, 0.01))
    s["sv_points"] = float(sum(1 for tp in m.timing if not tp.uninherited))
    s["sv_points_ps"] = s["sv_points"] / dur_s
    s["sv_log_std"] = float(math.sqrt(np.sum(w * (logsp - np.sum(w * logsp)) ** 2)))
    s["sv_off_frac"] = float(np.sum(w[np.abs(logsp) > 0.05]))
    s["sv_far_frac"] = float(np.sum(w[np.abs(logsp) > 0.4]))
    s["sv_stop_frac"] = float(np.sum(w[sp < 0.2]))
    s["sv_fast_frac"] = float(np.sum(w[sp > 2.0]))
    changes = np.abs(np.diff(logsp)) > 0.05
    s["sv_changes_ps"] = float(changes.sum() / dur_s)
    # SV that actually moves notes: changes weighted by notes on screen near them
    if len(st) > 1:
        note_idx = np.searchsorted(m.start, st[1:][changes])
        near = np.minimum(note_idx, n - 1)
        s["sv_changes_near_notes_ps"] = float(np.sum(np.abs(m.start[near] - st[1:][changes]) < 1500) / dur_s)
    else:
        s["sv_changes_near_notes_ps"] = 0.0
    return s


def evidence(m: ManiaMap, top: int = 3) -> dict[str, list[int]]:
    """Timestamps (ms) of the densest jack, stream, LN and SV sections, for review links."""
    rt, rc = _rows(m)
    out: dict[str, list[int]] = {}
    if len(rt) > 8:
        gaps = np.diff(rt)
        ov = np.array([len(rc[i] & rc[i - 1]) > 0 for i in range(1, len(rt))]) & (gaps < 220)
        st = (~ov) & (gaps < 220)
        for name, mask in (("jacks", ov), ("streams", st)):
            score = np.convolve(mask.astype(float), np.ones(32), "same")
            out[name] = _peaks(rt[1:], score, top)
    if m.is_ln.any():
        score = np.convolve(m.is_ln.astype(float), np.ones(32), "same")
        out["long notes"] = _peaks(m.start, score, top)
    st_, du, sp = scroll_profile(m, float(m.start.min()), float(m.end.max()))
    if len(st_) > 4:
        ch = np.r_[0, np.abs(np.diff(np.log(np.maximum(sp, 0.01))))]
        out["sv"] = _peaks(st_, np.convolve(ch, np.ones(8), "same"), top)
    return {k: v for k, v in out.items() if v}


def _peaks(times, score, top, min_sep=10000.0):
    picked = []
    for i in np.argsort(-score):
        if score[i] <= 0 or len(picked) >= top:
            break
        if all(abs(times[i] - p) > min_sep for p in picked):
            picked.append(times[i])
    return sorted(int(p) for p in picked)
