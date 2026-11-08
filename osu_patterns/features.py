"""Turn a parsed beatmap into (a) per-transition arrays and (b) pattern segments.

Everything downstream (heuristic tagger, ML features) is built on these two views:
  * Transitions: one row per consecutive pair of playable objects (rhythm, spacing, angle).
  * Segments: contiguous runs that read as one pattern (a burst, a jump section, ...),
    with timestamps in osu!'s modding format so they can be pasted into a mod post.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .parser import SPINNER, Beatmap, HitObject

SNAP_DENOMS = (1, 2, 3, 4, 6, 8, 12, 16)


def snap_divisor(ratio: float, tol: float = 0.07) -> int:
    """Smallest beat divisor d such that ratio (dt / beat) is a multiple of 1/d."""
    for d in SNAP_DENOMS:
        if abs(ratio * d - round(ratio * d)) < tol * max(1, d / 4) and round(ratio * d) > 0:
            return d
    return 0  # unsnapped / irregular


def osu_timestamp(ms: float) -> str:
    ms = int(round(ms))
    return f"{ms // 60000:02d}:{(ms // 1000) % 60:02d}:{ms % 1000:03d}"


@dataclass
class Transitions:
    """Arrays indexed by object i (transition i-1 -> i); index 0 is padding."""
    t: np.ndarray          # object start time (ms)
    dt: np.ndarray         # start-to-start interval (ms)
    gap: np.ndarray        # end-of-previous to start interval (ms)
    beat: np.ndarray       # ms per beat at object i
    ratio: np.ndarray      # dt / beat
    divisor: np.ndarray    # snap divisor (1, 2, 3, 4, 6, 8, ...; 0 = irregular)
    dist: np.ndarray       # movement distance end(i-1) -> start(i), in circle diameters
    dist_px: np.ndarray    # same, osu!pixels
    angle: np.ndarray      # angle at object i (deg): 0 = back-and-forth, 180 = straight line
    is_slider: np.ndarray
    combo: list[int]       # combo number of each object (for timestamps)


@dataclass
class Segment:
    kind: str              # e.g. "stream", "burst", "jump", "spaced_stream"
    start: int             # object index (inclusive)
    end: int               # object index (inclusive)
    t0: float
    t1: float
    info: dict

    @property
    def n(self) -> int:
        return self.end - self.start + 1

    def timestamp(self, combo: list[int]) -> str:
        nums = combo[self.start : self.end + 1]
        if len(nums) > 8:
            label = f"{nums[0]},{nums[1]},...,{nums[-1]}"
        else:
            label = ",".join(map(str, nums))
        return f"{osu_timestamp(self.t0)} ({label})"


def playable(bm: Beatmap) -> list[HitObject]:
    return [o for o in bm.objects if o.kind != SPINNER]


def compute_transitions(bm: Beatmap) -> Transitions:
    objs = playable(bm)
    n = len(objs)
    diam = 2 * bm.radius
    t = np.array([o.time for o in objs], dtype=float)
    beat = np.array([o.beat_length for o in objs], dtype=float)
    dt = np.zeros(n)
    gap = np.zeros(n)
    dist_px = np.zeros(n)
    angle = np.full(n, np.nan)
    if n > 1:
        dt[1:] = np.diff(t)
        gap[1:] = [objs[i].time - objs[i - 1].end_time for i in range(1, n)]
        dist_px[1:] = [math.hypot(objs[i].x - objs[i - 1].end_x, objs[i].y - objs[i - 1].end_y) for i in range(1, n)]
    for i in range(1, n - 1):
        a = (objs[i - 1].end_x - objs[i].x, objs[i - 1].end_y - objs[i].y)
        b = (objs[i + 1].x - objs[i].end_x, objs[i + 1].y - objs[i].end_y)
        na, nb = math.hypot(*a), math.hypot(*b)
        if na > 1 and nb > 1:
            c = (a[0] * b[0] + a[1] * b[1]) / (na * nb)
            angle[i] = math.degrees(math.acos(max(-1.0, min(1.0, c))))
    ratio = np.divide(dt, beat, out=np.zeros(n), where=beat > 0)
    divisor = np.array([snap_divisor(r) if i else 0 for i, r in enumerate(ratio)])

    combo, c = [], 0
    for i, o in enumerate(objs):
        c = 1 if (o.new_combo or i == 0) else c + 1
        combo.append(c)

    return Transitions(t, dt, gap, beat, ratio, divisor, dist_px / diam, dist_px, angle,
                       np.array([o.kind == "slider" for o in objs]), combo)


# ---------------------------------------------------------------- segmentation

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True in a transition mask -> (first_object, last_object)."""
    out, i, n = [], 1, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i - 1, j))
            i = j + 1
        else:
            i += 1
    return out


def stream_mask(tr: Transitions) -> np.ndarray:
    """Transition is a 'tap stream' step: 1/4-or-faster at a tappable speed, from a circle
    (a slider in the middle of a stream breaks it, but a slider may end a run)."""
    fast = (tr.ratio <= 0.27) & (tr.dt <= 165) & (tr.dt > 0)
    fast |= (tr.dt > 0) & (tr.dt <= 95)  # very fast 1/2 at high BPM still reads as a stream
    prev_slider = np.concatenate([[False], tr.is_slider[:-1]])
    return fast & ~prev_slider


def jump_mask(tr: Transitions) -> np.ndarray:
    """Transition is a jump: 1/2-ish timing (not a stream tap), large movement, short gap."""
    rhythmic = (tr.ratio > 0.27) & (tr.ratio <= 1.05) & (tr.dt <= 450)
    return rhythmic & (tr.dist >= 1.6) & (tr.gap < 400)


def segments(bm: Beatmap, tr: Transitions) -> list[Segment]:
    segs: list[Segment] = []
    sm = stream_mask(tr)
    for a, b in _runs(sm):
        n = b - a + 1
        idx = slice(a + 1, b + 1)
        spacing = float(np.mean(tr.dist[idx]))
        bpm_eq = 15000.0 / float(np.median(tr.dt[idx]))
        ang = tr.angle[a + 1 : b]
        kind = ("double" if n == 2 else "triple" if n == 3 else "quad" if n == 4
                else "burst" if n <= 9 else "stream" if n <= 32 else "deathstream")
        segs.append(Segment(kind, a, b, tr.t[a], tr.t[b], {
            "notes": n, "spacing": round(spacing, 2), "bpm_1_4": round(bpm_eq),
            "mean_angle": float(np.nanmean(ang)) if np.isfinite(ang).any() else float("nan"),
        }))

    jm = jump_mask(tr)
    for a, b in _runs(jm):
        if b - a + 1 < 4:  # need at least 3 consecutive jumps
            continue
        idx = slice(a + 1, b + 1)
        ang = tr.angle[a + 1 : b]
        segs.append(Segment("jumps", a, b, tr.t[a], tr.t[b], {
            "notes": b - a + 1,
            "spacing": round(float(np.mean(tr.dist[idx])), 2),
            "mean_angle": float(np.nanmean(ang)) if np.isfinite(ang).any() else float("nan"),
            "max_px": round(float(np.max(tr.dist_px[idx]))),
            "shape": polygon_shape(bm, a, b),
        }))
    segs.sort(key=lambda s: s.t0)
    return segs


def polygon_shape(bm: Beatmap, a: int, b: int) -> str:
    """Detect repeating geometric jump shapes: an object returns near the position k objects back."""
    objs = playable(bm)
    r = bm.radius
    best, best_rate = "freeform", 0.0
    for k, name in ((2, "back and forth"), (3, "triangles"), (4, "squares"), (5, "stars")):
        hits = tot = 0
        for i in range(a + k, b + 1):
            tot += 1
            if math.hypot(objs[i].x - objs[i - k].x, objs[i].y - objs[i - k].y) < 1.2 * r:
                hits += 1
        if tot >= 2 and hits / tot >= 0.6 and hits / tot > best_rate + 0.15:
            best, best_rate = name, hits / tot
    return best
