"""Heuristic tagger: interpretable scores for osu! user-tag names, plus evidence segments.

This is (1) a usable tagger on its own, with no training data, and (2) a source of
features / weak labels for the learned model in model.py. Scores are in [0, 1];
a tag is reported when its score >= 0.5. Thresholds are hand-set and meant to be
re-fit against real user tags (see scripts/calibrate.py).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .features import Segment, Transitions, compute_transitions, playable, segments
from .parser import Beatmap

# Names follow the osu! user-tag taxonomy (https://osu.ppy.sh/wiki/en/Beatmap/Beatmap_tags).
TAGS = [
    "skillset/jumps", "skillset/streams", "skillset/alt", "skillset/tech",
    "skillset/precision", "skillset/reading",
    "jumps/sharp", "jumps/wide", "jumps/linear", "jumps/back and forth",
    "jumps/triangles", "jumps/squares", "jumps/stars", "jumps/cross-screen", "jumps/stamina",
    "streams/doubles", "streams/bursts", "streams/stamina", "streams/speed",
    "streams/spaced streams", "streams/cutstreams", "streams/flow aim",
    "tech/slider tech", "reading/overlaps", "reading/visually dense",
]


def ramp(x: float, lo: float, hi: float) -> float:
    """0 at lo, 1 at hi, linear between (works for lo > hi too)."""
    if hi == lo:
        return float(x >= hi)
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


@dataclass
class TagResult:
    scores: dict[str, float]
    evidence: dict[str, list[Segment]] = field(default_factory=dict)
    stats: dict[str, float] = field(default_factory=dict)

    def tags(self, threshold: float = 0.5) -> list[str]:
        return [t for t, s in sorted(self.scores.items(), key=lambda kv: -kv[1]) if s >= threshold]


def analyse(bm: Beatmap) -> tuple[Transitions, list[Segment], dict[str, float]]:
    """Map-level statistics shared by the heuristic tagger and the ML feature vector."""
    tr = compute_transitions(bm)
    segs = segments(bm, tr)
    objs = playable(bm)
    n = max(1, len(objs))
    st: dict[str, float] = {}
    if len(objs) < 2:
        return tr, segs, {"n_objects": len(objs)}

    drain = (objs[-1].end_time - objs[0].time - sum(b - a for a, b in bm.breaks)) / 1000
    st["n_objects"] = len(objs)
    st["drain_s"] = max(1.0, drain)
    st["density"] = len(objs) / st["drain_s"]
    st["bpm"] = bm.main_bpm
    st["cs"], st["ar"], st["od"] = bm.cs, bm.ar, bm.od
    st["slider_ratio"] = float(np.mean(tr.is_slider))

    # rhythm
    div = tr.divisor[1:]
    cnt = Counter(div.tolist())
    for d in (1, 2, 3, 4, 6, 8, 0):
        st[f"snap_1_{d}" if d else "snap_irregular"] = cnt.get(d, 0) / len(div)
    p = np.array([v for v in cnt.values()], dtype=float) / len(div)
    st["rhythm_entropy"] = float(-(p * np.log2(p)).sum())
    st["triplet_mix"] = min(st["snap_1_3"] + st["snap_1_6"], st["snap_1_4"] + st["snap_1_8"]) * 2
    changes = np.sum(div[1:] != div[:-1]) / max(1, len(div) - 1)
    st["rhythm_change_rate"] = float(changes)

    # spacing / movement
    dist = tr.dist[1:]
    st["spacing_mean"] = float(np.mean(dist))
    st["spacing_p90"] = float(np.quantile(dist, 0.9))
    half = (tr.ratio > 0.4) & (tr.ratio < 0.6)
    st["spacing_half_mean"] = float(np.mean(tr.dist[half])) if half.any() else 0.0
    vel = tr.dist_px[1:] / np.maximum(tr.gap[1:], 30)
    st["velocity_p90"] = float(np.quantile(vel, 0.9))
    st["velocity_cv"] = float(np.std(vel) / (np.mean(vel) + 1e-6))
    ang = tr.angle[np.isfinite(tr.angle)]
    jm_obj = np.zeros(len(objs), bool)
    for s in segs:
        if s.kind == "jumps":
            jm_obj[s.start + 1 : s.end] = True
    jang = tr.angle[jm_obj & np.isfinite(tr.angle)]
    for lo, hi, name in ((0, 25, "b2f"), (0, 60, "sharp"), (60, 120, "mid"), (120, 181, "wide"), (155, 181, "linear")):
        st[f"jump_angle_{name}"] = float(np.mean((jang >= lo) & (jang < hi))) if len(jang) else 0.0
    st["angle_mean"] = float(np.mean(ang)) if len(ang) else 90.0
    st["cross_screen_rate"] = float(np.mean(tr.dist_px[1:] >= 300))

    # segment coverage (fraction of objects inside each pattern family)
    def cover(kinds, pred=lambda s: True):
        return sum(s.n for s in segs if s.kind in kinds and pred(s)) / n

    st["cov_doubles"] = cover({"double", "triple"})
    st["cov_quads"] = cover({"quad"})
    st["cov_bursts"] = cover({"quad", "burst"})
    st["cov_streams"] = cover({"stream", "deathstream"})
    st["cov_deathstream"] = cover({"deathstream"})
    st["cov_jumps"] = cover({"jumps"})
    st["cov_jump_stamina"] = cover({"jumps"}, lambda s: s.n >= 24)
    st["cov_spaced_streams"] = cover({"burst", "stream", "deathstream"}, lambda s: s.info["spacing"] >= 0.9)
    st["cov_flow_aim"] = cover({"burst", "stream", "deathstream"},
                               lambda s: s.info["spacing"] >= 0.55 and s.info["mean_angle"] >= 110)
    st["cov_short_spaced_groups"] = cover({"double", "triple", "quad", "burst"}, lambda s: s.info["spacing"] >= 0.8)
    for shape in ("triangles", "squares", "stars", "back and forth"):
        st[f"cov_shape_{shape}"] = cover({"jumps"}, lambda s, shape=shape: s.info["shape"] == shape)
    st["n_stream_groups_per_min"] = sum(1 for s in segs if s.kind in {"burst", "stream", "deathstream"}) / (st["drain_s"] / 60)
    sb = [s.info["bpm_1_4"] for s in segs if s.kind in {"burst", "stream", "deathstream"}]
    st["stream_bpm_p75"] = float(np.quantile(sb, 0.75)) if sb else 0.0
    longest = max((s.n for s in segs if s.kind in {"stream", "deathstream"}), default=0)
    st["longest_stream"] = float(longest)

    # cutstreams: two stream-ish runs separated by exactly one missing 1/4 (a 1/2 gap)
    runs = [s for s in segs if s.kind in {"quad", "burst", "stream", "deathstream"}]
    cuts = 0
    for a, b in zip(runs, runs[1:]):
        if b.start == a.end + 1 and 0.4 < tr.ratio[b.start] < 0.6:
            cuts += 1
    st["cutstreams_per_min"] = cuts / (st["drain_s"] / 60)

    # sliders: SV variety, complexity, quick slider->circle rhythms
    sl = [o for o in objs if o.kind == "slider"]
    svs = np.array([o.sv for o in sl]) if sl else np.array([1.0])
    st["sv_unique"] = float(len(np.unique(np.round(svs, 2))))
    st["sv_cv"] = float(np.std(svs) / (np.mean(svs) + 1e-6))
    st["slider_complex"] = float(np.mean([(o.control_points > 5) or o.slides > 2 for o in sl])) if sl else 0.0
    quick = (np.concatenate([[False], tr.is_slider[:-1]])) & (tr.gap > 0) & (tr.gap / tr.beat <= 0.27)
    st["slider_quick_exit"] = float(np.mean(quick[1:]))
    st["short_sliders_1_4"] = float(np.mean([((o.end_time - o.time) / o.slides) / o.beat_length <= 0.27 for o in sl])) if sl else 0.0

    # reading: overlaps of non-adjacent objects visible at once, visual density
    pre = bm.preempt
    r2 = (2 * bm.radius) ** 2
    ov = 0
    vis = []
    j0 = 0
    xs = np.array([o.x for o in objs]); ys = np.array([o.y for o in objs]); ts = tr.t
    for i in range(len(objs)):
        while ts[j0] < ts[i] - pre:
            j0 += 1
        vis.append(i - j0 + 1)
        for j in range(j0, i - 2):  # skip the two immediate predecessors
            d2 = (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2
            if 9 < d2 < r2:
                ov += 1
                break
    st["overlap_rate"] = ov / n
    st["visible_mean"] = float(np.mean(vis))
    st["visible_p90"] = float(np.quantile(vis, 0.9))
    st["preempt_beats"] = pre / (60000 / max(1.0, st["bpm"]))
    st["stack_rate"] = float(np.mean(tr.dist_px[1:] < 3))
    return tr, segs, st


def tag(bm: Beatmap) -> TagResult:
    tr, segs, st = analyse(bm)
    if st.get("n_objects", 0) < 10:
        return TagResult({t: 0.0 for t in TAGS}, {}, st)
    S: dict[str, float] = {}

    # streams family
    S["streams/doubles"] = ramp(st["cov_doubles"], 0.22, 0.35)
    S["streams/bursts"] = ramp(st["cov_bursts"], 0.10, 0.22)
    S["streams/stamina"] = max(ramp(st["longest_stream"], 24, 48), ramp(st["cov_deathstream"], 0.05, 0.15))
    S["streams/spaced streams"] = ramp(st["cov_spaced_streams"], 0.04, 0.12)
    S["streams/cutstreams"] = ramp(st["cutstreams_per_min"], 1.0, 4.0)
    S["streams/flow aim"] = ramp(st["cov_flow_aim"], 0.05, 0.15)
    stream_cov = st["cov_streams"] + 0.5 * st["cov_bursts"]
    S["skillset/streams"] = ramp(stream_cov, 0.10, 0.25)
    S["streams/speed"] = ramp(st["stream_bpm_p75"], 195, 230) * ramp(stream_cov, 0.03, 0.10)

    # jumps family
    S["skillset/jumps"] = max(ramp(st["cov_jumps"], 0.12, 0.30), ramp(st["spacing_half_mean"], 1.8, 2.8))
    jgate = ramp(st["cov_jumps"], 0.05, 0.15)
    S["jumps/sharp"] = jgate * ramp(st["jump_angle_sharp"], 0.65, 0.85)
    S["jumps/wide"] = jgate * ramp(st["jump_angle_wide"], 0.30, 0.50)
    S["jumps/linear"] = jgate * ramp(st["jump_angle_linear"], 0.20, 0.40)
    S["jumps/back and forth"] = jgate * ramp(st["cov_shape_back and forth"] / max(st["cov_jumps"], 1e-6), 0.15, 0.35)
    for shape in ("triangles", "squares", "stars"):
        S[f"jumps/{shape}"] = jgate * ramp(st[f"cov_shape_{shape}"] / max(st["cov_jumps"], 1e-6), 0.15, 0.35)
    S["jumps/cross-screen"] = jgate * ramp(st["cross_screen_rate"], 0.03, 0.10)
    S["jumps/stamina"] = ramp(st["cov_jump_stamina"], 0.10, 0.25)

    # alt: short 1/4 groups placed with real spacing, mixed into the map
    S["skillset/alt"] = ramp(st["cov_short_spaced_groups"], 0.08, 0.20) * (1 - 0.5 * S["skillset/streams"])

    # tech: rhythm and slider irregularity
    tech = (0.30 * ramp(st["rhythm_entropy"], 1.4, 2.2) + 0.20 * ramp(st["triplet_mix"], 0.02, 0.12)
            + 0.20 * ramp(st["sv_unique"], 3, 10) + 0.15 * ramp(st["slider_quick_exit"], 0.03, 0.12)
            + 0.15 * ramp(st["velocity_cv"], 0.8, 1.4))
    S["skillset/tech"] = ramp(tech, 0.35, 0.65)
    S["tech/slider tech"] = ramp(0.4 * ramp(st["sv_unique"], 6, 16) + 0.3 * ramp(st["slider_complex"], 0.10, 0.30)
                                 + 0.3 * ramp(st["slider_quick_exit"], 0.10, 0.30), 0.45, 0.75)

    S["skillset/precision"] = ramp(st["cs"], 4.8, 6.0)
    S["reading/overlaps"] = ramp(st["overlap_rate"], 0.10, 0.25)
    S["reading/visually dense"] = ramp(st["visible_p90"], 6, 10)
    S["skillset/reading"] = max(ramp(st["preempt_beats"], 2.6, 3.6) * ramp(st["density"], 3, 5),
                                0.8 * S["reading/overlaps"], ramp(st["ar"], 10.2, 10.6))

    ev: dict[str, list[Segment]] = {
        "skillset/streams": [s for s in segs if s.kind in {"stream", "deathstream"}],
        "streams/bursts": [s for s in segs if s.kind in {"quad", "burst"}],
        "streams/doubles": [s for s in segs if s.kind in {"double", "triple"}],
        "streams/stamina": [s for s in segs if s.kind in {"stream", "deathstream"} and s.n >= 24],
        "streams/spaced streams": [s for s in segs if s.kind in {"burst", "stream", "deathstream"} and s.info["spacing"] >= 0.9],
        "streams/flow aim": [s for s in segs if s.kind in {"burst", "stream", "deathstream"}
                             and s.info["spacing"] >= 0.55 and s.info["mean_angle"] >= 110],
        "skillset/jumps": [s for s in segs if s.kind == "jumps"],
        "skillset/alt": [s for s in segs if s.kind in {"double", "triple", "quad", "burst"} and s.info["spacing"] >= 0.8],
    }
    for shape in ("triangles", "squares", "stars", "back and forth"):
        ev[f"jumps/{shape}"] = [s for s in segs if s.kind == "jumps" and s.info["shape"] == shape]
    return TagResult({t: round(S.get(t, 0.0), 3) for t in TAGS}, ev, st)
