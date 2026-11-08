"""Minimal .osu (osu!standard) parser.

Only what pattern analysis needs: difficulty settings, timing, and hit objects
with approximate slider geometry (end position, duration, path length).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

CIRCLE, SLIDER, SPINNER = "circle", "slider", "spinner"


@dataclass
class TimingPoint:
    time: float
    beat_length: float  # ms per beat (uninherited) or raw negative value (inherited)
    meter: int
    uninherited: bool


@dataclass
class HitObject:
    kind: str
    time: float
    x: float
    y: float
    new_combo: bool
    end_time: float
    end_x: float
    end_y: float
    # slider-only
    slides: int = 0
    pixel_length: float = 0.0
    control_points: int = 0
    curve_type: str = ""
    sv: float = 1.0
    # filled later by the parser
    beat_length: float = 500.0


@dataclass
class Beatmap:
    path: str
    mode: int
    title: str
    artist: str
    version: str
    creator: str
    beatmap_id: int | None
    beatmapset_id: int | None
    cs: float
    ar: float
    od: float
    hp: float
    slider_multiplier: float
    stack_leniency: float
    timing: list[TimingPoint] = field(default_factory=list)
    objects: list[HitObject] = field(default_factory=list)
    breaks: list[tuple[float, float]] = field(default_factory=list)

    @property
    def radius(self) -> float:
        return 54.4 - 4.48 * self.cs

    @property
    def preempt(self) -> float:
        ar = self.ar
        return 1200 + 600 * (5 - ar) / 5 if ar < 5 else 1200 - 750 * (ar - 5) / 5

    @property
    def main_bpm(self) -> float:
        """BPM that covers the most playable time."""
        red = [t for t in self.timing if t.uninherited]
        if not red or not self.objects:
            return 0.0
        end = self.objects[-1].end_time
        dur: dict[float, float] = {}
        for i, t in enumerate(red):
            nxt = red[i + 1].time if i + 1 < len(red) else end
            bpm = round(60000 / t.beat_length, 1)
            dur[bpm] = dur.get(bpm, 0) + max(0.0, nxt - max(t.time, self.objects[0].time))
        return max(dur, key=dur.get) if dur else 60000 / red[0].beat_length


def _sections(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    cur = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            cur = line[1:-1]
            out[cur] = []
        elif cur:
            out[cur].append(line)
    return out


def _kv(lines: list[str]) -> dict[str, str]:
    d = {}
    for line in lines:
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = v.strip()
    return d


# ---------------------------------------------------------------- slider paths

def _bezier(pts: list[tuple[float, float]], n: int = 24) -> list[tuple[float, float]]:
    out = []
    for i in range(n + 1):
        t = i / n
        p = list(pts)
        while len(p) > 1:
            p = [((1 - t) * a[0] + t * b[0], (1 - t) * a[1] + t * b[1]) for a, b in zip(p, p[1:])]
        out.append(p[0])
    return out


def _arc(p0, p1, p2, n: int = 32):
    ax, ay = p0
    bx, by = p1
    cx, cy = p2
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-6:
        return None
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    r = math.hypot(ax - ux, ay - uy)
    t0 = math.atan2(ay - uy, ax - ux)
    t2 = math.atan2(cy - uy, cx - ux)
    # direction: sign of cross product of (p1-p0) x (p2-p1)
    cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
    if cross < 0:  # clockwise in screen coords means decreasing angle
        while t2 > t0:
            t2 -= 2 * math.pi
    else:
        while t2 < t0:
            t2 += 2 * math.pi
    return [(ux + r * math.cos(t0 + (t2 - t0) * i / n), uy + r * math.sin(t0 + (t2 - t0) * i / n)) for i in range(n + 1)]


def slider_path(start, curve_type, pts, pixel_length):
    """Approximate slider path as a polyline truncated to pixel_length."""
    allp = [start] + pts
    poly: list[tuple[float, float]]
    if curve_type == "P" and len(allp) == 3:
        poly = _arc(*allp) or allp
    elif curve_type in ("B", "C"):
        poly = []
        seg = [allp[0]]
        for p in allp[1:]:
            if p == seg[-1] and curve_type == "B":  # red anchor splits bezier
                poly += _bezier(seg)
                seg = [p]
            else:
                seg.append(p)
        poly += _bezier(seg) if curve_type == "B" else seg
    else:
        poly = allp
    # truncate / extend to pixel length
    out = [poly[0]]
    acc = 0.0
    for a, b in zip(poly, poly[1:]):
        seg = math.dist(a, b)
        if seg == 0:
            continue
        if acc + seg >= pixel_length:
            f = (pixel_length - acc) / seg
            out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
            return out
        acc += seg
        out.append(b)
    if len(out) >= 2 and acc < pixel_length:  # extend last segment linearly
        a, b = out[-2], out[-1]
        seg = math.dist(a, b) or 1
        f = (pixel_length - acc) / seg
        out[-1] = (b[0] + (b[0] - a[0]) * f, b[1] + (b[1] - a[1]) * f)
    return out


# ---------------------------------------------------------------- main parser

def parse(path_or_text: str | Path, *, is_text: bool = False) -> Beatmap:
    text = path_or_text if is_text else Path(path_or_text).read_text(encoding="utf-8-sig", errors="replace")
    s = _sections(text)
    gen, meta, diff = _kv(s.get("General", [])), _kv(s.get("Metadata", [])), _kv(s.get("Difficulty", []))
    od = float(diff.get("OverallDifficulty", 5))

    def _int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    bm = Beatmap(
        path="<text>" if is_text else str(path_or_text),
        mode=int(gen.get("Mode", 0)),
        title=meta.get("Title", ""),
        artist=meta.get("Artist", ""),
        version=meta.get("Version", ""),
        creator=meta.get("Creator", ""),
        beatmap_id=_int(meta.get("BeatmapID")),
        beatmapset_id=_int(meta.get("BeatmapSetID")),
        cs=float(diff.get("CircleSize", 4)),
        ar=float(diff.get("ApproachRate", od)),
        od=od,
        hp=float(diff.get("HPDrainRate", 5)),
        slider_multiplier=float(diff.get("SliderMultiplier", 1.4)),
        stack_leniency=float(gen.get("StackLeniency", 0.7)),
    )

    for line in s.get("Events", []):
        p = line.split(",")
        if p[0] in ("2", "Break") and len(p) >= 3:
            bm.breaks.append((float(p[1]), float(p[2])))

    for line in s.get("TimingPoints", []):
        p = line.split(",")
        if len(p) < 2:
            continue
        unin = (p[6].strip() == "1") if len(p) > 6 else True
        bm.timing.append(TimingPoint(float(p[0]), float(p[1]), _int(p[2]) or 4 if len(p) > 2 else 4, unin))
    bm.timing.sort(key=lambda t: (t.time, not t.uninherited))

    # walk timing points alongside objects
    tps = bm.timing
    ti = 0
    beat_len = next((t.beat_length for t in tps if t.uninherited), 500.0)
    sv = 1.0
    raw_objs = []
    for line in s.get("HitObjects", []):
        p = line.split(",")
        if len(p) < 4:
            continue
        raw_objs.append(p)
    raw_objs.sort(key=lambda p: float(p[2]))

    for p in raw_objs:
        x, y, t, typ = float(p[0]), float(p[1]), float(p[2]), int(p[3])
        while ti < len(tps) and tps[ti].time <= t + 1:
            tp = tps[ti]
            if tp.uninherited:
                beat_len, sv = tp.beat_length, 1.0
            elif tp.beat_length < 0:
                sv = max(0.1, min(10.0, -100.0 / tp.beat_length))
            ti += 1
        nc = bool(typ & 4)
        if typ & 2 and len(p) >= 8:
            curve = p[5].split("|")
            ctype = curve[0]
            pts = []
            for c in curve[1:]:
                if ":" in c:
                    a, b = c.split(":")[:2]
                    pts.append((float(a), float(b)))
            slides = max(1, int(p[6]))
            length = float(p[7])
            poly = slider_path((x, y), ctype, pts, length) if pts else [(x, y)]
            ex, ey = poly[-1] if slides % 2 == 1 else (x, y)
            dur = length / (bm.slider_multiplier * 100 * sv) * beat_len * slides
            bm.objects.append(HitObject(SLIDER, t, x, y, nc, t + dur, ex, ey, slides, length, len(pts), ctype, sv, beat_len))
        elif typ & 8:
            end = float(p[5]) if len(p) > 5 else t
            bm.objects.append(HitObject(SPINNER, t, 256, 192, nc, end, 256, 192, beat_length=beat_len))
        else:
            bm.objects.append(HitObject(CIRCLE, t, x, y, nc, t, x, y, beat_length=beat_len))
    return bm
