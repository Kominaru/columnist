"""Generate synthetic .osu text with known patterns, for tests and sanity checks."""
from __future__ import annotations

import math

HEADER = """osu file format v14

[General]
Mode: 0
StackLeniency: 0.7

[Metadata]
Title:{title}
Artist:synthetic
Creator:test
Version:{title}

[Difficulty]
HPDrainRate:5
CircleSize:{cs}
OverallDifficulty:8
ApproachRate:{ar}
SliderMultiplier:1.8
SliderTickRate:1

[Events]

[TimingPoints]
0,{beat},4,2,0,60,1,0

[HitObjects]
"""


def _circle(x, y, t, nc=False):
    return f"{int(x)},{int(y)},{int(t)},{5 if nc else 1},0,0:0:0:0:"


def make(pattern: str, bpm: float = 180, cs: float = 4, ar: float = 9, bars: int = 48) -> str:
    beat = 60000 / bpm
    lines, t = [], 1000.0
    if pattern == "streams":
        # 32-note 1/4 streams following a circle, separated by 1 beat rests
        for b in range(bars // 8):
            for i in range(32):
                a = i / 32 * 2 * math.pi
                lines.append(_circle(256 + 120 * math.cos(a), 192 + 100 * math.sin(a), t, i == 0))
                t += beat / 4
            t += beat
    elif pattern in ("triangles", "squares", "back and forth"):
        k = {"triangles": 3, "squares": 4, "back and forth": 2}[pattern]
        pts = [(256 + 180 * math.cos(2 * math.pi * j / k), 192 + 150 * math.sin(2 * math.pi * j / k)) for j in range(k)]
        for i in range(bars * 8):
            x, y = pts[i % k]
            lines.append(_circle(x, y, t, i % 8 == 0))
            t += beat / 2
    elif pattern == "alt":
        # spaced 1/4 triples between 1/2 jumps
        for i in range(bars * 2):
            base = (100 + (i * 97) % 300, 80 + (i * 61) % 220)
            for j in range(3):
                lines.append(_circle(base[0] + 70 * j, base[1], t, j == 0))
                t += beat / 4
            t += beat / 4 + beat / 2
    elif pattern == "simple":
        # 1/1 circles, small spacing: should get no skill tags
        for i in range(bars * 2):
            lines.append(_circle(200 + 60 * (i % 3), 192, t, i % 4 == 0))
            t += beat
    else:
        raise ValueError(pattern)
    return HEADER.format(title=pattern, cs=cs, ar=ar, beat=beat) + "\n".join(lines) + "\n"
