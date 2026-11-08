"""SV as a layer (Komi, 2026-11-02): compatible with Rice / LN categories, and gated on visible scroll changes."""
import numpy as np

from osu_patterns import categories, mania, reading


def chart(rows, timing=("0,500,4,1,0,60,1,0",), keys=4):
    objs = [f"{int((c + 0.5) * 512 / keys)},192,{t},1,0,0:0:0:0:" for t, cols in rows for c in cols]
    return ("osu file format v14\n\n[General]\nMode: 3\n\n[Metadata]\nTitle:T\nArtist:A\nCreator:C\nVersion:V\n\n"
            f"[Difficulty]\nCircleSize:{keys}\nOverallDifficulty:8\n\n[TimingPoints]\n" + "\n".join(timing)
            + "\n\n[HitObjects]\n" + "\n".join(objs))


ROWS = [(1000 + 125 * i, [i % 4]) for i in range(240)]   # 30 s of 1/4 stream at 120 BPM


def effect(timing=("0,500,4,1,0,60,1,0",), rows=ROWS):
    return reading.scroll_effect(mania.parse_mania(chart(rows, timing)))


def test_no_scroll_change_fails_gate():
    assert effect()["sve_effective"] == 0
    # one SV for the whole chart is a different scroll speed, not a change
    assert effect(("0,500,4,1,0,60,1,0", "0,-50,4,1,0,60,0,0"))["sve_effective"] == 0
    # a BPM change normalised by SV (240 BPM at 0.5x) looks the same on screen
    assert effect(("0,500,4,1,0,60,1,0", "15000,250,4,1,0,60,1,0", "15000,-200,4,1,0,60,0,0"))["sve_effective"] == 0
    # 1.1x is too small to see
    assert effect(("0,500,4,1,0,60,1,0", "10000,-90.9,4,1,0,60,0,0", "20000,-100,4,1,0,60,0,0"))["sve_effective"] == 0


def test_slowjam_fastjam_teleport_and_stutter_pass():
    slow = ("0,500,4,1,0,60,1,0", "12000,-400,4,1,0,60,0,0", "14000,-100,4,1,0,60,0,0")    # 0.25x burst
    fast = ("0,500,4,1,0,60,1,0", "12000,-25,4,1,0,60,0,0", "14000,-100,4,1,0,60,0,0")     # 4x burst
    tele = ("0,500,4,1,0,60,1,0", "12000,-10,4,1,0,60,0,0", "12030,-100,4,1,0,60,0,0")     # 10x for 30 ms
    # stutter: 2x then 0.5x inside every 1/4 gap; every row lands at its usual gap but moves while falling
    stut = ["0,500,4,1,0,60,1,0"] + [f"{t},-50,4,1,0,60,0,0\n{t + 42},-200,4,1,0,60,0,0" for t in range(10000, 14000, 125)] \
        + ["14000,-100,4,1,0,60,0,0"]
    for tp in (slow, fast, tele, stut):
        assert effect(tp)["sve_effective"] == 1, tp
    # an un-normalised BPM change moves the scroll too
    assert effect(("0,500,4,1,0,60,1,0", "15000,250,4,1,0,60,1,0"))["sve_effective"] == 1


def test_note_gate_is_causal_and_local():
    m = mania.parse_mania(chart(ROWS, ("0,500,4,1,0,60,1,0", "12000,-400,4,1,0,60,0,0", "14000,-100,4,1,0,60,0,0")))
    g = reading.sv_note_gate(m)
    t = m.start
    assert g[t < 11000].max() == 0           # nothing before the jam
    assert g[(t > 12500) & (t < 14000)].min() == 1
    assert g[t > 20000].max() == 0           # and it closes after it
    assert reading.sv_note_gate(mania.parse_mania(chart(ROWS))).max() == 0
    w = reading.sv_window_gate(m, [5000, 13000, 25000], 4000)
    assert w.tolist() == [0, 1, 0]


def test_sv_and_ln_show_together():
    probs = {n: 0.0 for n in categories.NAMES}
    probs.update({"Release": 0.7, "Reading": 0.6})
    shown, _ = categories.pick(probs, "LN", {})
    assert set(shown) == {"Release", "Reading"}          # SV no longer needs 0.8 on an LN map
    # an SV-slot map keeps its LN categories at their own cut
    sp = {"RC": 0.05, "LN": 0.35, "HB": 0.1, "SV": 0.5}
    shown, _ = categories.pick(probs, "SV", {}, slot_probs=sp)
    assert set(shown) == {"Release", "Reading"}
    assert categories.slot_label("SV", sp, shown) == "LN + SV"
    assert categories.slot_label("LN", None, shown) == "LN + SV"
    # no effective scroll change: never SV
    shown, _ = categories.pick(probs, "LN", {}, sv_effective=False)
    assert shown == ["Release"]
    g = categories.gate_sv({"Reading": 0.9, "Memo": 0.9, "tag:sliders/low sv": 0.9, "sliders/high sv": 0.9,
                            "Release": 0.7}, False)
    assert g == {"Reading": 0, "Memo": 0, "tag:sliders/low sv": 0, "sliders/high sv": 0, "Release": 0.7}
    sp2 = categories.gate_slot_probs(sp, False)
    assert sp2["SV"] == 0 and abs(sum(sp2.values()) - 1) < 1e-9 and max(sp2, key=sp2.get) == "LN"


def test_memo_gate_needs_a_speed_step_not_a_steady_slowjam():
    slow = ("0,500,4,1,0,60,1,0", "12000,-200,4,1,0,60,0,0", "24000,-100,4,1,0,60,0,0")   # steady 0.5x for 12 s
    m = mania.parse_mania(chart(ROWS, slow))
    g = reading.memo_note_gate(m)
    t = m.start
    assert g[(t > 15000) & (t < 23000)].max() == 0      # inside the jam: Reading only
    assert g[(t > 12100) & (t < 13500)].min() == 1      # just after the step into it
    ramp = ["0,500,4,1,0,60,1,0"] + [f"{12000 + 50 * i},{-100 - 2 * i},4,1,0,60,0,0" for i in range(40)]
    assert reading.memo_note_gate(mania.parse_mania(chart(ROWS, tuple(ramp)))).max() == 0   # a smooth ramp is no step
    assert reading.memo_window_gate(m, [5000, 12500, 20000], 4000).tolist() == [0, 1, 0]
