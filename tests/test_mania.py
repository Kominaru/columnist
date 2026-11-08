"""Synthetic osu!mania charts with known patterns -> mania.stats() and the slot rules."""
from osu_patterns import mania, slots


def chart(rows, keys=4, beat=375.0, step=0.25, ln_beats=0.0, sv=None):
    """rows: list of column tuples, one per 1/4 (step) beat. ln_beats > 0 makes holds."""
    lines = ["osu file format v14", "[General]", "Mode: 3", "[Metadata]", "Artist:a", "Title:t", "Version:v",
             "[Difficulty]", f"CircleSize:{keys}", "OverallDifficulty:8", "HPDrainRate:8", "[TimingPoints]",
             f"0,{beat},4,1,0,50,1,0"]
    for t, mult in sv or []:
        lines.append(f"{t},{-100 / mult},4,1,0,50,0,0")
    lines.append("[HitObjects]")
    for i, cols in enumerate(rows):
        t = int(1000 + i * beat * step)
        for c in cols:
            x = int((c + 0.5) * 512 / keys)
            if ln_beats:
                lines.append(f"{x},192,{t},128,0,{int(t + ln_beats * beat)}:0:0:0:0:")
            else:
                lines.append(f"{x},192,{t},1,0,0:0:0:0:")
    return mania.parse_mania("\n".join(lines))


def test_jacks_vs_streams():
    jack = mania.stats(chart([(0, 1)] * 200))
    stream = mania.stats(chart([(0,), (2,), (1,), (3,)] * 50))
    assert jack["jack_row"] > 0.9 and jack["chordjack_row"] > 0.9 and jack["stream_row"] < 0.05
    assert stream["stream_row"] > 0.9 and stream["jack_row"] < 0.05
    assert jack["jack_longest"] >= 100


def test_jumpstream_and_handstream():
    js = mania.stats(chart([(0, 2), (3,), (1, 2), (0,), (2, 3), (1,)] * 40))
    hs = mania.stats(chart([(0, 1, 3), (2,), (0, 1, 3), (2,)][:2] * 100))
    assert js["jumpstream_row"] > 0.4 and js["handstream_row"] == 0
    assert hs["handstream_row"] > 0.4


def test_ln_and_rule_slots():
    rice = mania.stats(chart([(0,), (2,), (1,), (3,)] * 50))
    ln = mania.stats(chart([(0,), (2,), (1,), (3,)] * 50, ln_beats=0.5))
    assert rice["ln_ratio"] == 0 and ln["ln_ratio"] == 1
    assert slots.rule_slot(rice) == "RC" and slots.rule_slot(ln) == "LN"


def test_sv_detected():
    sv = [(1000 + i * 375, 0.3 if i % 2 else 2.5) for i in range(100)]
    s = mania.stats(chart([(0,), (2,), (1,), (3,)] * 100, sv=sv))
    assert s["sv_log_std"] > 0.5 and s["sv_changes_ps"] > 1
    assert slots.rule_slot(s) == "SV"


def test_slot_normalisation():
    assert slots.normalise_slot("MWC/2024_4K", "Long Note") == "LN"
    assert slots.normalise_slot("MWC/2024_4K", "FreeMod") is None
    assert slots.normalise_slot("o!mLN/3", "Coordination") == "LN"
    assert slots.normalise_slot("JOKE/1", "Speed") is None
    assert slots.song_key("Camellia - BADBOY (Valedict) [XLMP]") == "camellia badboy"


def test_categories_pick():
    from osu_patterns import categories
    probs = {n: 0.0 for n in categories.NAMES}
    probs.update({"Chordjack": 0.9, "Streams": 0.7, "Release": 0.6, "TB": 0.9})
    shown, tb = categories.pick(probs, "RC", {})
    assert set(shown) == {"Chordjack", "Streams", "Mixed Rice"}
    assert "Release" not in shown and tb  # LN sub-category below 0.8 on a rice map stays hidden
    shown, _ = categories.pick({**probs, "Chordjack": 0.0, "Jumpstream/Handstream": 0.9}, "RC", {})
    assert "Mixed Rice" not in shown  # streams + jumpstream is one family


def test_category_labels_from_votes():
    import pandas as pd
    from osu_patterns import categories
    y = categories.labels_from_votes(pd.Series({1: {"style/handstream": 3, "skillset/tech": 1, "x": 9}, 2: {"x": 1}}))
    assert y.at[1, "Jumpstream/Handstream"] == 1 and pd.isna(y.at[1, "Technical"]) and y.at[1, "Release"] == 0
    assert y.loc[2].isna().all()
