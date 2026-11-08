import pytest

from osu_patterns.features import compute_transitions, osu_timestamp, snap_divisor
from osu_patterns.parser import parse, slider_path
from osu_patterns.tagger import tag
from tests.synth import make


def tags_of(pattern, **kw):
    return tag(parse(make(pattern, **kw), is_text=True))


def test_snap_divisor():
    assert snap_divisor(0.25) == 4
    assert snap_divisor(0.5) == 2
    assert snap_divisor(1 / 3) == 3
    assert snap_divisor(1 / 6) == 6
    assert snap_divisor(0.75) == 4


def test_timestamp():
    assert osu_timestamp(83456) == "01:23:456"


def test_slider_path_linear_length():
    pts = slider_path((0, 0), "L", [(100, 0)], 50)
    assert pts[-1] == pytest.approx((50, 0))


def test_streams_detected():
    r = tags_of("streams")
    assert r.scores["skillset/streams"] >= 0.9
    assert r.scores["skillset/jumps"] < 0.5
    assert r.evidence["skillset/streams"], "stream segments should carry timestamps"


@pytest.mark.parametrize("shape", ["triangles", "squares", "back and forth"])
def test_jump_shapes(shape):
    r = tags_of(shape)
    assert r.scores["skillset/jumps"] >= 0.9
    assert r.scores[f"jumps/{shape}"] >= 0.9
    others = {"triangles", "squares", "back and forth"} - {shape}
    for o in others:
        assert r.scores[f"jumps/{o}"] < 0.5, o
    assert r.scores["skillset/streams"] < 0.5


def test_alt_detected():
    r = tags_of("alt")
    assert r.scores["skillset/alt"] >= 0.5
    assert r.scores["skillset/streams"] < 0.5


def test_simple_map_has_no_skill_tags():
    r = tags_of("simple", bpm=120)
    assert [t for t in r.tags() if t.startswith("skillset/")] == []


def test_precision_from_cs():
    assert tags_of("simple", cs=6.5).scores["skillset/precision"] == 1.0


def test_combo_numbers():
    tr = compute_transitions(parse(make("triangles"), is_text=True))
    assert tr.combo[:9] == [1, 2, 3, 4, 5, 6, 7, 8, 1]
