"""End-to-end smoke test of the learned pipeline on synthetic maps with simulated tag votes."""
import numpy as np
import pandas as pd

from osu_patterns.cli import beatmapset_id_from_text, describe, to_trello
from osu_patterns.model import TagModel, feature_vector, make_labels
from osu_patterns.parser import parse
from tests.synth import make

PATTERN_TAGS = {
    "streams": ["skillset/streams"],
    "triangles": ["skillset/jumps", "jumps/triangles"],
    "squares": ["skillset/jumps", "jumps/squares"],
    "back and forth": ["skillset/jumps", "jumps/back and forth"],
    "alt": ["skillset/alt"],
    "simple": [],
}


def simulated_dataset(n_per=12, seed=0):
    rng = np.random.default_rng(seed)
    rows, meta = {}, []
    bid = 0
    for p, true in PATTERN_TAGS.items():
        for _ in range(n_per):
            bid += 1
            bm = parse(make(p, bpm=float(rng.uniform(150, 230)), cs=float(rng.uniform(3, 5)), bars=int(rng.integers(24, 48))), is_text=True)
            rows[bid] = feature_vector(bm)
            plays = rng.integers(0, 3)  # 0 = barely played -> few votes -> labels masked
            votes = {t: int(rng.poisson(8 if plays else 1)) for t in true}
            votes["style/clean"] = int(rng.poisson(10 if plays else 2))  # uninformative tag
            meta.append({"beatmap_id": bid, "beatmapset_id": bid // 2, "tags": votes})
    meta = pd.DataFrame(meta).set_index("beatmap_id")
    return pd.DataFrame.from_dict(rows, orient="index"), meta


def test_labels_mask_unannotated_maps():
    meta = pd.DataFrame({"tags": [{"a": 5}, {"a": 0, "b": 20}, {"b": 2}]}, index=[1, 2, 3])
    Y = make_labels(meta, ["a", "b"], min_votes=3, min_total=15)
    assert Y.loc[1, "a"] == 1 and np.isnan(Y.loc[1, "b"])  # 5 total votes: absent tag is unknown
    assert Y.loc[2, "a"] == 0 and Y.loc[2, "b"] == 1
    assert np.isnan(Y.loc[3, "a"])


def test_fit_and_predict_roundtrip(tmp_path):
    X, meta = simulated_dataset()
    tags = ["skillset/streams", "skillset/jumps", "jumps/triangles"]
    Y = make_labels(meta, tags, min_votes=3, min_total=10)
    m = TagModel.fit(X, Y, meta["beatmapset_id"], min_pos=5, n_splits=3)
    assert set(m.heads) == set(tags)
    assert m.report["skillset/streams"]["ap"] > 0.8
    m.save(tmp_path / "m.pkl")
    m2 = TagModel.load(tmp_path / "m.pkl")
    probs = m2.predict(parse(make("streams"), is_text=True))
    assert "skillset/streams" in m2.decide(probs)


def test_trello_output():
    r = describe(parse(make("triangles"), is_text=True))
    md = to_trello([r])
    assert "jumps" in md and "triangles" in md
    assert "00:0" in md  # carries an osu! timestamp


def test_set_id_from_card():
    assert beatmapset_id_from_text("pls check https://osu.ppy.sh/beatmapsets/2345678#osu/5000000") == 2345678
    assert beatmapset_id_from_text("no link") is None
