"""Pattern tags for osu!mania, named after osu!(lazer) user tags.

The slot model says *which pool slot* a map belongs to (RC / LN / HB / SV). Inside a slot,
poolers and players care about the skillset: jumpstream vs handstream vs chordjack, LN release
vs LN coordination, and so on. osu!(lazer) players vote exactly these as user tags on ranked
maps, so they are the training labels here (positive-unlabelled, as in model.py).

`heuristic_scores` is the rule baseline: each tag is a squashed version of the one or two
statistics that define it. The learned heads (model.TagModel) see every statistic plus these
scores and the slot probabilities.
"""
from __future__ import annotations

import numpy as np

from . import mania


def _sq(x: float, mid: float, width: float) -> float:
    """Logistic squash: 0.5 at mid, ~0.88 at mid + 2*width."""
    return float(1 / (1 + np.exp(-(x - mid) / max(width, 1e-9))))


def heuristic_scores(s: dict) -> dict[str, float]:
    rice = 1 - _sq(s["ln_ratio"], 0.35, 0.05)
    ln = _sq(s["ln_ratio"], 0.35, 0.05)
    fast = _sq(s["nps_per_key"], 4.0, 0.6)
    return {
        "style/jumpstream": rice * _sq(s["jumpstream_row"], 0.18, 0.04),
        "style/handstream": rice * _sq(s["handstream_row"], 0.10, 0.03),
        "style/chordstream": rice * _sq(s["jumpstream_row"] + s["handstream_row"], 0.25, 0.05),
        "style/quadstream": rice * _sq(s["row_quad"], 0.08, 0.03),
        "style/chordjack": rice * _sq(s["chordjack_row"], 0.15, 0.04),
        "skillset/speedjack": rice * _sq(s["jack_row"], 0.15, 0.04) * _sq(-s["jack_gap_ms"], -120, 15),
        "style/longjack": rice * _sq(s["jack_chain3"], 0.05, 0.02),
        "skillset/wristjack": rice * _sq(s["jack_note"], 0.25, 0.05),
        "skillset/streams": rice * _sq(s["stream_row"], 0.45, 0.08),
        "streams/speed": rice * fast,
        "streams/stamina": _sq(s["stamina"] * s["duration_s"], 60, 15),
        "skillset/tech": _sq(s["rhythm_variety"] + s["gap_ratio_change"], 3.5, 0.8),
        "style/LN density": ln * _sq(s["ln_count_ps"], 6, 1.5),
        "style/LN release": ln * _sq(s["ln_release_offgrid"], 0.5, 0.1),
        "style/LN coordination": ln * _sq(s["ln_held_mean"], 0.35, 0.08),
        "gimmick/LN inverse": ln * _sq(s["ln_inverse"], 0.6, 0.1),
        "style/generic hybrid": float(np.clip(1 - abs(s["ln_ratio"] - 0.25) / 0.15, 0, 1)),
        "sliders/complex sv": _sq(s["sv_changes_near_notes_ps"], 0.5, 0.15) * _sq(s["sv_log_std"], 0.2, 0.05),
        "sliders/low sv": _sq(s["sv_stop_frac"] + s["sv_far_frac"] * (s["sv_fast_frac"] < 0.05), 0.15, 0.05),
        "sliders/high sv": _sq(s["sv_fast_frac"], 0.1, 0.04),
    }


TAGS = list(heuristic_scores({k: 0.0 for k in [
    "ln_ratio", "nps_per_key", "jumpstream_row", "handstream_row", "row_quad", "chordjack_row",
    "jack_row", "jack_gap_ms", "jack_chain3", "jack_note", "stream_row", "stamina", "duration_s",
    "rhythm_variety", "gap_ratio_change", "ln_count_ps", "ln_release_offgrid",
    "ln_held_mean", "ln_inverse", "sv_changes_near_notes_ps", "sv_log_std", "sv_stop_frac",
    "sv_far_frac", "sv_fast_frac"]}))


def feature_vector(s: dict, slot_probs: dict | None = None) -> dict[str, float]:
    f = {k: float(v) for k, v in s.items()}
    f.update({f"h:{k}": v for k, v in heuristic_scores(s).items()})
    for k, v in (slot_probs or {}).items():
        f[f"slot:{k}"] = float(v)
    return f


def describe(m: mania.ManiaMap, slot_model=None, tag_model=None, category_model=None, threshold: float = 0.5) -> dict:
    """Everything a Trello comment needs for one difficulty."""
    from . import categories, reading
    from .slots import rule_scores, rule_slot
    s = mania.stats(m)
    gate = reading.scroll_effect(m)
    sv_ok = gate["sve_effective"] > 0
    out = {"version": m.version, "keys": m.keys, "bpm": mania.main_bpm(m), "stats": s, "sv_gate": gate}
    if slot_model is not None:
        probs = slot_model.predict_proba(s)
        out["slot_probs"] = categories.gate_slot_probs(probs, sv_ok)  # no SV slot without scroll changes
        out["slot"] = max(out["slot_probs"], key=out["slot_probs"].get)
    else:
        probs, out["slot"] = None, rule_slot(s)
        if out["slot"] == "SV" and not sv_ok:
            sc = rule_scores(s)
            out["slot"] = max(("RC", "HB", "LN"), key=sc.get)
    f = feature_vector(s, probs)
    if tag_model is not None:
        x = np.array([[f.get(c, np.nan) for c in tag_model.columns]])
        scores = {t: (float(h.predict_proba(x)[0, 1]) if t in tag_model.heads else f[f"h:{t}"])
                  for t, h in ((t, tag_model.heads.get(t)) for t in TAGS)}
        scores = categories.gate_sv(scores, sv_ok)
        chosen = [t for t, p in scores.items() if p >= tag_model.thresholds.get(t, threshold)]
    else:
        scores = categories.gate_sv(heuristic_scores(s), sv_ok)
        chosen = [t for t, p in scores.items() if p >= threshold]
    out["tag_scores"] = {k: round(v, 3) for k, v in sorted(scores.items(), key=lambda kv: -kv[1])}
    out["tags"] = sorted(chosen, key=lambda t: -scores[t])
    cat = categories.gate_sv(categories.predict(s, probs, category_model), sv_ok)
    thr = category_model.thresholds if category_model is not None else {}
    out["category_scores"] = {k: round(v, 3) for k, v in cat.items()}
    out["categories"], out["tb"] = categories.pick(cat, out["slot"], thr, slot_probs=out.get("slot_probs"),
                                                   sv_effective=sv_ok)
    out["slot_label"] = categories.slot_label(out["slot"], out.get("slot_probs"), out["categories"])
    out["evidence"] = mania.evidence(m)
    return out
