"""Time-resolved pattern analysis for the live visualizer (service/viz).

The production heads (scripts/train_live_heads.py) are chart-level classifiers. To see *where*
a chart does what, `analyze` slides a 16 s window over it and scores every window as if it were
a short chart of its own: the same features the heads were trained on (chart stats, rule scores,
slot probabilities and the pooled 16 s window stats, which for a 16 s chart are just its own
stats). 16 s is the window the heads' pooled features already use, so the inputs stay inside
the range the heads have seen.

Heads that measure the length of a chart (Stamina, TB, streams/stamina) are only reported for
the whole chart. On top of the learned heads, every note gets a local label from simple row
rules (jack, chordjack, stream, jumpstream, handstream, trill, LN) for colouring the notes.
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import joblib
import numpy as np

from . import categories, mania, mania_tags, note_model, reading, slots, windows

WIN_MS = windows.WIN_MS        # 16 s, the pooled-window scale the heads were trained with
HOP_MS = 1000.0
# window lengths offered in the player; 16 s matches the heads' pooled features, shorter ones follow the
# chart more closely but sit further from what the heads were trained on
WIN_SIZES = (4000.0, 8000.0, WIN_MS)
CHART_ONLY = {"Stamina", "TB", "tag:streams/stamina"}

# local per-note labels, in priority order
LOCAL = ["ln", "chordjack", "jack", "handstream", "jumpstream", "trill", "stream", "chord", "single"]


def _columnist():
    from . import note_model_columnist   # numpy only
    return note_model_columnist


def _v3():
    from . import note_model_v3   # needs torch (osu_patterns.deep.v3), so imported only when its model file is served
    return note_model_v3


class LiveModel:
    def __init__(self, models_dir: str | Path = "models"):
        d = Path(models_dir)
        self.slot = slots.SlotModel.load(d / "slots.pkl")
        b = joblib.load(d / "live_heads.pkl")
        self.columns, self.tasks, self.heads = b["columns"], b["tasks"], b["heads"]
        self.thresholds, self.report = b["thresholds"], b["report"]
        # section-level cuts (scripts/calibrate_live_windows.py); the chart cut where there is none
        self.window_thresholds = {t: b.get("window_thresholds", {}).get(t, v) for t, v in self.thresholds.items()}
        self.window_report = b.get("window_report", {})
        # note-level model (scripts/train_note_model.py), when it has been trained
        # the causal v3 network (scripts/train_v3.py --causal --final) takes over when it is there
        # the causal Columnist head (scripts/train_columnist_lens.py) takes over from both when it is there
        if (d / "note_model_columnist.npz").exists():
            self.notes = _columnist().NoteModelColumnist(d / "note_model_columnist.npz")
        elif (d / "note_model_v3.npz").exists():
            self.notes = _v3().NoteModelV3(d / "note_model_v3.npz")
        else:
            self.notes = note_model.NoteModel(d / "note_model.npz") if (d / "note_model.npz").exists() else None

    # ------------------------------------------------------------------ features
    def _row(self, m: mania.ManiaMap, stats: dict | None = None) -> dict:
        """Head inputs for one chart (or one window treated as a chart); slot probabilities filled later."""
        s = stats or mania.stats(m)
        f = {**mania_tags.feature_vector(s), **{f"h:{k}": v for k, v in categories.heuristic_scores(s).items()}}
        if float(m.start.max() - m.start.min()) < WIN_MS:
            # windows.pool of a chart shorter than one window is its own stats, with zero spread
            for k, v in s.items():
                if k not in windows.SKIP:
                    f[f"wmax:{k}"] = f[f"wp75:{k}"] = float(v)
                    f[f"wstd:{k}"] = 0.0
        else:
            f.update(windows.pool(windows.window_stats(m)))
        f["_stats"] = s
        return f

    def _with_slots(self, rows: list[dict]) -> list[dict]:
        X = np.array([[r["_stats"].get(c, np.nan) for c in self.slot.features] for r in rows])
        P = self.slot.clf.predict_proba(X)
        for r, p in zip(rows, P):
            r["slot_probs"] = dict(zip(self.slot.classes, p.round(3).tolist()))
            r.update({f"slot:{k}": float(v) for k, v in r["slot_probs"].items()})
        return rows

    def _score(self, rows: list[dict]) -> np.ndarray:
        X = np.array([[r.get(c, np.nan) for c in self.columns] for r in rows], dtype=np.float32)
        return np.stack([self.heads[t].predict_proba(X)[:, 1] for t in self.tasks], axis=1)

    # ------------------------------------------------------------------ analysis
    def _series(self, m: mania.ManiaMap, win: float, hop: float) -> dict:
        """Head scores on windows of `win` ms centred every `hop` ms."""
        t0, t1 = float(m.start.min()), float(m.end.max())
        min_notes = max(12, int(windows.MIN_NOTES * win / windows.WIN_MS))
        centres, rows = [], []
        for c in np.arange(t0, t1 + hop, hop):
            w = windows.submap(m, c - win / 2, c + win / 2)
            if len(w.start) < min_notes:
                continue
            try:
                rows.append(self._row(w))
            except (ValueError, IndexError, ZeroDivisionError):
                continue
            centres.append(float(c))
        series = {}
        if rows:
            self._with_slots(rows)
            # SV only where the scroll visibly changes inside the window (Komi, 2026-11-02)
            open_ = reading.sv_window_gate(m, centres, win)
            memo_open = reading.memo_window_gate(m, centres, win)
            for r, g in zip(rows, open_):
                r["slot_probs"] = categories.gate_slot_probs(r["slot_probs"], bool(g))
            P = self._score(rows)
            for j, t in enumerate(self.tasks):
                if t not in CHART_ONLY:
                    v = P[:, j] * open_ if t in categories.SV_KEYS else P[:, j]
                    if t in categories.MEMO_KEYS:   # Memo needs a speed step in the window, not a steady slowjam
                        v = v * memo_open
                    series[t] = v.round(3).tolist()
            for k in self.slot.classes:
                series[f"slot:{k}"] = [round(r["slot_probs"][k], 3) for r in rows]
        return {"t": centres, "win_ms": win, "series": series}

    def analyze(self, m: mania.ManiaMap, hop: float = HOP_MS, sizes=WIN_SIZES) -> dict:
        f = self._with_slots([self._row(m)])[0]
        gate = reading.scroll_effect(m)
        sv_ok = gate["sve_effective"] > 0
        slot_probs = categories.gate_slot_probs(f["slot_probs"], sv_ok)
        chart = categories.gate_sv(dict(zip(self.tasks, self._score([f])[0].round(4).tolist())), sv_ok)
        slot = max(slot_probs, key=slot_probs.get)
        shown = categories.pick({t: chart[t] for t in categories.NAMES if t in chart}, slot, self.thresholds,
                                slot_probs=slot_probs, sv_effective=sv_ok)[0]

        by_size = {str(int(w)): self._series(m, w, hop) for w in sizes}
        return {
            "chart": {"scores": chart, "slot_probs": slot_probs, "slot": slot,
                      "slot_label": categories.slot_label(slot, slot_probs, shown),
                      "categories": shown, "sv_gate": gate},
            "windows": by_size[str(int(WIN_MS))] if str(int(WIN_MS)) in by_size else next(iter(by_size.values())),
            "windows_by_size": by_size,
            "thresholds": {t: round(v, 4) for t, v in self.thresholds.items()},
            "window_thresholds": {t: round(v, 4) for t, v in self.window_thresholds.items()},
            "report": self.report,
            "notes": self.note_scores(m),
        }

    def note_scores(self, m: mania.ManiaMap) -> dict | None:
        """Per-note probabilities from the note model, as one base64 uint8 matrix (notes x tasks, row-major)."""
        if self.notes is None:
            return None
        r = self.notes.scores(m)
        # SV tasks only on notes at or just after a visible scroll change (causal), and never on a
        # chart without one (Komi, 2026-11-02)
        g = reading.sv_note_gate(m)
        sv = [j for j, t in enumerate(r["tasks"]) if t in categories.SV_KEYS]
        note = r["note"].copy()
        note[:, sv] *= g[:, None]
        mg = reading.memo_note_gate(m)
        note[:, [j for j, t in enumerate(r["tasks"]) if t in categories.MEMO_KEYS]] *= mg[:, None]
        chart = r.get("chart")
        if chart is None:   # v3: no chart pooling in the per-note net; the 90th percentile of the event scores
            chart = dict(zip(r["tasks"], np.percentile(r["event"], 90, axis=0).tolist()))
        chart = categories.gate_sv(chart, bool(g.any()))
        q = np.clip(np.round(note * 255), 0, 255).astype(np.uint8)
        return {"tasks": r["tasks"], "n": int(q.shape[0]), "p": base64.b64encode(q.tobytes()).decode("ascii"),
                "chart": {t: round(v, 4) for t, v in chart.items()},
                # best-F1 chart cuts from the held-out fold; floored at 0.2 so thinly labelled heads (Memo:
                # 10 test positives, cut 0.12) don't light up everywhere
                "thresholds": {t: round(max(v, 0.2), 4) for t, v in self.notes.thresholds.items()},
                "report": self.notes.report}


# ---------------------------------------------------------------------- per-note labels

def local_labels(m: mania.ManiaMap, tol: float = 2.0, fast_ms: float = 220.0) -> list[str]:
    """One label per note (index-aligned with m.start) from its row and the previous rows."""
    n = len(m.start)
    lab = ["single"] * n
    rows: list[list[int]] = []
    last_t = None
    for i, t in enumerate(m.start):
        if last_t is not None and t - last_t <= tol:
            rows[-1].append(i)
        else:
            rows.append([i])
            last_t = float(t)
    cols = [{int(m.col[i]) for i in r} for r in rows]
    times = [float(m.start[r[0]]) for r in rows]
    for k, r in enumerate(rows):
        size = len(cols[k])
        fast = k > 0 and times[k] - times[k - 1] < fast_ms
        ov = cols[k] & cols[k - 1] if k > 0 else set()
        trill = (k >= 3 and fast and times[k - 1] - times[k - 2] < fast_ms and times[k - 2] - times[k - 3] < fast_ms
                 and cols[k] == cols[k - 2] and cols[k - 1] == cols[k - 3] and not ov)
        for i in r:
            c = int(m.col[i])
            if fast and c in ov:
                lab[i] = "chordjack" if size >= 2 and len(cols[k - 1]) >= 2 else "jack"
            elif fast and not ov:
                lab[i] = "trill" if trill and size == 1 else (
                    "handstream" if size >= 3 else "jumpstream" if size == 2 else "stream")
            elif m.end[i] > m.start[i]:
                lab[i] = "ln"
            elif size >= 2:
                lab[i] = "chord"
    return lab


# ---------------------------------------------------------------------- .osu extras

def osu_meta(text: str) -> dict:
    """The few [General] / [Metadata] / [Events] fields the player needs."""
    meta, section = {}, ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section in ("General", "Metadata") and ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
        elif section == "Events" and line.startswith("0,0,") and "bg" not in meta:
            mm = re.match(r'0,0,"?([^",]+)"?', line)
            if mm:
                meta["bg"] = mm.group(1)
    return {"audio": meta.get("AudioFilename"), "preview": int(float(meta.get("PreviewTime", -1) or -1)),
            "artist": meta.get("Artist", ""), "title": meta.get("Title", ""), "creator": meta.get("Creator", ""),
            "version": meta.get("Version", ""), "beatmap_id": meta.get("BeatmapID"),
            "beatmapset_id": meta.get("BeatmapSetID"), "bg": meta.get("bg")}


def chart_payload(m: mania.ManiaMap) -> dict:
    """Notes, beat lines and the scroll-speed profile, for drawing."""
    t0, t1 = float(m.start.min()), float(m.end.max())
    st, du, sp = mania.scroll_profile(m, min(t0, 0.0), t1 + 5000)
    red = sorted((tp for tp in m.timing if tp.uninherited and tp.beat_length > 0), key=lambda tp: tp.time)
    return {
        "keys": m.keys, "od": m.od, "hp": m.hp,
        "notes": {"t": m.start.round(1).tolist(), "e": m.end.round(1).tolist(), "c": m.col.tolist(),
                  "k": local_labels(m)},
        "timing": [[round(tp.time, 2), round(tp.beat_length, 4)] for tp in red],
        "scroll": {"t": st.round(1).tolist(), "v": sp.round(4).tolist()},
        "start": t0, "end": t1,
    }
