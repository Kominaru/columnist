"""Torch-free row construction for the Columnist (4K rows, no BPM or snap). Shared by the torch network
(deep/columnist.py) and the numpy serving model (note_model_columnist.py), which runs without torch."""
from __future__ import annotations

import numpy as np

N_TOK = 256
MASK_ID = 256
N_GAPB = 24
WINDOW = 64            # rows each side that attention may use (causal: back only)


# ---------------------------------------------------------------------------------------------- rows

def chart_rows(notes, sv):
    """notes [(col, t_ms, end_ms | None)], sv [(t_ms, multiplier)] -> tok uint8, gap float16, svrow float16, times.

    Same as prep_dataset.chart_rows of the Columnist code, which built the pretraining store."""
    cols = np.array([n[0] for n in notes])
    t = np.array([n[1] for n in notes], dtype=np.int64)
    ends = np.array([-1 if n[2] is None else n[2] for n in notes], dtype=np.int64)
    hold = ends >= 0
    et = np.concatenate([t, ends[hold]])
    ec = np.concatenate([cols, cols[hold]])
    es = np.concatenate([np.where(hold, 2, 1), np.full(hold.sum(), 3)])
    times, inv = np.unique(et, return_inverse=True)
    states = np.zeros((len(times), 4), np.int64)
    np.maximum.at(states, (inv, ec), es)
    tok = (states * np.array([1, 4, 16, 64])).sum(1)
    gap = np.diff(times, prepend=times[0])
    if sv:
        st = np.array([s[0] for s in sv])
        sm = np.array([s[1] for s in sv], dtype=np.float32)
        o = np.argsort(st, kind="stable")
        st, sm = st[o], sm[o]
        j = np.searchsorted(st, times, side="right") - 1
        svrow = np.where(j >= 0, sm[np.clip(j, 0, None)], 1.0).astype(np.float32)
    else:
        svrow = np.ones(len(times), np.float32)
    return tok.astype(np.uint8), gap.clip(0, 60000).astype(np.float16), svrow.astype(np.float16), times


def map_rows(m):
    """ManiaMap (4K) -> (tok, gap, sv, row_time_ms, note_row): the model's rows, and for each note (index-aligned with
    m.start) the row it belongs to."""
    notes = [(int(c), int(round(float(s))), int(round(float(e))) if e > s else None)
             for c, s, e in zip(m.col, m.start, m.end)]
    notes = [(c, s, e if e is not None and e > s else None) for c, s, e in notes]
    sv = [(int(round(tp.time)), round(-100.0 / tp.beat_length, 3)) for tp in m.timing
          if not tp.uninherited and tp.beat_length < 0]
    order = sorted(range(len(notes)), key=lambda i: (notes[i][1], notes[i][0]))
    tok, gap, svr, times = chart_rows([notes[i] for i in order], sv)
    note_row = np.empty(len(notes), np.int64)
    for i, n in enumerate(notes):
        note_row[i] = int(np.searchsorted(times, n[1]))
    return tok, gap, svr, times, note_row
