"""Event-sequence cache of every 4K chart, with the visual channels, for scripts/train_note_model.py.

    python scripts/build_event_cache_4k.py        # -> data/events4k_vis.pkl (needs data/osu)
"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import pandas as pd
from osu_patterns.deep import data


def main():
    f = pd.read_csv('data/all_features.csv.gz', index_col=0, usecols=['beatmap_id', 'keys'])
    ids = sorted(set(f.index[f['keys'] == 4]))
    print(len(ids), flush=True)
    ev = data.load_events(ids, 'data/osu', 'data/events4k_vis.pkl', jobs=4, visual=True)
    print(len(ev), sum(len(e.t) for e in ev.values()) / len(ev))


if __name__ == "__main__":  # Windows multiprocessing re-imports this file
    main()
