# Pattern Lens: live pattern visualizer

Play a 4K osu!mania chart and watch what the classifier reads in each section, as it plays.

* **Load** a `.osz` (charts + audio + background) or a `.osu` by upload or drag-and-drop, or paste a
  beatmapset / beatmap link. Links download the `.osz` from a public mirror (`OSZ_MIRRORS`); if no mirror
  answers, the charts alone come from osu.ppy.sh and the player runs on a clock with optional clicks.
* **Playfield**: scrolling notes (SV applied, toggleable), coloured by a local row rule (jack, chordjack,
  stream, jumpstream, handstream, trill, LN, chord) or by column. The categories that are "on" right now
  are listed in the top-left corner.
* **Now panel**: slot probabilities (Rice / LN / Hybrid / SV) and the 9 section-level categories as bars,
  with a tick at each category's section cut; user tags above their cut; the mix of local labels on screen.
* **Whole chart**: the chart-level result (slot, categories, Stamina, Tiebreaker, user tags), as the
  n8n/Trello service would report it.
* **Timeline**: note density, slot and every category across the song; click to seek.
* Keys: space play/pause, ←/→ 5 s, ↑/↓ scroll speed. `?set=<id or link>` opens a map directly.

## How the section scores work

`osu_patterns/live.py` slides a 16 s window (1 s hop) over the chart and scores each window with the
production heads as if it were a short chart: the same features the heads were trained on (chart stats,
rule scores, slot probabilities, pooled 16 s window stats). 16 s is the window the heads' pooled features
already use. Stamina, TB and streams/stamina measure chart length and are only shown for the whole chart.

Heads: `scripts/train_live_heads.py`, the gbm+win+relax recipe (held-out mean AP 0.721 categories /
0.602 tags on the v2 folds) refit on every labelled chart. Section cuts: `scripts/calibrate_live_windows.py`
picks, per task, the cut on a chart's highest window score that best predicts the chart label on ~4,000
labelled 4K charts. Both are stored in `models/live_heads.pkl`.

## Per-note model

The per-note bars come from a note model in `models/`: `note_model_columnist.npz` (+ `.json`) when present, else
`note_model_v3.npz`, else `note_model.npz`. The Columnist head (`osu_patterns/deep/columnist.py`,
`scripts/train_columnist_lens.py`) is the pretrained Columnist row encoder run causally (a note's score depends on the
rows up to it and nothing after), with a per-row score for each pattern task, trained from chart labels by
attention-MIL. `scripts/compare_note_heads.py` compares it with the other note models on held-out songs. Stamina, TB,
Reading, Memo and the scroll-speed tags get no per-note output. Numpy inference: `osu_patterns/note_model_columnist.py`.

## Run

    pip install -r service/viz/requirements.txt
    uvicorn service.viz.app:app --port 8770                       # from the repo root

    docker build -f service/viz/Dockerfile -t pattern-lens .      # from the repo root
    docker compose -f service/viz/docker-compose.yml up -d --build

Settings (environment): `KEEP_HOURS` (uploaded maps are deleted after this, default 6), `MAX_UPLOAD_MB` (80),
`OSZ_MIRRORS` (comma-separated URL templates with `{id}`), `WORK_DIR`, `MODELS_DIR`.

Behind Cloudflare Tunnel, point the public hostname at `http://pattern-lens:8770` when cloudflared shares a
Docker network with the container, otherwise at `http://<nas-ip>:8770`.

## Similar charts (`/similar`)

Paste a beatmap link and get the 4K charts that play most like it. `osu_patterns/columnist_embed.py` is a numpy port of the
pretrained Columnist encoder (checked against the torch network to 1e-7); a chart embedding is the mean of its row encodings
(windows of 512 rows), z-scored and L2-normalised with the index statistics, and similarity is the cosine. The index
(`models/similar/`: `embeddings.npy` or `embeddings_q8.npy` + scale, `mean.npy`, `std.npy`, `charts.csv`, `model.npz`) holds
every 4K chart of the pretraining set; a beatmap in it uses its stored embedding, any other chart (new link or uploaded
`.osu`) is embedded when asked. `SIMILAR_DIR` overrides the folder.
