# Paper: learning tournament slots and pattern skillsets of osu!mania charts

- `main.pdf`: compiled paper. `main.tex`, `refs.bib`, `numbers.tex`, `tables/`, `fig/`: sources
  (build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`; needs TikZ and pgfplots).
- `fig/tikz/`: the explanatory figures, drawn in TikZ (`style.tex` holds the shared colours).
  `sh fig/tikz/render.sh <name>` renders one of them to PNG for checking.
- Slot results (tables `slots_*`, figures `confusion.pdf`, `ln_share.pdf`): run from the repository
  root after `scripts/train_slots.py`:

      python paper/experiments/paper_exp.py slots
      python paper/experiments/paper_figs.py ln_share confusion
      python paper/experiments/paper_tables.py slots_main features_table

- Category and tag results (tables `v2_*`, `fig/tikz/pertask.tex`, `numbers.tex`): run from the
  repository root after the v2 runs (`scripts/teacher_cv.py`, `scripts/train_mlp_v3.py`,
  `scripts/run_v2_suite.py`); it reads `results/v2` and `results/v5_x`:

      python paper/experiments/v2_tables.py --boot 500    # ~3 min; without --boot it keeps the last intervals

- `experiments/results/`: the slot outputs used in the paper (run of 2026-10-30).
