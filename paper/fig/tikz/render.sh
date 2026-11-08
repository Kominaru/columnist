#!/bin/sh
# render one figure body to PNG for checking: sh render.sh pipeline
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -jobname="prev_$1" "\def\fig{$1.tex}\input{preview.tex}" > "prev_$1.log" 2>&1 || { grep -A3 '^!' "prev_$1.log" | head -20; exit 1; }
pdftoppm -png -r 140 "prev_$1.pdf" "prev_$1" && echo ok
