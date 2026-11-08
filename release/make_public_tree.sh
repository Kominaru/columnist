#!/usr/bin/env bash
# Export the public release tree (code, docs, tests, release cards; no data, results, logs or model binaries) as one squashed commit.
# usage: release/make_public_tree.sh /path/to/new/empty/dir
set -euo pipefail
out="$1"; mkdir -p "$out"
git ls-files -z -- osu_patterns scripts service tests docs tools paper release LICENSE README.md requirements.txt \
  | xargs -0 -I{} sh -c 'mkdir -p "$0/$(dirname "{}")" && cp "{}" "$0/{}"' "$out"
git -C "$out" init -q && git -C "$out" add -A
git -C "$out" -c user.name=Columnist -c user.email=noreply@example.com commit -qm "Columnist: foundation model for osu!mania 4K charts"
echo "squashed tree in $out"
