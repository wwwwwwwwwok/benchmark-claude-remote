#!/bin/bash
# Compress a finished run's jsonl (user's decision, 2026-10-22): write <file>.gz next to it, check that it decompresses
# to the same bytes, then move the original to vlm_scores_to_delete/<model>/ for the user to delete.
# usage: bash compress.sh <path to main_<model>.jsonl>
set -euo pipefail
F=$1
M=$(basename "$(dirname "$F")")
DEL=/data/lab/cejli/sy/waymo_v143/vlm_scores_to_delete/$M
[ -f "$F" ] && [ ! -e "$F.gz" ] && [ ! -e "$DEL/$(basename "$F")" ]
gzip -c "$F" > "$F.gz"
gzip -t "$F.gz"
cmp <(gzip -dc "$F.gz") "$F"
echo "$(basename "$F"): $(stat -c %s "$F") bytes, sha256 $(sha256sum < "$F" | cut -c1-64)"
echo "$(basename "$F").gz: $(stat -c %s "$F.gz") bytes; decompresses to the same bytes (cmp), sha256 of the decompressed $(gzip -dc "$F.gz" | sha256sum | cut -c1-64)"
mkdir -p "$DEL"
mv "$F" "$DEL/"
echo "original moved to $DEL/ (for the user to delete)"
