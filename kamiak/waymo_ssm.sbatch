#!/bin/bash
# One block of PER_TASK segments of a downloaded split (Slurm array task k = files k*PER_TASK ... in sorted order):
#   1. extract each segment with code/wod_v143_extract.py into the scratch workspace (all five cameras, as on the
#      office PC), skipping segments done before, and copy its tables (context.json + parquet, ~4.3 MB) to lab storage,
#      so the SSMs can be recomputed after the workspace expires (the images stay on scratch only);
#   2. run code/check_v143_extract.py and code/waymo_ssm_frames.py on the block, with the defaults of the demo run
#      (results/waymo_demo: all five cameras, ego footprint), into <workspace>/ssm_parts/<split>/<block>/.
# code/merge_ssm_parts.py then joins the blocks; kamiak/waymo_submit.sh submits all steps in order. See kamiak/README.md.
#
# Usage (repository root):
#   sbatch --array=0-20%20 kamiak/waymo_ssm.sbatch <workspace> validation     # 202 segments -> blocks 0-20
#   sbatch --array=0-79%20 kamiak/waymo_ssm.sbatch <workspace> training       # 798 segments -> blocks 0-79
#   On the shared kamiak partition a block can be preempted: resubmit it (--array=<k>); extracted segments are skipped.
#SBATCH --job-name=waymo-ssm
#SBATCH --partition=kamiak
#SBATCH --account=cejli
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
WS=${1:?usage: sbatch --array=... kamiak/waymo_ssm.sbatch <workspace> validation|training}
SPLIT=${2:?usage: sbatch --array=... kamiak/waymo_ssm.sbatch <workspace> validation|training}
PER_TASK=${PER_TASK:-10}               # kamiak/waymo_submit.sh passes the same value
PY=/data/lab/cejli/sy/conda_envs/waymo/bin/python
TABLES=/data/lab/cejli/sy/waymo_v143/tables/$SPLIT
RAW=$WS/raw/$SPLIT
EXT=$WS/extracted/$SPLIT
DONE=$WS/extracted/${SPLIT}_done       # one empty file per finished segment
BLOCK=$(printf '%03d' "$SLURM_ARRAY_TASK_ID")
PART=$WS/ssm_parts/$SPLIT/$BLOCK

[[ -f "$RAW/DOWNLOAD_COMPLETE" ]] || { echo "$RAW: download not complete"; exit 1; }
# byte order, the same order as Python's sorted() in waymo_ssm_frames.py and merge_ssm_parts.py
mapfile -t FILES < <(cd "$RAW" && printf '%s\n' segment-*_with_camera_labels.tfrecord | LC_ALL=C sort)
first=$((SLURM_ARRAY_TASK_ID * PER_TASK))
(( first < ${#FILES[@]} )) || { echo "block $BLOCK starts after the last of ${#FILES[@]} files"; exit 1; }
mkdir -p "$EXT" "$DONE" "$TABLES" "$PART"
last=$((first + PER_TASK < ${#FILES[@]} ? first + PER_TASK - 1 : ${#FILES[@]} - 1))
echo "block $BLOCK of $SPLIT on $(hostname): files $first-$last of ${#FILES[@]} (counting from 0)"

SEGS=()
for f in "${FILES[@]:first:PER_TASK}"; do
    seg=${f#segment-}
    seg=${seg%_with_camera_labels.tfrecord}
    SEGS+=("$seg")
    if [[ -f "$DONE/$seg" ]]; then
        echo "$seg: extracted before"
        continue
    fi
    "$PY" -I code/wod_v143_extract.py "$RAW/$f" "$EXT"
    [[ -f "$EXT/$seg/context.json" ]] || { echo "$f: no output folder named $seg"; exit 1; }
    mkdir -p "$TABLES/$seg"
    cp "$EXT/$seg/context.json" "$EXT/$seg"/*.parquet "$TABLES/$seg/"
    touch "$DONE/$seg"
done

"$PY" -I code/check_v143_extract.py "$EXT" "$PART/checks" "${SEGS[@]}"
"$PY" -I code/waymo_ssm_frames.py "$EXT" "$PART/ssm" "${SEGS[@]}"
echo "block $BLOCK done: ${#SEGS[@]} segments"
