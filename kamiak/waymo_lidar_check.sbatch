#!/bin/bash
# Lidar check (code/check_lidar_points.py) on the leads of one block of segments: the lead's rear face in the TOP-lidar
# points vs the rear of its label box, and vehicle types from the lidar segmentation labels. It reads the raw
# tfrecords, so it has to run before the scratch workspace expires (2026-11-04 for waymo_20261021_205447).
# One array task per block of the SSM run (kamiak/waymo_ssm.sbatch, same block numbers and RUN); it uses that block's
# frame_ssm.csv and writes <workspace>/ssm_parts_<RUN>/<split>/<block>/lidar_check/. The blocks are then joined with
# /data/lab/cejli/sy/tmp_checks/merge_lidar_parts.py (see docs/handoff.md section 2).
#
# Usage (repository root):
#   sbatch --array=0-20%20 --export=ALL,RUN=lead_overlap kamiak/waymo_lidar_check.sbatch <workspace> validation
#   sbatch --array=0-79%20 --export=ALL,RUN=lead_overlap kamiak/waymo_lidar_check.sbatch <workspace> training
#SBATCH --job-name=waymo-lidar
#SBATCH --partition=kamiak
#SBATCH --account=cejli
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-02:00:00              # 2026-10-21 demo: about 22 s per segment, so about 4 min per block
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
WS=${1:?usage: sbatch --array=... --export=ALL,RUN=<name> kamiak/waymo_lidar_check.sbatch <workspace> validation|training}
SPLIT=${2:?usage: sbatch --array=... --export=ALL,RUN=<name> kamiak/waymo_lidar_check.sbatch <workspace> validation|training}
PY=/data/lab/cejli/sy/conda_envs/waymo/bin/python
BLOCK=$(printf '%03d' "$SLURM_ARRAY_TASK_ID")
PART=$WS/ssm_parts${RUN:+_$RUN}/$SPLIT/$BLOCK
[[ -f "$PART/ssm/frame_ssm.csv" ]] || { echo "$PART/ssm/frame_ssm.csv is missing: run kamiak/waymo_ssm.sbatch first"; exit 1; }
echo "lidar check, block $BLOCK of $SPLIT (RUN=${RUN:-}) on $(hostname)"
"$PY" -I code/check_lidar_points.py "$WS/raw/$SPLIT" "$WS/extracted/$SPLIT" "$PART/ssm/frame_ssm.csv" "$PART/lidar_check"
echo "block $BLOCK done"
