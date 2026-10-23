#!/bin/bash
# The formal experiment's analysis after a model's main run (user's decisions, 2026-10-22): main result on the 1,084
# frames (frames closing in slower than 0.1 m/s left out entirely), sensitivity on all 1,123 frames (no such rule);
# segments resampled, 2,000 resamples; groupings: all, time of day (Waymo's label), segment (weather not decided yet).
# usage (from the repository root): bash analyze.sh <output folder>
set -euo pipefail
PY=/data/lab/cejli/sy/conda_envs/waymo/bin/python
L=/data/lab/cejli/sy/waymo_v143
A=$1
mkdir -p "$A"
COMMON=(--run main --scene_labels "$L/scene_labels_waymo.csv" --group_by time_of_day --resample segment --n_boot 2000)
$PY -I code/analyze_vlm_ssm.py "$L/vlm_scores" "$L/results_lead_final_excl/frame_ssm_validation_training.csv" \
    "$A/main_1084" "${COMMON[@]}" --closing_min 0.1 > "$A/main_1084.log" 2>&1 &
$PY -I code/analyze_vlm_ssm.py "$L/vlm_scores" "$L/results_lead_final_excl/frame_ssm_validation_training.csv" \
    "$A/sensitivity_1123" "${COMMON[@]}" --closing_min 0 > "$A/sensitivity_1123.log" 2>&1 &
wait
tail -2 "$A/main_1084.log" "$A/sensitivity_1123.log"
