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
P_MAIN=$!
$PY -I code/analyze_vlm_ssm.py "$L/vlm_scores" "$L/results_lead_final_excl/frame_ssm_validation_training.csv" \
    "$A/sensitivity_1123" "${COMMON[@]}" --closing_min 0 > "$A/sensitivity_1123.log" 2>&1 &
P_SENS=$!
# wait for each run by its process id (user, 2026-10-22): a bare wait always returns 0. Both are waited for, then the
# script fails if either failed.
RC=0
wait $P_MAIN || { echo "main_1084 failed (exit $?), see $A/main_1084.log" >&2; RC=1; }
wait $P_SENS || { echo "sensitivity_1123 failed (exit $?), see $A/sensitivity_1123.log" >&2; RC=1; }
tail -n 2 "$A/main_1084.log" "$A/sensitivity_1123.log"    # -n: Kamiak's tail rejects "tail -2" with two files
exit $RC
