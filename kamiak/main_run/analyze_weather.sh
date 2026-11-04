#!/bin/bash
# The 7-model risk analysis again with the "same weather" grouping added (user, 2026-11-03; in the design of 10-21,
# CLAUDE.md "比较分组", never done before): exactly as analyze.sh (job 31445833 -> analysis/after_qwen3.5-4b/), but
# --scene_labels labels/scene_labels.csv (the user's weather labels of 2026-11-03; time_of_day and location from Waymo,
# the same text as scene_labels_waymo.csv) and --group_by weather,time_of_day. analyze.sh is not changed.
# usage (from the repository root): bash analyze_weather.sh <output folder>
set -euo pipefail
PY=/data/lab/cejli/sy/conda_envs/waymo/bin/python
L=/data/lab/cejli/sy/waymo_v143
A=$1
SL=$(pwd)/labels/scene_labels.csv
[ -f "$SL" ] || { echo "run it from the repository root (no $SL)" >&2; exit 1; }
mkdir -p "$A"
COMMON=(--run main --scene_labels "$SL" --group_by weather,time_of_day --resample segment --n_boot 2000)
$PY -I code/analyze_vlm_ssm.py "$L/vlm_scores" "$L/results_lead_final_excl/frame_ssm_validation_training.csv" \
    "$A/main_1084" "${COMMON[@]}" --closing_min 0.1 > "$A/main_1084.log" 2>&1 &
P_MAIN=$!
$PY -I code/analyze_vlm_ssm.py "$L/vlm_scores" "$L/results_lead_final_excl/frame_ssm_validation_training.csv" \
    "$A/sensitivity_1123" "${COMMON[@]}" --closing_min 0 > "$A/sensitivity_1123.log" 2>&1 &
P_SENS=$!
RC=0
wait $P_MAIN || { echo "main_1084 failed (exit $?), see $A/main_1084.log" >&2; RC=1; }
wait $P_SENS || { echo "sensitivity_1123 failed (exit $?), see $A/sensitivity_1123.log" >&2; RC=1; }
tail -n 2 "$A/main_1084.log" "$A/sensitivity_1123.log"
exit $RC
