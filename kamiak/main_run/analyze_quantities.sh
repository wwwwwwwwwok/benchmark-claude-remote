#!/bin/bash
# The quantities task's analysis (user's design, 2026-10-28; CLAUDE.md "物理量的正式分析"): main result on the 1,084
# frames, sensitivity on all 1,123, as analyze.sh; the risk scores and the check of the resamples from the risk
# analysis folder given (its main_1084/ and sensitivity_1123/). Every model folder of vlm_scores/ with a
# main_quantities_<model>.jsonl(.gz) is analysed.
# usage (from the repository root): bash analyze_quantities.sh <output folder> <risk analysis folder>
set -euo pipefail
PY=/data/lab/cejli/sy/conda_envs/waymo/bin/python
L=/data/lab/cejli/sy/waymo_v143
A=$1
R=$2
mkdir -p "$A"
COMMON=(--run main --scene_labels "$L/scene_labels_waymo.csv" --group_by time_of_day --n_boot 2000 --seed 20261021)
FS=$L/results_lead_final_excl/frame_ssm_validation_training.csv
$PY -I code/analyze_quantities.py "$L/vlm_scores" "$L/quantities/truth_main.csv" "$FS" "$R/main_1084" "$A/main_1084" \
    "${COMMON[@]}" --closing_min 0.1 > "$A/main_1084.log" 2>&1 &
P_MAIN=$!
$PY -I code/analyze_quantities.py "$L/vlm_scores" "$L/quantities/truth_main.csv" "$FS" "$R/sensitivity_1123" \
    "$A/sensitivity_1123" "${COMMON[@]}" --closing_min 0 > "$A/sensitivity_1123.log" 2>&1 &
P_SENS=$!
RC=0
wait $P_MAIN || { echo "main_1084 failed (exit $?), see $A/main_1084.log" >&2; RC=1; }
wait $P_SENS || { echo "sensitivity_1123 failed (exit $?), see $A/sensitivity_1123.log" >&2; RC=1; }
tail -n 2 "$A/main_1084.log" "$A/sensitivity_1123.log"
exit $RC
