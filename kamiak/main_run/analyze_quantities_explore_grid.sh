#!/bin/bash
# The exploratory version of the quantities analysis with Qwen3.6-27B with thinking on grid as well (user, 2026-11-03;
# CLAUDE.md, the 11-03 entry): the whole output is exploratory and differs from the formal analysis
# (analysis/quantities_after_27b_think/, job 31542131, final: qwen3.6-27b on sep_camera only) only by the cell
# qwen3.6-27b x grid. As analyze_quantities.sh (same arguments, both shares), but the grid replies added from their own
# file (--extra_quantities), --only_conditions qwen3.6-27b=grid,sep_camera, and a note in params.txt and
# exploratory_params.txt. Run it only after the grid run is checked and compressed (the .gz is read).
# usage (from the repository root): bash analyze_quantities_explore_grid.sh <output folder> <risk analysis folder>
set -euo pipefail
PY=/data/lab/cejli/sy/conda_envs/waymo/bin/python
L=/data/lab/cejli/sy/waymo_v143
A=$1
R=$2
mkdir -p "$A"
GRID=$L/vlm_scores/qwen3.6-27b/main_quantities_grid_qwen3.6-27b.jsonl.gz
NOTE='EXPLORATORY as a whole (user, 2026-11-03): differs from the formal analysis (analysis/quantities_after_27b_think/) only by Qwen3.6-27B with thinking on grid, run after the results without thinking because grid was more accurate there (distance and ego speed slightly, ego acceleration clearly, than sep_camera); sep_camera is the condition chosen by the rule; the formal results are in quantities_after_27b_think/'
COMMON=(--run main --scene_labels "$L/scene_labels_waymo.csv" --group_by time_of_day --n_boot 2000 --seed 20261021
        --only_conditions qwen3.6-27b=grid,sep_camera --extra_quantities "qwen3.6-27b=$GRID" --note "$NOTE")
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
