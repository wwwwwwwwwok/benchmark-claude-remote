#!/bin/bash
# EXPLORATORY: decision trees for Qwen3.6-27B's risk scores (user, 2026-11-04, decided before any tree was seen;
# CLAUDE.md "找模型自己的规则"). code/rule_tree.py on the main result (1,084 frames) and the sensitivity run (1,123), both
# at once, one process each, CPU only. Inputs as kamiak/main_run/analyze_rules.sh, plus the rule analysis's subfolders
# (their exploratory_oos_r2.csv: the linear fits beside the trees, and the check that the frames, ranks, cues and
# resamples are the rule analysis's). Needs scikit-learn in the waymo environment (added 2026-11-04 by
# kamiak/checks/add_sklearn_waymo.sh). About a minute each.
# usage (from the repository root): bash rule_tree.sh <rule analysis folder (analysis/rule_explore)> <output folder>
set -uo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/data/lab/cejli/sy/conda_envs/waymo/bin/python
L=/data/lab/cejli/sy/waymo_v143
R=$1
O=$2
FS=$L/results_lead_final_excl/frame_ssm_validation_training.csv
[ -f labels/scene_labels.csv ] || { echo "run it from the repository root (no labels/scene_labels.csv)" >&2; exit 1; }
for s in main_1084 sensitivity_1123; do
    [ -f $R/$s/exploratory_oos_r2.csv ] || { echo "$R/$s has no exploratory_oos_r2.csv: run analyze_rules_oos.sh first" >&2; exit 1; }
done
mkdir -p $O
PIDS=()
for s in main_1084 sensitivity_1123; do
    $PY -I code/rule_tree.py $L/analysis/after_qwen3.5-4b/$s $L/analysis/quantities_after_27b_think/$s $FS \
        $L/quantities/truth_main.csv labels/scene_labels.csv $R/nearby_counts.csv $R/$s $O/$s > $O/$s.log 2>&1 &
    PIDS+=($!)
done
RC=0
for p in "${PIDS[@]}"; do wait $p || RC=1; done
for s in main_1084 sensitivity_1123; do
    echo "== $s"
    tail -n 4 $O/$s.log
done
echo "RC=$RC end $(date +%T)"
exit $RC
