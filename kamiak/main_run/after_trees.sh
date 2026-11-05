#!/bin/bash
# EXPLORATORY, added after the trees were seen (user, 2026-11-04; CLAUDE.md "找模型自己的规则"): 1) the ceiling of
# Qwen3.6-27B's risk-score ranks from its 5 repeats (code/score_ceiling.py); 2) the A, B, C trees regrown on the 2,000
# resamples with every cut recorded (code/rule_tree.py --resampled_trees). Main result and sensitivity, 4 processes at
# once, CPU only, a minute or two. Inputs as kamiak/main_run/rule_tree.sh, plus that script's output folder (the trees'
# exploratory_tree_*.csv, only read).
# usage (from the repository root): bash after_trees.sh <rule analysis folder (analysis/rule_explore)>
#        <rule_tree.sh output folder (analysis/rule_explore/tree_27b)> <output folder>
set -uo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/data/lab/cejli/sy/conda_envs/waymo/bin/python
L=/data/lab/cejli/sy/waymo_v143
R=$1
T=$2
O=$3
FS=$L/results_lead_final_excl/frame_ssm_validation_training.csv
[ -f labels/scene_labels.csv ] || { echo "run it from the repository root (no labels/scene_labels.csv)" >&2; exit 1; }
for s in main_1084 sensitivity_1123; do
    [ -f $R/$s/exploratory_oos_r2.csv ] || { echo "$R/$s has no exploratory_oos_r2.csv" >&2; exit 1; }
    [ -f $T/$s/exploratory_tree_stability.csv ] || { echo "$T/$s has no exploratory_tree_stability.csv: run rule_tree.sh first" >&2; exit 1; }
done
mkdir -p $O
PIDS=()
for s in main_1084 sensitivity_1123; do
    $PY -I code/score_ceiling.py $L/analysis/after_qwen3.5-4b/$s $L/analysis/quantities_after_27b_think/$s $R/$s $T/$s \
        $O/$s > $O/${s}_ceiling.log 2>&1 &
    PIDS+=($!)
    $PY -I code/rule_tree.py $L/analysis/after_qwen3.5-4b/$s $L/analysis/quantities_after_27b_think/$s $FS \
        $L/quantities/truth_main.csv labels/scene_labels.csv $R/nearby_counts.csv $R/$s $T/$s \
        --resampled_trees $O/$s > $O/${s}_resampled_trees.log 2>&1 &
    PIDS+=($!)
done
RC=0
for p in "${PIDS[@]}"; do wait $p || RC=1; done
for s in main_1084 sensitivity_1123; do
    echo "== $s"
    tail -n 2 $O/${s}_ceiling.log
    tail -n 4 $O/${s}_resampled_trees.log
done
echo "RC=$RC end $(date +%T)"
exit $RC
