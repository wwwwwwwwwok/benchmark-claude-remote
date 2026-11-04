#!/bin/bash
# EXPLORATORY: out-of-sample R2 of the rule analysis by leaving one segment out (user, 2026-11-04, decided before the
# rule analysis's results were seen; CLAUDE.md "找模型自己的规则"). code/analyze_rules.py --oos with the same arguments
# as kamiak/main_run/analyze_rules.sh, into the same folder, after that run (it reads its exploratory_rule_r2*.csv to
# check that the in-sample R2 are the same): exploratory_oos_*.csv, exploratory_oos_params.txt. Main result and
# sensitivity at once; a few minutes, one process each.
# usage (from the repository root): bash analyze_rules_oos.sh <the folder of analyze_rules.sh>
set -uo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/data/lab/cejli/sy/conda_envs/waymo/bin/python
L=/data/lab/cejli/sy/waymo_v143
O=$1
FS=$L/results_lead_final_excl/frame_ssm_validation_training.csv
[ -f labels/scene_labels.csv ] || { echo "run it from the repository root (no labels/scene_labels.csv)" >&2; exit 1; }
for s in main_1084 sensitivity_1123; do
    [ -f $O/$s/exploratory_rule_r2.csv ] || { echo "$O/$s has no exploratory_rule_r2.csv: run analyze_rules.sh first" >&2; exit 1; }
done
PIDS=()
for s in main_1084 sensitivity_1123; do
    $PY -I code/analyze_rules.py $L/analysis/after_qwen3.5-4b/$s $L/analysis/quantities_after_27b_think/$s $FS \
        $L/quantities/truth_main.csv labels/scene_labels.csv $O/nearby_counts.csv $O/$s --oos \
        > $O/${s}_oos.log 2>&1 &
    PIDS+=($!)
done
RC=0
for p in "${PIDS[@]}"; do wait $p || RC=1; done
for s in main_1084 sensitivity_1123; do
    echo "== $s"
    tail -n 5 $O/${s}_oos.log
done
echo "RC=$RC end $(date +%T)"
exit $RC
