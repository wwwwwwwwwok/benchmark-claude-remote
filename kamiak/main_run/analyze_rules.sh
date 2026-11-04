#!/bin/bash
# EXPLORATORY: the risk scores' own rule (user, 2026-11-03; CLAUDE.md "找模型自己的规则") on the real data.
# 1) code/count_nearby.py -> <output folder>/nearby_counts.csv (road users in the cameras' view, occluded ones included;
#    and the version with at least 1 lidar point); 2) code/analyze_rules.py on the main result (1,084 frames) and the
#    sensitivity run (1,123): S with the user's weather labels (labels/scene_labels.csv), the risk analysis's 2,000
#    resamples, 8 processes each, both at once; its checks decide the exit code. The commands and arguments are those
#    of tmp_checks/rule_explore/run_real.sbatch, moved here unchanged (user, 2026-11-04).
# usage (from the repository root): bash analyze_rules.sh <output folder>
set -uo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/data/lab/cejli/sy/conda_envs/waymo/bin/python
L=/data/lab/cejli/sy/waymo_v143
O=$1
FS=$L/results_lead_final_excl/frame_ssm_validation_training.csv
[ -f labels/scene_labels.csv ] || { echo "run it from the repository root (no labels/scene_labels.csv)" >&2; exit 1; }
mkdir -p $O
echo "=== count_nearby $(date +%T)"
$PY -I code/count_nearby.py $L/vlm_inputs/segments vlm_frames/main_all9_targets.csv $O/nearby_counts.csv \
    > $O/count_nearby.out 2>&1 || { cat $O/count_nearby.out; echo "count_nearby failed: stop"; exit 1; }
cat $O/count_nearby.out
echo "=== analyze_rules, main and sensitivity at once $(date +%T)"
PIDS=()
for s in main_1084 sensitivity_1123; do
    $PY -I code/analyze_rules.py $L/analysis/after_qwen3.5-4b/$s $L/analysis/quantities_after_27b_think/$s $FS \
        $L/quantities/truth_main.csv labels/scene_labels.csv $O/nearby_counts.csv $O/$s --workers 8 \
        > $O/$s.log 2>&1 &
    PIDS+=($!)
done
RC=0
for p in "${PIDS[@]}"; do wait $p || RC=1; done
for s in main_1084 sensitivity_1123; do
    echo "== $s"
    tail -n 6 $O/$s.log
done
echo "RC=$RC end $(date +%T)"
exit $RC
