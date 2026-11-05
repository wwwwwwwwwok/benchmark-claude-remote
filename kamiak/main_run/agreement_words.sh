#!/bin/bash
# EXPLORATORY, added after the trees, the ceiling and the resampled trees were seen (user, 2026-11-05; CLAUDE.md
# "找模型自己的规则"): item 2, how alike the 5 conditions rank the frames (code/score_agreement.py, main result and
# sensitivity), and item 3, the words of the replies' main_hazard and reason in the high and the low tier
# (code/reason_words.py, both runs in one pass over the 27B risk replies). CPU only; the replies file is read once
# (about 10 GB compressed).
# usage (from the repository root): bash agreement_words.sh <after_trees folder (exploratory_ceiling.csv)> <output folder>
set -uo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/data/lab/cejli/sy/conda_envs/waymo/bin/python
L=/data/lab/cejli/sy/waymo_v143
C=$1
O=$2
for s in main_1084 sensitivity_1123; do
    [ -f $C/$s/exploratory_ceiling.csv ] || { echo "$C/$s has no exploratory_ceiling.csv" >&2; exit 1; }
done
mkdir -p $O
PIDS=()
for s in main_1084 sensitivity_1123; do
    $PY -I code/score_agreement.py $L/analysis/after_qwen3.5-4b/$s $L/analysis/quantities_after_27b_think/$s \
        $C/$s/exploratory_ceiling.csv $O/agreement/$s > $O/agreement_$s.log 2>&1 &
    PIDS+=($!)
done
$PY -I code/reason_words.py $L/vlm_scores/qwen3.6-27b $L/analysis/after_qwen3.5-4b $O/words > $O/words.log 2>&1 &
PIDS+=($!)
RC=0
for p in "${PIDS[@]}"; do wait $p || RC=1; done
for s in main_1084 sensitivity_1123; do echo "== agreement $s"; tail -n 2 $O/agreement_$s.log; done
echo "== words"; tail -n 3 $O/words.log
echo "RC=$RC end $(date +%T)"
exit $RC
