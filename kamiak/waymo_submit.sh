#!/bin/bash
# Submit the three steps for one split, each starting only after the previous one ended without errors (afterok):
#   download (kamiak/waymo_download.sbatch, dtn) -> extraction + checks + SSMs per block of 10 segments
#   (kamiak/waymo_ssm.sbatch, array, at most 20 blocks at a time) -> join the blocks (code/merge_ssm_parts.py).
# Run from the repository root; it only calls sbatch, so a login node is fine:
#   bash kamiak/waymo_submit.sh <workspace> validation
#   bash kamiak/waymo_submit.sh <workspace> training
# If a step fails, the later ones stay pending with reason DependencyNeverSatisfied: scancel them, fix, submit again
# (downloaded files and extracted segments are skipped). See kamiak/README.md.
set -euo pipefail
WS=${1:?usage: bash kamiak/waymo_submit.sh <workspace> validation|training}
SPLIT=${2:?usage: bash kamiak/waymo_submit.sh <workspace> validation|training}
case "$SPLIT" in
    validation) N=202 ;;
    training)   N=798 ;;
    *) echo "split must be validation or training"; exit 1 ;;
esac
[[ -d "$WS" ]] || { echo "$WS is not a folder: make the workspace first (mkworkspace)"; exit 1; }
PER_TASK=10
LAST=$(( (N + PER_TASK - 1) / PER_TASK - 1 ))
PY=/data/lab/cejli/sy/conda_envs/waymo/bin/python
OUT=/data/lab/cejli/sy/waymo_v143/results/$SPLIT
mkdir -p logs

dl=$(sbatch --parsable kamiak/waymo_download.sbatch "$WS" "$SPLIT")
ssm=$(sbatch --parsable --dependency=afterok:"$dl" --array=0-"$LAST"%20 --export=ALL,PER_TASK=$PER_TASK \
      kamiak/waymo_ssm.sbatch "$WS" "$SPLIT")
merge=$(sbatch --parsable --dependency=afterok:"$ssm" --job-name=waymo-merge --partition=kamiak --account=cejli \
        --cpus-per-task=1 --mem=8G --time=02:00:00 --output=logs/%x_%j.out \
        --wrap="$PY -I code/merge_ssm_parts.py $WS/ssm_parts/$SPLIT $OUT $N")
echo "$SPLIT ($N segments): download $dl -> extraction and SSMs $ssm (blocks 0-$LAST) -> merge $merge"
echo "results will be in $OUT; logs in logs/"
