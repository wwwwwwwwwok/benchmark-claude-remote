#!/bin/bash
# Submit the three steps for one split, each starting only after the previous one ended without errors (afterok):
#   download (kamiak/waymo_download.sbatch, dtn) -> extraction + checks + SSMs per block of 10 segments
#   (kamiak/waymo_ssm.sbatch, array, at most 20 blocks at a time) -> join the blocks (code/merge_ssm_parts.py).
# Run from the repository root; it only calls sbatch, so a login node is fine:
#   bash kamiak/waymo_submit.sh <workspace> validation
#   bash kamiak/waymo_submit.sh <workspace> training
# If a step fails, the later ones stay pending with reason DependencyNeverSatisfied: scancel them, fix, submit again
# (downloaded files and extracted segments are skipped). See kamiak/README.md.
# A split whose download is complete (DOWNLOAD_COMPLETE) is not downloaded again. RUN=<name> bash kamiak/waymo_submit.sh
# ... recomputes into <workspace>/ssm_parts_<name>/ and /data/lab/cejli/sy/waymo_v143/results_<name>/, keeping the
# earlier results (2026-10-21: RUN=lead_overlap).
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
RUN=${RUN:-}
OUT=/data/lab/cejli/sy/waymo_v143/results${RUN:+_$RUN}/$SPLIT
PARTS=$WS/ssm_parts${RUN:+_$RUN}/$SPLIT
mkdir -p logs

if [[ -f "$WS/raw/$SPLIT/DOWNLOAD_COMPLETE" ]]; then
    dl=none
    after=()
else
    dl=$(sbatch --parsable kamiak/waymo_download.sbatch "$WS" "$SPLIT")
    after=(--dependency=afterok:"$dl")
fi
ssm=$(sbatch --parsable "${after[@]}" --array=0-"$LAST"%20 --export=ALL,PER_TASK=$PER_TASK,RUN="$RUN" \
      kamiak/waymo_ssm.sbatch "$WS" "$SPLIT")
merge=$(sbatch --parsable --dependency=afterok:"$ssm" --job-name=waymo-merge --partition=kamiak --account=cejli \
        --cpus-per-task=1 --mem=8G --time=02:00:00 --output=logs/%x_%j.out \
        --wrap="$PY -I code/merge_ssm_parts.py $PARTS $OUT $N")
echo "$SPLIT ($N segments): download $dl -> extraction and SSMs $ssm (blocks 0-$LAST) -> merge $merge"
echo "results will be in $OUT; logs in logs/"
