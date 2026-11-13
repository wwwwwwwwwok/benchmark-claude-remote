#!/bin/bash
# Gemma 4 31B-it pilot report (CPU only). Run from the repository root with srun:
#   srun -p kamiak -c 2 --mem=16G -t 00:30:00 bash kamiak/checks/gemma4_compare/pilot_steps.sh
# Writes only tmp_checks/gemma4/pilot/. The times are the server being ready (logs/vllm_31703021.log, the last line
# before "Application startup complete") and the pilot file's last write (stat).
set -eo pipefail
P=/data/lab/cejli/sy/waymo_v143/vlm_scores/gemma-4-31b-it/pilot_gemma-4-31b-it.jsonl
O=/data/lab/cejli/sy/tmp_checks/gemma4/pilot
PY=/data/lab/cejli/sy/conda_envs/waymo/bin/python
[ ! -e $O ] || { echo "$O exists: stop"; exit 1; }
mkdir -p $O
echo "node $(hostname); repo at $(git log -1 --format='%h %ad' --date=iso)"
echo "pilot $(stat -c '%s bytes, last written %y' $P); sha256 $(sha256sum $P | cut -c1-64)"
$PY -I kamiak/checks/gemma4_compare/pilot_report.py $P '2026-11-12 22:35:06' '2026-11-12 23:32:36' \
    /data/lab/cejli/sy/tmp_checks/quantities/truth_pilot.csv \
    /data/lab/cejli/sy/waymo_v143/vlm_inputs/grid/targets_main.csv > $O/pilot_report.out
$PY -I kamiak/checks/check_pilot_settings.py $P > $O/settings.out
$PY -I kamiak/checks/count_cut_replies.py $P > $O/cut_replies.out
$PY -I kamiak/checks/summarize_scores.py $P > $O/summarize_scores.out
$PY -I kamiak/checks/camera_variants.py $P vlm_frames/main_all9_targets.csv > $O/camera_variants.out
for f in pilot_report settings cut_replies summarize_scores camera_variants; do echo "=== $f"; cat $O/$f.out; done
