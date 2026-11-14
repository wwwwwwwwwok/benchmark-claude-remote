#!/bin/bash
# Gemma 4 31B-it without thinking, pilot report (user, 2026-11-13; CPU only). Run from the repository root with srun:
#   srun -p kamiak -c 2 --mem=16G -t 00:30:00 bash kamiak/checks/gemma4_compare/pilot_steps_nothink.sh
# Writes only tmp_checks/gemma4/nothink/pilot/. The times are the server being ready (logs/vllm_31755378.log, the last
# line before "Application startup complete") and the pilot file's last write (stat). check_smoke_nothink.py on all
# 2,550 lines (no thinking, prompt tokens = the official prompts without thinking, settings, server log), then the
# thinking run's pilot steps as they are.
set -eo pipefail
P=/data/lab/cejli/sy/waymo_v143/vlm_scores/gemma-4-31b-it-nothink/pilot_gemma-4-31b-it-nothink.jsonl
O=/data/lab/cejli/sy/tmp_checks/gemma4/nothink/pilot
C=/data/lab/cejli/sy/tmp_checks/gemma4/nothink
PY=/data/lab/cejli/sy/conda_envs/waymo/bin/python
[ ! -e $O ] || { echo "$O exists: stop"; exit 1; }
mkdir -p $O
echo "node $(hostname); repo at $(git log -1 --format='%h %ad' --date=iso)"
echo "pilot $(stat -c '%s bytes, last written %y' $P); sha256 $(sha256sum $P | cut -c1-64)"
$PY -I kamiak/checks/gemma4_compare/check_smoke_nothink.py $P $C/official_prompts_nothink.json logs/vllm_31755378.log \
    2550 > $O/check_nothink.out || true
$PY -I kamiak/checks/gemma4_compare/pilot_report.py $P '2026-11-13 19:57:08' '2026-11-13 20:36:50' \
    /data/lab/cejli/sy/tmp_checks/quantities/truth_pilot.csv \
    /data/lab/cejli/sy/waymo_v143/vlm_inputs/grid/targets_main.csv > $O/pilot_report.out
$PY -I kamiak/checks/check_pilot_settings.py $P > $O/settings.out
$PY -I kamiak/checks/count_cut_replies.py $P > $O/cut_replies.out
$PY -I kamiak/checks/summarize_scores.py $P > $O/summarize_scores.out
$PY -I kamiak/checks/camera_variants.py $P vlm_frames/main_all9_targets.csv > $O/camera_variants.out
for f in check_nothink pilot_report settings cut_replies summarize_scores camera_variants; do echo "=== $f"; cat $O/$f.out; done
