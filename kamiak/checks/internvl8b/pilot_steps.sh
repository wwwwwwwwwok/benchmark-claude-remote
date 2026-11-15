#!/bin/bash
# InternVL3.5-8B pilot report (user, 2026-11-13; CPU only), as the Gemma no-think pilot's steps. From the repository root:
#   srun -p kamiak -c 2 --mem=16G -t 00:30:00 bash kamiak/checks/internvl8b/pilot_steps.sh <off|on> <pilot job id> \
#        '<server ready>' '<last write>'
# Writes only tmp_checks/internvl8b/<model>/pilot/.
set -eo pipefail
MODE=$1; JOB=$2; T0=$3; T1=$4
if [ "$MODE" = on ]; then SFX=-thinking; else SFX=; fi
MODEL=internvl3.5-8b$SFX
P=/data/lab/cejli/sy/waymo_v143/vlm_scores/$MODEL/pilot_$MODEL.jsonl
G38=/data/lab/cejli/sy/waymo_v143/vlm_scores/internvl3.5-38b$SFX/main_internvl3.5-38b$SFX.jsonl.gz
O=/data/lab/cejli/sy/tmp_checks/internvl8b/$MODEL/pilot
PY=/data/lab/cejli/sy/conda_envs/waymo/bin/python
[ ! -e $O ] || { echo "$O exists: stop"; exit 1; }
mkdir -p $O
echo "node $(hostname); repo at $(git log -1 --format='%h %ad' --date=iso)"
echo "pilot $(stat -c '%s bytes, last written %y' $P); sha256 $(sha256sum $P | cut -c1-64)"
$PY -I kamiak/checks/internvl8b/check_smoke_internvl8b.py $MODE $P $G38 logs/vllm_$JOB.log 2550 > $O/check.out || true
$PY -I kamiak/checks/gemma4_compare/pilot_report.py $P "$T0" "$T1" \
    /data/lab/cejli/sy/tmp_checks/quantities/truth_pilot.csv \
    /data/lab/cejli/sy/waymo_v143/vlm_inputs/grid/targets_main.csv > $O/pilot_report.out
$PY -I kamiak/checks/check_pilot_settings.py $P > $O/settings.out
$PY -I kamiak/checks/count_cut_replies.py $P > $O/cut_replies.out
$PY -I kamiak/checks/summarize_scores.py $P > $O/summarize_scores.out
$PY -I kamiak/checks/camera_variants.py $P vlm_frames/main_all9_targets.csv > $O/camera_variants.out
for f in check pilot_report settings cut_replies summarize_scores camera_variants; do echo "=== $f"; cat $O/$f.out; done
