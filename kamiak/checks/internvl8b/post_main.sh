#!/bin/bash
# InternVL3.5-8B main run, after it ends (user, 2026-11-14: validate, check and compress following kamiak/README.md
# "正式实验的主实验" step 4, as for the Gemma runs; no forced-answer job). From the repository root, CPU only:
#   srun -p kamiak -c 2 --mem=16G -t 01:00:00 bash kamiak/checks/internvl8b/post_main.sh <model> <job id>
# 1. the job log's last line; 2. check_run.py against the model's own pilot; 3. camera_variants.py; 4. answer_counts.py
# (stop before compressing if a frame x condition has no answer); 5. replies cut at max_tokens or not readable: listed
# with unparsable_report.py, and the run is NOT compressed (reported to the user first, as for the Gemma no-think run);
# 6. otherwise compress.sh, chmod a-w on the .gz; 7. the job and vLLM logs copied to vlm_scores/<model>/logs/.
set -eo pipefail
M=$1; J=$2
S=/data/lab/cejli/sy/waymo_v143/vlm_scores
F=$S/$M/main_$M.jsonl
P=$S/$M/pilot_$M.jsonl
O=/data/lab/cejli/sy/tmp_checks/internvl8b/$M/main
PY=/data/lab/cejli/sy/conda_envs/waymo/bin/python
[ ! -e $O ] || { echo "$O exists: stop"; exit 1; }
mkdir -p $O
echo "node $(hostname) $(date '+%F %T'); repo $(git log -1 --format=%h)"
echo "##### job log: $(grep -E '^done:' logs/qwen-vlm_$J.out | tail -1)"
echo "##### check_run.py"; $PY -I kamiak/main_run/check_run.py $F $P vlm_frames/main_all9_targets.csv | tee $O/check_run.out
echo "##### camera_variants.py"; $PY -I kamiak/checks/camera_variants.py $F vlm_frames/main_all9_targets.csv | tee $O/camera_variants.out
echo "##### answer_counts.py"; $PY -I kamiak/checks/answer_counts.py vlm_frames/main_all9_targets.csv $S/$M | tee $O/answer_counts.out
NBAD=$($PY -I -c "import json,sys; n=0
for l in open(sys.argv[1]):
    d=json.loads(l); n+= d.get('finish_reason')=='length' or d.get('parsed') is None
print(n)" $F)
echo "##### replies cut at max_tokens or not readable: $NBAD"
mkdir -p $S/$M/logs
cp -p logs/qwen-vlm_$J.out logs/vllm_$J.log $S/$M/logs/ && cmp logs/vllm_$J.log $S/$M/logs/vllm_$J.log && echo "logs copied to $S/$M/logs/"
if [ "$NBAD" != 0 ]; then
    $PY -I kamiak/checks/unparsable_report.py $F | tee $O/unparsable_report.out
    echo "NOT compressed: $NBAD replies to report to the user first"; exit 2
fi
if grep -qE 'cells with none: [1-9]|with no answer: [1-9]' $O/answer_counts.out; then echo "a cell has no answer: NOT compressed"; exit 2; fi
echo "##### compress"; bash kamiak/main_run/compress.sh $F | tee $O/compress.out
chmod a-w $F.gz && ls -l $F.gz
echo "end $(date '+%F %T')"
