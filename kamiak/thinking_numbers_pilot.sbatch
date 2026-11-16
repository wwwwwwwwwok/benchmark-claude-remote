#!/bin/bash
# Numbers in the thinking text, PILOT (exploratory, post-hoc; user, 2026-11-16; plan in CLAUDE.md "思考文字里的数"):
# Llama-3.1-8B-Instruct (rev 0e9e39f..., bf16, official chat template, prefix caching off as in the accepted run 31822093),
# 1 H100; thinking_numbers.py run twice on the 2,000 pilot sentences (one request at a time), compared byte for byte and
# with the compare step; summary of both runs. No full run: the user decides after the pilot.
# Submit from the repository root: sbatch kamiak/thinking_numbers_pilot.sbatch
#SBATCH --job-name=thinking-numbers-pilot
#SBATCH --partition=camas
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
MD=/data/lab/cejli/sy/models/Llama-3.1-8B-Instruct
S=/data/lab/cejli/sy/tmp_checks/reason_27b/thinking_numbers
W=$S/pilot_runs
PW=/data/lab/cejli/sy/conda_envs/waymo/bin/python
[ ! -e $W ] || { echo "$W exists: stop"; exit 1; }
mkdir -p $W logs
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "node $(hostname); repo $(git log -1 --format='%h %ad' --date=iso); uncommitted in code/ kamiak/: $(git status --short code kamiak | wc -l)"
echo "sha256: thinking_numbers.py $(sha256sum code/thinking_numbers.py | cut -c1-16), system prompt $(sha256sum code/thinking_numbers_system_prompt.txt | cut -c1-16), pilot.jsonl $(sha256sum $S/pilot.jsonl | cut -c1-16)"
source /opt/apps/miniconda3/3.12/etc/profile.d/conda.sh
conda activate /data/lab/cejli/sy/conda_envs/vllm_cu129
export VLLM_USE_FLASHINFER_SAMPLER=0
PORT=$((8000 + SLURM_JOB_ID % 1000))
vllm serve $MD --served-model-name llama-3.1-8b-instruct --port $PORT --tensor-parallel-size 1 --dtype bfloat16 \
    --max-model-len 4096 --gpu-memory-utilization 0.90 --no-enable-prefix-caching > $W/vllm_server.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until curl -sf "http://localhost:$PORT/v1/models" > /dev/null; do
    sleep 10; kill -0 $SERVER 2>/dev/null || { echo "vLLM server died, see $W/vllm_server.log"; exit 1; }
done
echo "server ready $(date +%T)"; grep -m1 "non-default args" $W/vllm_server.log | cut -c1-700
RC=0
for i in 1 2; do
    echo "=== run $i ($(date +%T))"
    python -I code/thinking_numbers.py run $S/pilot.jsonl "http://localhost:$PORT/v1" $W/run$i.jsonl || { RC=2; break; }
done
echo "=== end of runs $(date +%T)"
if [ $RC = 0 ]; then
    for f in run.jsonl run_reask.jsonl; do
        a=$W/${f/run/run1}; b=$W/${f/run/run2}
        cmp -s $a $b && echo "  same bytes: $(basename $a) = $(basename $b)" || { echo "  DIFFERENT: $(basename $a) $(basename $b)"; RC=1; }
    done
    $PW -I code/thinking_numbers.py compare $W/run1.jsonl $W/run2.jsonl || RC=1
    for i in 1 2; do echo "--- summary run $i"; $PW -I code/thinking_numbers.py summary $W/run$i.jsonl || RC=2; done
fi
echo "done $(date +%T); exit $RC"; exit $RC
