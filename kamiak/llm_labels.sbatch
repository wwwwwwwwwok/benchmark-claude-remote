#!/bin/bash
# LLM annotation of the 5,420 reasons (user, 2026-11-15; exploratory; plan in CLAUDE.md "LLM 标注理由"): one vLLM server
# with meta-llama/Llama-3.1-8B-Instruct (rev 0e9e39f..., bf16, official chat template), 1 H100; llm_labels.py run twice
# (one request at a time each), the two runs compared byte for byte; then llm_labels.py analyze on run 1 (result folder)
# and run 2 (a copy), compared. Submit from the repository root: sbatch kamiak/llm_labels.sbatch
#SBATCH --job-name=llm-labels
#SBATCH --partition=camas
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
MD=/data/lab/cejli/sy/models/Llama-3.1-8B-Instruct
W1=/data/lab/cejli/sy/tmp_checks/reason_27b/run1
W=/data/lab/cejli/sy/tmp_checks/reason_27b/llm_labels
L=/data/lab/cejli/sy/waymo_v143
OUT=$L/analysis/reason_27b/main_1084_llm_labels
PW=/data/lab/cejli/sy/conda_envs/waymo/bin/python
mkdir -p $W logs
for f in run1.jsonl run2.jsonl; do [ ! -e $W/$f ] || { echo "$W/$f exists: stop"; exit 1; }; done
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "node $(hostname); repo $(git log -1 --format='%h %ad' --date=iso); uncommitted in code/ kamiak/: $(git status --short code kamiak | wc -l)"
echo "sha256: llm_labels.py $(sha256sum code/llm_labels.py | cut -c1-16), system prompt $(sha256sum code/llm_labels_system_prompt.txt | cut -c1-16)"
source /opt/apps/miniconda3/3.12/etc/profile.d/conda.sh
conda activate /data/lab/cejli/sy/conda_envs/vllm_cu129
export VLLM_USE_FLASHINFER_SAMPLER=0
PORT=$((8000 + SLURM_JOB_ID % 1000))
vllm serve $MD --served-model-name llama-3.1-8b-instruct --port $PORT --tensor-parallel-size 1 --dtype bfloat16 \
    --max-model-len 4096 --gpu-memory-utilization 0.90 > $W/vllm_server.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until curl -sf "http://localhost:$PORT/v1/models" > /dev/null; do
    sleep 10; kill -0 $SERVER 2>/dev/null || { echo "vLLM server died, see $W/vllm_server.log"; exit 1; }
done
echo "server ready $(date +%T)"; grep -m1 "non-default args" $W/vllm_server.log | cut -c1-600
RC=0
for i in 1 2; do
    echo "=== run $i ($(date +%T))"
    python -I code/llm_labels.py run $W1 "http://localhost:$PORT/v1" $W/run$i.jsonl || { RC=1; break; }
done
echo "=== end of runs $(date +%T)"
if [ $RC = 0 ]; then
    for f in run.jsonl run_reask.jsonl; do
        a=$W/${f/run/run1}; b=$W/${f/run/run2}
        cmp -s $a $b && echo "  same bytes: $(basename $a) = $(basename $b)" || { echo "  DIFFERENT: $(basename $a) $(basename $b)"; RC=1; }
    done
    $PW -I code/llm_labels.py analyze $L $W1 $W/run1.jsonl $OUT || RC=1
    $PW -I code/llm_labels.py analyze $L $W1 $W/run2.jsonl $W/analyze_run2_out > /dev/null || RC=1
    for f in $(cd $W/analyze_run2_out && ls *.csv); do cmp -s $OUT/$f $W/analyze_run2_out/$f && echo "  same bytes: $f" || { echo "  DIFFERENT: $f"; RC=1; }; done
fi
echo "done $(date +%T); exit $RC"; exit $RC
