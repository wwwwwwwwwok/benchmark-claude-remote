#!/bin/bash
# Numbers in the thinking text, FULL RUN (exploratory, post-hoc; user, 2026-11-16 plan change in CLAUDE.md "思考文字里的数":
# about 10 hours left). Llama-3.1-8B-Instruct (rev 0e9e39f..., bf16, official chat template; prefix caching ON = vLLM default),
# 2 H100 (one vLLM server per GPU, same settings).
# 1. timing test on GPU 0: the 2,000 pilot sentences with 32 requests in flight; compared with the sequential pilot.
# 2. projected full time = 264,144 / (2 x pilot rate); above 4.5 hours: stop and report (no full run).
# 3. otherwise: the passing sentences split in two halves, one per GPU, 32 in flight each, ONCE; JSON check, one re-ask.
# 4. summaries and the per-answer / per-frame aggregation with the two subject rules.
# Submit from the repository root: sbatch kamiak/thinking_numbers_full.sbatch
#SBATCH --job-name=thinking-numbers-full
#SBATCH --partition=camas
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=07:00:00
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
MD=/data/lab/cejli/sy/models/Llama-3.1-8B-Instruct
S=/data/lab/cejli/sy/tmp_checks/reason_27b/thinking_numbers
W=$S/full_c32
L=/data/lab/cejli/sy/waymo_v143
OUT=$L/analysis/reason_27b/main_1084_thinking_numbers
PW=/data/lab/cejli/sy/conda_envs/waymo/bin/python
MAXH=4.5
[ ! -e $W ] || { echo "$W exists: stop"; exit 1; }
mkdir -p $W logs
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "node $(hostname); repo $(git log -1 --format='%h %ad' --date=iso); uncommitted in code/ kamiak/: $(git status --short code kamiak | wc -l)"
echo "sha256: thinking_numbers.py $(sha256sum code/thinking_numbers.py | cut -c1-16), system prompt $(sha256sum code/thinking_numbers_system_prompt.txt | cut -c1-16), sentences_pass.jsonl $(sha256sum $S/sentences_pass.jsonl | cut -c1-16)"
source /opt/apps/miniconda3/3.12/etc/profile.d/conda.sh
conda activate /data/lab/cejli/sy/conda_envs/vllm_cu129
export VLLM_USE_FLASHINFER_SAMPLER=0
P0=$((8000 + SLURM_JOB_ID % 1000)); P1=$((P0 + 1))
PIDS=""
trap 'for p in $PIDS; do kill $p 2>/dev/null; done' EXIT
serve() {  # $1 gpu, $2 port, $3 log
    CUDA_VISIBLE_DEVICES=$1 vllm serve $MD --served-model-name llama-3.1-8b-instruct --port $2 --tensor-parallel-size 1 \
        --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90 > $3 2>&1 &
    PIDS="$PIDS $!"; local pid=$!
    until curl -sf "http://localhost:$2/v1/models" > /dev/null; do
        sleep 10; kill -0 $pid 2>/dev/null || { echo "vLLM server on GPU $1 died, see $3"; return 1; }
    done
    echo "server GPU $1 ready $(date +%T)"; grep -m1 "non-default args" $3 | cut -c1-700
}
serve 0 $P0 $W/vllm_gpu0.log || exit 2
echo "=== 1. timing test: 2,000 pilot sentences, 32 in flight ($(date +%T))"
python -I code/thinking_numbers.py run $S/pilot.jsonl "http://localhost:$P0/v1" $W/pilot_c32.jsonl --concurrency 32 || exit 2
$PW -I code/thinking_numbers.py compare_pilot $S/pilot_runs/run1.jsonl $W/pilot_c32.jsonl
$PW -I code/thinking_numbers.py summary $W/pilot_c32.jsonl
N=$(wc -l < $S/sentences_pass.jsonl)
read RATE HOURS <<< $($PW -I -c "import json,sys; s=json.load(open('$W/pilot_c32_seconds.json')); r=2000/s['wall']; print(f'{r:.4f}', f'{$N/(2*r)/3600:.3f}')")
echo "pilot rate $RATE sentences per second on one GPU; projected full run $N sentences on two GPUs: $HOURS hours (limit $MAXH)"
if $PW -I -c "import sys; sys.exit(0 if $HOURS <= $MAXH else 1)"; then :; else echo "projected time above $MAXH hours: stop, no full run"; exit 3; fi
echo "=== 2. full run, two halves ($(date +%T))"
$PW -I code/thinking_numbers.py split $S/sentences_pass.jsonl $W/half1.jsonl $W/half2.jsonl || exit 2
serve 1 $P1 $W/vllm_gpu1.log || exit 2
python -I code/thinking_numbers.py run $W/half1.jsonl "http://localhost:$P0/v1" $W/run_half1.jsonl --concurrency 32 > $W/run_half1.out 2>&1 &
A=$!
python -I code/thinking_numbers.py run $W/half2.jsonl "http://localhost:$P1/v1" $W/run_half2.jsonl --concurrency 32 > $W/run_half2.out 2>&1 &
B=$!
RC=0; wait $A || RC=2; wait $B || RC=2
cat $W/run_half1.out $W/run_half2.out
echo "=== end of full run $(date +%T)"
[ $RC = 0 ] || { echo "a half failed: stop"; exit 2; }
for h in 1 2; do echo "--- summary half $h"; $PW -I code/thinking_numbers.py summary $W/run_half$h.jsonl; done
echo "=== 3. aggregation (subject rules)"
mkdir -p $OUT
$PW -I code/thinking_numbers.py aggregate $L $S/sentences_pass.jsonl $W/exploratory_thinking_numbers_aggregate.csv \
    $W/run_half1.jsonl $W/run_half2.jsonl || exit 2
cp -p $W/exploratory_thinking_numbers_aggregate.csv $OUT/
echo "done $(date +%T); exit 0"
