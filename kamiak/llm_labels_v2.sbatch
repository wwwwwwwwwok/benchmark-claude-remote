#!/bin/bash
# LLM annotation, prompt v2 final (LEAD / OTHER / IMAGE_ONLY; exploratory; plan and the user's decisions of 2026-11-15 in
# CLAUDE.md): meta-llama/Llama-3.1-8B-Instruct (rev 0e9e39f..., bf16, official chat template), 1 H100, quote rule C.
# Pass 1: vLLM with --no-enable-prefix-caching; llm_labels_v2.py run twice (one request at a time), compared byte for byte.
# If the two runs differ, pass 2: also --enforce-eager, run twice again. If they still differ: stop and report.
# Each pass analyzes both runs into tmp_checks (label rates for both runs are printed). Tables are copied to the result
# folder only when all checks pass: the two runs byte-identical and the two runs' tables byte-identical.
# The failed run 31820897 stays in tmp_checks/reason_27b/llm_labels/v2/ (not touched).
# Submit from the repository root: sbatch kamiak/llm_labels_v2.sbatch
#SBATCH --job-name=llm-labels-v2c
#SBATCH --partition=camas
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
MD=/data/lab/cejli/sy/models/Llama-3.1-8B-Instruct
W1=/data/lab/cejli/sy/tmp_checks/reason_27b/run1
W=/data/lab/cejli/sy/tmp_checks/reason_27b/llm_labels/v2_rule_c
L=/data/lab/cejli/sy/waymo_v143
OUT=$L/analysis/reason_27b/main_1084_llm_labels
PW=/data/lab/cejli/sy/conda_envs/waymo/bin/python
[ ! -e $W ] || { echo "$W exists: stop"; exit 1; }
ls $OUT/exploratory_llm_v2_* > /dev/null 2>&1 && { echo "$OUT already has exploratory_llm_v2_ files: stop"; exit 1; }
mkdir -p $W logs
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "node $(hostname); repo $(git log -1 --format='%h %ad' --date=iso); uncommitted in code/ kamiak/: $(git status --short code kamiak | wc -l)"
echo "sha256: llm_labels_v2.py $(sha256sum code/llm_labels_v2.py | cut -c1-16), llm_labels.py $(sha256sum code/llm_labels.py | cut -c1-16), system prompt v2 $(sha256sum code/llm_labels_v2_system_prompt.txt | cut -c1-16)"
source /opt/apps/miniconda3/3.12/etc/profile.d/conda.sh
conda activate /data/lab/cejli/sy/conda_envs/vllm_cu129
export VLLM_USE_FLASHINFER_SAMPLER=0
PORT=$((8000 + SLURM_JOB_ID % 1000))
SERVER=
trap '[ -n "$SERVER" ] && kill $SERVER 2>/dev/null || true' EXIT

do_pass() {  # $1 pass folder, rest = extra vLLM args; returns 0 when the two runs are byte-identical, 1 if not, 2 on error
    local P=$W/$1; shift
    mkdir -p $P
    echo "===== $(basename $P): extra vLLM args: $* ($(date +%T))"
    vllm serve $MD --served-model-name llama-3.1-8b-instruct --port $PORT --tensor-parallel-size 1 --dtype bfloat16 \
        --max-model-len 4096 --gpu-memory-utilization 0.90 "$@" > $P/vllm_server.log 2>&1 &
    SERVER=$!
    until curl -sf "http://localhost:$PORT/v1/models" > /dev/null; do
        sleep 10; kill -0 $SERVER 2>/dev/null || { echo "vLLM server died, see $P/vllm_server.log"; return 2; }
    done
    echo "server ready $(date +%T)"; grep -m1 "non-default args" $P/vllm_server.log | cut -c1-700
    for i in 1 2; do
        echo "=== run $i ($(date +%T))"
        python -I code/llm_labels_v2.py run $W1 "http://localhost:$PORT/v1" $P/run$i.jsonl || return 2
    done
    kill $SERVER; wait $SERVER 2>/dev/null; SERVER=
    for t in $(seq 30); do [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) -lt 1000 ] && break; sleep 5; done
    echo "=== end of runs, server stopped $(date +%T)"
    local same=0
    for f in run.jsonl run_reask.jsonl; do
        a=$P/${f/run/run1}; b=$P/${f/run/run2}
        cmp -s $a $b && echo "  same bytes: $(basename $a) = $(basename $b)" || { echo "  DIFFERENT: $(basename $a) $(basename $b)"; same=1; }
    done
    $PW -I code/llm_labels_v2.py compare $P/run1.jsonl $P/run2.jsonl
    for i in 1 2; do
        $PW -I code/llm_labels_v2.py analyze $L $W1 $P/run$i.jsonl $P/tables_run$i > $P/analyze_run$i.out || return 2
        echo "  run $i label rates:"; sed 's/^/    /' $P/tables_run$i/exploratory_llm_v2_label_rates.csv
        echo "  run $i failures by type:"; sed 's/^/    /' $P/tables_run$i/exploratory_llm_v2_failures.csv
    done
    if [ $same = 0 ]; then
        for f in $(cd $P/tables_run1 && ls *.csv); do
            cmp -s $P/tables_run1/$f $P/tables_run2/$f && echo "  same bytes: $f" || { echo "  DIFFERENT: $f"; same=1; }
        done
    fi
    return $same
}

do_pass pass1_no_prefix_caching --no-enable-prefix-caching; RC=$?
PASS=pass1_no_prefix_caching
if [ $RC = 1 ]; then
    echo "pass 1: the two runs differ; pass 2 adds --enforce-eager"
    do_pass pass2_no_prefix_caching_eager --no-enable-prefix-caching --enforce-eager; RC=$?
    PASS=pass2_no_prefix_caching_eager
    [ $RC = 1 ] && echo "pass 2: the two runs still differ: stop (tables only in $W/$PASS)"
fi
if [ $RC = 0 ]; then
    mkdir -p $OUT
    cp -p $W/$PASS/tables_run1/exploratory_llm_v2_* $OUT/ && echo "all checks passed: tables of $PASS run 1 copied to $OUT" || RC=2
fi
echo "done $(date +%T); exit $RC"; exit $RC
