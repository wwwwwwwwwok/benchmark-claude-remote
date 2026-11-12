#!/bin/bash
# Gemma 4 31B-it before its pilot (user, 2026-11-12): small test, forced-answer rendering, check against the official
# transformers code (prompt tokens, image tokens and pixel values the same; greedy generations token by token, the
# standard of the Qwen-Drive check). Stops with exit code 1 at the first step with a check not met. Needs the CPU
# steps first (cpu_steps.sh: requests.json, official_prompts.json). Submit from the repository root:
#   sbatch kamiak/checks/gemma4_compare/gemma4_check.sbatch
# 1 server started with kamiak/qwen_vlm.sbatch's gemma-4-31b-it case: the small test (code/run_vlm.py with the
#   arguments kamiak/qwen_vlm.sbatch gives it, plus --limit 2 --repeats 1: 10 requests) and its checks
#   (check_smoke.py); then the 5 comparison requests greedily (qwen_drive_compare2/vllm_server_greedy.py, as is);
# 2 server started with kamiak/force_answer.sbatch's gemma-4-31b-it case: the forced-answer rendering
#   (force_render_check.py); then code/force_answer.py on the small test as in the runs (answers the replies cut at
#   max_tokens without an answer, if any);
# 3 the official code answers the 5 requests greedily (official_greedy.py, both GPUs);
# 4 vLLM offline scores the official generations and saves what its vision encoder receives (vllm_offline_score.py);
# 5 the report against (a)(b)(c) (report.py).
#SBATCH --job-name=gemma4-check
#SBATCH --partition=camas
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
set -eo pipefail
G=/data/lab/cejli/sy/tmp_checks/gemma4
C=$G/compare_out
CK=kamiak/checks/gemma4_compare
M=/data/lab/cejli/sy/models/gemma-4-31B-it
MODEL=gemma-4-31b-it
SMOKE=results/vlm_scores/$MODEL/smoke_$MODEL.jsonl
GT=results/vlm_inputs/targets.csv
ST=results/vlm_inputs_separate/targets.csv
[ ! -e $C ] || { echo "$C exists: stop"; exit 1; }
[ ! -e $SMOKE ] || { echo "$SMOKE exists: stop"; exit 1; }
[ -s $G/requests.json ] && [ -s $G/official_prompts.json ] || { echo "run cpu_steps.sh first"; exit 1; }
mkdir -p $C results/vlm_scores/$MODEL logs
cp $G/requests.json $C/requests.json
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "node $(hostname); repo at $(git log -1 --format='%h %ad' --date=iso); uncommitted in code/ and kamiak/: $(git status --short code kamiak | wc -l)"

source /opt/apps/miniconda3/3.12/etc/profile.d/conda.sh
conda activate /data/lab/cejli/sy/conda_envs/vllm_cu129
export VLLM_USE_FLASHINFER_SAMPLER=0
PORT=$((8000 + SLURM_JOB_ID % 1000))
serve() {   # serve <sbatch file whose case block to use> <log>: start the server as that file does, wait until ready
    eval "$(sed -n '/^case "\$MODEL" in/,/^esac/p' "$1")"
    echo "case of $1: MODEL_DIR=$MODEL_DIR SERVE_ARGS=${SERVE_ARGS[*]} SAMPLING=${SAMPLING[*]} FA_ARGS=${FA_ARGS[*]}"
    vllm serve "$MODEL_DIR" --served-model-name "$MODEL" --port "$PORT" --tensor-parallel-size 2 \
        --limit-mm-per-prompt '{"image": 25}' --gpu-memory-utilization 0.90 "${SERVE_ARGS[@]}" > "$2" 2>&1 &
    SERVER=$!
    until curl -sf "http://localhost:$PORT/v1/models" > /dev/null; do
        sleep 15
        kill -0 $SERVER 2>/dev/null || { echo "vLLM server died, see $2"; exit 1; }
    done
    echo "server ready $(date +%T)"
}
stop() { kill $SERVER; wait $SERVER || true; sleep 20; }

echo "=== 1 server (kamiak/qwen_vlm.sbatch's case): small test $(date +%T)"
FA_ARGS=()
serve kamiak/qwen_vlm.sbatch $C/vllm_server.log
python -I code/run_vlm.py --grid_targets $GT --sep_targets $ST --out $SMOKE \
    --conditions grid,sep_time,sep_camera,ctrl_grid,ctrl_sep --repeats 5 \
    --base_url "http://localhost:$PORT/v1" --model "$MODEL" \
    "${SAMPLING[@]}" --top_logprobs 5 --workers 32 --send --limit 2 --repeats 1
python -I $CK/check_smoke.py $SMOKE $G/official_prompts.json $C/vllm_server.log 10 | tee $C/smoke_check.out
echo "=== 1b the 5 comparison requests, greedy $(date +%T)"
python -I kamiak/checks/qwen_drive_compare2/vllm_server_greedy.py $C/requests.json "http://localhost:$PORT/v1" \
    "$MODEL" $C 24576
stop

echo "=== 2 server (kamiak/force_answer.sbatch's case): forced-answer rendering $(date +%T)"
serve kamiak/force_answer.sbatch $C/vllm_force_server.log
python -I $CK/force_render_check.py $SMOKE $GT $ST $M "http://localhost:$PORT/v1" "$MODEL" $C/force_render.json \
    | tee $C/force_render.out
echo "=== 2b forced answers for the small test (only replies cut at max_tokens without an answer) $(date +%T)"
python -I -u code/force_answer.py --pilot $SMOKE --grid_targets $GT --sep_targets $ST \
    --out results/vlm_scores/$MODEL/smoke_${MODEL}_forced_answers.jsonl \
    --base_url "http://localhost:$PORT/v1" --model "$MODEL" \
    "${SAMPLING[@]}" "${FA_ARGS[@]}" --max_tokens 1024 --top_logprobs 5 --send
stop

echo "=== 3 official code, greedy, both GPUs $(date +%T)"
python -I $CK/official_greedy.py $M $C/requests.json $C --extra-path $G/pylib 2>&1 | grep -v "Loading weights" \
    | tee $C/official.log

echo "=== 4 vLLM offline scoring $(date +%T)"
VLLM_ALLOW_INSECURE_SERIALIZATION=1 python -I $CK/vllm_offline_score.py $M $C/requests.json $C $C:new \
    > $C/vllm_offline.log 2>&1 || { echo "offline scoring failed, see $C/vllm_offline.log"; exit 1; }
grep -E "^new |^hooks" $C/vllm_offline.log || true

echo "=== 5 report $(date +%T)"
python -I $CK/report.py $C/requests.json $C | tee $C/report.txt
echo "end $(date +%T)"
