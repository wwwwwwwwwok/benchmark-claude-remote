#!/bin/bash
# Qwen3.6-27B without thinking, quantities task (user, 2026-10-29; CLAUDE.md "正式实验怎么跑" 2026-10-29): smoke test,
# greedy control, pilot and - when the pilot meets the user's three conditions - the main run, one after the other in
# this job. The server is started as kamiak/qwen_vlm.sbatch starts it for MODEL=qwen3.6-27b-nothink (the case is read
# from that file), and run_vlm.py gets the arguments qwen_vlm.sbatch gives it, with --task quantities --top_logprobs 0.
#  1 server with --enable-log-requests (smoke test only: its log shows every request's SamplingParams)
#  2 probe (kamiak/checks/probe_prompt_tail.py): the 25 smoke requests with return_token_ids
#  3 smoke test: first frame x 5 conditions x 5 quantities x 1, the run's settings; check_nothink_runs.py smoke. If
#    the replies are taken for thinking, the server is restarted without --reasoning-parser and 2-3 run again
#    (user: drop it then, and report).
#  4 greedy control: the same 25 at temperature 0 with presence_penalty 1.5, then 0, then 1.5 again (Claude's addition:
#    does greedy decoding repeat itself?), one worker each; check_nothink_runs.py greedy. An answer that differs
#    between 1.5 and 0 stops the job (the user decides).
#  5 server restarted without request logging; 6 pilot: 102 frames x 5 x 5 x 5 = 12,750 requests;
#  7 check_nothink_runs.py gate: 0 errors, the settings and no thinking on every line, frames with a lead read >= 98 %,
#    room for the main run; stop when one is not met;
#  8 main run: 1,123 frames x 5 x 5 x 5 = 140,375 requests, in the lab folder.
# A step not passed ends the job with exit 1; what was written stays. STAGE=main (sbatch --export=ALL,STAGE=main)
# skips 1-4 and 6, checks the pilot again and resumes the main run (e.g. after the time limit).
# The final sampling of vLLM's code is checked before, on CPU (check_defaults.py; tmp_checks/run_vlm_compare2/).
# Submit from the repository root: sbatch kamiak/main_run/quantities_27b_nothink.sbatch
#SBATCH --job-name=q27b-nothink
#SBATCH --partition=camas
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
STAGE=${STAGE:-all}
MODEL=qwen3.6-27b-nothink
eval "$(sed -n '/^case "\$MODEL" in/,/^esac/p' kamiak/qwen_vlm.sbatch)"
N_GPUS=2
SERVED_NAME=$MODEL
L=/data/lab/cejli/sy/waymo_v143
V=$L/vlm_inputs
R=results/vlm_scores/$MODEL
Q=/data/lab/cejli/sy/tmp_checks/quantities/nothink_27b
MAIN_DIR=$L/vlm_scores/$MODEL
PILOT=$R/pilot_quantities_$MODEL.jsonl
MAIN=$MAIN_DIR/main_quantities_$MODEL.jsonl
THINK_SMOKE=results/vlm_scores/qwen3.6-27b/smoke_quantities_qwen3.6-27b.jsonl
TRUTH=/data/lab/cejli/sy/tmp_checks/quantities/truth_pilot.csv
EXPECT='{"model": "qwen3.6-27b-nothink", "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 24576}'
mkdir -p logs "$R" "$Q" "$MAIN_DIR"
source /opt/apps/miniconda3/3.12/etc/profile.d/conda.sh
conda activate /data/lab/cejli/sy/conda_envs/vllm_cu129
export VLLM_USE_FLASHINFER_SAMPLER=0
PORT=$((8000 + SLURM_JOB_ID % 1000))
echo "MODEL=$MODEL MODEL_DIR=$MODEL_DIR STAGE=$STAGE"
echo "SERVE_ARGS=${SERVE_ARGS[*]}"
echo "SAMPLING=${SAMPLING[*]}"
echo "repo at $(git log -1 --format='%h %ad' --date=iso); uncommitted in code/ kamiak/: $(git status --short code/ kamiak/ | wc -l)"
sha256sum code/run_vlm.py kamiak/qwen_vlm.sbatch
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

SERVER=
trap 'kill $SERVER 2>/dev/null || true' EXIT
serve() {   # serve <log> [more vllm arguments]: start the server as kamiak/qwen_vlm.sbatch does; wait until ready
    local log=$1
    shift
    echo "$(date '+%F %T') starting vLLM: ${SERVE_ARGS[*]} $*; log $log"
    vllm serve "$MODEL_DIR" --served-model-name "$SERVED_NAME" --port "$PORT" \
        --tensor-parallel-size "$N_GPUS" --limit-mm-per-prompt '{"image": 25}' \
        --gpu-memory-utilization 0.90 "${SERVE_ARGS[@]}" "$@" > "$log" 2>&1 &
    SERVER=$!
    until curl -sf "http://localhost:$PORT/v1/models" > /dev/null; do
        sleep 15
        kill -0 $SERVER 2>/dev/null || { echo "vLLM server died, see $log"; exit 1; }
    done
    echo "$(date '+%F %T') server ready; $(grep -m1 -o 'speculative_config=[^,]*' "$log" || echo 'no speculative_config line')"
}
stop_server() { kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true; SERVER=; sleep 20; }
# as kamiak/qwen_vlm.sbatch passes them, then its "$@" (--task quantities --top_logprobs 0); targets and --out below
ARGS=(--conditions grid,sep_time,sep_camera,ctrl_grid,ctrl_sep --repeats 5 --base_url "http://localhost:$PORT/v1"
      --model "$SERVED_NAME" "${SAMPLING[@]}" --top_logprobs 5 --workers 32 --send --task quantities --top_logprobs 0)
PILOT_T=(--grid_targets results/vlm_inputs/targets.csv --sep_targets results/vlm_inputs_separate/targets.csv)
MAIN_T=(--grid_targets "$V/grid/targets_main.csv" --sep_targets "$V/separate/targets_main.csv")
SMOKE=(--limit 1 --repeats 1)
GREEDY=("1.5 pp1.5" "0 pp0" "1.5 pp1.5_again")

if [ "$STAGE" = all ]; then
    for f in "$R"/smoke*_quantities_$MODEL.jsonl "$PILOT" "$MAIN" "$Q"/probe*; do
        [ ! -e "$f" ] || { echo "$f exists: stop"; exit 1; }
    done
    echo "=== 1 server for the smoke test, request logging on"
    serve "logs/vllm_${SLURM_JOB_ID}_smoke.log" --enable-log-requests
    try=1
    while :; do
        P=$Q/probe$([ $try = 1 ] || echo $try)
        S=$R/smoke$([ $try = 1 ] || echo $try)_quantities_$MODEL.jsonl
        echo "=== 2 probe, attempt $try $(date '+%F %T')"
        python -I kamiak/checks/probe_prompt_tail.py code/run_vlm.py "$MODEL_DIR" "$P.json" "${ARGS[@]}" \
            "${PILOT_T[@]}" "${SMOKE[@]}" --workers 1 --out "${P}_lines.jsonl"
        echo "=== 3 smoke test, attempt $try $(date '+%F %T')"
        python -I code/run_vlm.py "${ARGS[@]}" "${PILOT_T[@]}" "${SMOKE[@]}" --out "$S" | tail -n 2
        rc=0
        python -I kamiak/checks/check_nothink_runs.py smoke "$S" "$THINK_SMOKE" "$P.json" "$EXPECT" \
            > "$Q/smoke_check$try.out" 2>&1 || rc=$?
        cat "$Q/smoke_check$try.out"
        if [ $rc = 3 ] && [ $try = 1 ]; then
            echo "=== replies taken for thinking: restart without --reasoning-parser (user, 2026-10-29)"
            stop_server
            NEW=()
            skip=0
            for x in "${SERVE_ARGS[@]}"; do
                if [ $skip = 1 ]; then skip=0; elif [ "$x" = --reasoning-parser ]; then skip=1; else NEW+=("$x"); fi
            done
            SERVE_ARGS=("${NEW[@]}")
            serve "logs/vllm_${SLURM_JOB_ID}_smoke2.log" --enable-log-requests
            try=2
            continue
        fi
        [ $rc = 0 ] || { echo "smoke test not passed (exit $rc): stop"; exit 1; }
        break
    done
    echo "=== 4 greedy control $(date '+%F %T')"
    G=()
    for g in "${GREEDY[@]}"; do
        read -r PP TAG <<< "$g"
        F=$R/smoke_greedy_${TAG}_quantities_$MODEL.jsonl
        python -I code/run_vlm.py "${ARGS[@]}" "${PILOT_T[@]}" "${SMOKE[@]}" --temperature 0 --presence_penalty "$PP" \
            --workers 1 --out "$F" | tail -n 1
        G+=("$F")
    done
    rc=0
    python -I kamiak/checks/check_nothink_runs.py greedy "${G[@]}" > "$Q/greedy_check.out" 2>&1 || rc=$?
    cat "$Q/greedy_check.out"
    echo "sampling the server received (request log of the smoke server):"
    cat logs/vllm_${SLURM_JOB_ID}_smoke*.log | grep -o 'params: SamplingParams(.*), lora_request' | sort | uniq -c \
        | tee "$Q/smoke_sampling_params.txt" || true
    [ $rc = 0 ] || { echo "greedy answers differ between presence_penalty 1.5 and 0 (or a run is incomplete): stop"; exit 1; }
    stop_server
fi

echo "=== 5 server for the pilot and the main run"
serve "logs/vllm_${SLURM_JOB_ID}.log"
if [ "$STAGE" = all ]; then
    echo "=== 6 pilot $(date '+%F %T')"
    python -I code/run_vlm.py "${ARGS[@]}" "${PILOT_T[@]}" --out "$PILOT" 2>&1 | tee "$Q/pilot_run.log"
    echo "pilot written $(date '+%F %T'): $(wc -l < "$PILOT") lines, $(stat -c %s "$PILOT") bytes"
fi
echo "=== 7 pilot gate $(date '+%F %T')"
rc=0
python -I kamiak/checks/check_nothink_runs.py gate "$PILOT" "$TRUTH" "$Q/pilot_run.log" "$EXPECT" "$MAIN_DIR" \
    > "$Q/pilot_gate.out" 2>&1 || rc=$?
cat "$Q/pilot_gate.out"
[ $rc = 0 ] || { echo "pilot gate not passed: stop before the main run"; exit 1; }
echo "=== 8 main run $(date '+%F %T')"
python -I code/run_vlm.py "${ARGS[@]}" "${MAIN_T[@]}" --out "$MAIN"
echo "main written $(date '+%F %T'): $(wc -l < "$MAIN") lines, $(stat -c %s "$MAIN") bytes"
echo "end $(date '+%F %T')"
