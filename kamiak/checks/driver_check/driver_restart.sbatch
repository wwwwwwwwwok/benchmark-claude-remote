#!/bin/bash
# Driver check, restart control (user, 2026-11-12): on the new driver, the 25 greedy requests of 10-29 in two separate
# server sessions (start, 25 requests, stop; start again, 25 requests, stop), sent exactly as driver_greedy.sbatch sends
# them (kamiak/qwen_vlm.sbatch's qwen3.6-27b-nothink case + --enable-log-requests; run_vlm.py with --task quantities
# --top_logprobs 0 --limit 1 --repeats 1 --temperature 0 --presence_penalty 1.5 --workers 1). Then driver_compare.py:
# session 1 vs session 2 (against the two old runs), and session 1 vs the earlier new-driver run greedy_new1.jsonl
# (job 31701426, a third session). Writes only tmp_checks/driver_check/restart/.
# Submit from the repository root: sbatch kamiak/checks/driver_check/driver_restart.sbatch
#SBATCH --job-name=driver-restart
#SBATCH --partition=camas
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -eo pipefail
D0=/data/lab/cejli/sy/tmp_checks/driver_check
D=$D0/restart
R=results/vlm_scores/qwen3.6-27b-nothink
[ ! -e $D ] || { echo "$D exists: stop"; exit 1; }
mkdir -p $D logs
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "node $(hostname); repo at $(git log -1 --format='%h %ad' --date=iso); uncommitted in code/ and kamiak/: $(git status --short code kamiak | wc -l)"
echo "run_vlm.py sha256 $(sha256sum code/run_vlm.py | cut -c1-64)"
source /opt/apps/miniconda3/3.12/etc/profile.d/conda.sh
conda activate /data/lab/cejli/sy/conda_envs/vllm_cu129
export VLLM_USE_FLASHINFER_SAMPLER=0
N_GPUS=2
MODEL=qwen3.6-27b-nothink
SERVED_NAME=$MODEL
eval "$(sed -n '/^case "\$MODEL" in/,/^esac/p' kamiak/qwen_vlm.sbatch)"
echo "MODEL_DIR=$MODEL_DIR SERVE_ARGS=${SERVE_ARGS[*]} SAMPLING=${SAMPLING[*]}"
PORT=$((8000 + SLURM_JOB_ID % 1000))
SERVER=
trap 'kill $SERVER 2>/dev/null || true' EXIT
for k in 1 2; do
    echo "=== session $k: start the server $(date '+%F %T')"
    vllm serve "$MODEL_DIR" --served-model-name "$SERVED_NAME" --port "$PORT" \
        --tensor-parallel-size "$N_GPUS" --limit-mm-per-prompt '{"image": 25}' \
        --gpu-memory-utilization 0.90 "${SERVE_ARGS[@]}" --enable-log-requests > $D/vllm_server_$k.log 2>&1 &
    SERVER=$!
    until curl -sf "http://localhost:$PORT/v1/models" > /dev/null; do
        sleep 15
        kill -0 $SERVER 2>/dev/null || { echo "vLLM server died, see $D/vllm_server_$k.log"; exit 1; }
    done
    echo "$(date '+%F %T') server ready; $(grep -m1 -o 'speculative_config=[^,]*' $D/vllm_server_$k.log)"
    ARGS=(--conditions grid,sep_time,sep_camera,ctrl_grid,ctrl_sep --repeats 5 --base_url "http://localhost:$PORT/v1"
          --model "$SERVED_NAME" "${SAMPLING[@]}" --top_logprobs 5 --workers 32 --send --task quantities --top_logprobs 0)
    python -I code/run_vlm.py "${ARGS[@]}" --grid_targets results/vlm_inputs/targets.csv \
        --sep_targets results/vlm_inputs_separate/targets.csv --limit 1 --repeats 1 --temperature 0 \
        --presence_penalty 1.5 --workers 1 --out $D/session$k.jsonl | tail -n 1
    grep -o 'params: SamplingParams(.*), lora_request' $D/vllm_server_$k.log | sort | uniq -c > $D/sampling_params_$k.txt || true
    kill $SERVER; wait $SERVER 2>/dev/null || true; SERVER=
    sleep 20
done
echo "=== session 1 vs session 2 (with the two old runs) $(date '+%F %T')"
python -I kamiak/checks/driver_check/driver_compare.py $D/session1.jsonl $D/session2.jsonl \
    $R/smoke_greedy_pp1.5_quantities_$MODEL.jsonl $R/smoke_greedy_pp1.5_again_quantities_$MODEL.jsonl \
    | tee $D/compare_sessions.out || true
echo "=== session 1 vs the earlier new-driver run (job 31701426) $(date '+%F %T')"
python -I kamiak/checks/driver_check/driver_compare.py $D/session1.jsonl $D0/greedy_new1.jsonl \
    $R/smoke_greedy_pp1.5_quantities_$MODEL.jsonl $R/smoke_greedy_pp1.5_again_quantities_$MODEL.jsonl \
    | tee $D/compare_with_31701426.out || true
echo "end $(date '+%F %T')"
