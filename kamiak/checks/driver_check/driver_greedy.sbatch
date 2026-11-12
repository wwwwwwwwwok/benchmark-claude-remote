#!/bin/bash
# Driver check (user, 2026-11-12): camas has NVIDIA driver 615.71.10 since its 11-12 maintenance (every earlier run:
# 535.274.02). Qwen3.6-27B without thinking answers the 25 greedy requests of 10-29 twice, exactly as step 4 of
# kamiak/main_run/quantities_27b_nothink.sbatch sent them in job 31488638 (sn17, 535.274.02): the server started with
# kamiak/qwen_vlm.sbatch's qwen3.6-27b-nothink case (unchanged since a2043fe) plus --enable-log-requests (as that smoke
# server), code/run_vlm.py (unchanged since a2043fe, sha256 4fca1d03...) with the arguments qwen_vlm.sbatch gives it,
# --task quantities --top_logprobs 0, the first pilot frame (--limit 1 --repeats 1), --temperature 0
# --presence_penalty 1.5 --workers 1. Then driver_compare.py against the two old runs. Writes only into the check
# folder. Submit from the repository root: sbatch kamiak/checks/driver_check/driver_greedy.sbatch
#SBATCH --job-name=driver-check
#SBATCH --partition=camas
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -eo pipefail
D=/data/lab/cejli/sy/tmp_checks/driver_check
R=results/vlm_scores/qwen3.6-27b-nothink
[ ! -e $D/greedy_new1.jsonl ] && [ ! -e $D/greedy_new2.jsonl ] || { echo "earlier outputs in $D: stop"; exit 1; }
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
vllm serve "$MODEL_DIR" --served-model-name "$SERVED_NAME" --port "$PORT" \
    --tensor-parallel-size "$N_GPUS" --limit-mm-per-prompt '{"image": 25}' \
    --gpu-memory-utilization 0.90 "${SERVE_ARGS[@]}" --enable-log-requests > $D/vllm_server.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until curl -sf "http://localhost:$PORT/v1/models" > /dev/null; do
    sleep 15
    kill -0 $SERVER 2>/dev/null || { echo "vLLM server died, see $D/vllm_server.log"; exit 1; }
done
echo "$(date '+%F %T') server ready; $(grep -m1 -o 'speculative_config=[^,]*' $D/vllm_server.log)"
ARGS=(--conditions grid,sep_time,sep_camera,ctrl_grid,ctrl_sep --repeats 5 --base_url "http://localhost:$PORT/v1"
      --model "$SERVED_NAME" "${SAMPLING[@]}" --top_logprobs 5 --workers 32 --send --task quantities --top_logprobs 0)
PILOT_T=(--grid_targets results/vlm_inputs/targets.csv --sep_targets results/vlm_inputs_separate/targets.csv)
for k in 1 2; do
    echo "=== greedy run $k $(date '+%F %T')"
    python -I code/run_vlm.py "${ARGS[@]}" "${PILOT_T[@]}" --limit 1 --repeats 1 --temperature 0 \
        --presence_penalty 1.5 --workers 1 --out $D/greedy_new$k.jsonl | tail -n 1
done
echo "sampling the server received (request log):"
grep -o 'params: SamplingParams(.*), lora_request' $D/vllm_server.log | sort | uniq -c | tee $D/sampling_params.txt || true
echo "old (job 31488638, smoke server, 10-29):"
cat /data/lab/cejli/sy/tmp_checks/quantities/nothink_27b/smoke_sampling_params.txt
echo "=== compare $(date '+%F %T')"
python -I kamiak/checks/driver_check/driver_compare.py $D/greedy_new1.jsonl $D/greedy_new2.jsonl \
    $R/smoke_greedy_pp1.5_quantities_$MODEL.jsonl $R/smoke_greedy_pp1.5_again_quantities_$MODEL.jsonl \
    | tee $D/driver_compare.out
echo "end $(date '+%F %T')"
