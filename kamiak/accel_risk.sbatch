#!/bin/bash
# EXPLORATORY new experiment (user, 2026-11-09): Qwen3.6-27B's risk score with only the ego car's acceleration added to
# the question (code/run_vlm_accel.py: "Your car's acceleration is <true a_ego, 1 decimal> meters per second squared
# (negative means slowing down)." before the risk question); everything else exactly as in the main risk run: 1,123
# frames x 5 input conditions x 5 repeats = 28,075 requests, thinking on, temperature 1.0, top_p 0.95, top_k 20,
# max_tokens 24576, log-probabilities 5, 32 workers, the same vLLM server settings; then the forced answers (the server
# without --reasoning-parser, max_tokens 1024). Kept apart from the speed experiment (/data/lab/cejli/sy/speed_risk,
# the teacher's account): its own output folder vlm_scores/qwen3.6-27b-accel/ and job name.
# Time: about 51 h (the main 27B risk run); the camas GPU nodes have a maintenance reservation from 2026-11-12 10:30,
# so --time-min lets Slurm start it with a shorter limit that ends before; a rerun resumes (finished requests skipped).
# Submit from the repository root:  sbatch kamiak/accel_risk.sbatch
#SBATCH --job-name=accel-risk
#SBATCH --partition=camas
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2-12:00:00
#SBATCH --time-min=12:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
CONDA_ENV=/data/lab/cejli/sy/conda_envs/vllm_cu129
N_GPUS=2
MODEL=qwen3.6-27b
MODEL_DIR=/data/lab/cejli/sy/models/Qwen3.6-27B
SERVE_ARGS=(--max-model-len 40960 --reasoning-parser qwen3 --gdn-prefill-backend triton)     # qwen_vlm.sbatch, qwen3.6-27b
SERVE_ARGS_FORCED=(--max-model-len 40960 --gdn-prefill-backend triton)                       # force_answer.sbatch, qwen3.6-27b
SAMPLING=(--temperature 1.0 --top_p 0.95 --top_k 20)
L=/data/lab/cejli/sy/waymo_v143
V=$L/vlm_inputs
TARGETS=(--grid_targets $V/grid/targets_main.csv --sep_targets $V/separate/targets_main.csv)
TRUTH=$L/quantities/truth_main.csv
OUTDIR=$L/vlm_scores/qwen3.6-27b-accel
OUT=$OUTDIR/main_accel_$MODEL.jsonl
FORCED=$OUTDIR/main_accel_${MODEL}_forced_answers.jsonl

source /opt/apps/miniconda3/3.12/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p logs "$OUTDIR"
PORT=$((8000 + SLURM_JOB_ID % 1000))
echo "node $(hostname), start $(date '+%F %T'); MODEL=$MODEL OUT=$OUT; repository $(git log --oneline -n 1 | cut -c1-60)"
git status --short code/ kamiak/
sha256sum code/run_vlm.py code/force_answer.py code/run_vlm_accel.py "$TRUTH"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

serve() {        # $1: a label for the log file, then the server arguments
    local label=$1; shift
    vllm serve "$MODEL_DIR" --served-model-name "$MODEL" --port "$PORT" \
        --tensor-parallel-size "$N_GPUS" --limit-mm-per-prompt '{"image": 25}' \
        --gpu-memory-utilization 0.90 "$@" > "logs/vllm_${SLURM_JOB_ID}_${label}.log" 2>&1 &
    SERVER=$!
    echo "waiting for the vLLM server on port $PORT ($*) ..."
    until curl -sf "http://localhost:$PORT/v1/models" > /dev/null; do
        sleep 15
        kill -0 $SERVER 2>/dev/null || { echo "vLLM server died, see logs/vllm_${SLURM_JOB_ID}_*.log"; exit 1; }
    done
    echo "server ready $(date '+%F %T')"
}
trap 'kill ${SERVER:-0} 2>/dev/null || true' EXIT

# 1. the requests (first the assembled first request and the counts, nothing sent)
python -I code/run_vlm_accel.py "$TRUTH" run "${TARGETS[@]}" --out "$OUT" \
    --conditions grid,sep_time,sep_camera,ctrl_grid,ctrl_sep --repeats 5 --model "$MODEL" --show 1
serve requests "${SERVE_ARGS[@]}"
python -I -u code/run_vlm_accel.py "$TRUTH" run "${TARGETS[@]}" --out "$OUT" \
    --conditions grid,sep_time,sep_camera,ctrl_grid,ctrl_sep --repeats 5 \
    --base_url "http://localhost:$PORT/v1" --model "$MODEL" "${SAMPLING[@]}" --max_tokens 24576 \
    --top_logprobs 5 --workers 32 --send
kill $SERVER; wait $SERVER 2>/dev/null || true
# 2. the forced answers: the server again, without --reasoning-parser
echo "replies stopped at max_tokens: $(grep -c '"finish_reason": "length"' "$OUT" || true)"
serve forced "${SERVE_ARGS_FORCED[@]}"
python -I -u code/run_vlm_accel.py "$TRUTH" forced --pilot "$OUT" "${TARGETS[@]}" --out "$FORCED" \
    --base_url "http://localhost:$PORT/v1" --model "$MODEL" "${SAMPLING[@]}" --max_tokens 1024 --top_logprobs 5 --send
echo "done $(date '+%F %T'); $(wc -l < "$OUT") replies, $(cat "$FORCED" 2>/dev/null | wc -l) forced answers"
