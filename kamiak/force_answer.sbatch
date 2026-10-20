#!/bin/bash
# Final answers for the pilot replies that ran out of tokens while still thinking (finish_reason "length"):
# code/force_answer.py sends each one's thinking back with Qwen's early-stopping text so the model answers from it.
# Model settings as in kamiak/qwen_vlm.sbatch, except that the server runs without --reasoning-parser (see
# code/force_answer.py). force_answer.py checks that the sampling below matches the pilot records.
#
# Submit from the repository root:
#   sbatch --export=ALL,MODEL=qwen3.6-27b kamiak/force_answer.sbatch
#   sbatch --export=ALL,MODEL=qwen3-vl-32b-thinking kamiak/force_answer.sbatch
#SBATCH --job-name=force-answer
#SBATCH --partition=camas
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

CONDA_ENV=/data/lab/cejli/sy/conda_envs/vllm_cu129
N_GPUS=2
MODEL=${MODEL:?set MODEL=qwen3.6-27b or MODEL=qwen3-vl-32b-thinking}

case "$MODEL" in
    qwen3-vl-32b-thinking)
        MODEL_DIR=/data/lab/cejli/sy/models/Qwen3-VL-32B-Thinking
        SERVE_ARGS=(--max-model-len 40960)
        SAMPLING=(--temperature 0.8 --top_p 0.95 --top_k 20)
        ;;
    internvl3.5-38b-thinking)   # the reply-starts-with-<think> template only matters for normal requests; here the
                                # thinking is sent back as the start of the reply. A sep reply that filled the context
                                # leaves no room for the early-stopping text; vLLM refuses it (2026-10-20: 1 reply).
        MODEL_DIR=/data/lab/cejli/sy/models/InternVL3_5-38B
        SERVE_ARGS=(--max-model-len 40960 --trust-remote-code --chat-template kamiak/internvl_thinking_chat_template.jinja)
        SAMPLING=(--temperature 0.6)
        ;;
    qwen3.6-27b)
        MODEL_DIR=/data/lab/cejli/sy/models/Qwen3.6-27B
        SERVE_ARGS=(--max-model-len 40960 --gdn-prefill-backend triton)
        SAMPLING=(--temperature 1.0 --top_p 0.95 --top_k 20)
        ;;
    *) echo "unknown MODEL=$MODEL (only the models that think)"; exit 1 ;;
esac
SERVED_NAME=$MODEL
PILOT=results/vlm_scores/$MODEL/pilot_$MODEL.jsonl
OUT=results/vlm_scores/$MODEL/pilot_${MODEL}_forced_answers.jsonl

source /opt/apps/miniconda3/3.12/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p logs
PORT=$((8000 + SLURM_JOB_ID % 1000))
echo "MODEL=$MODEL MODEL_DIR=$MODEL_DIR PILOT=$PILOT OUT=$OUT"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader   # same GPUs as the pilot runs

vllm serve "$MODEL_DIR" --served-model-name "$SERVED_NAME" --port "$PORT" \
    --tensor-parallel-size "$N_GPUS" --limit-mm-per-prompt '{"image": 25}' \
    --gpu-memory-utilization 0.90 "${SERVE_ARGS[@]}" \
    > "logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

echo "waiting for the vLLM server on port $PORT ..."
until curl -sf "http://localhost:$PORT/v1/models" > /dev/null; do
    sleep 15
    kill -0 $SERVER 2>/dev/null || { echo "vLLM server died, see logs/vllm_${SLURM_JOB_ID}.log"; exit 1; }
done
echo "server ready"

python -I -u code/force_answer.py --pilot "$PILOT" \
    --grid_targets results/vlm_inputs/targets.csv \
    --sep_targets results/vlm_inputs_separate/targets.csv \
    --out "$OUT" \
    --base_url "http://localhost:$PORT/v1" --model "$SERVED_NAME" \
    "${SAMPLING[@]}" --max_tokens 1024 --top_logprobs 5 --send "$@"
