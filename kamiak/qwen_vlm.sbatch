#!/bin/bash
# Run the VLM pilot on Kamiak: start a vLLM server for an open-weight Qwen model on H100 GPUs, then send the requests
# with code/run_vlm.py. See kamiak/README.md for the one-time setup. Lines marked CHECK depend on Kamiak's current
# configuration (verify with `sinfo -p camas`, `scontrol show partition camas`, `scontrol show node <gpu node>`).
#
# Submit from the repository root:   sbatch kamiak/qwen_vlm.sbatch
#SBATCH --job-name=qwen-vlm
#SBATCH --partition=camas              # 4x H100 80GB per node, GPU jobs not preemptable (CHECK)
#SBATCH --gres=gpu:2                   # 2 GPUs -> tensor parallel 2 for a 32B model (CHECK the gres syntax)
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00              # CHECK the partition time limit
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

# ---- edit these ----------------------------------------------------------------------------------------------
VENV=/data/lab/CHANGE_ME/venvs/vllm                         # python venv with vllm (README step 3)
MODEL_DIR=/data/lab/CHANGE_ME/models/Qwen3-VL-32B-Instruct  # downloaded weights (README step 4)
SERVED_NAME=qwen3-vl-32b                                    # name recorded in every output line
RUN=pilot_qwen3-vl-32b                                      # output file results/vlm_scores/$RUN.jsonl
N_GPUS=2
# ----------------------------------------------------------------------------------------------------------------

source "$VENV/bin/activate"
mkdir -p logs results/vlm_scores
PORT=$((8000 + SLURM_JOB_ID % 1000))

# vLLM OpenAI-compatible server; up to 25 images per request, prompts of ~14k tokens
# (older vLLM versions take --limit-mm-per-prompt image=25 instead of the JSON form)
vllm serve "$MODEL_DIR" --served-model-name "$SERVED_NAME" --port "$PORT" \
    --tensor-parallel-size "$N_GPUS" --max-model-len 32768 --limit-mm-per-prompt '{"image": 25}' \
    --gpu-memory-utilization 0.90 > "logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

echo "waiting for the vLLM server on port $PORT ..."
until curl -sf "http://localhost:$PORT/v1/models" > /dev/null; do
    sleep 15
    kill -0 $SERVER 2>/dev/null || { echo "vLLM server died, see logs/vllm_${SLURM_JOB_ID}.log"; exit 1; }
done
echo "server ready"

# all five pilot conditions, 3 repeats each; already finished requests are skipped, so a rerun resumes
python -I code/run_vlm.py \
    --grid_targets results/vlm_inputs/targets.csv \
    --sep_targets results/vlm_inputs_separate/targets.csv \
    --out "results/vlm_scores/$RUN.jsonl" \
    --conditions grid,sep_time,sep_camera,ctrl_grid,ctrl_sep --repeats 3 \
    --base_url "http://localhost:$PORT/v1" --model "$SERVED_NAME" \
    --temperature 1.0 --top_logprobs 5 --workers 16 --send
