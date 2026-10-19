#!/bin/bash
# Run the VLM pilot on Kamiak: start a vLLM server for an open-weight Qwen model on H100 GPUs, then send the requests
# with code/run_vlm.py. See kamiak/README.md for the one-time setup.
#
# 2026-10-19 verified against the live cluster (sinfo -p camas / scontrol show partition camas / scontrol show
# node sn16): camas = sn[16-18], 4x H100 80GB each, GRES type name is "tesla" (gpu:tesla:4), so a plain `gpu:2`
# request resolves fine since there's only one GRES type per node. MaxTime=7-00:00:00, so 1-00:00:00 below is fine.
# All 3 nodes see real contention from other users (e.g. sn16 fully allocated at check time) - the job may queue.
#
# Submit from the repository root (Slurm does not create the log folder, so make it first):
#   mkdir -p logs && sbatch kamiak/qwen_vlm.sbatch
# Arguments after the script name are passed on to run_vlm.py and override the ones below, e.g. a small test run:
#   sbatch kamiak/qwen_vlm.sbatch --limit 2 --repeats 1 --out results/vlm_scores/smoke.jsonl --max_tokens 2000
#   (--max_tokens raised from the 300 default: Qwen3.6 thinking tokens count against the same budget, see below)
#SBATCH --job-name=qwen-vlm
#SBATCH --partition=camas              # 4x H100 80GB per node, GPU jobs not preemptable
#SBATCH --gres=gpu:2                   # 2 GPUs -> tensor parallel 2; confirmed syntax, see note above
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00              # within camas MaxTime=7-00:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

# ---- edit these ----------------------------------------------------------------------------------------------
CONDA_ENV=/data/lab/cejli/sy/conda_envs/vllm                 # conda env with vllm (2026-10-19: created via `conda
                                                              # create -p ... python=3.11` + srun pip install, not
                                                              # a plain venv — see docs/handoff.md 2026-10-19 entry)
MODEL_DIR=/data/lab/cejli/sy/models/Qwen3.6-27B              # downloaded weights (2026-10-19: switched from
                                                              # Qwen3-VL-32B-Instruct to Qwen3.6-27B, user's decision)
SERVED_NAME=qwen3.6-27b                                      # name recorded in every output line
RUN=pilot_qwen3.6-27b                                        # output file results/vlm_scores/$RUN.jsonl
N_GPUS=2
# ----------------------------------------------------------------------------------------------------------------

source /opt/apps/miniconda3/3.12/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
mkdir -p logs results/vlm_scores
PORT=$((8000 + SLURM_JOB_ID % 1000))

# vLLM OpenAI-compatible server; up to 25 images per request, prompts of ~14k tokens
# (older vLLM versions take --limit-mm-per-prompt image=25 instead of the JSON form)
# --reasoning-parser qwen3: Qwen3.6 is a hybrid thinking model; this keeps <think>...</think> out of
# message.content (it lands in message.reasoning_content instead) so code/run_vlm.py's JSON parser isn't
# fed the reasoning text. Decided 2026-10-19: keep thinking ON for this pilot, see docs/handoff.md.
vllm serve "$MODEL_DIR" --served-model-name "$SERVED_NAME" --port "$PORT" \
    --tensor-parallel-size "$N_GPUS" --max-model-len 32768 --limit-mm-per-prompt '{"image": 25}' \
    --gpu-memory-utilization 0.90 --reasoning-parser qwen3 > "logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

echo "waiting for the vLLM server on port $PORT ..."
until curl -sf "http://localhost:$PORT/v1/models" > /dev/null; do
    sleep 15
    kill -0 $SERVER 2>/dev/null || { echo "vLLM server died, see logs/vllm_${SLURM_JOB_ID}.log"; exit 1; }
done
echo "server ready"

# all five pilot conditions, 3 repeats each; already finished requests are skipped, so a rerun resumes.
# "$@" = extra arguments given to sbatch after the script name; argparse keeps the last value, so they override.
python -I code/run_vlm.py \
    --grid_targets results/vlm_inputs/targets.csv \
    --sep_targets results/vlm_inputs_separate/targets.csv \
    --out "results/vlm_scores/$RUN.jsonl" \
    --conditions grid,sep_time,sep_camera,ctrl_grid,ctrl_sep --repeats 3 \
    --base_url "http://localhost:$PORT/v1" --model "$SERVED_NAME" \
    --temperature 1.0 --top_logprobs 5 --workers 16 --send "$@"
