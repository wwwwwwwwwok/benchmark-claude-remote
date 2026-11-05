#!/bin/bash
# EXPLORATORY, step 2 "only words and numbers" (user, 2026-11-05; CLAUDE.md, the entry "第 2 步 ... 用户 11-05 定"):
# Qwen3.6-27B (thinking on) scores 625 described situations x 5 repeats (code/run_text_vlm.py), then the replies cut at
# max_tokens get their forced answers (run_text_vlm.py --forced, as code/force_answer.py). NOT SUBMITTED: the user
# decides when (after the grid run and the jobs queued behind it, when the GPUs are free).
# The model, the server and the sampling are those of the 27B risk run: kamiak/qwen_vlm.sbatch, MODEL=qwen3.6-27b
# (vllm_cu129, 2 x H100 on camas, tensor parallel 2, --max-model-len 40960 --reasoning-parser qwen3
# --gdn-prefill-backend triton, --limit-mm-per-prompt image 25, --gpu-memory-utilization 0.90, VLLM_USE_FLASHINFER_SAMPLER
# 0, 32 workers; temperature 1.0, top_p 0.95, top_k 20, max_tokens 24576, log-probabilities 5) and, for the forced
# answers, kamiak/force_answer.sbatch, MODEL=qwen3.6-27b (the server without --reasoning-parser, max_tokens 1024).
# kamiak/qwen_vlm.sbatch itself is not changed (the grid run's resubmission needs it). A rerun resumes: the requests
# already in the output file are skipped, and so are the forced answers already written.
# Submit from the repository root:  sbatch kamiak/text_vlm.sbatch
#SBATCH --job-name=text-vlm
#SBATCH --partition=camas
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00              # estimated about 6 hours (27B risk run: 6.6 s per request); resumes if resubmitted
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
CONDA_ENV=/data/lab/cejli/sy/conda_envs/vllm_cu129
N_GPUS=2
MODEL=qwen3.6-27b
MODEL_DIR=/data/lab/cejli/sy/models/Qwen3.6-27B
SERVE_ARGS=(--max-model-len 40960 --reasoning-parser qwen3 --gdn-prefill-backend triton)     # qwen_vlm.sbatch, qwen3.6-27b
SERVE_ARGS_FORCED=(--max-model-len 40960 --gdn-prefill-backend triton)                       # force_answer.sbatch, qwen3.6-27b
L=/data/lab/cejli/sy/waymo_v143
OUTDIR=$L/vlm_scores_text/$MODEL
OUT=$OUTDIR/situations_$MODEL.jsonl
COMMON=(--truth $L/quantities/truth_main.csv --frames $L/analysis/after_qwen3.5-4b/main_1084/ssm_target_frames.csv
        --out "$OUT" --model "$MODEL")

source /opt/apps/miniconda3/3.12/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p logs "$OUTDIR"
PORT=$((8000 + SLURM_JOB_ID % 1000))
echo "MODEL=$MODEL MODEL_DIR=$MODEL_DIR OUT=$OUT; repository $(git log --oneline -n 1 | cut -c1-60)"
git status --short code/ kamiak/
sha256sum code/run_text_vlm.py code/run_vlm.py code/force_answer.py
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

# 1. the requests (the settings are checked against the first line of the 27B risk run's replies first)
python -I code/run_text_vlm.py "${COMMON[@]}" --check_run $L/vlm_scores/$MODEL/main_$MODEL.jsonl.gz --show 1 \
    --design_csv "$OUTDIR/design.csv"
serve requests "${SERVE_ARGS[@]}"
python -I -u code/run_text_vlm.py "${COMMON[@]}" --base_url "http://localhost:$PORT/v1" --top_logprobs 5 --workers 32 --send
kill $SERVER; wait $SERVER 2>/dev/null || true
# 2. the forced answers: the server again, without --reasoning-parser
N_CUT=$(grep -c '"finish_reason": "length"' "$OUT" || true)
echo "replies stopped at max_tokens (with or without an answer): $N_CUT"
serve forced "${SERVE_ARGS_FORCED[@]}"
python -I -u code/run_text_vlm.py "${COMMON[@]}" --base_url "http://localhost:$PORT/v1" --top_logprobs 5 --forced
echo "done $(date '+%F %T'); lines: $(wc -l < "$OUT") replies, $(cat "${OUT%.jsonl}_forced_answers.jsonl" 2>/dev/null | wc -l) forced answers"
