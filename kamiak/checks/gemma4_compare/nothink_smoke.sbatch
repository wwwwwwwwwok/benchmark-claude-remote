#!/bin/bash
# Gemma 4 31B-it without thinking (user, 2026-11-13): small test before its pilot. One server started exactly as
# kamiak/qwen_vlm.sbatch's gemma-4-31b-it-nothink case starts it; code/run_vlm.py with the arguments qwen_vlm.sbatch
# gives it plus --limit 2 --repeats 1 (10 requests, the pilot frames); then check_smoke_nothink.py (no thinking, prompt
# tokens = the official prompts without thinking of render_nothink.py, settings, server log). Exit code 1 when a check
# is not met, so a pilot queued with afterok does not start. Needs render_nothink.py's official_prompts_nothink.json.
# Submit from the repository root: sbatch kamiak/checks/gemma4_compare/nothink_smoke.sbatch
#SBATCH --job-name=gemma4-nothink-smoke
#SBATCH --partition=camas
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -eo pipefail
MODEL=gemma-4-31b-it-nothink
D=/data/lab/cejli/sy/waymo_v143/vlm_scores/$MODEL
C=/data/lab/cejli/sy/tmp_checks/gemma4/nothink
SMOKE=$D/smoke_$MODEL.jsonl
GT=results/vlm_inputs/targets.csv
ST=results/vlm_inputs_separate/targets.csv
[ ! -e $SMOKE ] || { echo "$SMOKE exists: stop"; exit 1; }
[ -s $C/official_prompts_nothink.json ] || { echo "run render_nothink.py first"; exit 1; }
mkdir -p $D logs
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "node $(hostname); repo at $(git log -1 --format='%h %ad' --date=iso); uncommitted in code/ and kamiak/: $(git status --short code kamiak | wc -l)"
echo "sha256: run_vlm.py $(sha256sum code/run_vlm.py | cut -c1-16), qwen_vlm.sbatch $(sha256sum kamiak/qwen_vlm.sbatch | cut -c1-16)"

source /opt/apps/miniconda3/3.12/etc/profile.d/conda.sh
conda activate /data/lab/cejli/sy/conda_envs/vllm_cu129
export VLLM_USE_FLASHINFER_SAMPLER=0
PORT=$((8000 + SLURM_JOB_ID % 1000))
eval "$(sed -n '/^case "\$MODEL" in/,/^esac/p' kamiak/qwen_vlm.sbatch)"
echo "case: MODEL_DIR=$MODEL_DIR SERVE_ARGS=${SERVE_ARGS[*]} SAMPLING=${SAMPLING[*]}"
vllm serve "$MODEL_DIR" --served-model-name "$MODEL" --port "$PORT" --tensor-parallel-size 2 \
    --limit-mm-per-prompt '{"image": 25}' --gpu-memory-utilization 0.90 "${SERVE_ARGS[@]}" > $C/vllm_smoke_server.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until curl -sf "http://localhost:$PORT/v1/models" > /dev/null; do
    sleep 15
    kill -0 $SERVER 2>/dev/null || { echo "vLLM server died, see $C/vllm_smoke_server.log"; exit 1; }
done
echo "server ready $(date +%T)"
python -I code/run_vlm.py --grid_targets $GT --sep_targets $ST --out $SMOKE \
    --conditions grid,sep_time,sep_camera,ctrl_grid,ctrl_sep --repeats 5 \
    --base_url "http://localhost:$PORT/v1" --model "$MODEL" \
    "${SAMPLING[@]}" --top_logprobs 5 --workers 32 --send --limit 2 --repeats 1
python -I kamiak/checks/gemma4_compare/check_smoke_nothink.py $SMOKE $C/official_prompts_nothink.json \
    $C/vllm_smoke_server.log 10 | tee $C/smoke_check.out
echo "done $(date +%T)"
