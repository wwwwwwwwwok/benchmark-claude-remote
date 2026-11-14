#!/bin/bash
# InternVL3.5-8B (user, 2026-11-13): small test before its pilot, and the GPU-count test. One server started exactly as
# kamiak/qwen_vlm.sbatch's case for $MODEL starts it (2 GPUs, tensor parallel 2, as the 38B); GPU memory recorded
# after loading; code/run_vlm.py with qwen_vlm.sbatch's arguments plus --limit 2 --repeats 1 (10 requests, the pilot
# frames); then check_smoke_internvl8b.py. Exit code 1 when a check is not met, so a pilot queued with afterok does not
# start. Submit from the repository root:
#   sbatch --export=ALL,MODEL=internvl3.5-8b,MODE=off kamiak/checks/internvl8b/smoke.sbatch
#SBATCH --job-name=internvl8b-smoke
#SBATCH --partition=camas
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -eo pipefail
: "${MODEL:?}" "${MODE:?}"
D=/data/lab/cejli/sy/waymo_v143/vlm_scores/$MODEL
C=/data/lab/cejli/sy/tmp_checks/internvl8b/$MODEL
G38=/data/lab/cejli/sy/waymo_v143/vlm_scores/internvl3.5-38b$([ "$MODE" = on ] && echo -thinking)/main_internvl3.5-38b$([ "$MODE" = on ] && echo -thinking).jsonl.gz
SMOKE=$D/smoke_$MODEL.jsonl
GT=results/vlm_inputs/targets.csv
ST=results/vlm_inputs_separate/targets.csv
[ ! -e $SMOKE ] || { echo "$SMOKE exists: stop"; exit 1; }
[ -s $G38 ] || { echo "$G38 missing"; exit 1; }
mkdir -p $D $C logs
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "node $(hostname); repo at $(git log -1 --format='%h %ad' --date=iso); uncommitted in code/ and kamiak/: $(git status --short code kamiak | wc -l)"
echo "sha256: run_vlm.py $(sha256sum code/run_vlm.py | cut -c1-16), qwen_vlm.sbatch $(sha256sum kamiak/qwen_vlm.sbatch | cut -c1-16)"
grep -n '^N_GPUS=' kamiak/qwen_vlm.sbatch

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
echo "GPU memory after loading (0.90 of each card is reserved by vLLM for weights + KV cache):"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
python -I code/run_vlm.py --grid_targets $GT --sep_targets $ST --out $SMOKE \
    --conditions grid,sep_time,sep_camera,ctrl_grid,ctrl_sep --repeats 5 \
    --base_url "http://localhost:$PORT/v1" --model "$MODEL" \
    "${SAMPLING[@]}" --top_logprobs 5 --workers 32 --send --limit 2 --repeats 1
python -I kamiak/checks/internvl8b/check_smoke_internvl8b.py $MODE $SMOKE $G38 $C/vllm_smoke_server.log 10 | tee $C/smoke_check.out
echo "done $(date +%T)"
