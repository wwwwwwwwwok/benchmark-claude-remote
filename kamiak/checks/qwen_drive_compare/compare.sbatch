#!/bin/bash
# Qwen-Drive-1.0-4B: vLLM against the official code, greedy, on the 5 requests of one frame (user, 2026-10-26).
# Steps: 0 build the requests (run_vlm.build); 1 official code on GPU 0; 2 the vLLM server, started with the
# arguments of kamiak/qwen_vlm.sbatch's qwen-drive-1.0-4b case, answers them; 3 vLLM offline scores the official
# generation and saves what its vision encoder receives; 4 the report. Submit from the repository root.
#SBATCH --job-name=drive-compare
#SBATCH --partition=camas
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
set -eo pipefail
Q=/data/lab/cejli/sy/tmp_checks/qwen_drive
C=$Q/compare_out
[ ! -e $C ] || { echo "$C exists: stop"; exit 1; }
mkdir -p $C
DPY=/data/lab/cejli/sy/conda_envs/qwen_drive/bin/python
VPY=/data/lab/cejli/sy/conda_envs/vllm_cu129/bin/python
SRC=$Q/src_28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0/src
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
git log -1 --format='repo at %h %ad' --date=iso

echo "=== 0 requests $(date +%T)"
$VPY -I $Q/compare/build_requests.py $Q/compare_inputs/grid/targets_compare.csv \
    $Q/compare_inputs/separate/targets_compare.csv $C/requests.json

echo "=== 1 official code, GPU 0 $(date +%T)"
CUDA_VISIBLE_DEVICES=0 $DPY -I $Q/compare/official_greedy.py $SRC /data/lab/cejli/sy/models/Qwen-Drive-1.0-4B \
    $C/requests.json $C 2>&1 | grep -v "Loading weights"

echo "=== 2 vLLM server $(date +%T)"
MODEL=qwen-drive-1.0-4b
eval "$(sed -n '/^case "\$MODEL" in/,/^esac/p' kamiak/qwen_vlm.sbatch)"
echo "MODEL_DIR=$MODEL_DIR SERVE_ARGS=${SERVE_ARGS[*]}"
source /opt/apps/miniconda3/3.12/etc/profile.d/conda.sh
conda activate /data/lab/cejli/sy/conda_envs/vllm_cu129
export VLLM_USE_FLASHINFER_SAMPLER=0
PORT=$((8000 + SLURM_JOB_ID % 1000))
vllm serve "$MODEL_DIR" --served-model-name "$MODEL" --port "$PORT" --tensor-parallel-size 2 \
    --limit-mm-per-prompt '{"image": 25}' --gpu-memory-utilization 0.90 "${SERVE_ARGS[@]}" > $C/vllm_server.log 2>&1 &
SERVER=$!
until curl -sf "http://localhost:$PORT/v1/models" > /dev/null; do
    sleep 15
    kill -0 $SERVER 2>/dev/null || { echo "vLLM server died, see $C/vllm_server.log"; exit 1; }
done
echo "server ready $(date +%T)"
python -I $Q/compare/vllm_server_greedy.py $C/requests.json "http://localhost:$PORT/v1" "$MODEL" $C
kill $SERVER; wait $SERVER || true
sleep 20

echo "=== 3 vLLM offline scoring $(date +%T)"
VLLM_ALLOW_INSECURE_SERIALIZATION=1 python -I $Q/compare/vllm_offline_score.py "$MODEL_DIR" $C/requests.json $C \
    > $C/vllm_offline.log 2>&1
grep -v "^\(INFO\|WARNING\|DEBUG\)" $C/vllm_offline.log | grep -v "it/s]" | tail -12 || true

echo "=== 4 report $(date +%T)"
python -I $Q/compare/report.py $C/requests.json $C | tee $C/report.txt
echo "end $(date +%T)"
