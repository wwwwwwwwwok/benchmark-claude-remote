#!/bin/bash
# Steps 3-6 of the second Qwen-Drive check again (2026-10-27): steps 1 and 2 of job 31440684 finished and their
# outputs stay in compare_out2/; its step 3 failed because the vision hooks could not be pickled into vLLM's workers
# (now in vision_hooks.py). Same steps as compare2.sbatch from 3 on. Submit from the repository root.
#SBATCH --job-name=drive-compare2b
#SBATCH --partition=camas
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -eo pipefail
Q=/data/lab/cejli/sy/tmp_checks/qwen_drive
C=$Q/compare_out2
C1=$Q/compare_out
for f in official_grid.json vllm_server_ctrl_sep.json vllm_server.log; do [ -e $C/$f ] || { echo "missing $C/$f"; exit 1; }; done
[ -z "$(ls -A $C/vllm_vision 2>/dev/null)" ] || { echo "$C/vllm_vision is not empty: stop"; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "repo at $(git log -1 --format='%h %ad' --date=iso); uncommitted in kamiak/qwen_vlm.sbatch: $(git status --short kamiak/qwen_vlm.sbatch | wc -l)"
source /opt/apps/miniconda3/3.12/etc/profile.d/conda.sh
conda activate /data/lab/cejli/sy/conda_envs/vllm_cu129
export VLLM_USE_FLASHINFER_SAMPLER=0
PORT=$((8000 + SLURM_JOB_ID % 1000))

echo "=== 3 vLLM offline scoring $(date +%T)"
MODEL=qwen-drive-1.0-4b
eval "$(sed -n '/^case "\$MODEL" in/,/^esac/p' kamiak/qwen_vlm.sbatch)"
VLLM_ALLOW_INSECURE_SERIALIZATION=1 python -I $Q/compare2/vllm_offline_score.py "$MODEL_DIR" $C/requests.json $C \
    $C:new $C1:think > $C/vllm_offline.log 2>&1 || { echo "offline scoring failed, see $C/vllm_offline.log"; exit 1; }
grep -E "^(new|think) |^hooks" $C/vllm_offline.log || true

echo "=== 4 vLLM server, Qwen3.5-4B (startup log only) $(date +%T)"
MODEL=qwen3.5-4b
eval "$(sed -n '/^case "\$MODEL" in/,/^esac/p' kamiak/qwen_vlm.sbatch)"
echo "MODEL=$MODEL MODEL_DIR=$MODEL_DIR SERVE_ARGS=${SERVE_ARGS[*]} SAMPLING=${SAMPLING[*]}"
vllm serve "$MODEL_DIR" --served-model-name "$MODEL" --port "$PORT" --tensor-parallel-size 2 \
    --limit-mm-per-prompt '{"image": 25}' --gpu-memory-utilization 0.90 "${SERVE_ARGS[@]}" > $C/vllm_server_qwen3.5-4b.log 2>&1 &
SERVER=$!
until curl -sf "http://localhost:$PORT/v1/models" > /dev/null; do
    sleep 15
    kill -0 $SERVER 2>/dev/null || { echo "vLLM server died, see $C/vllm_server_qwen3.5-4b.log"; exit 1; }
done
echo "server ready $(date +%T)"
kill $SERVER; wait $SERVER || true

echo "=== 5 default sampling and stop tokens $(date +%T)"
python -I $Q/compare2/check_defaults.py /data/lab/cejli/sy/models/Qwen-Drive-1.0-4B-vlm \
    /data/lab/cejli/sy/models/Qwen3.5-4B 2>&1 | grep -v "^\[transformers\]\|INFO\|WARNING" > $C/check_defaults.out
tail -2 $C/check_defaults.out

echo "=== 6 report $(date +%T)"
python -I $Q/compare2/report.py $C/requests.json $C $C1 | tee $C/report.txt
echo "end $(date +%T)"
