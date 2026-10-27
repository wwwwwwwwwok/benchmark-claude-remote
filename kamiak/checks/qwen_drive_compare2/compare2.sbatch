#!/bin/bash
# Second Qwen-Drive-1.0-4B check (user, 2026-10-27: both 4B models without thinking, settings as qwen3-vl-32b):
# 1 the official code answers the same 5 requests greedily with the template's enable_thinking=False (GPU 0);
# 2 the vLLM server, started with kamiak/qwen_vlm.sbatch's qwen-drive-1.0-4b case, answers them greedily;
# 3 vLLM offline scores the official answers and the first check's official generations with thinking, and saves
#   what its vision encoder receives; 4 the Qwen3.5-4B server is started with its case, for its startup log only;
# 5 both models' final sampling and stop tokens with vLLM's own code; 6 the report. Submit from the repo root.
#SBATCH --job-name=drive-compare2
#SBATCH --partition=camas
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
set -eo pipefail
Q=/data/lab/cejli/sy/tmp_checks/qwen_drive
C=$Q/compare_out2
C1=$Q/compare_out
[ ! -e $C ] || { echo "$C exists: stop"; exit 1; }
mkdir -p $C
cp $C1/requests.json $C/requests.json                   # the same 5 requests as the first check
DPY=/data/lab/cejli/sy/conda_envs/qwen_drive/bin/python
SRC=$Q/src_28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0/src
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "repo at $(git log -1 --format='%h %ad' --date=iso); uncommitted in kamiak/qwen_vlm.sbatch: $(git status --short kamiak/qwen_vlm.sbatch | wc -l)"

echo "=== 1 official code, no thinking, GPU 0 $(date +%T)"
CUDA_VISIBLE_DEVICES=0 $DPY -I $Q/compare2/official_greedy.py $SRC /data/lab/cejli/sy/models/Qwen-Drive-1.0-4B \
    $C/requests.json $C --enable-thinking false --max-tokens 1024 2>&1 | grep -v "Loading weights" || true

source /opt/apps/miniconda3/3.12/etc/profile.d/conda.sh
conda activate /data/lab/cejli/sy/conda_envs/vllm_cu129
export VLLM_USE_FLASHINFER_SAMPLER=0
PORT=$((8000 + SLURM_JOB_ID % 1000))
serve() {   # serve <model> <log>: start the server with the model's case of kamiak/qwen_vlm.sbatch, wait until ready
    MODEL=$1
    eval "$(sed -n '/^case "\$MODEL" in/,/^esac/p' kamiak/qwen_vlm.sbatch)"
    echo "MODEL=$MODEL MODEL_DIR=$MODEL_DIR SERVE_ARGS=${SERVE_ARGS[*]} SAMPLING=${SAMPLING[*]}"
    vllm serve "$MODEL_DIR" --served-model-name "$MODEL" --port "$PORT" --tensor-parallel-size 2 \
        --limit-mm-per-prompt '{"image": 25}' --gpu-memory-utilization 0.90 "${SERVE_ARGS[@]}" > "$2" 2>&1 &
    SERVER=$!
    until curl -sf "http://localhost:$PORT/v1/models" > /dev/null; do
        sleep 15
        kill -0 $SERVER 2>/dev/null || { echo "vLLM server died, see $2"; exit 1; }
    done
    echo "server ready $(date +%T)"
}

echo "=== 2 vLLM server, Qwen-Drive $(date +%T)"
serve qwen-drive-1.0-4b $C/vllm_server.log
DRIVE_DIR=$MODEL_DIR
python -I $Q/compare2/vllm_server_greedy.py $C/requests.json "http://localhost:$PORT/v1" qwen-drive-1.0-4b $C 1024
kill $SERVER; wait $SERVER || true
sleep 20

echo "=== 3 vLLM offline scoring $(date +%T)"
VLLM_ALLOW_INSECURE_SERIALIZATION=1 python -I $Q/compare2/vllm_offline_score.py "$DRIVE_DIR" $C/requests.json $C \
    $C:new $C1:think > $C/vllm_offline.log 2>&1 || { echo "offline scoring failed, see $C/vllm_offline.log"; exit 1; }
grep -E "^(new|think) |^hooks" $C/vllm_offline.log || true

echo "=== 4 vLLM server, Qwen3.5-4B (startup log only) $(date +%T)"
serve qwen3.5-4b $C/vllm_server_qwen3.5-4b.log
kill $SERVER; wait $SERVER || true

echo "=== 5 default sampling and stop tokens $(date +%T)"
python -I $Q/compare2/check_defaults.py /data/lab/cejli/sy/models/Qwen-Drive-1.0-4B-vlm \
    /data/lab/cejli/sy/models/Qwen3.5-4B 2>&1 | grep -v "^\[transformers\]\|INFO\|WARNING" > $C/check_defaults.out
tail -2 $C/check_defaults.out

echo "=== 6 report $(date +%T)"
python -I $Q/compare2/report.py $C/requests.json $C $C1 | tee $C/report.txt
echo "end $(date +%T)"
