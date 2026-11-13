#!/bin/bash
# Diagnosis after the Gemma 4 check 31701268 (user, 2026-11-12): the official transformers code answers the same 5
# requests greedily with the pixel values cast to bfloat16 before the model (official_greedy.py --cast-pixels-bf16,
# as vLLM does), so both sides' vision input is the same; vLLM offline scores those generations; report.py against
# (a)(b)(c); diag_b_steps.py lists the differing positions in steps of 0.125. The vLLM side of (a) and (c) is unchanged
# (its server greedy answers are copied from compare_out). Exit code 1 unless (a) and (b) are met.
# Submit from the repository root: sbatch kamiak/checks/gemma4_compare/gemma4_check_bf16.sbatch
#SBATCH --job-name=gemma4-bf16
#SBATCH --partition=camas
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -eo pipefail
G=/data/lab/cejli/sy/tmp_checks/gemma4
C1=$G/compare_out
C=$G/compare_out_bf16
CK=kamiak/checks/gemma4_compare
M=/data/lab/cejli/sy/models/gemma-4-31B-it
[ ! -e $C ] || { echo "$C exists: stop"; exit 1; }
mkdir -p $C
cp $C1/requests.json $C1/vllm_server_*.json $C/
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "node $(hostname); repo at $(git log -1 --format='%h %ad' --date=iso); uncommitted in code/ and kamiak/: $(git status --short code kamiak | wc -l)"
source /opt/apps/miniconda3/3.12/etc/profile.d/conda.sh
conda activate /data/lab/cejli/sy/conda_envs/vllm_cu129
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "=== 1 official code, pixel values cast to bfloat16, greedy, both GPUs $(date +%T)"
python -I $CK/official_greedy.py $M $C/requests.json $C --extra-path $G/pylib --cast-pixels-bf16 2>&1 \
    | grep -v "Loading weights" | tee $C/official.log

echo "=== 2 vLLM offline scoring $(date +%T)"
VLLM_ALLOW_INSECURE_SERIALIZATION=1 python -I $CK/vllm_offline_score.py $M $C/requests.json $C $C:new \
    > $C/vllm_offline.log 2>&1 || { echo "offline scoring failed, see $C/vllm_offline.log"; exit 1; }
grep -E "^new |^hooks" $C/vllm_offline.log || true

echo "=== 3 differing positions in steps of 0.125 $(date +%T)"
python -I $CK/diag_b_steps.py $C | tee $C/diag_b_steps.out

echo "=== 4 report $(date +%T)"
python -I $CK/report.py $C/requests.json $C | tee $C/report.txt
echo "end $(date +%T)"
