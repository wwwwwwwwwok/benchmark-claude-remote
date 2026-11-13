#!/bin/bash
# Reference for the Gemma 4 check (user, 2026-11-12): the official transformers code against itself, sdpa vs eager
# (official_two_attn.py), on the diagnosis's 5 official greedy sequences with bfloat16 pixel values. Writes
# tmp_checks/gemma4/two_attn/. Submit from the repository root: sbatch kamiak/checks/gemma4_compare/gemma4_two_attn.sbatch
#SBATCH --job-name=gemma4-attn
#SBATCH --partition=camas
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -eo pipefail
G=/data/lab/cejli/sy/tmp_checks/gemma4
C=$G/two_attn
[ ! -e $C ] || { echo "$C exists: stop"; exit 1; }
mkdir -p $C
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "node $(hostname); repo at $(git log -1 --format='%h %ad' --date=iso); uncommitted in code/ and kamiak/: $(git status --short code kamiak | wc -l)"
source /opt/apps/miniconda3/3.12/etc/profile.d/conda.sh
conda activate /data/lab/cejli/sy/conda_envs/vllm_cu129
python -I kamiak/checks/gemma4_compare/official_two_attn.py /data/lab/cejli/sy/models/gemma-4-31B-it \
    $G/compare_out_bf16/requests.json $G/compare_out_bf16 $C --extra-path $G/pylib 2>&1 \
    | grep -v "Loading weights" | tee $C/two_attn.out
echo "end $(date +%T)"
