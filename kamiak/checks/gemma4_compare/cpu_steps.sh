#!/bin/bash
# CPU steps before the Gemma 4 31B-it GPU runs (user, 2026-11-12): image sizes and image tokens with the model's own
# processor (prep_inputs.py), the comparison inputs and requests (build_requests.py from the Qwen-Drive check, as is),
# prompts, image tokens per input and the forced-answer rendering (render_check.py). Stops at the first step with a
# check not met. usage (repository root, CPU node): bash kamiak/checks/gemma4_compare/cpu_steps.sh
set -euo pipefail
G=/data/lab/cejli/sy/tmp_checks/gemma4
M=/data/lab/cejli/sy/models/gemma-4-31B-it
PY=/data/lab/cejli/sy/conda_envs/vllm_cu129/bin/python
C=kamiak/checks/gemma4_compare
echo "node $(hostname), $(date '+%F %T'); repo $(git log -1 --format='%h')"
echo "##### prep_inputs.py"
$PY -I $C/prep_inputs.py $M /data/lab/cejli/sy/waymo_v143/vlm_inputs \
    results/vlm_inputs/targets.csv results/vlm_inputs_separate/targets.csv $G/compare_inputs
echo "##### build_requests.py"
$PY -I kamiak/checks/qwen_drive_compare/build_requests.py $G/compare_inputs/grid/targets_compare.csv \
    $G/compare_inputs/separate/targets_compare.csv $G/requests.json
$PY -I kamiak/checks/qwen_drive_compare/build_requests.py $G/compare_inputs/grid/targets_original.csv \
    $G/compare_inputs/separate/targets_original.csv $G/requests_original.json
echo "##### render_check.py"
$PY -I $C/render_check.py $M $G/requests.json $G/requests_original.json $G
echo "done $(date '+%F %T')"
