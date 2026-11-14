#!/bin/bash
# InternVL3.5-8B (user, 2026-11-13), before its GPU runs, CPU only, nothing sent: the requests code/run_vlm.py would send
# for internvl3.5-8b against internvl3.5-38b and internvl3.5-8b-thinking against internvl3.5-38b-thinking, request by
# request (compare_sampling_only.py, as is: a stand-in for the OpenAI client, image data URLs as sha256), for the pilot
# list (102 frames, 2,550 requests) and the main list (1,123 frames, 28,075); the arguments are those
# kamiak/qwen_vlm.sbatch gives run_vlm.py for each case (its case block evaluated). Only the model name may differ; the
# servers' arguments (SERVE_ARGS) must be the same and the model folders differ (38B -> 8B).
# usage (repository root, CPU node): bash kamiak/checks/internvl8b/compare_requests.sh <work folder>
set -euo pipefail
W=$1
PY=/data/lab/cejli/sy/conda_envs/vllm_cu129/bin/python
V=/data/lab/cejli/sy/waymo_v143/vlm_inputs
mkdir -p "$W"
echo "node $(hostname), $(date '+%F %T'); repo $(git log -1 --format='%h'); qwen_vlm.sbatch sha256 $(sha256sum kamiak/qwen_vlm.sbatch | cut -c1-16); run_vlm.py sha256 $(sha256sum code/run_vlm.py | cut -c1-16)"
case_of() { MODEL=$1; eval "$(sed -n '/^case "\$MODEL" in/,/^esac/p' kamiak/qwen_vlm.sbatch)"; }
RC=0
for pair in "internvl3.5-38b internvl3.5-8b" "internvl3.5-38b-thinking internvl3.5-8b-thinking"; do
    set -- $pair; MA=$1; MB=$2
    case_of $MA; SA=("${SAMPLING[@]}"); VA=("${SERVE_ARGS[@]}"); DA=$MODEL_DIR
    case_of $MB; SB=("${SAMPLING[@]}"); VB=("${SERVE_ARGS[@]}"); DB=$MODEL_DIR
    echo "##### $MA vs $MB"
    echo "model folder: A $DA; B $DB"
    echo "SERVE_ARGS A: ${VA[*]}"
    echo "SERVE_ARGS B: ${VB[*]}"
    [ "${VA[*]}" = "${VB[*]}" ] && echo "SERVE_ARGS same: yes" || { echo "SERVE_ARGS same: NO"; RC=1; }
    [ "${SA[*]}" = "${SB[*]}" ] && echo "SAMPLING ${SA[*]}; same: yes" || { echo "SAMPLING A ${SA[*]}; B ${SB[*]}; same: NO"; RC=1; }
    for set in pilot main; do
        if [ $set = pilot ]; then G=results/vlm_inputs/targets.csv; S=results/vlm_inputs_separate/targets.csv
        else G=$V/grid/targets_main.csv; S=$V/separate/targets_main.csv; fi
        COMMON=(--grid_targets $G --sep_targets $S --conditions grid,sep_time,sep_camera,ctrl_grid,ctrl_sep --repeats 5
                --base_url http://localhost:8000/v1)
        echo "=== $set ($(date +%T))"
        mkdir -p "$W/$MB"
        $PY -I kamiak/checks/compare_sampling_only.py code/run_vlm.py "$W/$MB" $set \
            "${COMMON[@]}" --model $MA "${SA[@]}" --top_logprobs 5 \
            --vs "${COMMON[@]}" --model $MB "${SB[@]}" --top_logprobs 5 || RC=1
    done
done
echo "end $(date '+%F %T'); exit $RC"
exit $RC
