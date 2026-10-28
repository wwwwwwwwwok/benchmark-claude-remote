#!/bin/bash
# Risk requests of run_vlm.py before (run_vlm_old.py = code/run_vlm.py at c2a1813, unchanged since 2026-10-20) and
# after adding --task quantities (user, 2026-10-28): kamiak/checks/compare_risk_requests.py for every model's
# settings in kamiak/qwen_vlm.sbatch, on the pilot targets and on the main targets, as the submission script passes
# them; plus the InternVL reference smoke test with --system_prefix_file. Submit from the repository root.
#SBATCH --job-name=compare-requests
#SBATCH --partition=kamiak
#SBATCH --account=cejli
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
PY=/data/lab/cejli/sy/conda_envs/waymo/bin/python
T=/data/lab/cejli/sy/tmp_checks/run_vlm_compare
V=/data/lab/cejli/sy/waymo_v143/vlm_inputs
cmp <(git show c2a1813:code/run_vlm.py) $T/run_vlm_old.py && echo "old = run_vlm.py at c2a1813"
echo "new: code/run_vlm.py, $(git diff --stat code/run_vlm.py | tail -1); sha256 $(sha256sum < code/run_vlm.py | cut -c1-16)"
COND=(--conditions grid,sep_time,sep_camera,ctrl_grid,ctrl_sep --repeats 5)
PIDS=()
for MODEL in qwen3-vl-32b qwen3-vl-32b-thinking qwen3.6-27b internvl3.5-38b internvl3.5-38b-thinking qwen-drive-1.0-4b qwen3.5-4b; do
    eval "$(sed -n '/^case "\$MODEL" in/,/^esac/p' kamiak/qwen_vlm.sbatch)"
    BASE=(--base_url http://localhost:8000/v1 --model "$MODEL" "${SAMPLING[@]}" --top_logprobs 5)
    $PY -I kamiak/checks/compare_risk_requests.py $T/run_vlm_old.py code/run_vlm.py $T pilot_$MODEL \
        --grid_targets results/vlm_inputs/targets.csv --sep_targets results/vlm_inputs_separate/targets.csv \
        "${COND[@]}" "${BASE[@]}" > $T/pilot_$MODEL.out 2>&1 & PIDS+=($!)
    $PY -I kamiak/checks/compare_risk_requests.py $T/run_vlm_old.py code/run_vlm.py $T main_$MODEL \
        --grid_targets $V/grid/targets_main.csv --sep_targets $V/separate/targets_main.csv \
        "${COND[@]}" "${BASE[@]}" > $T/main_$MODEL.out 2>&1 & PIDS+=($!)
done
MODEL=internvl3.5-38b-thinking
eval "$(sed -n '/^case "\$MODEL" in/,/^esac/p' kamiak/qwen_vlm.sbatch)"
$PY -I kamiak/checks/compare_risk_requests.py $T/run_vlm_old.py code/run_vlm.py $T smoke_prefix_$MODEL \
    --grid_targets results/vlm_inputs/targets.csv --sep_targets results/vlm_inputs_separate/targets.csv \
    "${COND[@]}" --base_url http://localhost:8000/v1 --model "$MODEL" "${SAMPLING[@]}" --top_logprobs 5 \
    --limit 2 --repeats 1 --system_prefix_file code/internvl_thinking_system_prompt.txt > $T/smoke_prefix_$MODEL.out 2>&1 & PIDS+=($!)
RC=0
for p in "${PIDS[@]}"; do wait $p || RC=1; done
for f in $T/pilot_*.out $T/main_*.out $T/smoke_prefix_*.out; do echo "== $(basename $f)"; cat $f; done
echo "end $(date +%T), exit codes all 0: $([ $RC = 0 ] && echo yes || echo no)"
