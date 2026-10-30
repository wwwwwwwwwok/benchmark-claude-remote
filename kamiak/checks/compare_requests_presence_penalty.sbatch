#!/bin/bash
# Before the Qwen3.6-27B quantities runs without thinking (user, 2026-10-29), on CPU:
#  1 run_vlm.py before (run_vlm_old.py = code/run_vlm.py at 98a8fa4, its last change) and after adding
#    --presence_penalty: kamiak/checks/compare_risk_requests.py for every risk model's settings in
#    kamiak/qwen_vlm.sbatch on the pilot and the main targets, plus the InternVL reference smoke test with
#    --system_prefix_file (the 15 groups of 2026-10-28); and (Claude's addition) the quantities requests of the three
#    models run so far, on the pilot targets, as they were sent (--task quantities --top_logprobs 0).
#  2 the 27B requests without thinking against those with thinking (kamiak/checks/compare_sampling_only.py, pilot
#    targets): the same apart from the model name and sampling.
#  3 the final sampling of vLLM's code (kamiak/checks/qwen_drive_compare2/check_defaults.py, vLLM environment) for
#    the run's request, the two greedy controls and the pilot with thinking; check_nothink_runs.py defaults.
#  4 dry run (no --send) of the pilot and the main run without thinking: 12,750 and 140,375 requests.
# Exit 1 if anything is not as expected: the GPU job waits for this one with afterok. Submit from the repository root.
#SBATCH --job-name=nothink-precheck
#SBATCH --partition=kamiak
#SBATCH --account=cejli
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
PY=/data/lab/cejli/sy/conda_envs/waymo/bin/python
VPY=/data/lab/cejli/sy/conda_envs/vllm_cu129/bin/python
T=/data/lab/cejli/sy/tmp_checks/run_vlm_compare2
V=/data/lab/cejli/sy/waymo_v143/vlm_inputs
FAIL=0
cmp <(git show 98a8fa4:code/run_vlm.py) $T/run_vlm_old.py && echo "old = run_vlm.py at 98a8fa4" || FAIL=1
echo "new: code/run_vlm.py at $(git log -1 --format=%h -- code/run_vlm.py), uncommitted changes: $(git status --short code/run_vlm.py | wc -l); sha256 $(sha256sum < code/run_vlm.py | cut -c1-16)"
COND=(--conditions grid,sep_time,sep_camera,ctrl_grid,ctrl_sep --repeats 5)
PILOT_T=(--grid_targets results/vlm_inputs/targets.csv --sep_targets results/vlm_inputs_separate/targets.csv)
MAIN_T=(--grid_targets $V/grid/targets_main.csv --sep_targets $V/separate/targets_main.csv)
PIDS=()
for MODEL in qwen3-vl-32b qwen3-vl-32b-thinking qwen3.6-27b internvl3.5-38b internvl3.5-38b-thinking qwen-drive-1.0-4b qwen3.5-4b; do
    eval "$(sed -n '/^case "\$MODEL" in/,/^esac/p' kamiak/qwen_vlm.sbatch)"
    BASE=(--base_url http://localhost:8000/v1 --model "$MODEL" "${SAMPLING[@]}" --top_logprobs 5)
    $PY -I kamiak/checks/compare_risk_requests.py $T/run_vlm_old.py code/run_vlm.py $T pilot_$MODEL \
        "${PILOT_T[@]}" "${COND[@]}" "${BASE[@]}" > $T/pilot_$MODEL.out 2>&1 & PIDS+=($!)
    $PY -I kamiak/checks/compare_risk_requests.py $T/run_vlm_old.py code/run_vlm.py $T main_$MODEL \
        "${MAIN_T[@]}" "${COND[@]}" "${BASE[@]}" > $T/main_$MODEL.out 2>&1 & PIDS+=($!)
done
MODEL=internvl3.5-38b-thinking
eval "$(sed -n '/^case "\$MODEL" in/,/^esac/p' kamiak/qwen_vlm.sbatch)"
$PY -I kamiak/checks/compare_risk_requests.py $T/run_vlm_old.py code/run_vlm.py $T smoke_prefix_$MODEL \
    "${PILOT_T[@]}" "${COND[@]}" --base_url http://localhost:8000/v1 --model "$MODEL" "${SAMPLING[@]}" --top_logprobs 5 \
    --limit 2 --repeats 1 --system_prefix_file code/internvl_thinking_system_prompt.txt > $T/smoke_prefix_$MODEL.out 2>&1 & PIDS+=($!)
for MODEL in qwen-drive-1.0-4b qwen3.5-4b qwen3.6-27b; do
    eval "$(sed -n '/^case "\$MODEL" in/,/^esac/p' kamiak/qwen_vlm.sbatch)"
    $PY -I kamiak/checks/compare_risk_requests.py $T/run_vlm_old.py code/run_vlm.py $T quantities_pilot_$MODEL \
        "${PILOT_T[@]}" "${COND[@]}" --base_url http://localhost:8000/v1 --model "$MODEL" "${SAMPLING[@]}" \
        --top_logprobs 5 --task quantities --top_logprobs 0 > $T/quantities_pilot_$MODEL.out 2>&1 & PIDS+=($!)
done
# 2: with thinking (A) against without (B), each with its own case of kamiak/qwen_vlm.sbatch
MODEL=qwen3.6-27b
eval "$(sed -n '/^case "\$MODEL" in/,/^esac/p' kamiak/qwen_vlm.sbatch)"
A=(--base_url http://localhost:8000/v1 --model "$MODEL" "${SAMPLING[@]}" --top_logprobs 5 --task quantities --top_logprobs 0)
MODEL=qwen3.6-27b-nothink
eval "$(sed -n '/^case "\$MODEL" in/,/^esac/p' kamiak/qwen_vlm.sbatch)"
B=(--base_url http://localhost:8000/v1 --model "$MODEL" "${SAMPLING[@]}" --top_logprobs 5 --task quantities --top_logprobs 0)
$PY -I kamiak/checks/compare_sampling_only.py code/run_vlm.py $T nothink_vs_think_pilot \
    "${PILOT_T[@]}" "${COND[@]}" "${A[@]}" --vs "${PILOT_T[@]}" "${COND[@]}" "${B[@]}" > $T/nothink_vs_think_pilot.out 2>&1 & PIDS+=($!)
# 3
D=/data/lab/cejli/sy/models/Qwen3.6-27B
for s in '{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 24576}' \
         '{"temperature": 0.0, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 24576}' \
         '{"temperature": 0.0, "top_p": 0.8, "top_k": 20, "presence_penalty": 0.0, "max_tokens": 24576}' \
         '{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 24576}'; do
    $VPY -I kamiak/checks/qwen_drive_compare2/check_defaults.py $D --sent "$s"
done > $T/check_defaults_full.out 2>&1
grep -E '^(request values|/data|  )' $T/check_defaults_full.out > $T/check_defaults.out
cat $T/check_defaults.out
$PY -I kamiak/checks/check_nothink_runs.py defaults $T/check_defaults.out || FAIL=1
# 4
$PY -I code/run_vlm.py "${PILOT_T[@]}" "${COND[@]}" "${B[@]}" --out $T/dry_pilot.jsonl --show 1 > $T/dry_pilot.out 2>&1
$PY -I code/run_vlm.py "${MAIN_T[@]}" "${COND[@]}" "${B[@]}" --out $T/dry_main.jsonl --show 1 > $T/dry_main.out 2>&1
grep -E '^[0-9]+ requests' $T/dry_pilot.out $T/dry_main.out
grep -q '^12750 requests (102 targets x 5 conditions x 5 quantities x 5 repeats)' $T/dry_pilot.out || FAIL=1
grep -q '^140375 requests (1123 targets x 5 conditions x 5 quantities x 5 repeats)' $T/dry_main.out || FAIL=1
[ ! -e $T/dry_pilot.jsonl ] && [ ! -e $T/dry_main.jsonl ] || { echo "the dry run wrote a file"; FAIL=1; }
RC=0
for p in "${PIDS[@]}"; do wait $p || RC=1; done
[ $RC = 0 ] || FAIL=1
for f in $T/pilot_*.out $T/main_*.out $T/smoke_prefix_*.out $T/quantities_pilot_*.out; do
    echo "== $(basename $f)"; cat $f
    grep -q 'identical in order: True; output lines identical (without latency_s): True' $f \
        && grep -q 'dry-run printout identical: True' $f || { echo "   NOT IDENTICAL"; FAIL=1; }
done
echo "== nothink_vs_think_pilot.out"; cat $T/nothink_vs_think_pilot.out
grep -q '^SAME APART FROM SAMPLING: True' $T/nothink_vs_think_pilot.out || FAIL=1
echo "groups compared: $(ls $T/pilot_*.out $T/main_*.out $T/smoke_prefix_*.out $T/quantities_pilot_*.out | wc -l)"
echo "end $(date '+%F %T'), all as expected: $([ $FAIL = 0 ] && echo yes || echo no)"
exit $FAIL
