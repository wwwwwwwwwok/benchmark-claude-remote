#!/bin/bash
# Pre-submission checks of the 4 remaining main runs against each model's own pilot (user, 2026-10-22), as done for
# Qwen3-VL-32B-Instruct. Run from the repository root.
# 1. messages: dry runs (nothing sent) of run_vlm.py as it was at each pilot and as it is now, on the main inputs; the
#    25 assembled requests of the first target (5 conditions x 5 repeats, texts and image paths) and the run size;
# 2. what kamiak/qwen_vlm.sbatch sends now (its own case block, evaluated per model) next to the vLLM arguments each
#    pilot's server log recorded;
# 3. the sampling settings recorded in every line of each pilot file;
# 4. files in the model folders and in the vLLM environment changed after the pilot started.
set -euo pipefail
T=/data/lab/cejli/sy/tmp_checks/presubmit
PY=/data/lab/cejli/sy/conda_envs/waymo/bin/python
V=/data/lab/cejli/sy/waymo_v143/vlm_inputs
[ ! -e $T/dry ] || { echo "$T/dry exists: stop"; exit 1; }
mkdir -p $T/dry

echo "=== 1. messages: run_vlm.py at each pilot vs now (dry runs, nothing sent)"
# 3a14955: Qwen3.6-27B's first pilot job; 1c151a3: its rerun of 101 requests (committed after that rerun) and the
# Qwen3-VL-32B-Thinking pilot; 1feef17: both InternVL3.5-38B pilots (thinking off and on); HEAD: now
for v in 3a14955 1c151a3 1feef17 HEAD; do
    git show $v:code/run_vlm.py > $T/dry/run_vlm_$v.py
    $PY -I $T/dry/run_vlm_$v.py --grid_targets $V/grid/targets_main.csv --sep_targets $V/separate/targets_main.csv \
        --out $T/dry/never_written.jsonl --conditions grid,sep_time,sep_camera,ctrl_grid,ctrl_sep --repeats 5 \
        --model m --show 25 > $T/dry/dry_$v.txt
    echo "  $v: sha256 $(sha256sum < $T/dry/dry_$v.txt | cut -c1-16), $(grep -c '^=====' $T/dry/dry_$v.txt) requests" \
         "shown, $(grep -c '^<image' $T/dry/dry_$v.txt) image lines; $(grep 'requests (' $T/dry/dry_$v.txt)"
done
for v in 3a14955 1c151a3 1feef17; do
    if cmp -s $T/dry/dry_$v.txt $T/dry/dry_HEAD.txt; then echo "  $v: same output as HEAD, byte for byte"
    else echo "  $v: DIFFERENT from HEAD:"; diff $T/dry/dry_$v.txt $T/dry/dry_HEAD.txt | head -20 || true; fi
done
grep -E '^  [a-z_]+ *:|total image' $T/dry/dry_HEAD.txt
[ ! -e $T/dry/never_written.jsonl ] && echo "  nothing written (no output file)"

echo; echo "=== 2. vLLM and sampling arguments: kamiak/qwen_vlm.sbatch now vs the pilot's server log"
echo "  last commit of the sbatch: $(git log -1 --format='%h %ad' --date=iso -- kamiak/qwen_vlm.sbatch); of the" \
     "InternVL thinking template: $(git log -1 --format='%h %ad' --date=iso -- kamiak/internvl_thinking_chat_template.jinja);" \
     "uncommitted changes: $(git status --short -- kamiak/qwen_vlm.sbatch kamiak/internvl_thinking_chat_template.jinja code/run_vlm.py | wc -l)"
grep -nE 'FLASHINFER_SAMPLER|tensor-parallel-size|gpu-memory-utilization|--conditions|top_logprobs' kamiak/qwen_vlm.sbatch | sed 's/^/  /'
CASE=$(sed -n '/^case "\$MODEL" in/,/^esac/p' kamiak/qwen_vlm.sbatch)
log() { grep -m1 -o "non-default args: .*" "$1" | sed -E "s/'port': [0-9]+, //"; }
for m in qwen3-vl-32b-thinking:results/vlm_scores/qwen3-vl-32b-thinking/logs/vllm_31326781.log \
         internvl3.5-38b:results/vlm_scores/internvl3.5-38b/logs/vllm_31327948.log \
         internvl3.5-38b-thinking:logs/vllm_31328458.log \
         qwen3.6-27b:results/vlm_scores/qwen3.6-27b/logs/vllm_31307048.log \
         qwen3.6-27b:results/vlm_scores/qwen3.6-27b/logs/vllm_31315513.log; do
    MODEL=${m%%:*} bash -c "$CASE"'
        echo "  $MODEL now:   $MODEL_DIR | serve ${SERVE_ARGS[*]} | sampling ${SAMPLING[*]}"'
    echo "  ${m%%:*} pilot $(basename ${m#*:}): $(log ${m#*:}) $(grep -m1 -oE 'version [0-9]+\.[0-9]+\.[0-9]+' ${m#*:})"
done

echo; echo "=== 3. sampling recorded in each pilot file"
$PY -I $T/check_pilot_settings.py results/vlm_scores/qwen3-vl-32b-thinking/pilot_qwen3-vl-32b-thinking.jsonl \
    results/vlm_scores/internvl3.5-38b/pilot_internvl3.5-38b.jsonl \
    results/vlm_scores/internvl3.5-38b-thinking/pilot_internvl3.5-38b-thinking.jsonl \
    results/vlm_scores/qwen3.6-27b/pilot_qwen3.6-27b.jsonl

echo; echo "=== 4. files changed after the pilot started"
chk() {   # chk <folder> <pilot start> <find option: -L follows links>
    local newer
    newer=$(find $3 "$1" -newermt "$2" -type f -not -path '*/__pycache__/*' 2>/dev/null || true)
    echo "  $1 (since $2): $(printf '%s' "$newer" | grep -c . || true) files changed;" \
         "$(find $3 "$1" -type f 2>/dev/null | wc -l) files, $(find $3 "$1" -type f -printf '%s\n' 2>/dev/null | awk '{s += $1} END {printf "%.0f", s}') bytes"
    printf '%s\n' "$newer" | grep . | head -5 | sed 's/^/    /' || true
}
chk /data/lab/cejli/sy/models/Qwen3-VL-32B-Thinking '2026-10-20 09:02:50' -L
chk /data/lab/cejli/sy/models/InternVL3_5-38B '2026-10-20 12:24:00' -L
chk /data/lab/cejli/sy/models/Qwen3.6-27B '2026-10-19 15:53:12' -L
chk /data/lab/cejli/sy/conda_envs/vllm_cu129 '2026-10-19 15:53:12' -P
ls -la --time-style=full-iso /data/lab/cejli/sy/conda_envs/vllm_cu129/conda-meta/history | sed 's/^/  /'
