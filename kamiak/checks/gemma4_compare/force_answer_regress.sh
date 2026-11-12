#!/bin/bash
# Regression test of code/force_answer.py's new --thinking_format option (user, 2026-11-12: without the new option,
# Qwen's forced-answer output must be byte for byte the same as now). The version before the change (<old commit>) and
# the changed one (working copy) each run under fixed_clock.py (time.time fixed, so latency_s is the same) against
# their own mock_chat_server.py, on the Qwen3.6-27B pilot (3 replies cut at max_tokens without an answer) with the
# pilot's sampling. Compared byte for byte: the request bodies the server received, the output file, everything the
# program printed. Then the changed version with --thinking_format gemma4 on the same input, only to show the last
# message of the requests it builds (Qwen's thinking text; not a regression item).
# usage (repository root, CPU node): bash kamiak/checks/gemma4_compare/force_answer_regress.sh <out folder> <old commit>
set -euo pipefail
OUT=$1
OLD=$(git rev-parse --short "$2")
[ ! -e "$OUT" ] || { echo "$OUT exists: stop"; exit 1; }
mkdir -p "$OUT/old"
PY=/data/lab/cejli/sy/conda_envs/vllm_cu129/bin/python
C=kamiak/checks/gemma4_compare
echo "node $(hostname), $(date '+%F %T'); old = $OLD:code/force_answer.py, new = working copy"
git show "$OLD:code/force_answer.py" > "$OUT/old/force_answer.py"
git show "$OLD:code/run_vlm.py" > "$OUT/old/run_vlm.py"
cmp "$OUT/old/run_vlm.py" code/run_vlm.py && echo "run_vlm.py: $OLD = working copy"
P=results/vlm_scores/qwen3.6-27b/pilot_qwen3.6-27b.jsonl
ARGS=(--pilot "$P" --grid_targets results/vlm_inputs/targets.csv --sep_targets results/vlm_inputs_separate/targets.csv
      --model qwen3.6-27b --temperature 1.0 --top_p 0.95 --top_k 20 --max_tokens 1024 --top_logprobs 5 --send)

run() {   # run <name> <script> [extra arguments]
    local name=$1 script=$2
    shift 2
    local port=$((18000 + RANDOM % 1000))
    $PY -I $C/mock_chat_server.py $port "$OUT/received_$name.txt" &
    local srv=$!
    until curl -sf "http://localhost:$port/v1/models" > /dev/null; do sleep 1; done
    $PY -I $C/fixed_clock.py "$script" "${ARGS[@]}" --out "$OUT/forced_$name.jsonl" \
        --base_url "http://localhost:$port/v1" "$@" > "$OUT/stdout_$name.txt" 2>&1
    kill $srv
    wait $srv 2>/dev/null || true
    echo "$name: $(wc -l < "$OUT/received_$name.txt") requests received, $(wc -l < "$OUT/forced_$name.jsonl") lines written"
}
run old "$OUT/old/force_answer.py"
run new code/force_answer.py
run gemma4 code/force_answer.py --thinking_format gemma4

bad=0
for f in received_%s.txt forced_%s.jsonl stdout_%s.txt; do
    a=$(printf "$f" old); b=$(printf "$f" new)
    if cmp -s "$OUT/$a" "$OUT/$b"; then
        echo "SAME      $a = $b ($(stat -c %s "$OUT/$a") bytes, sha256 $(sha256sum < "$OUT/$a" | cut -c1-16)...)"
    else
        echo "DIFFERENT $a / $b"; bad=$((bad + 1))
    fi
done
echo "--- printed by the old version:"; cat "$OUT/stdout_old.txt"
echo "--- gemma4 mode, last message of each request:"
$PY -I - "$OUT/received_gemma4.txt" "$OUT/forced_gemma4.jsonl" <<'EOF'
import json, sys
for line in open(sys.argv[1], 'rb'):
    m = json.loads(line)['messages'][-1]
    r = m.get('reasoning', '')
    print(f"role {m['role']}, keys {sorted(m)}, content {m.get('content')!r}, reasoning {len(r):,} characters, "
          f"ends with {r[-140:]!r}")
for line in open(sys.argv[2]):
    print('early_stopping_text recorded:', repr(json.loads(line)['early_stopping_text']))
    break
EOF
echo "regression items not met: $bad"
exit $bad
