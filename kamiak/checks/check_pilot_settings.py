"""Pre-submission check of the remaining main runs (user, 2026-10-22): the sampling settings each pilot file recorded
(every line) and the number of top log-probabilities per generated token (first line), next to what
kamiak/qwen_vlm.sbatch sends now (printed by check_presubmit.sh from its own case block).
usage: python -I check_pilot_settings.py <pilot jsonl> ...
"""
import collections
import json
import sys

for path in sys.argv[1:]:
    settings, prefix, n = collections.Counter(), collections.Counter(), 0
    n_top = None
    with open(path, 'rb') as f:
        for raw in f:
            n += 1
            if n_top is None:
                lp = json.loads(raw).get('logprobs') or []
                n_top = collections.Counter(len(t['top']) for t in lp)
            i = raw.find(b', "logprobs": ')
            if i >= 0:
                j = raw.find(b', "latency_s": ', i)
                raw = raw[:i] + (raw[j:] if j > 0 else b'}')
            d = json.loads(raw)
            settings[(d.get('model'), d.get('temperature'), d.get('top_p'), d.get('top_k'), d.get('max_tokens'))] += 1
            prefix[d.get('system_prefix_file')] += 1
    print(f'{path}: {n} lines')
    for (m, t, p, k, mt), c in settings.items():
        print(f'  model {m}, temperature {t}, top_p {p}, top_k {k}, max_tokens {mt}: {c} lines')
    print(f'  system_prefix_file: {dict(prefix)}; top log-probabilities per token in line 1: {dict(n_top)}')
