"""EXPLORATORY (user, 2026-11-15; plan in CLAUDE.md "理由分析" -> "LLM 标注理由", recorded as given before this program
was written): LLM annotation of Qwen3.6-27B's 5,420 stated reasons (sep_camera, main set) as PHYSICAL / ENVIRONMENT.
Step 1 here: the validation sample for the user (no model needed).
  sample <round-1 work folder> <to_office folder> <key out_dir>: 200 reasons drawn at random without replacement,
    numpy default_rng(20261115).choice(5420, 200, replace=False) over round 1's reasons.jsonl rows (the answers_base.csv
    order; reasons sha256 c718132a... checked), items 1..200 in draw order. Writes <to_office>/reason_labels_sample.csv
    (item, reason, physical, environment; the last two empty; no score, no theme, no other label) and
    <key out_dir>/exploratory_llm_validation_key.csv (item, row, segment, frame_idx, repeat) to match the user's labels
    later (kept out of the download folder).
usage: python -I llm_labels.py sample <round-1 work folder> <to_office folder> <key out_dir>
"""
import csv
import hashlib
import json
import os
import sys

import numpy as np

REASONS_SHA = 'c718132acf962585cb1463f0fc1736b77d984b598808498f2bd8e5a3c7345373'
SEED, N_SAMPLE = 20261115, 200


def read_reasons(W1):
    recs = [json.loads(x) for x in open(os.path.join(W1, 'reasons.jsonl'), encoding='utf-8')]
    h = hashlib.sha256()
    for r in recs:
        h.update(r['reason'].encode('utf-8') + b'\n')
    if h.hexdigest() != REASONS_SHA or [r['row'] for r in recs] != list(range(len(recs))) or len(recs) != 5420:
        sys.exit(f'reasons.jsonl is not round 1\'s (sha256 {h.hexdigest()}, {len(recs)} rows)')
    return recs


def sample(W1, T, K):
    recs = read_reasons(W1)
    os.makedirs(T, exist_ok=True)
    os.makedirs(K, exist_ok=True)
    p_s, p_k = os.path.join(T, 'reason_labels_sample.csv'), os.path.join(K, 'exploratory_llm_validation_key.csv')
    for p in (p_s, p_k):
        if os.path.exists(p):
            sys.exit(f'{p} exists: stopped (not overwritten)')
    idx = np.random.default_rng(SEED).choice(len(recs), N_SAMPLE, replace=False)
    with open(p_s, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['item', 'reason', 'physical', 'environment'])
        for i, j in enumerate(idx, 1):
            w.writerow([i, recs[j]['reason'], '', ''])
    with open(p_k, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['item', 'row', 'segment', 'frame_idx', 'repeat'])
        for i, j in enumerate(idx, 1):
            r = recs[j]
            w.writerow([i, int(j), r['segment'], r['frame_idx'], r['repeat']])
    print(f'EXPLORATORY validation sample: {N_SAMPLE} of {len(recs)} reasons, default_rng({SEED}); {p_s} '
          f'({os.path.getsize(p_s)} bytes, {len(set(idx.tolist()))} distinct rows, {len({recs[j]["segment"] for j in idx})} '
          f'segments); key {p_k}')


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == 'sample':
        sample(*sys.argv[2:5])
    else:
        sys.exit(__doc__)
