"""Diagnosis of (b) in steps of 0.125 (user, 2026-11-12): at every position where vLLM's top-1 differs from the official
token, the official top-2 log-probability difference ln(p1 / p2) and vLLM's ln(q1 / q2) (= the logit differences) in
steps of 0.125 (the bfloat16 spacing of logits between 16 and 32), whether the official top-2 is an exact tie, whether
the official token is vLLM's 2nd; and the number of exact official ties over all scored positions.
usage: python -I diag_b_steps.py <out folder with official_<c>.json and vllm_scored_new_<c>.json>
"""
import collections
import json
import math
import os
import sys

C = sys.argv[1]
CONDS = ['grid', 'sep_time', 'sep_camera', 'ctrl_grid', 'ctrl_sep']
rows, ties_all, n_all = [], 0, 0
for c in CONDS:
    off = json.load(open(os.path.join(C, f'official_{c}.json')))
    sco = json.load(open(os.path.join(C, f'vllm_scored_new_{c}.json')))
    for j, (g, row) in enumerate(zip(off['generated'], sco['top5'])):
        p1, p2 = off['top2'][j][1]
        n_all += 1
        ties_all += p1 == p2
        if row[0] and row[0][0] == g:
            continue
        o_ln = math.log(p1 / p2) if p2 > 0 else float('inf')
        v_ln = row[1][0] - row[1][1] if len(row[1]) > 1 else float('nan')
        rows.append(dict(cond=c, pos=j, gap=p1 - p2, o_steps=o_ln / 0.125, v_steps=v_ln / 0.125, tie=p1 == p2,
                         second=len(row[0]) > 1 and row[0][1] == g))
print(f'{len(rows)} differing positions of {n_all}; exact official ties among them {sum(r["tie"] for r in rows)}, over all '
      f'scored positions {ties_all}; official token = vLLM\'s 2nd at {sum(r["second"] for r in rows)}; official p1 - p2 '
      f'>= 0.1 at {sum(r["gap"] >= 0.1 for r in rows)}')
whole = sum(abs(r['o_steps'] - round(r['o_steps'])) < 1e-3 for r in rows)
whole_v = sum(abs(r['v_steps'] - round(r['v_steps'])) < 1e-3 for r in rows)
print(f'whole numbers of 0.125 steps: official {whole} of {len(rows)}, vLLM {whole_v} of {len(rows)}')
print('official steps (count): ' + str(dict(sorted(collections.Counter(round(r['o_steps'], 3) for r in rows).items()))))
print('vLLM steps (count):     ' + str(dict(sorted(collections.Counter(round(r['v_steps'], 3) for r in rows).items()))))
print('condition, position, official p1 - p2, official steps, vLLM steps, official tie, official = vLLM 2nd:')
for r in rows:
    print(f"  {r['cond']:10s} {r['pos']:4d}  {r['gap']:.4f}  {r['o_steps']:.3f}  {r['v_steps']:.3f}  {r['tie']}  {r['second']}")
