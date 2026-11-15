"""InternVL3.5-8B thinking vs InternVL3.5-38B thinking (and 8B non-thinking): reply length on the same items (user,
2026-11-14). The items are the (segment, frame_idx, condition, repeat) present in every file given (the main runs: the
same 28,075 requests); the main replies only (no forced answers). Per run and per condition: generated tokens
(usage.completion_tokens) and reasoning characters (len of the `reasoning` field; None counts as 0): median and 90th
percentile (numpy, linear), mean, max; also the items cut at max_tokens.
usage: python -I reasoning_length.py <name>=<jsonl[.gz]> [<name>=<jsonl[.gz]> ...]
"""
import gzip
import json
import sys

import numpy as np

COND = ['grid', 'sep_time', 'sep_camera', 'ctrl_grid', 'ctrl_sep']
runs = {}
for a in sys.argv[1:]:
    name, path = a.split('=', 1)
    op = gzip.open if path.endswith('.gz') else open
    d = {}
    with op(path, 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            k = (r['segment'], int(r['frame_idx']), r['condition'], int(r['repeat']))
            d[k] = ((r.get('usage') or {}).get('completion_tokens'), len(r.get('reasoning') or ''), r.get('finish_reason'))
    runs[name] = d
    print(f'{name}: {path}: {len(d)} items')
common = sorted(set.intersection(*[set(d) for d in runs.values()]))
print(f'items in every file: {len(common)}\n')
print(f'{"run":28s} {"condition":10s} {"n":>6s} | {"tokens med":>10s} {"p90":>8s} {"mean":>9s} {"max":>6s} | '
      f'{"reasoning chars med":>19s} {"p90":>8s} {"mean":>9s} {"max":>7s} | cut')
for name, d in runs.items():
    for c in COND + ['all']:
        ks = [k for k in common if c == 'all' or k[2] == c]
        t = np.array([d[k][0] for k in ks], float)
        r = np.array([d[k][1] for k in ks], float)
        cut = sum(d[k][2] == 'length' for k in ks)
        print(f'{name:28s} {c:10s} {len(ks):6d} | {np.median(t):10.1f} {np.percentile(t, 90):8.1f} {t.mean():9.2f} '
              f'{t.max():6.0f} | {np.median(r):19.1f} {np.percentile(r, 90):8.1f} {r.mean():9.2f} {r.max():7.0f} | {cut}')
    print()
