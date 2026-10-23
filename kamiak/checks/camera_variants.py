"""Answers whose main_hazard_camera is not one of the allowed values, per condition and written form, in a run file;
also how many fall on the 39 frames left out of the main result (closing_below_0_1 in the main list).
usage: python -I camera_variants.py <run jsonl[.gz]> <main list csv>
"""
import collections
import gzip
import json
import sys

import pandas as pd

ALLOWED = {'front-left', 'front', 'front-right', 'left-side', 'right-side', 'none'}
lst = pd.read_csv(sys.argv[2], comment='#')
slow = set(zip(lst.loc[lst.closing_below_0_1.astype(bool), 'segment'], lst.loc[lst.closing_below_0_1.astype(bool), 'frame_idx']))
cnt, cnt_slow, n = collections.Counter(), collections.Counter(), collections.Counter()
with (gzip.open if sys.argv[1].endswith('.gz') else open)(sys.argv[1], 'rb') as f:
    for raw in f:
        i = raw.find(b', "logprobs": ')
        if i >= 0:
            j = raw.find(b', "latency_s": ', i)
            raw = raw[:i] + (raw[j:] if j > 0 else b'}')
        d = json.loads(raw)
        n[d['condition']] += 1
        cam = (d.get('parsed') or {}).get('main_hazard_camera')
        if cam not in ALLOWED:
            cnt[(d['condition'], cam)] += 1
            if (d['segment'], d['frame_idx']) in slow:
                cnt_slow[(d['condition'], cam)] += 1
print(f'lines per condition: {dict(n)}; slow frames in the list: {len(slow)}')
for (c, cam), k in sorted(cnt.items()):
    print(f'  {c:10s} {cam!r}: {k} (on the 39 left-out frames: {cnt_slow[(c, cam)]})')
print(f'  total {sum(cnt.values())}')
