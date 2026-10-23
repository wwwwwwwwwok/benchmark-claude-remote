"""Answers whose main_hazard_camera is not one of the allowed values, per condition and written form, in a run file;
also how many fall on the 39 frames left out of the main result (closing_below_0_1 in the main list). Counted apart,
one line each (user, 2026-10-23): replies without an answer (parsed empty) and answers without a camera
(main_hazard_camera missing or null); the total counts only the written forms.
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
apart, apart_slow = collections.Counter(), collections.Counter()      # (kind, condition)
with (gzip.open if sys.argv[1].endswith('.gz') else open)(sys.argv[1], 'rb') as f:
    for raw in f:
        i = raw.find(b', "logprobs": ')
        if i >= 0:
            j = raw.find(b', "latency_s": ', i)
            raw = raw[:i] + (raw[j:] if j > 0 else b'}')
        d = json.loads(raw)
        n[d['condition']] += 1
        p = d.get('parsed')
        if not p or p.get('main_hazard_camera') is None:
            k = ('no answer (parsed empty)' if not p else 'answer without a camera (missing or null)', d['condition'])
            apart[k] += 1
            apart_slow[k] += (d['segment'], d['frame_idx']) in slow
            continue
        cam = p['main_hazard_camera']
        if cam not in ALLOWED:
            cnt[(d['condition'], cam)] += 1
            if (d['segment'], d['frame_idx']) in slow:
                cnt_slow[(d['condition'], cam)] += 1
print(f'lines per condition: {dict(n)}; slow frames in the list: {len(slow)}')
# sorted as text (user, 2026-10-23): sorted() stops with a TypeError when a camera is not a string
for (c, cam), k in sorted(cnt.items(), key=lambda x: (x[0][0], str(x[0][1]))):
    print(f'  {c:10s} {cam!r}: {k} (on the 39 left-out frames: {cnt_slow[(c, cam)]})')
print(f'  total {sum(cnt.values())} (written forms only)')
for kind in ('no answer (parsed empty)', 'answer without a camera (missing or null)'):
    per = {c: k for (t, c), k in sorted(apart.items()) if t == kind}
    print(f'  {kind}: {sum(per.values())} {per} (on the 39 left-out frames: '
          f'{sum(k for (t, c), k in apart_slow.items() if t == kind)})')
