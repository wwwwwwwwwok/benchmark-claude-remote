"""The risk analysis with the weather grouping added (kamiak/main_run/analyze_weather.sh; user, 2026-11-03) against the
one without it (analysis/after_qwen3.5-4b/), for main_1084/ and sensitivity_1123/:
 - every CSV of the old run: every line in the new one, character for character and in the same order (the old lines a
   subsequence of the new), and every line the new one has more has 'weather' in its grouping column. Where the new
   table has an extra column 'weather' (the program writes each grouping's labels into ssm_target_frames.csv and
   agreement_by_segment.csv), that column is dropped first and the rest must then be the same, cell for cell, with no
   line more; reported as such.
 - the files of both runs;
 - the resampled segments: their sha256 (numpy default_rng(seed), one rng.choice over the sorted segments per
   resample, as analyze_vlm_ssm.py draws them) from each run's params.txt (seed, number, segment list), the two the
   same, and the same as the one the quantities analysis records for these frames (when given);
 - params.txt and the run logs: the lines that differ, to read by eye.
usage: python -I compare_weather_grouping.py <old analysis folder> <new analysis folder> [<quantities analysis folder>]
"""
import csv
import difflib
import hashlib
import io
import os
import re
import sys

import numpy as np
import pandas as pd

old, new = sys.argv[1:3]
quant = sys.argv[3] if len(sys.argv) > 3 else None
bad = 0


def differing(a_path, b_path):
    a, b = open(a_path).read().splitlines(), open(b_path).read().splitlines()
    return [l for l in difflib.unified_diff(a, b, lineterm='', n=0) if l[:1] in '+-' and l[:3] not in ('+++', '---')]


def draws_sha256(params):
    txt = open(params, encoding='utf-8').read()
    m = re.search(r'^bootstrap: (\d+) resamples \(segments resampled\), seed (\d+)', txt, re.M)
    segs = []
    for line in txt.split('scene labels per segment:\n', 1)[1].splitlines():
        mm = re.match(r'^  (\S+): ', line)
        if not mm:
            break
        segs.append(mm.group(1))
    seg_list = np.array(sorted(segs))
    rng = np.random.default_rng(int(m.group(2)))
    h = hashlib.sha256()
    for _ in range(int(m.group(1))):
        h.update(np.searchsorted(seg_list, rng.choice(seg_list, len(seg_list))).astype(np.int32).tobytes())
    return h.hexdigest(), int(m.group(1)), int(m.group(2)), len(seg_list), list(seg_list) == segs


for sub in ('main_1084', 'sensitivity_1123'):
    a_names = sorted(os.listdir(os.path.join(old, sub)))
    b_names = sorted(os.listdir(os.path.join(new, sub)))
    only_old = [x for x in a_names if x not in b_names]
    only_new = [x for x in b_names if x not in a_names]
    bad += len(only_old) + len(only_new)
    print(f'== {sub}: {len(a_names)} files old, {len(b_names)} new; only in the old run {only_old}, only in the new '
          f'{only_new}')
    for name in a_names:
        if name in only_old or not name.endswith('.csv'):
            continue
        pa, pb = os.path.join(old, sub, name), os.path.join(new, sub, name)
        a, b = open(pa).read().splitlines(), open(pb).read().splitlines()
        if a[0] == b[0]:
            j, more = 0, []
            for line in b:
                if j < len(a) and line == a[j]:
                    j += 1
                else:
                    more.append(line)
            gcol = next(csv.reader([a[0]])).index('grouping') if 'grouping' in next(csv.reader([a[0]])) else None
            weather = [next(csv.reader(io.StringIO(line)))[gcol] == 'weather' for line in more] if gcol is not None \
                else [False] * len(more)
            ok = j == len(a) and all(weather)
            bad += not ok
            print(f'  {name}: old {len(a)} lines, all in the new in order: {j == len(a)} ({j} found); new {len(b)}, '
                  f'{len(more)} more, all with grouping weather: {all(weather)}')
            if not ok:
                if j < len(a):
                    print(f'    first old line not found in order: {a[j][:300]}')
                for line, w in zip(more, weather):
                    if not w:
                        print(f'    a line more that is not grouping weather: {line[:300]}')
                        break
        else:
            da = pd.read_csv(pa, dtype=str, keep_default_na=False)
            db = pd.read_csv(pb, dtype=str, keep_default_na=False)
            extra = [c for c in db.columns if c not in da.columns]
            ok = extra == ['weather'] and list(db.drop(columns=extra).columns) == list(da.columns) and \
                db.drop(columns=extra).equals(da)
            bad += not ok
            print(f'  {name}: the new table has the column(s) {extra} more; without it the same as the old one, cell for '
                  f'cell ({len(da)} rows): {ok}' + (f'; weather values {sorted(db.weather.unique())}' if 'weather' in
                                                     db.columns else ''))
    # params.txt: the per-segment label lines that only gain the weather label are counted (and checked: the same
    # segment, the same time_of_day); every other differing line is printed
    pa_, pb_ = (open(os.path.join(x, sub, 'params.txt')).read().splitlines() for x in (old, new))
    lab_a = {m.group(1): m.group(2) for m in (re.match(r'^  (\S+): (time_of_day=.*)$', s) for s in pa_) if m}
    lab_b = {m.group(1): (m.group(2), m.group(3)) for m in
             (re.match(r'^  (\S+): weather=([^,]+), (time_of_day=.*)$', s) for s in pb_) if m}
    seg_ok = set(lab_a) == set(lab_b) and all(lab_b[s][1] == lab_a[s] for s in lab_a)
    bad += not seg_ok
    print(f'  params.txt, scene labels per segment: {len(lab_a)} old, {len(lab_b)} new; every segment the same with the '
          f'same time_of_day, the weather label added: {seg_ok}; weather {pd.Series([v[0] for v in lab_b.values()]).value_counts().to_dict()}')
    seg_lines = {f'  {s}: {v}' for s, v in lab_a.items()} | {f'  {s}: weather={v[0]}, {v[1]}' for s, v in lab_b.items()}
    d_ = [x for x in difflib.unified_diff(pa_, pb_, lineterm='', n=0) if x[:1] in '+-' and x[:3] not in ('+++', '---')
          and x[1:] not in seg_lines]
    print(f'  params.txt: {len(d_)} other differing lines (old -, new +)')
    for line in d_:
        print(f'    {line[:300]}')
    ha, hb = draws_sha256(os.path.join(old, sub, 'params.txt')), draws_sha256(os.path.join(new, sub, 'params.txt'))
    print(f'  resampled segments: old sha256 {ha[0]} ({ha[1]} resamples, seed {ha[2]}, {ha[3]} segments, list sorted '
          f'{ha[4]}); new {hb[0]} ({hb[1]}, seed {hb[2]}, {hb[3]}, {hb[4]}); the same: {ha[0] == hb[0]}')
    bad += ha[0] != hb[0]
    if quant:
        m = re.search(r'drawn segment indices ([0-9a-f]{64})', open(os.path.join(quant, sub, 'params.txt')).read())
        print(f'  the quantities analysis records {m.group(1) if m else None}: the same: {bool(m) and m.group(1) == ha[0]}')
        bad += not (m and m.group(1) == ha[0])
for name in ('main_1084.log', 'sensitivity_1123.log'):
    d_ = differing(os.path.join(old, name), os.path.join(new, name))
    print(f'== {name}: {len(d_)} differing lines (old -, new +; bootstrap progress lines not shown)')
    for line in d_[:60]:
        if 'bootstrap' not in line:
            print(f'    {line[:300]}')
print(f'\ncomparisons not met, or files in one run only: {bad}')
