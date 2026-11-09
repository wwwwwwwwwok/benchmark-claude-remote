"""Before the acceleration experiment (user, 2026-11-09): every request (1,123 frames x 5 conditions) built by
run_vlm_accel.py equals run_vlm.py's own request part by part, except the last text = the sentence + newline + the
original question; force_answer.py's build replaced too. Also the acceleration texts' distribution.
usage: python -I check_accel.py <repository code folder> <truth_main.csv> <vlm_inputs folder>"""
import os
import sys

import pandas as pd

CODE, TRUTH, V = sys.argv[1:4]
sys.path.insert(0, CODE)
import force_answer as fa  # noqa: E402
import run_vlm as rv  # noqa: E402
import run_vlm_accel as ra  # noqa: E402

t = pd.read_csv(TRUTH, float_precision='round_trip')
acc = {(s, int(k)): ra.accel_text(v) for s, k, v in zip(t.segment, t.frame_idx, t.a_ego)}
orig = rv.build
new = ra.make_build(acc, orig)
tg = pd.read_csv(os.path.join(V, 'grid', 'targets_main.csv'))
ts = pd.read_csv(os.path.join(V, 'separate', 'targets_main.csv'))
x = tg.merge(ts[['segment', 'frame_idx', 'images_by_time']], on=['segment', 'frame_idx'], validate='one_to_one')
bad = n = 0
for r in x.itertuples():
    grid = [os.path.join(V, 'grid', p) for p in r.images.split(';')]
    bt = [os.path.join(V, 'separate', p) for p in r.images_by_time.split(';')]
    sep = {cam: [bt[i * 5 + c] for i in range(5)] for c, cam in enumerate(rv.CAMERAS)}
    for cond in rv.CONDITIONS:
        a, b = orig(cond, grid, sep), new(cond, grid, sep)
        s = ra.SENTENCE.format(a=acc[(r.segment, int(r.frame_idx))])
        n += 1
        bad += int(b != a[:-1] + [('text', s + '\n' + a[-1][1])])
vals = pd.Series(list(acc.values()))
print(f'{len(acc)} frames; acceleration texts: {vals.nunique()} values, from {min(vals, key=float)} to '
      f'{max(vals, key=float)}; "0.0" {int((vals == "0.0").sum())}; "-0.0" {int((vals == "-0.0").sum())}; negative '
      f'{int(vals.str.startswith("-").sum())}')
print(f'requests built: {n} (x 5 repeats = {n * 5}); different from the original plus the sentence: {bad}')
sys.argv = ['run_vlm_accel.py', TRUTH, 'nothing']
try:
    ra.main()
except SystemExit:
    pass
print(f"force_answer.py's build replaced: {fa.build is not orig}; checks not met: {int(bad > 0 or n != 5615)}")
