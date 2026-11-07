"""Before the new experiment (user, 2026-11-06): 1) write <run dir>/ego_speed_main.csv (segment, frame_idx, v_ego of
the main list's 1,123 frames, from the quantities truth csv; stop if a speed is missing); 2) build every request of the
run (1,123 frames x 5 conditions) with run_vlm.py's own build and with run_vlm_speed.py's, and check that they are the
same part by part except the last text, which must be the sentence, a newline, then the original question; 3) check
that force_answer.py's build is the replaced one after run_vlm_speed.py sets it up.
usage: python -I check_speed.py <run dir> <truth_main.csv> <vlm_inputs folder>"""
import os
import sys

import pandas as pd

D, TRUTH, V = sys.argv[1:4]
sys.path.insert(0, os.path.join(D, 'code'))
import force_answer as fa  # noqa: E402
import run_vlm as rv  # noqa: E402
import run_vlm_speed as rs  # noqa: E402

out = os.path.join(D, 'ego_speed_main.csv')
tr = pd.read_csv(TRUTH, float_precision='round_trip')[['segment', 'frame_idx', 'v_ego']]
assert not tr.v_ego.isna().any() and len(tr) == 1123, (len(tr), int(tr.v_ego.isna().sum()))
if not os.path.exists(out):
    tr.to_csv(out, index=False)
t = pd.read_csv(out)
speed = {(s, int(k)): rs.speed_text(v) for s, k, v in zip(t.segment, t.frame_idx, t.v_ego)}
orig = rv.build
new = rs.make_build(speed, orig)
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
        want = a[:-1] + [('text', f"Your car's speed is {speed[(r.segment, int(r.frame_idx))]} meters per second.\n"
                          + a[-1][1])]
        n += 1
        bad += int(b != want)
print(f'speed csv: {len(t)} frames, v_ego {t.v_ego.min():.3f} to {t.v_ego.max():.3f} m/s, written as '
      f'{min(speed.values(), key=float)} to {max(speed.values(), key=float)}')
print(f'requests built: {n} (frames x conditions; x 5 repeats = {n * 5}); different from the original plus the '
      f'sentence: {bad}')
sys.argv = ['run_vlm_speed.py', out, 'nothing']
try:
    rs.main()
except SystemExit:
    pass
print(f"force_answer.py's build replaced: {fa.build is not orig and fa.build.__name__ == 'build'}; "
      f"run_vlm.py's: {rv.build is not orig}")
print('checks not met: ' + str(int(bad > 0 or n != 5615)))
