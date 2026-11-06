"""new_cues_e.py with --max_length against without (user, 2026-11-06): the csv with the option must have exactly the
columns of the one without plus max_vehicle_length_30m (last), every other cell the same as text; printed: the new
column's distribution and the frames with a vehicle of 7.0 m or more within 30 m (described only).
usage: python -I compare_e2.py <new_cues_e.csv> <new_cues_e2.csv> <risk analysis main_1084 folder> <D's new cues csv>"""
import os
import sys

import numpy as np
import pandas as pd

A, B, RISK, DC = sys.argv[1:5]
a = pd.read_csv(A, dtype=str, keep_default_na=False)
b = pd.read_csv(B, dtype=str, keep_default_na=False)
col = 'max_vehicle_length_30m'
ok = list(b.columns) == list(a.columns) + [col] and b.drop(columns=[col]).equals(a)
print(f'{"ok" if ok else "FAIL"}: the other {len(a.columns)} columns of {len(a)} rows the same as text; new column last')
v = b[col].astype(float)
main = pd.read_csv(os.path.join(RISK, 'ssm_target_frames.csv'))[['segment', 'frame_idx']]
d = pd.read_csv(DC)
NEW_D = ['speed_limit_mps', 'dist_crosswalk', 'dist_stop_sign', 'dist_lane_crossing']
fit = main.merge(d[~d[NEW_D].isna().any(axis=1)][['segment', 'frame_idx']], on=['segment', 'frame_idx'])
bb = b.assign(**{col: v, 'frame_idx': b.frame_idx.astype(int)})
for name, fr in (('all 1,873', bb[['segment', 'frame_idx']]), ('fit 1,065', fit)):
    x = fr.merge(bb, on=['segment', 'frame_idx'])[col].to_numpy(float)
    q = np.quantile(x, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    print(f'  {name}: {len(x)} frames; min / 10% / q1 / median / q3 / 90% / max ' + ' / '.join(f'{z:.3f}' for z in q) +
          f'; no vehicle within 30 m (0): {int((x == 0).sum())}; a vehicle of 7.0 m or more: {int((x >= 7.0).sum())}')
sys.exit(0 if ok else 1)
