"""How smooth are Waymo's labelled accelerations of other vehicles, compared with our ego acceleration at
different smoothing windows? Same criteria as before (Punzo et al. 2011): share of 1 s windows with more than
one jerk sign flip; also the spread of the acceleration.
usage: python -I label_accel_smoothness.py <code_dir> <extracted_root>"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

code_dir, ex_root = sys.argv[1:3]
sys.path.insert(0, code_dir)
import wod_common as wc   # noqa: E402

DT = 0.1


def flip_share(a):
    j = np.diff(a) / DT
    s = np.sign(j)
    inv = (s[1:] * s[:-1] < 0).astype(int)
    per_s = np.convolve(inv, np.ones(10), mode='valid')
    return (per_s > 1).sum(), len(per_s)


segs = sorted(d for d in os.listdir(ex_root) if os.path.isdir(os.path.join(ex_root, d)))
lab_f = lab_n = 0
lab_a = []
n_tracks = 0
for name in segs:
    o = pd.read_parquet(os.path.join(ex_root, name, 'objects.parquet'))
    v = o[o['type'] == 'vehicle'].sort_values('frame_idx')
    for oid, g in v.groupby('id'):
        frames = g.frame_idx.to_numpy()
        if len(g) < 30 or np.any(np.diff(frames) != 1):
            continue
        vel = g[['speed_x', 'speed_y']].to_numpy(float)
        acc = g[['accel_x', 'accel_y']].to_numpy(float)
        sp = np.hypot(*vel.T)
        if sp.min() < 1.0:                     # moving the whole time, so the travel direction is defined
            continue
        a_t = (acc * vel).sum(1) / sp           # rate of change of speed
        f, n = flip_share(a_t)
        lab_f += f; lab_n += n; lab_a.append(a_t); n_tracks += 1
print(f'Waymo labels, {n_tracks} moving vehicle tracks (>= 3 s, speed > 1 m/s): '
      f'sign-flip > 1 in 1 s: {lab_f / lab_n:.1%}, sd(a) {np.std(np.concatenate(lab_a)):.3f} m/s2')

for win in (11, 21, 31, 41):
    f_all = n_all = 0
    for name in segs:
        seg = wc.Segment(ex_root, name)
        a_vec = savgol_filter(seg.ego_v, win, 2, deriv=1, delta=DT, axis=0)
        a = np.array([wc.speed_and_accel(seg.ego_v[k], a_vec[k], seg.yaw[k])[1] for k in range(len(seg.ego_v))])
        f, n = flip_share(a)
        f_all += f; n_all += n
    print(f'ego, SG window {win * DT:.1f} s: sign-flip > 1 in 1 s: {f_all / n_all:.1%}')
