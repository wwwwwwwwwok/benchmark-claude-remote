"""Choose the ego-acceleration smoothing window with criteria from the literature:
- Thiemann et al. 2008: variance of the acceleration vs smoothing width; noise disappears fast, real signal slowly.
- Punzo et al. 2011 (as used by Hu et al. 2022 on Waymo): |jerk| > 15 m/s^3 is infeasible; more than one jerk
  sign inversion within 1 s is anomalous.
usage: python -I accel_window_criteria.py <code_dir> <extracted_root>"""
import os
import sys
import numpy as np
from scipy.signal import savgol_filter

code_dir, ex_root = sys.argv[1:3]
sys.path.insert(0, code_dir)
import wod_common as wc   # noqa: E402

segs = sorted(d for d in os.listdir(ex_root) if os.path.isdir(os.path.join(ex_root, d)))
WINDOWS = [None, 5, 7, 9, 11, 15, 21, 31, 41, 61]
print(f'{"window":>14s} {"sd(a) m/s2":>10s} {"|jerk|>15":>10s} {"sign-flip >1 in 1 s":>20s} {"max |a|":>8s}')
for win in WINDOWS:
    sd, big_jerk, flips, amax, n_all, n_win = [], 0, 0, 0.0, 0, 0
    for name in segs:
        seg = wc.Segment(ex_root, name)
        v, t = seg.ego_v, seg.t
        dt = np.median(np.diff(t))
        if win is None:
            a_vec = np.gradient(v, t, axis=0)
        else:
            a_vec = savgol_filter(v, win, 2, deriv=1, delta=dt, axis=0)
        a = np.array([wc.speed_and_accel(v[k], a_vec[k], seg.yaw[k])[1] for k in range(len(v))])
        j = np.diff(a) / dt
        sd.append(a)
        big_jerk += int((np.abs(j) > 15).sum()); n_all += len(j)
        s = np.sign(j)
        inv = (s[1:] * s[:-1] < 0).astype(int)                 # a sign inversion between consecutive jerks
        per_s = np.convolve(inv, np.ones(int(round(1 / dt))), mode='valid')
        flips += int((per_s > 1).sum()); n_win += len(per_s)
        amax = max(amax, np.abs(a).max())
    label = 'raw diff' if win is None else f'{win} fr ({win * 0.1:.1f} s)'
    print(f'{label:>14s} {np.std(np.concatenate(sd)):10.3f} {big_jerk / n_all:10.1%} {flips / n_win:20.1%} {amax:8.2f}')
