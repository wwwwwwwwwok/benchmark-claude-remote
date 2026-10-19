"""Which smoothing window reproduces Waymo's labelled accelerations?
For moving vehicle tracks, derive the acceleration ourselves with a Savitzky-Golay filter of different
windows (a) from the labelled velocities (first derivative) and (b) from the box positions in the global
frame (second derivative), and compare with the labelled acceleration (component along the direction of travel).
usage: python -I label_accel_match.py <code_dir> <extracted_root>"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

code_dir, ex_root = sys.argv[1:3]
sys.path.insert(0, code_dir)
import wod_common as wc   # noqa: E402

DT = 0.1
WINDOWS = [5, 7, 9, 11, 15, 21, 31, 41]
EDGE = 20            # frames dropped at each end so every window is compared on the same, edge-free frames
MIN_LEN = 60

segs = sorted(d for d in os.listdir(ex_root) if os.path.isdir(os.path.join(ex_root, d)))
rows = {('speed', w): [] for w in WINDOWS} | {('position', w): [] for w in WINDOWS}
ref, n_tracks = [], 0
for name in segs:
    seg = wc.Segment(ex_root, name)
    o = seg.objects[seg.objects['type'] == 'vehicle']
    # global (map-frame) box centres for every object row
    R = seg.R[o.frame_idx.to_numpy()]
    t = seg.pose[o.frame_idx.to_numpy(), :3, 3] + seg.offset[o.frame_idx.to_numpy()]
    c = o[['cx', 'cy', 'cz']].to_numpy(float)
    gxy = (np.einsum('nij,nj->ni', R, c) + t)[:, :2]
    o = o.assign(gx=gxy[:, 0], gy=gxy[:, 1]).sort_values('frame_idx')
    for oid, g in o.groupby('id'):
        f = g.frame_idx.to_numpy()
        if len(g) < MIN_LEN or np.any(np.diff(f) != 1):
            continue
        v = g[['speed_x', 'speed_y']].to_numpy(float)
        sp = np.hypot(*v.T)
        if sp.min() < 1.0:
            continue
        u = v / sp[:, None]                                    # direction of travel
        a_lab = (g[['accel_x', 'accel_y']].to_numpy(float) * u).sum(1)
        p = g[['gx', 'gy']].to_numpy(float)
        keep = slice(EDGE, len(g) - EDGE)
        ref.append(a_lab[keep]); n_tracks += 1
        for w in WINDOWS:
            a_v = savgol_filter(v, w, 2, deriv=1, delta=DT, axis=0)
            a_p = savgol_filter(p, w, 3, deriv=2, delta=DT, axis=0)
            rows[('speed', w)].append((a_v * u).sum(1)[keep])
            rows[('position', w)].append((a_p * u).sum(1)[keep])

ref = np.concatenate(ref)
print(f'{n_tracks} moving vehicle tracks (>= {MIN_LEN} frames, > 1 m/s), {len(ref)} frames compared')
print(f'{"derived from":>12s} {"window":>8s} {"corr r":>7s} {"RMSE":>6s} {"mean |diff|":>11s} {"best lag":>9s}')
for (src, w), parts in rows.items():
    a = np.concatenate(parts)
    r = np.corrcoef(a, ref)[0, 1]
    rmse = np.sqrt(np.mean((a - ref) ** 2))
    mad = np.mean(np.abs(a - ref))
    lags = range(-10, 11)
    best = max(lags, key=lambda L: np.corrcoef(np.roll(a, L)[15:-15], ref[15:-15])[0, 1])
    print(f'{src:>12s} {w * DT:6.1f} s {r:7.3f} {rmse:6.3f} {mad:11.3f} {best * DT:+7.1f} s')
