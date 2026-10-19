"""Pin down the window: label accel vs 2nd derivative of positions (poly 2 and 3); label speed vs 1st derivative."""
import os, sys
import numpy as np
from scipy.signal import savgol_filter
code_dir, ex_root = sys.argv[1:3]
sys.path.insert(0, code_dir)
import wod_common as wc
DT, EDGE, MIN_LEN = 0.1, 20, 60
WINDOWS = [15, 17, 19, 21, 23, 25, 27, 31]
segs = sorted(d for d in os.listdir(ex_root) if os.path.isdir(os.path.join(ex_root, d)))
acc = {(po, w): [] for po in (2,) for w in [17, 19, 21]}
spd = {(po, w): [] for po in (1, 2) for w in [15, 17, 19, 21, 23, 25, 31]}
ref_a, ref_v = [], []
for name in segs:
    seg = wc.Segment(ex_root, name)
    o = seg.objects[seg.objects['type'] == 'vehicle']
    fi = o.frame_idx.to_numpy()
    gxy = (np.einsum('nij,nj->ni', seg.R[fi], o[['cx', 'cy', 'cz']].to_numpy(float)) + seg.pose[fi, :3, 3])[:, :2]
    o = o.assign(gx=gxy[:, 0], gy=gxy[:, 1]).sort_values('frame_idx')
    for oid, g in o.groupby('id'):
        f = g.frame_idx.to_numpy()
        if len(g) < MIN_LEN or np.any(np.diff(f) != 1):
            continue
        v = g[['speed_x', 'speed_y']].to_numpy(float); sp = np.hypot(*v.T)
        if sp.min() < 1.0:
            continue
        u = v / sp[:, None]; k = slice(EDGE, len(g) - EDGE)
        p = g[['gx', 'gy']].to_numpy(float)
        ref_a.append((g[['accel_x', 'accel_y']].to_numpy(float) * u).sum(1)[k]); ref_v.append(v[k])
        for (po, w) in acc:
            acc[(po, w)].append((savgol_filter(p, w, po, deriv=2, delta=DT, axis=0) * u).sum(1)[k])
        for (po, w) in spd:
            spd[(po, w)].append(savgol_filter(p, w, po, deriv=1, delta=DT, axis=0)[k])
ra, rv = np.concatenate(ref_a), np.concatenate(ref_v)
print('label ACCELERATION vs 2nd derivative of positions')
for (po, w), parts in acc.items():
    a = np.concatenate(parts)
    print(f'  poly {po}, window {w * DT:.1f} s: r {np.corrcoef(a, ra)[0, 1]:.4f}, RMSE {np.sqrt(np.mean((a - ra) ** 2)):.3f} m/s2')
print('label SPEED vs 1st derivative of positions')
for (po, w), parts in spd.items():
    v = np.concatenate(parts)
    print(f'  poly {po}, window {w * DT:.1f} s: RMSE {np.sqrt(np.mean(np.sum((v - rv) ** 2, 1))):.4f} m/s')
