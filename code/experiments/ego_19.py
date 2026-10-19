"""Ego acceleration processed like Waymo's labels (SG 1.9 s on positions) vs on velocity vs current 1.1 s."""
import os, sys
import numpy as np, pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import kendalltau
code_dir, ex_root, csv = sys.argv[1:4]
sys.path.insert(0, code_dir)
import wod_common as wc
import waymo_ssm_frames as wsf
f = pd.read_csv(csv)
variants = {'current: velocity, 1.1 s': [], 'velocity, 1.9 s': [], 'positions (like labels), 1.9 s': []}
flips = {k: [0, 0] for k in variants}
for name in f.segment.unique():
    seg = wc.Segment(ex_root, name)
    v, p = seg.ego_v, seg.ego_map[:, :2]
    cand = {'current: velocity, 1.1 s': savgol_filter(v, 11, 2, deriv=1, delta=0.1, axis=0)[:, :2],
            'velocity, 1.9 s': savgol_filter(v, 19, 2, deriv=1, delta=0.1, axis=0)[:, :2],
            'positions (like labels), 1.9 s': savgol_filter(p, 19, 2, deriv=2, delta=0.1, axis=0)}
    for k, a in cand.items():
        at = np.array([wc.speed_and_accel(v[i], a[i], seg.yaw[i])[1] for i in range(len(v))])
        variants[k].append(at)
        j = np.diff(at) / 0.1; s = np.sign(j); inv = (s[1:] * s[:-1] < 0).astype(int)
        per = np.convolve(inv, np.ones(10), mode='valid'); flips[k][0] += (per > 1).sum(); flips[k][1] += len(per)
for k in variants:
    variants[k] = np.concatenate(variants[k])
d = f[f.lead_map.notna()]
idx = d.index.to_numpy()
def mc(a):
    out = [wsf.lead_ssms(g, vf, vl, af, al) for g, vf, vl, af, al in zip(d.gap, d.v_ego, d.v_lead, a[idx], d.a_lead)]
    return np.array([o['MTTC'] for o in out]), np.array([o['CI'] for o in out])
m0, c0 = mc(variants['current: velocity, 1.1 s'])
ref = variants['positions (like labels), 1.9 s']
for k, a in variants.items():
    m, c = mc(a)
    print(f'{k:32s} | vs labels-style: r {np.corrcoef(a, ref)[0,1]:.3f}, RMSE {np.sqrt(np.mean((a-ref)**2)):.3f} | jerk-flip {flips[k][0]/flips[k][1]:.0%} | '
          f'MTTC tau vs current {kendalltau(np.where(np.isinf(m),1e9,m), np.where(np.isinf(m0),1e9,m0)).statistic:.3f}, CI {kendalltau(c, c0).statistic:.3f}')
