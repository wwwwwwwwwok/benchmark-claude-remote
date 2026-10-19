"""How much the ego-acceleration smoothing window changes a_ego, MTTC and CI on the demo.
usage: python -I accel_window_check.py <code_dir> <extracted_root> <frame_ssm.csv>"""
import sys
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import kendalltau

code_dir, ex_root, csv = sys.argv[1:4]
sys.path.insert(0, code_dir)
import wod_common as wc          # noqa: E402
import waymo_ssm_frames as wsf   # noqa: E402

f = pd.read_csv(csv)
WINDOWS = {'11 frames (1.1 s, current)': 11, '21 frames (2.1 s)': 21, '31 frames (3.1 s)': 31, '41 frames (4.1 s)': 41}
a_ego = {k: [] for k in WINDOWS}
for name in f.segment.unique():
    seg = wc.Segment(ex_root, name)
    v = seg.ego_v
    dt = np.median(np.diff(seg.t))
    for label, win in WINDOWS.items():
        a = np.gradient(v, seg.t, axis=0) if win is None else savgol_filter(v, win, 2, deriv=1, delta=dt, axis=0)
        a_ego[label] += [wc.speed_and_accel(v[k], a[k], seg.yaw[k])[1] for k in range(len(v))]
f = f.assign(**{f'a_{k}': np.array(vals) for k, vals in a_ego.items()})
cur = 'a_11 frames (1.1 s, current)'
lf = f.lead_map.notna(); print('check (lead frames): current window reproduces a_ego in frame_ssm.csv:', np.allclose(f.loc[lf, cur], f.loc[lf, 'a_ego'], atol=1e-9), '| MTTC reproduced:', None)
d = f[f.lead_map.notna()].copy()
big = 1e9


def metrics(a):
    out = [wsf.lead_ssms(g, vf, vl, af, al) for g, vf, vl, af, al in zip(d.gap, d.v_ego, d.v_lead, a, d.a_lead)]
    return pd.Series([o['MTTC'] for o in out], index=d.index), pd.Series([o['CI'] for o in out], index=d.index)


m0, c0 = metrics(d[cur]); print('MTTC reproduced from frame_ssm:', np.allclose(m0.replace(np.inf, big), d.MTTC.replace(np.inf, big)), '| CI reproduced:', np.allclose(c0, d.CI))
print(f'{"window":28s} {"sd of a_ego":>12s} {"max |diff| vs 1.1 s":>20s} {"MTTC tau":>9s} {"CI tau":>8s} {"finite MTTC":>12s}')
for k in WINDOWS:
    col = f'a_{k}'
    m, c = metrics(d[col])
    tau_m = kendalltau(m.replace(np.inf, big), m0.replace(np.inf, big)).statistic
    tau_c = kendalltau(c, c0).statistic
    print(f'{k:28s} {f[col].std():12.3f} {np.abs(f[col] - f[cur]).max():20.3f} {tau_m:9.3f} {tau_c:8.3f} {int(np.isfinite(m).sum()):12d}')
