"""How wrong is the acceleration near the end of a track? Savitzky-Golay (19 frames, quadratic) as used by Waymo for
the label kinematics and by us for the ego. Pretend a long track ends (or starts) at frame k and compare the edge
estimate at k (and j frames before the edge) with the centred estimate at k from the full track.
Acceleration = component along the direction of travel (what MTTC/CI use).
usage: python -I exp_edge_accel.py <repo_code_dir> <extracted_root>"""
import sys

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.stdout.reconfigure(encoding='utf-8')
code, root = sys.argv[1:3]
sys.path.insert(0, code)
import wod_common as wc  # noqa: E402

W, O, DT = 19, 2, 0.1
SEGS = ['13184115878756336167_1354_000_1374_000', '14486517341017504003_3406_349_3426_349',
        '16229547658178627464_380_000_400_000']


def along(xy, idx):
    v = savgol_filter(xy, W, O, deriv=1, delta=DT, axis=0)[idx]
    a = savgol_filter(xy, W, O, deriv=2, delta=DT, axis=0)[idx]
    return float(a @ v / np.linalg.norm(v))


def profile(series_list, label):
    rows = []
    for xy in series_list:
        n = len(xy)
        for k in range(30, n - 30, 3):
            c = along(xy, k)
            for j in (0, 3, 6, 9):           # frames between k and the (pretended) end / start
                rows.append(dict(j=j, end=abs(along(xy[:k + 1 + j], k) - c),
                                 start=abs(along(xy[k - j:], j) - c), centred=abs(c)))
    d = pd.DataFrame(rows)
    print(f'=== {label}: {len(series_list)} tracks, {len(d) // 4} cut points; '
          f'|centred acceleration| median {d.centred.median():.3f}, p95 {d.centred.quantile(0.95):.3f} m/s2')
    for j, g in d.groupby('j'):
        print(f'  {j} frames from the edge: |error| end median {g.end.median():.3f} p95 {g.end.quantile(0.95):.3f} | '
              f'start median {g.start.median():.3f} p95 {g.start.quantile(0.95):.3f} m/s2')


veh, ego = [], []
for name in SEGS:
    seg = wc.Segment(root, name)
    ego.append(seg.pose[:, :2, 3].copy())
    o = seg.objects[seg.objects['type'] == 'vehicle']
    k = o['frame_idx'].to_numpy()
    xyw = np.einsum('nij,nj->ni', seg.R[k], o[['cx', 'cy', 'cz']].to_numpy(float)) + seg.pose[k, :3, 3]
    o = o.assign(wx=xyw[:, 0], wy=xyw[:, 1], spd=np.hypot(o['speed_x'], o['speed_y']))
    for tid, g in o.sort_values('frame_idx').groupby('id'):
        f = g['frame_idx'].to_numpy()
        if len(g) >= 80 and (np.diff(f) == 1).all() and g.spd.min() > 2.0:   # long, continuous, always moving
            veh.append(g[['wx', 'wy']].to_numpy())
profile(veh, 'labelled vehicles (box centres)')
profile(ego, 'ego (pose)')
