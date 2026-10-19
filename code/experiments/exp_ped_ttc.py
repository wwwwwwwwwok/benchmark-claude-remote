"""(1) Pedestrian label speeds on the demo: distribution, and whether boxes at 0.2-0.5 m/s really move.
(2) 1D TTC to the lead vs 2D-TTC to the same lead: grid (10 s cap, 0.05 s step) vs exact first contact.
usage: python -I exp_ped_ttc.py <repo_code_dir> <extracted_root> <frame_ssm.csv>"""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
code, root, csv = sys.argv[1:4]
sys.path.insert(0, code)
import wod_common as wc  # noqa: E402
import waymo_ssm_frames as w  # noqa: E402


def ray_time(p, u, poly):
    best = np.inf
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        e, wv = b - a, a - p
        den = e[0] * u[1] - e[1] * u[0]
        if abs(den) < 1e-12:
            continue
        s = (e[0] * wv[1] - e[1] * wv[0]) / den
        r = (u[0] * wv[1] - u[1] * wv[0]) / den
        if s >= 0 and 0 <= r <= 1:
            best = min(best, s)
    return best


def ttc2d_exact(obj, rel_v):
    if w.poly_dist(w.EGO_BOX, obj) == 0:
        return 0.0
    if np.hypot(*rel_v) < 1e-12:
        return np.inf
    return min(min(ray_time(p, rel_v, w.EGO_BOX) for p in obj), min(ray_time(q, -rel_v, obj) for q in w.EGO_BOX))


f = pd.read_csv(csv)
peds, lead_rows = [], []
for name in sorted(f.segment.unique()):
    seg = wc.Segment(root, name)
    w.use_ego(*wc.footprint_from_lidars(seg.context))
    o = seg.objects
    k = o['frame_idx'].to_numpy()
    xyw = np.einsum('nij,nj->ni', seg.R[k], o[['cx', 'cy', 'cz']].to_numpy(float)) + seg.pose[k, :3, 3]
    o = o.assign(wx=xyw[:, 0], wy=xyw[:, 1], spd=np.hypot(o['speed_x'], o['speed_y']))
    p = o[o['type'] == 'pedestrian'].sort_values(['id', 'frame_idx'])
    g = p.groupby('id')
    p = p.assign(dx=g['wx'].shift(-5) - g['wx'].shift(5), dy=g['wy'].shift(-5) - g['wy'].shift(5))
    p['disp1s'] = np.hypot(p.dx, p.dy)
    peds.append(p)
    # 2D-TTC to the lead itself, in frames where the 1D TTC is finite
    fl = f[(f.segment == name) & np.isfinite(f.TTC)]
    for r in fl.itertuples():
        q = o[(o.frame_idx == r.frame_idx) & (o.id == r.lead_map)].iloc[0]
        R = seg.R[r.frame_idx]
        ego_v_veh = (R.T @ seg.ego_v[r.frame_idx])[:2]
        _, _, v_g, _ = seg.to_map(r.frame_idx, o[(o.frame_idx == r.frame_idx) & (o.id == r.lead_map)])
        rel = R[:2, :2].T @ v_g[0] - ego_v_veh
        box = w.rect(q.cx, q.cy, q.length, q.width, q.heading)
        lead_rows.append(dict(TTC=r.TTC, grid=w.ttc_2d(box, rel, 10.0), exact=ttc2d_exact(box, rel)))

p = pd.concat(peds)
print('=== 1. pedestrians: %d boxes, %d tracks' % (len(p), p['id'].nunique()))
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 10]
cnt = pd.cut(p.spd, bins, right=False).value_counts().sort_index()
print((cnt / cnt.sum()).round(4).to_string())
for lo, hi in ((0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.8), (0.8, 10)):
    q = p[(p.spd >= lo) & (p.spd < hi)].dropna(subset=['disp1s'])
    print(f'  speed {lo}-{hi} m/s: {len(q)} boxes; centre moved over the surrounding 1 s: median {q.disp1s.median():.3f} m, '
          f'share > 0.2 m {(q.disp1s > 0.2).mean():.3f}')
mv = p[p.spd >= 0.5]
print('  pedestrians >= 0.5 m/s: speed percentiles 15/50/85 = %s m/s' % np.percentile(mv.spd, [15, 50, 85]).round(3))

d = pd.DataFrame(lead_rows)
print('=== 2. lead frames with finite 1D TTC: %d; TTC > 10 s in %d' % (len(d), (d.TTC > 10).sum()))
big = d[d.TTC > 10]
print('  in those: 2D-TTC to the lead, grid finite %d, exact finite %d; exact/TTC median %.4f (range %.4f-%.4f)' % (
    np.isfinite(big.grid).sum(), np.isfinite(big.exact).sum(), (big.exact / big.TTC).median(),
    (big.exact / big.TTC).min(), (big.exact / big.TTC).max()))
small = d[d.TTC <= 10]
print('  TTC <= 10 s: %d frames; grid finite %d, exact finite %d; |exact - TTC| max %.4f s' % (
    len(small), np.isfinite(small.grid).sum(), np.isfinite(small.exact).sum(), np.abs(small.exact - small.TTC).max()))
