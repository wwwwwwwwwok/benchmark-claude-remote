"""Experiments for the user's questions (2026-10-15):
 1) label-speed noise of static objects, and what changes if the 0.5 m/s 'moving / travel direction' threshold
    becomes 0.1 m/s;
 2) exact 2D-TTC (ray casting, no horizon and no time step, as in Jiao's fast 2D-TTC) vs the current grid search
    (10 s cap, 0.05 s step);
 3) which 2D partners lie beyond 75 m (the paper's fully labelled radius).
usage: python -I exp_lowspeed_ttc2d.py <repo_code_dir> <extracted_root> <frame_ssm.csv>"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.stdout.reconfigure(encoding='utf-8')
code, root, csv = sys.argv[1:4]
sys.path.insert(0, code)
import wod_common as wc  # noqa: E402
import waymo_ssm_frames as w  # noqa: E402

CAMS = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']


def ray_time(p, u, poly):
    """Smallest s >= 0 with p + s*u on an edge of poly (u is a velocity, so s is in seconds)."""
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
    """First contact of two translating rectangles: a corner of one reaches an edge of the other."""
    if w.poly_dist(w.EGO_BOX, obj) == 0:
        return 0.0
    if np.hypot(*rel_v) < 1e-12:
        return np.inf
    t1 = min(ray_time(p, rel_v, w.EGO_BOX) for p in obj)    # object corners moving into the ego
    t2 = min(ray_time(q, -rel_v, obj) for q in w.EGO_BOX)   # ego corners, seen from the object
    return min(t1, t2)


f = pd.read_csv(csv)
rows, static = [], []
for name in sorted(f.segment.unique()):
    seg = wc.Segment(root, name)
    w.use_ego(*wc.footprint_from_lidars(seg.context))
    # static-object label-speed noise: tracks whose world centre never moves more than 0.3 m
    o = seg.objects
    k = o['frame_idx'].to_numpy()
    c = o[['cx', 'cy', 'cz']].to_numpy(float)
    xyw = np.einsum('nij,nj->ni', seg.R[k], c) + seg.pose[k, :3, 3]
    o = o.assign(wx=xyw[:, 0], wy=xyw[:, 1], spd=np.hypot(o['speed_x'], o['speed_y']))
    ext = o.groupby('id').agg(type=('type', 'first'), dx=('wx', lambda s: s.max() - s.min()),
                              dy=('wy', lambda s: s.max() - s.min()), n=('wx', 'size'))
    st_ids = ext[(np.hypot(ext.dx, ext.dy) < 0.3) & (ext.n >= 50)].index
    static.append(o[o['id'].isin(st_ids)][['type', 'spd']])
    for kk in range(len(seg.frames)):
        o_all = seg.objects_at(kk, cameras=CAMS, types=['vehicle', 'pedestrian', 'cyclist'])
        if not len(o_all):
            continue
        R = seg.R[kk]
        ego_v_veh = (R.T @ seg.ego_v[kk])[:2]
        _, _, v_g, _ = seg.to_map(kk, o_all)
        for (oid, typ, cx, cy, L, W, h), vg in zip(
                o_all[['id', 'type', 'cx', 'cy', 'length', 'width', 'heading']].itertuples(index=False), v_g):
            r = np.hypot(cx, cy)
            if r > 80:
                continue
            q = w.rect(cx, cy, L, W, h)
            v_obj = R[:2, :2].T @ vg
            rel = v_obj - ego_v_veh
            so, se = np.hypot(*v_obj), np.hypot(*ego_v_veh)
            rows.append(dict(segment=name, frame_idx=kk, id=oid, type=typ, r=r, v_obj=so,
                             grid=w.ttc_2d(q, rel, 10.0), exact=ttc2d_exact(q, rel),
                             act05=w.act(q, rel, h, so >= 0.5, se >= 0.5)[0],
                             act01=w.act(q, rel, h, so >= 0.1, se >= 0.1)[0]))
d = pd.DataFrame(rows)

print('=== 1. label-speed noise of static tracks (world centre moves < 0.3 m over >= 50 frames)')
s = pd.concat(static)
for t, g in s.groupby('type'):
    print(f'  {t:10s} boxes {len(g):6d}: speed percentiles 50/90/95/99/max = '
          + ' / '.join(f'{x:.3f}' for x in np.percentile(g.spd, [50, 90, 95, 99, 100])) + ' m/s'
          + f'; share >= 0.1: {(g.spd >= 0.1).mean():.4f}, >= 0.5: {(g.spd >= 0.5).mean():.4f}')
print('  lead frames: min v_lead %.3f m/s, min v_ego %.3f m/s' % (f.v_lead.min(), f.v_ego.min()))
print('  objects in 2D loop with 0.1 <= speed < 0.5:', int(((d.v_obj >= 0.1) & (d.v_obj < 0.5)).sum()), 'of', len(d),
      '| by type:', d[(d.v_obj >= 0.1) & (d.v_obj < 0.5)].type.value_counts().to_dict())
fa = d.groupby(['segment', 'frame_idx']).agg(act05=('act05', 'min'), act01=('act01', 'min')).reset_index()
chg = ~np.isclose(fa.act05.replace(np.inf, 1e9), fa.act01.replace(np.inf, 1e9))
print(f'  frame ACT: changed in {chg.sum()} of {len(fa)} frames; finite {np.isfinite(fa.act05).sum()} -> '
      f'{np.isfinite(fa.act01).sum()}; < 1.5 s {(fa.act05 < 1.5).sum()} -> {(fa.act01 < 1.5).sum()}; '
      f'tau-b {kendalltau(fa.act05, fa.act01, variant="b").statistic:.4f}')

print('=== 2. 2D-TTC: exact vs grid (10 s cap, 0.05 s step)')
both = d[np.isfinite(d.grid)]
print(f'  object pairs with grid finite: {len(both)}; exact finite there: {np.isfinite(both.exact).sum()}; '
      f'max |exact - grid| = {np.abs(both.exact - both.grid).max():.4f} s; exact <= grid always: {(both.exact <= both.grid + 1e-9).all()}')
more = d[~np.isfinite(d.grid) & np.isfinite(d.exact)]
print(f'  pairs grid inf but exact finite (> 10 s): {len(more)}; exact range {more.exact.min():.3f}-{more.exact.max():.3f} s')
ff = d.groupby(['segment', 'frame_idx']).agg(grid=('grid', 'min'), exact=('exact', 'min')).reset_index()
print(f'  frames: finite grid {np.isfinite(ff.grid).sum()}, finite exact {np.isfinite(ff.exact).sum()} of {len(ff)}; '
      f'frame min exact <= 10 s: {(ff.exact <= 10).sum()}; tau-b frame min {kendalltau(ff.grid, ff.exact, variant="b").statistic:.4f}')
cap_only = ff[np.isfinite(ff.grid)]
print(f'  among grid-finite frames, max |exact - grid| = {np.abs(cap_only.exact - cap_only.grid).max():.4f} s')
print('  exact frame-min percentiles (finite):', np.percentile(ff.exact[np.isfinite(ff.exact)], [0, 10, 50, 90, 100]).round(3))
print('  check vs frame_ssm TTC2D:', np.allclose(
    ff.merge(f[['segment', 'frame_idx', 'TTC2D']], on=['segment', 'frame_idx']).pipe(lambda x: x.grid.replace(np.inf, -1) - x.TTC2D.replace(np.inf, -1)), 0))

print('=== 3. 2D partners beyond 75 m')
for col in ('TTC2D_obj', 'ACT_obj', 'min_dist_obj'):
    m = f[['segment', 'frame_idx', col]].dropna().merge(d[['segment', 'frame_idx', 'id', 'r']],
                                                        left_on=['segment', 'frame_idx', col],
                                                        right_on=['segment', 'frame_idx', 'id'])
    print(f'  {col}: {len(m)} frames, partner beyond 75 m in {(m.r > 75).sum()}')
