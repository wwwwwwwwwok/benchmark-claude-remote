"""Is Waymo's labelling frontier a sharp 'nearest point of the box <= 77 m from the vehicle origin'?
 (1) over ALL labelled boxes: the largest nearest-point distance, per segment;
 (2) for far track starts/ends: would the object have been beyond 77 m one frame earlier/later (so the frontier
     alone explains why it was not labelled there)?
usage: python -I exp_label_edge2.py <repo_code_dir> <extracted_root>"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
code, root = sys.argv[1:3]
sys.path.insert(0, code)
import wod_common as wc  # noqa: E402


def near_vec(o):
    """nearest point of each box (vehicle frame) to the origin; vectorised over rows"""
    c, s = np.cos(o.heading.to_numpy()), np.sin(o.heading.to_numpy())
    L, W, cx, cy = (o[k].to_numpy() for k in ('length', 'width', 'cx', 'cy'))
    sx, sy = np.array([1, 1, -1, -1]), np.array([1, -1, -1, 1])
    X = cx[:, None] + (sx * L[:, None] / 2) * c[:, None] - (sy * W[:, None] / 2) * s[:, None]
    Y = cy[:, None] + (sx * L[:, None] / 2) * s[:, None] + (sy * W[:, None] / 2) * c[:, None]
    best = np.full(len(o), np.inf)
    for i in range(4):
        px, py, qx, qy = X[:, i], Y[:, i], X[:, (i + 1) % 4], Y[:, (i + 1) % 4]
        ax, ay = qx - px, qy - py
        u = np.clip(-(px * ax + py * ay) / (ax * ax + ay * ay), 0, 1)
        best = np.minimum(best, np.hypot(px + u * ax, py + u * ay))
    return best


FRONTIER = 77.0
ends = []
for name in sorted(os.listdir(root)):
    seg = wc.Segment(root, name)
    o = seg.objects[seg.objects['type'].isin(['vehicle', 'pedestrian', 'cyclist'])].copy()
    o['near'] = near_vec(o)
    o['centre'] = np.hypot(o.cx, o.cy)
    print(f'{name[:6]}: {len(o)} boxes; nearest point max {o.near.max():.3f} m, > 77.0 m: {(o.near > FRONTIER).sum()}, '
          f'> 76.5 m: {(o.near > 76.5).sum()}; centre max {o.centre.max():.3f}')
    last = o['frame_idx'].max()
    for tid, g in o.sort_values('frame_idx').groupby('id'):
        for end, r in (('first', g.iloc[0]), ('last', g.iloc[-1])):
            k = int(r.frame_idx)
            if (end == 'first' and k == 0) or (end == 'last' and k == last) or r.centre <= 65:
                continue
            R = seg.R[k]
            v_rel = (R.T @ np.array([r.speed_x, r.speed_y, r.speed_z]))[:2] - (R.T @ seg.ego_v[k])[:2]
            radial = float(np.dot(v_rel, [r.cx, r.cy]) / r.centre)       # > 0: moving away from the ego
            step = 0.1 * radial * (-1 if end == 'first' else 1)         # change of distance to the unlabelled frame
            ends.append(dict(end=end, near=r.near, other=r.near + step, radial=radial, pts=r.num_top_lidar_points))
d = pd.DataFrame(ends)
for e in ('first', 'last'):
    g = d[d.end == e]
    ok = g.other > FRONTIER - 0.05
    print(f'{e:5s} far track ends: {len(g)}; neighbouring unlabelled frame beyond 77 m (frontier explains it): '
          f'{ok.sum()} ({ok.mean():.1%}); the rest: nearest {g[~ok].near.min():.2f}-{g[~ok].near.max():.2f} m, '
          f'top-lidar points median {g[~ok].pts.median():.0f}')
