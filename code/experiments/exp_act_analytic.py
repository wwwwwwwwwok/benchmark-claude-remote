"""ACT closing-in rate: current finite difference (boxes moved 0.1 s) vs the analytic projection used by Jiao's
SSMsOnPlane (relative velocity projected on the shortest-distance direction; acceleration and yaw rate = 0).
Same collision-course condition (paper Eq. 15) in both.
usage: python -I exp_act_analytic.py <repo_code_dir> <extracted_root>"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.stdout.reconfigure(encoding='utf-8')
code, root = sys.argv[1:3]
sys.path.insert(0, code)
import wod_common as wc  # noqa: E402
import waymo_ssm_frames as w  # noqa: E402

CAMS = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']


def closest_pair(A, B):
    """Closest points (pa on A, pb on B) of two disjoint convex polygons: min over vertex-to-edge pairs."""
    best = (np.inf, None, None)
    for P, Q, flip in ((A, B, False), (B, A, True)):
        for p in P:
            for a, b in zip(Q, np.roll(Q, -1, axis=0)):
                ab = b - a
                t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
                q = a + t * ab
                d = np.linalg.norm(p - q)
                if d < best[0]:
                    best = (d, q, p) if flip else (d, p, q)
    return best


def act_analytic(obj, rel_v, obj_heading, obj_moving, ego_moving):
    d0 = w.poly_dist(w.EGO_BOX, obj)
    if d0 == 0:
        return 0.0
    d, pe, po = closest_pair(w.EGO_BOX, obj)
    u = (po - pe) / d                       # from the ego towards the object
    closing = -float(np.dot(rel_v, u))      # object moving towards the ego along u
    if closing <= 1e-6 or not w.on_collision_course(obj, obj_heading, obj_moving, ego_moving):
        return np.inf
    return d0 / closing


rows = []
for name in sorted(['13184115878756336167_1354_000_1374_000', '14486517341017504003_3406_349_3426_349',
                    '16229547658178627464_380_000_400_000']):
    seg = wc.Segment(root, name)
    w.use_ego(*wc.footprint_from_lidars(seg.context))
    for k in range(len(seg.frames)):
        o_all = seg.objects_at(k, cameras=CAMS, types=['vehicle', 'pedestrian', 'cyclist'])
        if not len(o_all):
            continue
        R = seg.R[k]
        ego_v_veh = (R.T @ seg.ego_v[k])[:2]
        ego_moving = np.hypot(*ego_v_veh) >= w.V_MOVING
        _, _, v_g, _ = seg.to_map(k, o_all)
        for (oid, cx, cy, L, W, h), vg in zip(o_all[['id', 'cx', 'cy', 'length', 'width', 'heading']].itertuples(index=False), v_g):
            if np.hypot(cx, cy) > 80:
                continue
            q = w.rect(cx, cy, L, W, h)
            v_obj = R[:2, :2].T @ vg
            rel = v_obj - ego_v_veh
            mov = np.hypot(*v_obj) >= w.V_MOVING
            rows.append(dict(segment=name, frame_idx=k, id=oid, fd=w.act(q, rel, h, mov, ego_moving)[0],
                             an=act_analytic(q, rel, h, mov, ego_moving)))
d = pd.DataFrame(rows)
both = d[np.isfinite(d.fd) & np.isfinite(d.an)]
print(f'object pairs {len(d)}; finite: finite-difference {np.isfinite(d.fd).sum()}, analytic {np.isfinite(d.an).sum()}, both {len(both)}')
rel_err = np.abs(both.an - both.fd) / both.fd
print(f'  |analytic - fd| max {np.abs(both.an - both.fd).max():.4f} s; relative: median {rel_err.median():.2e}, '
      f'p99 {rel_err.quantile(0.99):.2e}, max {rel_err.max():.2e}')
f = d.groupby(['segment', 'frame_idx']).agg(fd=('fd', 'min'), an=('an', 'min')).reset_index()
chg = ~np.isclose(f.fd.replace(np.inf, 1e9), f.an.replace(np.inf, 1e9), rtol=1e-3, atol=1e-3)
print(f'frames {len(f)}: finite fd {np.isfinite(f.fd).sum()}, analytic {np.isfinite(f.an).sum()}; '
      f'differ by > 0.1% in {chg.sum()} frames; < 1.5 s: {(f.fd < 1.5).sum()} vs {(f.an < 1.5).sum()}; '
      f'tau-b {kendalltau(f.fd, f.an, variant="b").statistic:.4f}')
print(f.loc[chg].assign(diff=lambda x: x.an - x.fd).head(10).to_string(index=False))
