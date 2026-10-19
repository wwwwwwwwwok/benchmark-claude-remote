"""What the ACT paper's collision-course condition (Venthuruthiyil & Chunchu 2022, Sec. 3.1.3, Eq. 15) does to
our ACT on the demo. Condition: a line from the approaching vehicle's closest corner, along its heading, must pass
through the other vehicle's body; otherwise the pair is not on a collision course (ACT = inf).
usage: python -I act_course_check.py <code_dir> <extracted_root> <frame_ssm.csv>"""
import sys
import numpy as np
import pandas as pd

code_dir, ex_root, ssm_csv = sys.argv[1:4]
sys.path.insert(0, code_dir)
import wod_common as wc          # noqa: E402
import waymo_ssm_frames as wsf   # noqa: E402

CAMS = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']


def ray_hits(p, u, poly):
    """Does the ray p + s*u (s >= 0) cross the polygon (any edge)?"""
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        e = b - a
        den = u[0] * (-e[1]) - u[1] * (-e[0])
        if abs(den) < 1e-12:
            continue
        w = a - p
        s = (w[0] * (-e[1]) - w[1] * (-e[0])) / den
        r = (u[0] * w[1] - u[1] * w[0]) / den
        if s >= 0 and 0 <= r <= 1:
            return True
    return False


def on_course(ego_box, ego_moving, q, h_obj, obj_moving):
    d = np.linalg.norm(ego_box[:, None, :] - q[None, :, :], axis=-1)
    ie, io = np.unravel_index(d.argmin(), d.shape)          # closest corner pair (Eq. 7)
    hit = False
    if ego_moving:                                          # ego heading = +x in the vehicle frame
        hit |= ray_hits(ego_box[ie], np.array([1.0, 0.0]), q)
    if obj_moving:
        hit |= ray_hits(q[io], np.array([np.cos(h_obj), np.sin(h_obj)]), ego_box)
    return hit


fs = pd.read_csv(ssm_csv)
rows = []
for name in fs.segment.unique():
    seg = wc.Segment(ex_root, name)
    wsf.use_ego(*wc.footprint_from_lidars(seg.context))
    for k in range(len(seg.frames)):
        o = seg.objects_at(k, cameras=CAMS, types=['vehicle', 'pedestrian', 'cyclist'])
        R = seg.R[k]
        ego_v_veh = (R.T @ seg.ego_v[k])[:2]
        ego_moving = np.hypot(*ego_v_veh) >= 0.5
        best_old, best_new, obj_old, obj_new = np.inf, np.inf, None, None
        if len(o):
            _, _, v_g, _ = seg.to_map(k, o)
            for (oid, typ, cx, cy, L, W, h), vg in zip(o[['id', 'type', 'cx', 'cy', 'length', 'width', 'heading']].itertuples(index=False), v_g):
                if np.hypot(cx, cy) > 80:
                    continue
                q = wsf.rect(cx, cy, L, W, h)
                v_obj = R[:2, :2].T @ vg
                a_val, _ = wsf.act(q, v_obj - ego_v_veh)
                if a_val < best_old:
                    best_old, obj_old = a_val, (oid, typ)
                if np.isfinite(a_val) and on_course(wsf.EGO_BOX, ego_moving, q, h, np.hypot(*v_obj) >= 0.5):
                    if a_val < best_new:
                        best_new, obj_new = a_val, (oid, typ)
        rows.append(dict(segment=name[:8], frame=k, act_old=best_old, act_new=best_new,
                         obj_new_type=obj_new[1] if obj_new else None,
                         new_is_lead=(obj_new is not None and obj_new[0] == fs.loc[(fs.segment == name) & (fs.frame_idx == k), 'lead_map'].iloc[0])))
t = pd.DataFrame(rows)
for c in ('act_old', 'act_new'):
    print(f'{c}: frames with ACT < 1.5 s: {(t[c] < 1.5).mean():.0%}, < 3 s: {(t[c] < 3).mean():.0%}, finite: {np.isfinite(t[c]).mean():.0%}, median {t[c].median():.2f} s')
print(t.groupby('segment').apply(lambda g: pd.Series({'old_<1.5s': (g.act_old < 1.5).mean(), 'new_<1.5s': (g.act_new < 1.5).mean(),
                                                     'new_<3s': (g.act_new < 3).mean(), 'new_finite': np.isfinite(g.act_new).mean()}), include_groups=False).round(2).to_string())
fin = t[np.isfinite(t.act_new)]
print('\nobject that sets the frame ACT after the condition:', fin.obj_new_type.value_counts().to_dict(),
      '| it is the lead vehicle in', f'{fin.new_is_lead.mean():.0%}', 'of those frames')
low = t[t.act_new < 3]
print('frames with new ACT < 3 s by segment:', low.segment.value_counts().to_dict())
