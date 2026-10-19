"""(a) exact 2D-TTC to the lead vs 1D TTC where TTC > 10 s (finite ones only);
(b) ACT with a pedestrian 'moving' threshold of 0.3 m/s (vehicles/cyclists/ego stay 0.5) vs 0.5 for all.
usage: python -I exp_ped_thr.py <repo_code_dir> <extracted_root> <frame_ssm.csv>"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
sys.stdout.reconfigure(encoding='utf-8')
code, root, csv = sys.argv[1:4]
sys.path.insert(0, code)
import wod_common as wc  # noqa: E402
import waymo_ssm_frames as w  # noqa: E402
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_ped_ttc.py'), encoding='utf-8').read().split('f = pd.read_csv(csv)')[0].split('import waymo_ssm_frames as w  # noqa: E402')[1])
CAMS = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
f = pd.read_csv(csv)
lead, acts = [], []
for name in sorted(f.segment.unique()):
    seg = wc.Segment(root, name)
    w.use_ego(*wc.footprint_from_lidars(seg.context))
    for k in range(len(seg.frames)):
        o_all = seg.objects_at(k, cameras=CAMS, types=['vehicle', 'pedestrian', 'cyclist'])
        if not len(o_all):
            continue
        R = seg.R[k]
        ego_v_veh = (R.T @ seg.ego_v[k])[:2]
        ego_moving = np.hypot(*ego_v_veh) >= 0.5
        _, _, v_g, _ = seg.to_map(k, o_all)
        fr = f[(f.segment == name) & (f.frame_idx == k)].iloc[0]
        a5 = a3 = np.inf
        for (oid, typ, cx, cy, L, W, h), vg in zip(o_all[['id', 'type', 'cx', 'cy', 'length', 'width', 'heading']].itertuples(index=False), v_g):
            if np.hypot(cx, cy) > wc.LABEL_RANGE:
                continue
            q = w.rect(cx, cy, L, W, h)
            v_obj = R[:2, :2].T @ vg
            rel = v_obj - ego_v_veh
            so = np.hypot(*v_obj)
            a5 = min(a5, w.act(q, rel, h, so >= 0.5, ego_moving)[0])
            a3 = min(a3, w.act(q, rel, h, so >= (0.3 if typ == 'pedestrian' else 0.5), ego_moving)[0])
            if oid == fr.lead_map and np.isfinite(fr.TTC) and fr.TTC > 10:
                lead.append((fr.TTC, ttc2d_exact(q, rel)))
        acts.append((a5, a3))
d = pd.DataFrame(lead, columns=['TTC', 'exact'])
fin = d[np.isfinite(d.exact)]
print('(a) lead frames with TTC > 10 s: %d; exact 2D-TTC finite in %d; exact/TTC for those: median %.4f, range %.4f-%.4f' % (
    len(d), len(fin), (fin.exact / fin.TTC).median(), (fin.exact / fin.TTC).min(), (fin.exact / fin.TTC).max()))
a = pd.DataFrame(acts, columns=['ped05', 'ped03'])
chg = ~np.isclose(a.ped05.replace(np.inf, 1e9), a.ped03.replace(np.inf, 1e9))
print('(b) ACT frames changed with a 0.3 m/s pedestrian threshold: %d of %d; finite %d -> %d; < 1.5 s %d -> %d; < 3 s %d -> %d; tau-b %.4f' % (
    chg.sum(), len(a), np.isfinite(a.ped05).sum(), np.isfinite(a.ped03).sum(), (a.ped05 < 1.5).sum(), (a.ped03 < 1.5).sum(),
    (a.ped05 < 3).sum(), (a.ped03 < 3).sum(), kendalltau(a.ped05, a.ped03, variant='b').statistic))
print(a[chg].round(3).to_string())
