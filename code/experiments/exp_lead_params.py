"""Sensitivity of the map-based lead choice to (a) the same-direction threshold (30 deg) and (b) the ego lane-matching
parameters (2.5 m, 45 deg, heading weight 2, stay bonus 1, split overlap 1 m). Lead-only, no 2D metrics.
usage: python -I exp_lead_params.py <repo_code_dir> <extracted_root>"""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
code, root = sys.argv[1:3]
sys.path.insert(0, code)
import wod_common as wc  # noqa: E402
from wod_common import wrap  # noqa: E402

CAMS = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
SEGS = ['13184115878756336167_1354_000_1374_000', '14486517341017504003_3406_349_3426_349',
        '16229547658178627464_380_000_400_000']
P = dict(max_d=2.5, max_dh=45.0, w_h=2.0, bonus=1.0, overlap_d=1.0, lead_dh=30.0)


def candidates(self, p, heading):
    t = np.clip(((p - self.A) * self.AB).sum(1) / self.L2, 0, 1)
    Q = self.A + t[:, None] * self.AB
    dist = np.hypot(*(p - Q).T)
    dh = np.abs(wrap(self.hdg - heading))
    ok = np.flatnonzero((dist < P['max_d']) & (dh < np.deg2rad(P['max_dh'])))
    out = {}
    for i in ok:
        lane = self.lane_of[i]
        cost = dist[i] + P['w_h'] * dh[i]
        if lane not in out or cost < out[lane][0]:
            s = self.lanes[lane]['s'][self.seg_idx[i]] + t[i] * np.sqrt(self.L2[i])
            cross = self.AB[i, 0] * (p[1] - self.A[i, 1]) - self.AB[i, 1] * (p[0] - self.A[i, 0])
            out[lane] = (cost, s, np.sign(cross) * dist[i])
    return out


def match_sequence(self, Pts, H):
    lanes, ss, cands, prev = [], [], [], None
    for p, h in zip(Pts, H):
        c = self.candidates(p, h)
        cands.append(c)
        if not c:
            lanes.append(None); ss.append(np.nan); prev = None
            continue
        def cost(item):
            lane, (cst, _, _) = item
            b = P['bonus'] if prev is not None and (lane == prev or lane in self.lanes[prev]['exits']) else 0.0
            return cst - b
        lane, (_, s, _) = min(c.items(), key=cost)
        lanes.append(lane); ss.append(s); prev = lane
    for k in range(len(lanes) - 2, -1, -1):
        a, b = lanes[k], lanes[k + 1]
        if a is None or b is None or a == b or b in self.lanes[a]['exits']:
            continue
        if b in cands[k] and abs(cands[k][b][2]) < P['overlap_d']:
            lanes[k], ss[k] = b, cands[k][b][1]
    return lanes, np.array(ss)


wc.LaneMap.candidates = candidates
wc.LaneMap.match_sequence = match_sequence
segs = {n: wc.Segment(root, n) for n in SEGS}
for n, s in segs.items():
    wc.set_ego_geometry(*wc.footprint_from_lidars(s.context))   # same footprint in all demo segments (front 4.070)


def leads():
    wc.LEAD_MAX_DHEADING = np.deg2rad(P['lead_dh'])
    out = {}
    for n, seg in segs.items():
        wc.set_ego_geometry(*wc.footprint_from_lidars(seg.context))
        lmap = wc.LaneMap(seg.map)
        ego_lanes, ego_s = lmap.match_sequence(seg.ego_map[:, :2], seg.yaw)
        for k in range(len(seg.frames)):
            o = seg.objects_at(k, cameras=CAMS, types=['vehicle', 'pedestrian', 'cyclist'])
            o = o[wc.box_nearest_dist(o['cx'], o['cy'], o['length'], o['width'], o['heading']) <= wc.LABEL_RANGE]
            o = o[o['type'].isin(wc.LEAD_TYPES)].reset_index(drop=True)
            r = wc.lead_map(seg, k, o, lmap, ego_lanes, ego_s)
            out[(n[:6], k)] = (o.loc[r['i'], 'id'] if r else None, ego_lanes[k])
    return out


base = leads()
print('baseline: lead frames', sum(v[0] is not None for v in base.values()), 'of', len(base))
for key, vals in (('lead_dh', [15, 20, 45, 60, 90]), ('max_d', [2.0, 3.0]), ('max_dh', [30, 60]),
                  ('w_h', [1.0, 4.0]), ('bonus', [0.0, 0.5, 2.0]), ('overlap_d', [0.5, 1.5])):
    for v in vals:
        old = P[key]; P[key] = v
        cur = leads()
        P[key] = old
        dl = [k for k in base if base[k][0] != cur[k][0]]
        de = [k for k in base if base[k][1] != cur[k][1]]
        print(f'  {key} = {v}: lead differs in {len(dl)} frames, ego lane differs in {len(de)} frames'
              + (f'  e.g. {dl[:4]}' if dl else ''))
