"""Ground truth of the quantities task (user, 2026-10-28) on a list of target frames:
    distance   = gap (frame_ssm: ego front bumper to the lead's rear, along the lane) + the segment's distance from the
                 front bumper to the front camera (footprint_from_lidars front x - camera_position FRONT x, both from
                 the segment's own context.json; no default: a segment whose calibration cannot be read is an error)
    ego_speed  = v_ego, lead_speed = v_lead, ego_accel = a_ego, lead_accel = a_lead (frame_ssm as it is; frame_ssm
                 fills these columns only on frames with a lead, so they are empty on frames without one)
Also prints the front-bumper-to-camera distance over the segments of the list, and checks the bumper position against
the "ego <segment> = front ..." lines of the SSM run's params.txt (the gap was measured from it).
usage: python -I quantity_truth.py <frame_ssm.csv> <targets csv (segment, frame_idx)> <tables root> <out csv>
                                   [--ssm_params <params.txt> ...]
       tables root: the folder with <split>/<segment>/context.json
"""
import argparse
import glob
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wod_common import camera_position, footprint_from_lidars  # noqa: E402

TRUTH = {'distance': 'distance', 'ego_speed': 'v_ego', 'lead_speed': 'v_lead', 'ego_accel': 'a_ego',
         'lead_accel': 'a_lead'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('frame_ssm')
    ap.add_argument('targets')
    ap.add_argument('tables')
    ap.add_argument('out')
    ap.add_argument('--ssm_params', nargs='*', default=[])
    a = ap.parse_args()
    tg = pd.read_csv(a.targets, comment='#')[['segment', 'frame_idx']]
    segs = sorted(tg.segment.unique())
    rows = []
    for s in segs:
        found = glob.glob(os.path.join(a.tables, '*', s, 'context.json'))
        if len(found) != 1:
            sys.exit(f'{s}: {len(found)} context.json under {a.tables}, need exactly 1')
        with open(found[0], encoding='utf-8') as f:
            ctx = json.load(f)
        front, _, _ = footprint_from_lidars(ctx)                  # raises if the lidar calibration is missing
        cam_x, cam_y = camera_position(ctx, 'FRONT')              # raises if there is no FRONT camera
        rows.append(dict(segment=s, split=os.path.basename(os.path.dirname(os.path.dirname(found[0]))),
                         front_x=front, camera_x=cam_x, camera_y=cam_y, camera_offset=front - cam_x))
    seg = pd.DataFrame(rows)

    fs = pd.read_csv(a.frame_ssm, usecols=['segment', 'frame_idx', 'edge_frame', 'reversing_ego', 'reversing_lead',
                                           'gap', 'v_ego', 'v_lead', 'a_ego', 'a_lead'])
    t = tg.merge(fs, on=['segment', 'frame_idx'], how='left', validate='one_to_one', indicator=True)
    if (t['_merge'] != 'both').any():
        sys.exit(f'{int((t["_merge"] != "both").sum())} target frames are not in {a.frame_ssm}')
    t = t.drop(columns='_merge').merge(seg[['segment', 'camera_offset']], on='segment', how='left')
    t['distance'] = t['gap'] + t['camera_offset']
    t['has_lead'] = t['gap'].notna()
    t.to_csv(a.out, index=False)

    print(f'{a.targets}: {len(t)} frames, {len(segs)} segments; frame_ssm {a.frame_ssm}')
    o = seg['camera_offset']
    q = o.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    print(f'front bumper to front camera (m) over the {len(seg)} segments: min {o.min():.4f}, 5% {q[0.05]:.4f}, '
          f'25% {q[0.25]:.4f}, median {q[0.5]:.4f}, 75% {q[0.75]:.4f}, 95% {q[0.95]:.4f}, max {o.max():.4f}')
    print('  values (rounded to 1 mm): ' + ', '.join(f'{v:.3f} m: {n}' for v, n in
                                                     o.round(3).value_counts().sort_index().items()))
    for c in ('front_x', 'camera_x', 'camera_y'):
        print(f'  {c}: ' + ', '.join(f'{v:.3f}: {n}' for v, n in seg[c].round(3).value_counts().sort_index().items()))
    if a.ssm_params:
        logged = {}
        for p in a.ssm_params:
            for line in open(p, encoding='utf-8'):
                m = re.match(r'ego (\S+) = front (-?[0-9.]+) m', line)
                if m:
                    logged[m.group(1)] = float(m.group(2))
        miss = [s for s in segs if s not in logged]
        diff = [s for s in segs if s in logged and abs(logged[s] - round(float(seg.set_index('segment').at[s, 'front_x']), 3)) > 1e-9]
        print(f'  bumper position against the SSM params.txt: {len(segs) - len(miss) - len(diff)} of {len(segs)} the '
              f'same (to 1 mm), {len(diff)} different, {len(miss)} not listed')
    print(f'frames with a lead (gap): {int(t.has_lead.sum())} of {len(t)}; edge frames {int(t.edge_frame.sum())}; '
          f'ego reversing {int(t.reversing_ego.sum())}, lead reversing {int(t.reversing_lead.sum())}')
    for q_, c in TRUTH.items():
        v = t[c].dropna()
        if len(v):
            print(f'  {q_:10s} ({c}): {len(v)} frames, median {v.median():.3f}, range {v.min():.3f} to {v.max():.3f}, '
                  f'5-95% {v.quantile(0.05):.3f} to {v.quantile(0.95):.3f}')
        else:
            print(f'  {q_:10s} ({c}): no frames')
    for c in ('a_ego', 'a_lead'):
        v = t[c].dropna()
        print(f'  {c} by +-0.5 m/s2: speeding up {int((v > 0.5).sum())}, steady {int((v.abs() <= 0.5).sum())}, '
              f'slowing down {int((v < -0.5).sum())}')
    print(f'written {a.out}')


if __name__ == '__main__':
    main()
