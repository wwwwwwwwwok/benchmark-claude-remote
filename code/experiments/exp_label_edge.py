"""Where does Waymo labelling start/stop? For tracks that enter (first frame > 0) or leave (last frame < end) the
labels during a segment while far away, look at the distance at the first / last labelled frame, measured to the
box centre and to the nearest point of the box, from the vehicle origin (rear axle) and from the TOP lidar.
If an object is first labelled at 77 m, it existed unlabelled between 80 and 77 m -> that band is incomplete.
usage: python -I exp_label_edge.py <extracted_root>"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
root = sys.argv[1]


def nearest(cx, cy, L, W, h, ox, oy):
    c, s = np.cos(h), np.sin(h)
    dx = np.array([1, 1, -1, -1]) * L / 2
    dy = np.array([1, -1, -1, 1]) * W / 2
    P = np.stack([cx + dx * c - dy * s - ox, cy + dx * s + dy * c - oy], 1)
    best = np.inf
    for i in range(4):
        p, q = P[i], P[(i + 1) % 4]
        ab = q - p
        u = np.clip(-p @ ab / (ab @ ab), 0, 1)
        best = min(best, np.linalg.norm(p + u * ab))
    return best


rows = []
for name in sorted(os.listdir(root)):
    o = pd.read_parquet(os.path.join(root, name, 'objects.parquet'))
    ctx = json.load(open(os.path.join(root, name, 'context.json'), encoding='utf-8'))
    o = o[o['type'].isin(['vehicle', 'pedestrian', 'cyclist'])]
    last = o['frame_idx'].max()
    top = [lc for lc in ctx.get('laser_calibrations', []) if str(lc.get('name')).upper() in ('TOP', '1')]
    tx = np.array(top[0]['extrinsic']).reshape(4, 4)[:2, 3] if top else np.array([1.43, 0.0])
    for tid, g in o.sort_values('frame_idx').groupby('id'):
        for end, r in (('first', g.iloc[0]), ('last', g.iloc[-1])):
            if (end == 'first' and r.frame_idx == 0) or (end == 'last' and r.frame_idx == last):
                continue                                   # the track runs past the segment boundary
            rows.append(dict(segment=name[:6], id=tid, type=r.type, end=end, n=len(g),
                             centre=np.hypot(r.cx, r.cy),
                             near=nearest(r.cx, r.cy, r.length, r.width, r.heading, 0, 0),
                             near_lidar=nearest(r.cx, r.cy, r.length, r.width, r.heading, *tx),
                             pts=r.num_top_lidar_points))
d = pd.DataFrame(rows)
far = d[d.centre > 65]
print('track ends beyond 65 m (centre): %d (first %d, last %d) of %d track ends inside segments' % (
    len(far), (far.end == 'first').sum(), (far.end == 'last').sum(), len(d)))
for col in ('centre', 'near', 'near_lidar'):
    print(f'  {col:10s}: percentiles 0/5/25/50/75/95/100 = ' + ' / '.join(f'{x:.2f}' for x in np.percentile(far[col], [0, 5, 25, 50, 75, 95, 100])))
print('  share with nearest point (vehicle origin) <= 75 m: %.3f; centre <= 80 m: %.3f; centre <= 75: %.3f' % (
    (far.near <= 75).mean(), (far.centre <= 80).mean(), (far.centre <= 75).mean()))
print(far.sort_values('near')[['segment', 'type', 'end', 'n', 'centre', 'near', 'near_lidar', 'pts']].head(12).round(2).to_string(index=False))
