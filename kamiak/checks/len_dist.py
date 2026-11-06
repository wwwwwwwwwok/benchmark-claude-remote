"""Read-only (user, 2026-11-06, E item 3): is 7.0 m between small and large vehicles? Box lengths of the vehicles of the
comparison list's 303 segments (vlm_frames/comparison_lead7_targets.csv):
  tracks: every vehicle track of the segments (objects.parquet, type vehicle), its box length (the largest - smallest
          length of a track shows whether a track keeps one size);
  in view: the vehicles of the 1,873 target frames in the five cameras' view (Segment.objects_at, as count_nearby.py),
          the frame's lead left out, box centre within 30 m of the ego origin: one row per frame and vehicle.
Printed for each: quantiles and the counts in 0.5 m bins from 3 to 15 m.
usage: python -I len_dist.py <code folder> <segments folder (segments_1873)> <comparison_lead7_targets.csv>"""
import sys

import numpy as np
import pandas as pd

CODE, SEGS, TG = sys.argv[1:4]
sys.path.insert(0, CODE)
import wod_common as wc  # noqa: E402

CAMERAS = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
tg = pd.read_csv(TG, comment='#')[['segment', 'frame_idx', 'lead_map']]
tracks, inst = [], []
spread = []
for sname, g in tg.groupby('segment', sort=True):
    seg = wc.Segment(SEGS, sname)
    v = seg.objects[seg.objects['type'] == 'vehicle']
    for tid, q in v.groupby('id'):
        tracks.append(float(q['length'].iloc[0]))
        spread.append(float(q['length'].max() - q['length'].min()))
    for r in g.itertuples():
        o = seg.objects_at(int(r.frame_idx), cameras=CAMERAS, types=['vehicle'])
        o = o[o['id'] != r.lead_map]
        d = np.hypot(o['cx'].to_numpy(float), o['cy'].to_numpy(float))
        inst.extend(o['length'].to_numpy(float)[d <= 30].tolist())
QS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
edges = np.arange(3.0, 15.01, 0.5)
for name, x in (('tracks (one length each)', np.array(tracks)), ('in view within 30 m, per frame', np.array(inst))):
    print(f'== {name}: {len(x)}; min {x.min():.3f}, max {x.max():.3f}')
    print('  quantiles ' + ', '.join(f'{int(q * 100)}% {v:.3f}' for q, v in zip(QS, np.quantile(x, QS))))
    h, _ = np.histogram(x, bins=edges)
    print(f'  below 3.0 m {int((x < 3).sum())}; at least 15.0 m {int((x >= 15).sum())}')
    print('  0.5 m bins: ' + ', '.join(f'[{a:.1f},{a + 0.5:.1f}) {int(c)}' for a, c in zip(edges[:-1], h)))
    print(f'  at least 7.0 m: {int((x >= 7.0).sum())} ({(x >= 7.0).mean() * 100:.2f} %); 6.0-8.0 m: '
          f'{int(((x >= 6) & (x < 8)).sum())}')
sp = np.array(spread)
print(f'tracks whose length changes within the segment: {int((sp > 1e-9).sum())} of {len(sp)} (largest change '
      f'{sp.max():.6f} m)')
