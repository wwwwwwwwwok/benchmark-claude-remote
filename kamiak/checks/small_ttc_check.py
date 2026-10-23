"""Frames with a small lead-vehicle TTC (2026-10-21): is the "lead" really ahead in the ego's way, or a stopped vehicle
beside the path that the ego drives past? For every frame with TTC < --max_ttc (not edge frames, lead-vehicle SSMs
with values), from the segment tables: the lead box in the ego (vehicle) frame at that frame (longitudinal cx, lateral
cy), the lateral clearance between the two boxes if both keep their heading (|cy| - ego half width - lead width / 2,
ignoring the lead's yaw), whether the ego later drives past it (the lead's front ends up behind the ego's rear bumper in
a later frame, and the lateral clearance then), and the lead picked by the ego-path corridor.

usage: python -I small_ttc_check.py <tables root of a split> <frame_ssm.csv> [...pairs] [--max_ttc 3]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
LEAD = ['TTC', 'THW', 'DRAC', 'MTTC', 'CI', 'PICUD', 'PSD']
ALL9 = LEAD + ['TTC2D', 'ACT']


def has_value(d, s):
    """As in ssm_value_frames.py: the lead-vehicle SSMs have no value where the ego or the lead reverses (files with
    the reversing columns, 2026-10-21)."""
    v = d[s].to_numpy(float)
    val = (np.nan_to_num(v, nan=0.0) > 0) if s in ('DRAC', 'CI') else np.isfinite(v)
    if s in LEAD and 'reversing_ego' in d:
        val &= ~(d['reversing_ego'].astype(bool) | d['reversing_lead'].astype(bool)).to_numpy()
    return val


def ego_box(root, seg):
    with open(os.path.join(root, seg, 'context.json'), encoding='utf-8') as f:
        ctx = json.load(f)
    x = {c['name']: np.asarray(c['extrinsic'], float).reshape(4, 4)[:3, 3] for c in ctx['laser_calibrations']}
    return x['FRONT'][0], x['REAR'][0], abs(x['SIDE_LEFT'][1])


def main():
    args = sys.argv[1:]
    max_ttc = 3.0
    if '--max_ttc' in args:
        i = args.index('--max_ttc')
        max_ttc = float(args[i + 1])
        args = args[:i] + args[i + 2:]
    rows = []
    for root, fcsv in zip(args[0::2], args[1::2]):
        f = pd.read_csv(fcsv, dtype={'lead_map': str, 'lead_path': str})
        f = f[~f['edge_frame'].astype(bool)]
        sel = np.logical_and.reduce([has_value(f, s) for s in LEAD])
        f = f[sel].copy()
        f['all9'] = np.logical_and.reduce([has_value(f, s) for s in ALL9])
        f = f[f['TTC'] < max_ttc]
        for seg, g in f.groupby('segment'):
            o = pd.read_parquet(os.path.join(root, seg, 'objects.parquet'),
                                columns=['frame_idx', 'id', 'cx', 'cy', 'length', 'width', 'heading'])
            front, rear, half_w = ego_box(root, seg)
            by_id = {i: d.set_index('frame_idx') for i, d in o[o['id'].isin(set(g['lead_map']))].groupby('id')}
            for r in g.itertuples():
                d = by_id[r.lead_map]
                b = d.loc[r.frame_idx]
                clear = abs(b.cy) - half_w - b.width / 2
                later = d[d.index > r.frame_idx]
                passed = later[later['cx'] + later['length'] / 2 < rear]
                rows.append(dict(segment=seg, frame_idx=r.frame_idx, all9=r.all9, TTC=r.TTC, gap=r.gap, v_ego=r.v_ego,
                                 v_lead=r.v_lead, a_ego=r.a_ego, TTC2D=r.TTC2D, ACT=r.ACT, cx=b.cx, cy=b.cy,
                                 lat_clear=clear, passed=len(passed) > 0,
                                 pass_clear=(abs(passed['cy'].iloc[0]) - half_w - passed['width'].iloc[0] / 2)
                                 if len(passed) else np.nan,
                                 same_as_path=r.lead_map == r.lead_path))
    d = pd.DataFrame(rows)
    for lab, x in (('7 lead SSMs with values', d), ('all 9 SSMs with values', d[d['all9']])):
        print(f'\n=== TTC < {max_ttc:g} s, {lab}: {len(x)} frames, {x["segment"].nunique()} segments, '
              f'{x[["segment"]].assign(i=1).groupby("segment").size().max()} frames at most in one segment')
        if not len(x):
            continue
        q = lambda c: ' / '.join(f'{v:.3f}' for v in x[c].quantile([0, .1, .5, .9, 1]))
        for c in ('gap', 'v_ego', 'v_lead', 'a_ego', 'cy', 'lat_clear', 'TTC2D', 'ACT'):
            print(f'  {c:10s} min / p10 / median / p90 / max: {q(c)}')
        print(f'  lead nearly stopped (< 0.5 m/s): {int((x.v_lead < 0.5).sum())}; lateral clearance > 0 (boxes would '
              f'pass side by side): {int((x.lat_clear > 0).sum())}; ego later drives past the lead: '
              f'{int(x.passed.sum())} (clearance when passing: median {x.pass_clear.median():.3f} m, min '
              f'{x.pass_clear.min():.3f} m); same lead as the ego-path corridor: {int(x.same_as_path.sum())}; '
              f'TTC2D finite and > 3 s: {int((np.isfinite(x.TTC2D) & (x.TTC2D > 3)).sum())}')
        both = x[(x.v_lead < 0.5) & x.passed]
        print(f'  lead nearly stopped AND ego later drives past it: {len(both)} frames, {both["segment"].nunique()} '
              f'segments')


if __name__ == '__main__':
    main()
