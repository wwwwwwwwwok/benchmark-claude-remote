"""Trial of a rule for slow leads (user, 2026-10-22; computed only, the program is not changed), on the leads of the
centre-based lateral-overlap rule: a lead slower than 0.5 m/s at the frame stays the lead only if it is faster than
0.5 m/s in some frame of the segment, or the ego-path pairing picks the same vehicle; otherwise the frame has no lead
(no search for the next candidate). Reports how many frames it removes (all / not edge, per split and together), how
many of them have a lead that the ego later drives past (its front ends up behind the ego's rear bumper in a later
frame), the effect on the frames where the 9 / 7 lead SSMs have values, and the verdict on the acceptance frames.

usage: python -I trial_slow_rule.py <results root (split/ssm/frame_ssm.csv)> <tables root> <acceptance frames.csv>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
V_SLOW = 0.5
LEAD = ['TTC', 'THW', 'DRAC', 'MTTC', 'CI', 'PICUD', 'PSD']
ALL9 = LEAD + ['TTC2D', 'ACT']


def has_value(d, s):
    v = d[s].to_numpy(float)
    val = (np.nan_to_num(v, nan=0.0) > 0) if s in ('DRAC', 'CI') else np.isfinite(v)
    if s in LEAD:
        val &= ~(d['reversing_ego'].astype(bool) | d['reversing_lead'].astype(bool)).to_numpy()
    return val


def rear_x(root, seg):
    with open(os.path.join(root, seg, 'context.json'), encoding='utf-8') as f:
        ctx = json.load(f)
    return {c['name']: np.asarray(c['extrinsic'], float).reshape(4, 4)[0, 3] for c in ctx['laser_calibrations']}['REAR']


def main():
    res, tables, acc = sys.argv[1:4]
    parts = []
    for split in ('validation', 'training'):
        f = pd.read_csv(os.path.join(res, split, 'ssm', 'frame_ssm.csv'), dtype={'lead_map': str, 'lead_path': str})
        f['split'] = split
        f['max_speed'] = np.nan
        f['passed'] = False
        for name, g in f[f['lead_map'].notna()].groupby('segment'):
            o = pd.read_parquet(os.path.join(tables, split, name, 'objects.parquet'),
                                columns=['frame_idx', 'id', 'cx', 'length', 'speed_x', 'speed_y'])
            o['speed'] = np.hypot(o['speed_x'], o['speed_y'])
            mx = o.groupby('id')['speed'].max()
            f.loc[g.index, 'max_speed'] = g['lead_map'].map(mx).to_numpy()
            rx = rear_x(tables + '/' + split, name)
            front = (o['cx'] + o['length'] / 2).to_numpy()
            fi, ids = o['frame_idx'].to_numpy(), o['id'].to_numpy()
            behind = pd.Series(front < rx)
            first_behind = pd.DataFrame({'id': ids, 'k': fi})[behind.to_numpy()].groupby('id')['k'].max()
            f.loc[g.index, 'passed'] = [bool(i in first_behind.index and first_behind[i] > k)
                                        for i, k in zip(g['lead_map'], g['frame_idx'])]
        parts.append(f)
    f = pd.concat(parts, ignore_index=True)
    lead = f['lead_map'].notna()
    slow = lead & (f['v_lead'] < V_SLOW)
    moves = f['max_speed'] > V_SLOW
    same = lead & (f['lead_map'] == f['lead_path'])
    f['removed'] = slow & ~moves & ~same
    f['slow'] = slow
    edge = f['edge_frame'].astype(bool)
    all9 = np.logical_and.reduce([has_value(f, s) for s in ALL9])
    all7 = np.logical_and.reduce([has_value(f, s) for s in LEAD])
    for title, m in (('validation', f.split == 'validation'), ('training', f.split == 'training'), ('1,000 segments', f.split.notna())):
        x = f[m]
        r = x[x.removed]
        print(f'\n=== {title}: frames with a map lead {int(x.lead_map.notna().sum())}, lead slower than {V_SLOW} m/s '
              f'{int(x.slow.sum())}; of those, kept because it moves in the segment {int((x.slow & moves[m]).sum())}, '
              f'kept because the ego-path pairing picks it {int((x.slow & ~moves[m] & same[m]).sum())}')
        print(f'  removed by the trial rule: {len(r)} frames ({int((~edge[m] & x.removed).sum())} not edge) in '
              f'{r.segment.nunique()} segments, {r[["segment", "lead_map"]].drop_duplicates().shape[0]} vehicles; '
              f'the ego later drives past the lead in {int(r.passed.sum())}; map fallback (no ego lane) '
              f'{int(r.map_fallback.sum())}')
        nn = ~edge[m]
        print(f'  not edge frames with all 9 SSMs: {int((all9[m] & nn).sum())} -> {int((all9[m] & nn & ~x.removed).sum())};'
              f' with the 7 lead SSMs: {int((all7[m] & nn).sum())} -> {int((all7[m] & nn & ~x.removed).sum())};'
              f' of them TTC < 3 s: {int((all7[m] & nn & (x.TTC < 3)).sum())} -> '
              f'{int((all7[m] & nn & (x.TTC < 3) & ~x.removed).sum())}; all 9 and TTC < 3 s: '
              f'{int((all9[m] & nn & (x.TTC < 3)).sum())} -> {int((all9[m] & nn & (x.TTC < 3) & ~x.removed).sum())}')
    a = pd.read_csv(acc)
    t = a[['name', 'segment', 'frame_idx']].merge(f, on=['segment', 'frame_idx'], how='left')
    t['verdict'] = np.where(t.lead_map.isna(), 'no lead (centre rule)', np.where(t.removed, 'removed by the trial',
                                                                                     'kept'))
    pd.set_option('display.width', 250)
    print('\nacceptance frames (centre rule leads, then the trial rule):')
    print(t[['name', 'lead_map', 'lead_path', 'v_lead', 'max_speed', 'passed', 'TTC', 'verdict']]
          .assign(lead_map=t.lead_map.str[:6], lead_path=t.lead_path.str[:6])
          .to_string(index=False, float_format=lambda x: f'{x:.3f}'))


if __name__ == '__main__':
    main()
