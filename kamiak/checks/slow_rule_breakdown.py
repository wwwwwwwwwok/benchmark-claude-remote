"""The slow-lead rule (2026-10-22) in the official run, against the run without it (centre-based overlap, provisional on
scratch) and against the trial computed before (trial_slow_rule.py: a slow lead stays if the path pairing picks it or
it is faster than 0.5 m/s in any frame of its track).
Checks that the official run differs from the provisional one only by the frames the rule should drop (every other row
identical as text), and reports how many frames the rule drops; how many more than the trial, split into "the ego
drives past it later" and "it only moves at the ends of its track"; and writes the dropped frames with their reasons.

usage: python -I slow_rule_breakdown.py <provisional root> <official root> <tables root> <out csv>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
V = 0.5
HALF = 9


def rear_x(path):
    with open(path, encoding='utf-8') as f:
        ctx = json.load(f)
    return {c['name']: np.asarray(c['extrinsic'], float).reshape(4, 4)[0, 3] for c in ctx['laser_calibrations']}['REAR']


def facts(tables, split, seg):
    o = pd.read_parquet(os.path.join(tables, split, seg, 'objects.parquet'),
                        columns=['frame_idx', 'id', 'cx', 'length', 'speed_x', 'speed_y'])
    sp = np.hypot(o['speed_x'], o['speed_y'])
    span = o.groupby('id')['frame_idx'].agg(['min', 'max'])
    lo, hi = o['id'].map(span['min']), o['id'].map(span['max'])
    inner = (o['frame_idx'] - lo >= HALF) & (hi - o['frame_idx'] >= HALF)
    moved_all = set(o.loc[sp > V, 'id'])
    moved_inner = set(o.loc[inner & (sp > V), 'id'])
    rx = rear_x(os.path.join(tables, split, seg, 'context.json'))
    last_behind = o[o['cx'] + o['length'] / 2 < rx].groupby('id')['frame_idx'].max().to_dict()
    return moved_all, moved_inner, last_behind


def main():
    prov_root, off_root, tables, out = sys.argv[1:5]
    rows, bad = [], 0
    for split in ('validation', 'training'):
        rd = lambda r: pd.read_csv(os.path.join(r, split, 'ssm', 'frame_ssm.csv'), dtype=str, keep_default_na=False)
        p, o = rd(prov_root), rd(off_root)
        if not p[['segment', 'frame_idx']].equals(o[['segment', 'frame_idx']]):
            sys.exit(f'{split}: frames differ')
        p['v'] = pd.to_numeric(p['v_lead'].replace('', np.nan))
        lead = p['lead_map'] != ''
        drop = pd.Series(False, index=p.index)
        info = {}
        for seg, g in p[lead & (p['v'] < V)].groupby('segment'):
            ma, mi, lb = facts(tables, split, seg)
            for i, r in g.iterrows():
                k, lid = int(r['frame_idx']), r['lead_map']
                same = r['lead_path'] == lid
                passed = lb.get(lid, -1) > k
                keep = same or (lid in mi and not passed)
                drop[i] = not keep
                info[i] = dict(same=same, moved_any=lid in ma, moved_inner=lid in mi, passed=passed)
        # the official run must equal the provisional one except in the dropped frames
        changed = (p.drop(columns=['v']) != o).any(axis=1)
        unexpected = changed & ~drop
        missing = drop & (o['lead_map'] != '')
        bad += int(unexpected.sum()) + int(missing.sum())
        print(f'{split}: rows changed {int(changed.sum())}, dropped by the rule {int(drop.sum())}; changed but not '
              f'dropped {int(unexpected.sum())}; dropped but still with a lead {int(missing.sum())}')
        if unexpected.any():
            c = (p.drop(columns=['v']) != o)[unexpected]
            print('  columns differing in unexpected rows:', c.columns[c.any()].tolist())
        for i in drop.index[drop]:
            r = p.loc[i]
            rows.append(dict(split=split, segment=r['segment'], frame_idx=int(r['frame_idx']), lead=r['lead_map'],
                             lead_speed=r['v'], TTC=float(r['TTC'] or 'nan'), edge_frame=r['edge_frame'] == 'True',
                             **info[i]))
    d = pd.DataFrame(rows)
    d['trial_dropped'] = ~d['same'] & ~d['moved_any']
    d['reason'] = np.where(~d['moved_any'], 'never moves', np.where(~d['moved_inner'] & d['passed'],
                           'moves only at its track ends, and passed', np.where(~d['moved_inner'],
                           'moves only at its track ends', 'moves, but the ego drives past it')))
    d.to_csv(out, index=False)
    print(f'\nconsistency: {"OK" if bad == 0 else f"{bad} rows not as expected"}')
    for title, x in (('validation', d[d.split == 'validation']), ('training', d[d.split == 'training']),
                     ('1,000 segments', d)):
        extra = x[~x.trial_dropped]
        print(f'\n=== {title}: dropped {len(x)} frames ({int((~x.edge_frame).sum())} not edge) in {x.segment.nunique()} '
              f'segments, {x[["segment", "lead"]].drop_duplicates().shape[0]} vehicles; the ego later drives past the '
              f'lead in {int(x.passed.sum())}')
        print(f'  as in the trial (never faster than 0.5 m/s, path pairing not the same): {int(x.trial_dropped.sum())}')
        print(f'  more than the trial: {len(extra)} ({int((~extra.edge_frame).sum())} not edge): moves away from its '
              f'track ends but the ego drives past it {int((extra.moved_inner & extra.passed).sum())}; moves only within '
              f'9 frames of its track ends {int((~extra.moved_inner & ~extra.passed).sum())}; both (moves only at the '
              f'ends, and passed) {int((~extra.moved_inner & extra.passed).sum())}')
        print('  reasons: ' + ', '.join(f'{k} {v}' for k, v in x.reason.value_counts().items()))


if __name__ == '__main__':
    main()
