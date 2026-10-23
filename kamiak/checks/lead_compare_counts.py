"""How the leads changed between the old run (results/) and the lateral-overlap run (results_lead_overlap/), 2026-10-21:
per split and for both together, per pairing method: frames whose lead is unchanged, removed (no lead any more),
switched to another vehicle, or added; over all frames and over frames that are not edge frames in the new run. For the
map lead: how many of the changed frames had a nearly stopped old lead (< 0.5 m/s) and an old TTC < 3 s.

usage: python -I lead_compare_counts.py <old results root> <new results root>
"""
import os
import sys

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')


def main():
    old_root, new_root = sys.argv[1:3]
    parts = []
    for split in ('validation', 'training'):
        rd = lambda r: pd.read_csv(os.path.join(r, split, 'ssm', 'frame_ssm.csv'),
                                   dtype={'lead_map': str, 'lead_path': str, 'lead_straight': str})
        o, n = rd(old_root), rd(new_root)
        if not o[['segment', 'frame_idx']].equals(n[['segment', 'frame_idx']]):
            sys.exit(f'{split}: frames differ')
        d = pd.DataFrame({'split': split, 'segment': n['segment'], 'edge_new': n['edge_frame'].astype(bool),
                          'old_TTC': o['TTC'], 'old_v_lead': o['v_lead']})
        for m in ('map', 'path', 'straight'):
            a, b = o[f'lead_{m}'], n[f'lead_{m}']
            d[m] = 'unchanged'
            d.loc[a.notna() & b.isna(), m] = 'removed'
            d.loc[a.notna() & b.notna() & (a != b), m] = 'switched'
            d.loc[a.isna() & b.notna(), m] = 'added'
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    for title, x in (('validation', d[d.split == 'validation']), ('training', d[d.split == 'training']), ('1,000 segments', d)):
        print(f'\n=== {title}: {len(x)} frames ({int((~x.edge_new).sum())} not edge frames in the new run)')
        for m in ('map', 'path', 'straight'):
            c = x[m].value_counts()
            ce = x.loc[~x.edge_new, m].value_counts()
            print(f'  lead_{m:8s}: ' + ', '.join(f'{k} {int(c.get(k, 0))} ({int(ce.get(k, 0))})'
                                                for k in ('unchanged', 'removed', 'switched', 'added'))
                  + '   (in brackets: not edge frames)')
        ch = x[x['map'].isin(['removed', 'switched'])]
        print(f'  map lead changed: {len(ch)} frames in {ch.segment.nunique()} segments; old lead nearly stopped (< 0.5 m/s) '
              f'{int((ch.old_v_lead < 0.5).sum())}; old TTC < 3 s {int((ch.old_TTC < 3).sum())}, of them not edge '
              f'{int(((ch.old_TTC < 3) & ~ch.edge_new).sum())}')


if __name__ == '__main__':
    main()
