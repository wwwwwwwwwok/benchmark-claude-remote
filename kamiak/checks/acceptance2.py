"""Second acceptance batch for the final lead rules (user, 2026-10-22), seed 20261022, one frame per segment as far as
possible:
  R: 10 random frames whose lead the slow-lead rule dropped (from slow_rule_breakdown.py's list), as many as possible
     (up to 5) of them "moves, but the ego drives past it"; the red box is the dropped lead;
  S: 10 random frames whose lead is slower than 0.5 m/s and kept (not edge, not reversing);
  T: the 5 frames with the smallest TTC (not edge, not reversing).
For each a FRONT camera sheet (1 s before, now, 1/2/3 s after, up to the end of the segment; front_sequence.make_sheet)
and a row in frames.csv. Claude does not open the sheets.

usage: python -I acceptance2.py <official results root> <dropped.csv> <extracted root> <out dir>
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser('~/benchmark-claude-remote/code'))
import wod_common as wc  # noqa: E402
from front_sequence import make_sheet  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
SEED = 20261022


def pick(cand, n, rng, used):
    segs = [s for s in cand['segment'].unique() if s not in used]
    rng.shuffle(segs)
    rows = []
    for s in segs[:n]:
        g = cand[cand['segment'] == s]
        rows.append(g.iloc[rng.integers(len(g))])
        used.add(s)
    if len(rows) < n:
        rest = cand.drop(index=[r.name for r in rows])
        rows += [rest.loc[i] for i in rng.choice(rest.index, min(n - len(rows), len(rest)), replace=False)]
    return pd.DataFrame(rows)


def main():
    res, dropped_csv, ext, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    f = pd.concat([pd.read_csv(os.path.join(res, s, 'ssm', 'frame_ssm.csv'), dtype={'lead_map': str, 'lead_path': str})
                   .assign(split=s) for s in ('validation', 'training')], ignore_index=True)
    ok = ~f['edge_frame'].astype(bool) & ~f['reversing_ego'].astype(bool) & ~f['reversing_lead'].astype(bool)
    dr = pd.read_csv(dropped_csv)
    dr = dr[~dr['edge_frame']]
    used = set()
    passed = dr[dr['moved_inner'] & dr['passed']]
    R1 = pick(passed, min(5, passed['segment'].nunique()), rng, used)
    R2 = pick(dr.drop(index=R1.index), 10 - len(R1), rng, used)
    R = pd.concat([R1, R2]).assign(group='R_dropped', box=lambda x: x['lead'])
    S = pick(f[ok & f['lead_map'].notna() & (f['v_lead'] < wc.LEAD_SLOW_SPEED)], 10, rng, used)
    S = S.assign(group='S_slow_kept', box=S['lead_map'])
    t = f[ok & np.isfinite(f['TTC'])].sort_values('TTC')
    T = []
    for _, r in t.iterrows():
        if r['segment'] not in used:
            T.append(r)
            used.add(r['segment'])
        if len(T) == 5:
            break
    T = pd.DataFrame(T)
    T = T.assign(group='T_smallest_TTC', box=T['lead_map'])
    sel = pd.concat([R, S, T], ignore_index=True)
    sel.insert(0, 'name', [f'{g[0]}{i + 1:02d}_{s[:8]}_f{int(k):03d}' for i, (g, s, k) in
                           enumerate(zip(sel['group'], sel['segment'], sel['frame_idx']))])
    fr = f.set_index(['segment', 'frame_idx'])
    for c in ('lead_map', 'lead_path', 'TTC', 'v_lead', 'v_ego', 'gap', 'ego_speed'):
        sel['now_' + c] = [fr.at[(s, int(k)), c] for s, k in zip(sel['segment'], sel['frame_idx'])]
    keep = ['name', 'group', 'split', 'segment', 'frame_idx', 'box', 'reason', 'lead_speed', 'TTC', 'moved_inner', 'passed',
            'now_lead_map', 'now_lead_path', 'now_TTC', 'now_v_lead', 'now_gap', 'now_ego_speed']
    sel = sel[[c for c in keep if c in sel.columns]]
    sel.to_csv(os.path.join(out, 'frames.csv'), index=False)
    for (split, name), g in sel.groupby(['split', 'segment'], sort=False):
        seg = wc.Segment(os.path.join(ext, split), name)
        for _, r in g.iterrows():
            head = (f'{r["group"]}: ' + (f'dropped lead ({r["reason"]}), its speed {r["lead_speed"]:.2f} m/s, TTC '
                                         f'{r["TTC"]:.2f} s' if r['group'] == 'R_dropped' else
                                         f'lead {r["now_v_lead"]:.2f} m/s, gap {r["now_gap"]:.2f} m, TTC {r["now_TTC"]:.3f} s'))
            make_sheet(seg, int(r['frame_idx']), r['box'], os.path.join(out, r['name'] + '.jpg'), head)
    pd.set_option('display.width', 250)
    print(sel.assign(segment=sel.segment.str[:12], box=sel.box.str[:6], now_lead_map=sel.now_lead_map.str[:6],
                     now_lead_path=sel.now_lead_path.str[:6]).to_string(index=False, float_format=lambda x: f'{x:.3f}'))


if __name__ == '__main__':
    main()
