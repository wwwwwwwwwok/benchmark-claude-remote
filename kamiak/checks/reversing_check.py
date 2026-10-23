"""How often the ego or the lead vehicle moves backwards (2026-10-21). The lead-vehicle SSMs use speed magnitudes
(wod_common.speed_and_accel), so a vehicle reversing still gets a positive speed: an ego backing away from its lead gets a
positive closing speed and a finite TTC although the gap grows; a lead reversing towards the ego gets none.
"Moving backwards": speed >= 0.5 m/s (below it the velocity direction is noise, as in speed_and_accel) and direction of
travel more than 90 deg from the heading. Ego velocity = the Savitzky-Golay estimate used for the SSMs; lead velocity =
the label's speed (global frame), lead heading = box heading + ego yaw; the lead is lead_map of frame_ssm.csv.
"Has a value" as in analyze_vlm_ssm.py: TTC, MTTC finite; DRAC > 0.
Reported separately for the ego and the lead (user, 2026-10-21): frames and segments; of them, frames with a lead and
frames with a TTC / MTTC / DRAC value; the same for target frames (every 0.5 s, frame_idx >= 20, not edge frames).

usage: python -I reversing_check.py <extracted root> <frame_ssm.csv> [<extracted root> <frame_ssm.csv> ...]
       [--list <csv>]   (one pair per split; prints each split and all splits together)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser('~/benchmark-claude-remote/code'))
from wod_common import Segment, wrap  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
V_MIN = 0.5


def frames_of(root, fcsv):
    f = pd.read_csv(fcsv, dtype={'lead_map': str})
    rows = []
    for name, g in f.groupby('segment', sort=True):
        seg = Segment(root, name)
        v = seg.ego_v[:, :2]
        sp = np.hypot(v[:, 0], v[:, 1])
        ang = np.degrees(np.abs(wrap(np.arctan2(v[:, 1], v[:, 0]) - seg.yaw)))
        ego_back = (sp >= V_MIN) & (ang > 90)
        o = seg.objects.set_index(['frame_idx', 'id'])
        for r in g.itertuples():
            k = int(r.frame_idx)
            lead_back, lead_speed = False, np.nan
            if isinstance(r.lead_map, str):
                x = o.loc[(k, r.lead_map)]
                lv = np.array([x['speed_x'], x['speed_y']], float)
                lead_speed = float(np.hypot(*lv))
                lh = x['heading'] + seg.yaw[k]
                lead_back = bool(lead_speed >= V_MIN and abs(wrap(np.arctan2(lv[1], lv[0]) - lh)) > np.pi / 2)
            rows.append(dict(segment=name, frame_idx=k, ego_speed=sp[k], lead_speed=lead_speed,
                             ego_back=bool(ego_back[k]), lead_back=lead_back, has_lead=isinstance(r.lead_map, str),
                             edge=bool(r.edge_frame), TTC=r.TTC, MTTC=r.MTTC, DRAC=r.DRAC, closing=r.closing,
                             gap=r.gap))
    d = pd.DataFrame(rows)
    d['target'] = ~d['edge'] & (d['frame_idx'] % 5 == 0) & (d['frame_idx'] >= 20)
    d['TTC_value'] = np.isfinite(d['TTC'])
    d['MTTC_value'] = np.isfinite(d['MTTC'])
    d['DRAC_value'] = np.nan_to_num(d['DRAC'].to_numpy(float), nan=0.0) > 0
    return d


def report(d, title):
    print(f'\n=== {title}: {d["segment"].nunique()} segments, {len(d)} frames, target frames {int(d["target"].sum())}')
    print(f'  {"":10s} {"frames":>13s} {"segments":>8s} {"with lead":>9s} {"TTC":>5s} {"MTTC":>5s} {"DRAC":>5s} | '
          f'{"target frames":>13s} {"with lead":>9s} {"TTC":>5s} {"MTTC":>5s} {"DRAC":>5s}')
    for who, lab in (('ego_back', 'ego'), ('lead_back', 'lead')):
        b = d[d[who]]
        t = b[b['target']]
        cnt = lambda x: [int(x['has_lead'].sum()), int(x['TTC_value'].sum()), int(x['MTTC_value'].sum()),
                         int(x['DRAC_value'].sum())]
        a1, a2 = cnt(b), cnt(t)
        print(f'  {lab + " back":10s} {len(b):13d} {b["segment"].nunique():8d} {a1[0]:9d} {a1[1]:5d} {a1[2]:5d} '
              f'{a1[3]:5d} | {len(t):13d} {a2[0]:9d} {a2[1]:5d} {a2[2]:5d} {a2[3]:5d}')
    for who in ('ego_back', 'lead_back'):
        b = d[d[who]]
        if len(b):
            print(f'  {who} per segment (frames / with a TTC value / target frames): ' +
                  ', '.join(f'{s} {len(g)}/{int(g["TTC_value"].sum())}/{int(g["target"].sum())}'
                            for s, g in b.groupby('segment')))


def main():
    args = sys.argv[1:]
    out = None
    if '--list' in args:
        i = args.index('--list')
        out = args[i + 1]
        args = args[:i] + args[i + 2:]
    parts = []
    for root, fcsv in zip(args[0::2], args[1::2]):
        d = frames_of(root, fcsv)
        d['split'] = os.path.basename(os.path.normpath(root))
        report(d, d['split'].iloc[0])
        parts.append(d)
    if len(parts) > 1:
        report(pd.concat(parts, ignore_index=True), 'all splits')
    if out:
        a = pd.concat(parts, ignore_index=True)
        a[a['ego_back'] | a['lead_back']].to_csv(out, index=False)
        print(f'\nreversing frames written to {out}')


if __name__ == '__main__':
    main()
