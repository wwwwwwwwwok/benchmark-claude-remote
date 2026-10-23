"""Frames where every SSM has a value (docs/handoff.md section 3, "正式版 VLM 问哪些帧"; user's request 2026-10-21).
  - "Has a value" as in code/analyze_vlm_ssm.py: TTC, THW, MTTC, PICUD, PSD, TTC2D, ACT finite; DRAC, CI > 0.
  - Edge frames (edge_frame) are left out everywhere.
  - Target frames as in code/make_vlm_inputs.py: frame_idx divisible by 5, frame_idx >= 20, not an edge frame.
  - Two sets of SSMs: all 9, and the 7 lead-vehicle ones (TTC, THW, DRAC, MTTC, CI, PICUD, PSD).
  - If the file has reversing_ego / reversing_lead (lateral-overlap run, 2026-10-21): the 7 lead-vehicle SSMs have no
    value where the ego or the lead reverses (left out, user's decision); TTC2D and ACT keep those frames.
  - Robustness check (user, 2026-10-21): frames where the map and the ego-path pairings pick the same vehicle.
For each set x {all frames, target frames}: frames, segments, runs (consecutive frames of one segment: frame_idx steps of 1
for all frames, of 5 for target frames), lead vehicles (segment + lead_map id), Kendall tau-b between the SSMs over these
frames (every SSM turned to larger = riskier, as in analyze_vlm_ssm.py), TTC range and the frames with TTC < 3 s, < 5 s.

usage: python -I ssm_value_frames.py <frame_ssm.csv> [<frame_ssm.csv> ...] [--segments <name> ...] [--out <dir>]
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.stdout.reconfigure(encoding='utf-8')
SSMS = ['TTC', 'THW', 'DRAC', 'MTTC', 'CI', 'PICUD', 'PSD', 'TTC2D', 'ACT']
LEAD_SSMS = ['TTC', 'THW', 'DRAC', 'MTTC', 'CI', 'PICUD', 'PSD']
SMALLER_RISKIER = {'TTC', 'THW', 'MTTC', 'PICUD', 'PSD', 'TTC2D', 'ACT'}
POSITIVE_IS_VALUE = {'DRAC', 'CI'}


CLOSING_MIN = 0.0    # set by --closing_min: below it TTC and DRAC have no value (user's rule of 2026-10-22, 0.1 m/s)


def has_value(df, s):
    v = df[s].to_numpy(float)
    val = (np.nan_to_num(v, nan=0.0) > 0) if s in POSITIVE_IS_VALUE else np.isfinite(v)
    if s in LEAD_SSMS and 'reversing_ego' in df:
        val &= ~(df['reversing_ego'].astype(bool) | df['reversing_lead'].astype(bool)).to_numpy()
    if s in ('TTC', 'DRAC') and CLOSING_MIN > 0:
        c = df['closing'].to_numpy(float)
        val &= ~((c > 0) & (c < CLOSING_MIN))
    return val


def same_lead(df):
    return (df['lead_map'].notna() & (df['lead_map'] == df['lead_path'])).to_numpy()


def runs(df, step):
    n = 0
    for _, g in df.groupby('segment'):
        k = np.sort(g['frame_idx'].to_numpy())
        n += 1 + int((np.diff(k) != step).sum())
    return n


def tau_matrix(df, ssms):
    m = pd.DataFrame(np.nan, index=ssms, columns=ssms)
    for a, b in itertools.combinations(ssms, 2):
        x = -df[a].to_numpy(float) if a in SMALLER_RISKIER else df[a].to_numpy(float)
        y = -df[b].to_numpy(float) if b in SMALLER_RISKIER else df[b].to_numpy(float)
        t = kendalltau(x, y, variant='b').statistic if len(df) >= 2 else np.nan
        m.loc[a, b] = m.loc[b, a] = t
    for s in ssms:
        m.loc[s, s] = 1.0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', nargs='+')
    ap.add_argument('--segments', nargs='*', help='only these segments')
    ap.add_argument('--out', help='also write the tau-b matrices and the frame lists here')
    ap.add_argument('--closing_min', type=float, default=0.0,
                    help='m/s: TTC and DRAC have no value below this closing speed (0 = off)')
    a = ap.parse_args()
    global CLOSING_MIN
    CLOSING_MIN = a.closing_min
    print(f'closing speed rule: TTC and DRAC without a value below {CLOSING_MIN} m/s' if CLOSING_MIN > 0 else
          'closing speed rule: off')
    parts = []
    for p in a.csv:
        d = pd.read_csv(p, dtype={'lead_map': str, 'lead_path': str})
        d['source'] = p
        parts.append(d)
    f = pd.concat(parts, ignore_index=True)
    if a.segments:
        missing = sorted(set(a.segments) - set(f['segment']))
        if missing:
            sys.exit(f'segments not in the files: {missing}')
        f = f[f['segment'].isin(a.segments)].reset_index(drop=True)
    if f.duplicated(['segment', 'frame_idx']).any():
        sys.exit('a (segment, frame_idx) appears twice')
    print(f'{len(f)} frames, {f["segment"].nunique()} segments, from {len(a.csv)} file(s)')
    edge = f['edge_frame'].astype(bool)
    ok = f[~edge].copy()
    tgt = ok[(ok['frame_idx'] % 5 == 0) & (ok['frame_idx'] >= 20)].copy()
    print(f'not edge frames: {len(ok)} (edge frames left out: {int(edge.sum())}); target frames: {len(tgt)}')
    if 'reversing_ego' in f:
        rev = ok['reversing_ego'].astype(bool) | ok['reversing_lead'].astype(bool)
        print(f'reversing (not edge frames): ego {int(ok["reversing_ego"].sum())}, lead {int(ok["reversing_lead"].sum())}, '
              f'either {int(rev.sum())} frames, left out of the lead-vehicle SSMs; of them target frames '
              f'{int((tgt["reversing_ego"].astype(bool) | tgt["reversing_lead"].astype(bool)).sum())}')
    for name, d in (('all frames', ok), ('target frames', tgt)):
        print(f'  {name}: frames with a value per SSM: ' +
              ', '.join(f'{s} {int(has_value(d, s).sum())}' for s in SSMS) +
              f'; frames with a lead (map pairing): {int(d["lead_map"].notna().sum())}, of them the ego-path pairing '
              f'picks the same vehicle in {int(same_lead(d).sum())}')
    if a.out:
        os.makedirs(a.out, exist_ok=True)

    for set_name, ssms in (('all 9 SSMs', SSMS), ('the 7 lead-vehicle SSMs', LEAD_SSMS)):
        print(f'\n===== frames where {set_name} have a value =====')
        for name, d, step in (('all frames', ok, 1), ('target frames', tgt, 5)):
            sel = np.logical_and.reduce([has_value(d, s) for s in ssms])
            v = d[sel]
            leads = v[['segment', 'lead_map']].drop_duplicates()
            print(f'\n--- {name}: {len(v)} / {len(d)} ({len(v) / len(d):.2%})')
            if not len(v):
                continue
            by_src = v.groupby('source').size()
            if len(by_src) > 1:
                print('  by file: ' + ', '.join(f'{os.path.basename(os.path.dirname(os.path.dirname(k)))} {n}'
                                               for k, n in by_src.items()))
            print(f'  segments {v["segment"].nunique()} (of {d["segment"].nunique()}), runs {runs(v, step)}, '
                  f'lead vehicles {len(leads)}; the ego-path pairing picks the same lead in {int(same_lead(v).sum())}')
            ttc = v['TTC'].to_numpy(float)
            print(f'  TTC: {np.nanmin(ttc):.3f} - {np.nanmax(ttc):.3f} s, median {np.nanmedian(ttc):.3f} s; '
                  f'< 3 s: {int((ttc < 3).sum())} frames, < 5 s: {int((ttc < 5).sum())} frames')
            m = tau_matrix(v, ssms)
            print('  Kendall tau-b (larger = riskier for every SSM):')
            print('  ' + m.to_string(float_format=lambda x: f'{x:.3f}').replace('\n', '\n  '))
            if a.out:
                tag = f'{"9" if ssms is SSMS else "7"}ssm_{"all" if step == 1 else "targets"}'
                m.to_csv(os.path.join(a.out, f'tau_b_{tag}.csv'))
                v.drop(columns='source').to_csv(os.path.join(a.out, f'frames_{tag}.csv'), index=False)


if __name__ == '__main__':
    main()
