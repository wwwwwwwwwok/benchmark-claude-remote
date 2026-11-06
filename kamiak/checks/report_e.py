"""Read-only (user, 2026-11-06, E): per new quantity of new_cues_e.py, on the main result's 1,084 frames, on the 1,065
fit frames of D (every cue of D known) and on all 1,873 frames of the comparison list: minimum, quartiles, maximum,
frames recorded as 100 (dist_merge), missing, nonzero; and the share of the sum of squares between segments
(analyze_rules.between_share on the rank / n, ties the mean rank, the frames with a value; as report_new_cues.py did for
D), with the random reference (segments - 1) / (frames - 1). Also the frames whose heading change over the next 3 s is
missing (the user: more than 20 in the main result -> report first), and the params' checks.
usage: python -I report_e.py <new_cues_e.csv> <risk analysis folder (after_qwen3.5-4b)> <D's new cues csv> <code folder>"""
import os
import sys

import numpy as np
import pandas as pd

E, RISK, DC, CODE = sys.argv[1:5]
sys.path.insert(0, CODE)
import analyze_rules as ar  # noqa: E402
import new_cues as nc  # noqa: E402
import new_cues_e as ne  # noqa: E402

e = pd.read_csv(E)
print(open(E[:-4] + '_params.txt', encoding='utf-8').read())
main = pd.read_csv(os.path.join(RISK, 'main_1084', 'ssm_target_frames.csv'))[['segment', 'frame_idx']]
d = pd.read_csv(DC)
fitk = main.merge(d[~d[nc.NEW].isna().any(axis=1)][['segment', 'frame_idx']], on=['segment', 'frame_idx'])  # main's
sets = {'main 1,084': main, 'fit 1,065': fitk, 'all 1,873': e[['segment', 'frame_idx']]}
for name, fr in sets.items():
    t = fr.merge(e, on=['segment', 'frame_idx'], how='left', validate='one_to_one')
    nseg = t.segment.nunique()
    print(f'===== {name}: {len(t)} frames in {nseg} segments; random reference {(nseg - 1) / (len(t) - 1):.3f}; ego on '
          f'no mapped lane {int(t.ego_lane.isna().sum())}')
    for c in ne.NEW:
        v = t[c].to_numpy(float)
        ok = ~np.isnan(v)
        q = np.quantile(v[ok], [0, 0.25, 0.5, 0.75, 1]) if ok.any() else [np.nan] * 5
        bs = ar.between_share(ar.rank01(v[ok]), np.unique(t.segment[ok], return_inverse=True)[1]) if ok.sum() > 1 \
            else np.nan
        extra = f' | at 100: {int((v[ok] == 100).sum()):4d}' if c == 'dist_merge' else ''
        print(f'  {c:34s} min {q[0]:8.3f} q1 {q[1]:8.3f} median {q[2]:8.3f} q3 {q[3]:8.3f} max {q[4]:8.3f}{extra} | '
              f'missing {int((~ok).sum()):4d} | nonzero {int((v[ok] != 0).sum()):4d} | between segments {bs:.3f}')
    m7 = t[t.heading_change_next3s_deg.isna()]
    print(f'  heading change next 3 s missing: {len(m7)} frames in {m7.segment.nunique()} segments'
          + (' (more than 20: the user asked to be told first)' if name == 'main 1,084' and len(m7) > 20 else ''))
    if name != 'all 1,873':
        for R in nc.RADII:
            print(f'    R {R} m: pedestrians + cyclists on the road / all {int(t[f"n_ped_cyc_road_{R}m"].sum())} / '
                  f'{int(t[f"n_ped_cyc_{R}m"].sum())}; stationary vehicles in a lane / all '
                  f'{int(t[f"n_vehicle_stationary_inlane_{R}m"].sum())} / {int(t[f"n_vehicle_stationary_{R}m"].sum())}')
print('===== the 750 frames outside the main list: side vehicles toward the lane / beside '
      f'{int(e[~e.in_main.astype(bool)].n_side_vehicle_toward.sum())} / {int(e[~e.in_main.astype(bool)].n_side_vehicle.sum())}')
