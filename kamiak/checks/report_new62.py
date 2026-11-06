"""Read-only (user, 2026-11-06, item 2.1): why the comparison list's 249 frames of the 62 segments outside the main list
were not in the main experiment, and how they compare with the main result's 1,065 fit frames.
Why: the main list (code/make_vlm_frame_lists.py) keeps the target frames where all nine SSMs have a value, the
comparison list those where the seven lead-vehicle SSMs have one; so these frames lack 2D-TTC or ACT (or both). Counted
on the list's own columns (a value = finite), and over every target frame of the 62 segments in frame_ssm (target frames
as code/make_vlm_inputs.py: frame_idx divisible by 5, >= 20, not edge_frame): how many have all nine (0 expected).
Compared (quantiles 10/25/50/75/90 %, mean; for the counts also the share of frames with none): the true ego speed,
closing speed (ego - lead) and distance (truth csv, quantity_truth.py), the pedestrians + cyclists and vehicles within
10/20/30/50 m (count_nearby.py), and closing speed below 0.1 m/s (the main result leaves those out).
usage: python -I report_new62.py <repository root> <frame_ssm.csv> <main truth csv> <new truth csv> <main nearby csv>
       <new nearby csv> <D's new cues csv (main)>"""
import os
import sys

import numpy as np
import pandas as pd

ROOT, FS, TM, TN, NM, NN, DC = sys.argv[1:8]
SSM9 = ['TTC', 'THW', 'DRAC', 'MTTC', 'CI', 'PICUD', 'PSD', 'TTC2D', 'ACT']
LEAD7 = SSM9[:7]
c = pd.read_csv(os.path.join(ROOT, 'vlm_frames', 'comparison_lead7_targets.csv'), comment='#')
m = pd.read_csv(os.path.join(ROOT, 'vlm_frames', 'main_all9_targets.csv'), comment='#')
new = c[~c.segment.isin(set(m.segment))]
fin = lambda df, cols: np.isfinite(df[cols].to_numpy(float))  # noqa: E731
print(f'== why not in the main list: {len(new)} frames in {new.segment.nunique()} segments')
print(f'  lead-vehicle SSMs (7) all with a value: {int(fin(new, LEAD7).all(1).sum())}; 2D-TTC without a value '
      f'{int((~fin(new, ["TTC2D"])[:, 0]).sum())}, ACT without {int((~fin(new, ["ACT"])[:, 0]).sum())}, both without '
      f'{int((~fin(new, ["TTC2D", "ACT"])).all(1).sum())}')
sys.path.insert(0, os.path.join(ROOT, 'code'))
import make_vlm_frame_lists as mfl  # noqa: E402  (its own "has a value": DRAC, CI > 0; no lead SSM when reversing)
fs = pd.read_csv(FS, low_memory=False)
fs = fs[fs.segment.isin(set(new.segment))]
tg = fs[(fs.frame_idx % 5 == 0) & (fs.frame_idx >= 20) & (~fs.edge_frame.astype(bool))]
hv = {s: mfl.has_value(tg, s) for s in mfl.ALL9}
all9 = np.column_stack([hv[s] for s in mfl.ALL9]).all(1)
all7 = np.column_stack([hv[s] for s in mfl.LEAD]).all(1)
print(f'  every target frame of these segments in frame_ssm: {len(tg)}; with all nine SSMs (make_vlm_frame_lists.'
      f'has_value) {int(all9.sum())}; with the seven lead-vehicle ones {int(all7.sum())} (the list has {len(new)})')
print('  of the target frames, the ones without a value, per SSM: ' + ', '.join(
    f'{s} {int((~hv[s]).sum())}' for s in mfl.ALL9) + f'; with no lead-vehicle SSM at all (no lead) '
    f'{int((~np.column_stack([hv[s] for s in mfl.LEAD]).any(1)).sum())}')
print(f'  closing speed below 0.1 m/s (left out of the main result): {int(new.closing_below_0_1.astype(bool).sum())}')
m84 = pd.read_csv(TM)
d = pd.read_csv(DC)
NEW_D = ['speed_limit_mps', 'dist_crosswalk', 'dist_stop_sign', 'dist_lane_crossing']
fit = m.merge(d, on=['segment', 'frame_idx'])
fit = fit[~fit.closing_below_0_1.astype(bool) & ~fit[NEW_D].isna().any(axis=1)][['segment', 'frame_idx']]
A = fit.merge(m84, on=['segment', 'frame_idx']).merge(pd.read_csv(NM), on=['segment', 'frame_idx'])
B = new[['segment', 'frame_idx']].merge(pd.read_csv(TN), on=['segment', 'frame_idx']) \
    .merge(pd.read_csv(NN), on=['segment', 'frame_idx'])
print(f'\n== main result fit frames {len(A)} ({A.segment.nunique()} segments) against the new frames {len(B)} '
      f'({B.segment.nunique()} segments): 10% / 25% / median / 75% / 90%, mean')
for df in (A, B):
    df['closing'] = df['v_ego'] - df['v_lead']
rows = [('ego speed (m/s)', 'v_ego'), ('closing speed (m/s)', 'closing'), ('distance (m)', 'distance')] + \
    [(f'{w} within {r} m', f'n_{k}_{r}m') for r in (10, 20, 30, 50) for w, k in (('ped + cyc', 'ped_cyc'),
                                                                                 ('vehicles', 'vehicle'))]
for label, col in rows:
    out = []
    for df in (A, B):
        v = df[col].to_numpy(float)
        q = np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])
        z = f', none {np.mean(v == 0) * 100:.1f} %' if col.startswith('n_') else ''
        out.append(' / '.join(f'{x:.2f}' for x in q) + f', mean {v.mean():.2f}{z}')
    print(f'  {label:22s} main {out[0]}\n  {"":22s} new  {out[1]}')
