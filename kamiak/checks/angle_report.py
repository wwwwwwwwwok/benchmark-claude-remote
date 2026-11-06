"""Read-only (user, 2026-11-05, check 1 before fitting D): the angle at the lane crossing used for dist_lane_crossing
(new_cues_angle.csv, --with_angle), on the main result's frames and the sensitivity run's: distribution, frames under
30 deg among the frames with a crossing found within 100 m, and the 5 % rule.
usage: python -I angle_report.py <new_cues_angle.csv> <new_cues.csv> <risk analysis folder>"""
import os
import sys

import numpy as np
import pandas as pd

A, N, R = sys.argv[1:4]
an = pd.read_csv(A)
old = pd.read_csv(N)
same = an.drop(columns=['cross_angle_deg']).equals(old)
print(f'new_cues_angle.csv without cross_angle_deg equals new_cues.csv: {same}')
for sub in ('main_1084', 'sensitivity_1123'):
    fr = pd.read_csv(os.path.join(R, sub, 'ssm_target_frames.csv'))[['segment', 'frame_idx']]
    t = fr.merge(an, on=['segment', 'frame_idx'], how='left', validate='one_to_one')
    found = t.dist_lane_crossing < 100
    a_ = t.cross_angle_deg[found].to_numpy(float)
    q = np.quantile(a_, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    lt = int((a_ < 30).sum())
    print(f'{sub}: crossings found within 100 m {int(found.sum())} frames (of {len(t)}; none within 100 m '
          f'{int((t.dist_lane_crossing == 100).sum())}, missing {int(t.dist_lane_crossing.isna().sum())}); angle NaN '
          f'among them {int(np.isnan(a_).sum())}')
    print('  angle (deg) min / 10% / 25% / median / 75% / 90% / max: ' + ' / '.join(f'{v:.1f}' for v in q))
    print(f'  under 30 deg: {lt} = {lt / found.sum() * 100:.1f} % of the frames with a crossing (rule: more than 5 % -> '
          f'the main version counts only crossings of 30 deg or more); under 10 deg {int((a_ < 10).sum())}, 10-20 '
          f'{int(((a_ >= 10) & (a_ < 20)).sum())}, 20-30 {int(((a_ >= 20) & (a_ < 30)).sum())}')
