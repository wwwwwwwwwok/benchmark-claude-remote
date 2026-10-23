"""Acceptance checks of an analyze.sh output folder pair (user, 2026-10-22): frame counts, every condition x SSM has a
result in every grouping (tau-b, its interval, Spearman, agreement), the SSM differences have intervals, no AUC.
usage: python -I main_run_accept.py <analysis folder> <expected frames>
"""
import os
import sys

import pandas as pd

d, n_want = sys.argv[1], int(sys.argv[2])
ag = pd.read_csv(os.path.join(d, 'agreement.csv'))
dif = pd.read_csv(os.path.join(d, 'ssm_differences.csv'))
sf = pd.read_csv(os.path.join(d, 'ssm_target_frames.csv'))
fs = pd.read_csv(os.path.join(d, 'vlm_frame_scores.csv'))
print(f'{os.path.basename(d)}: ssm_target_frames {len(sf)} rows, {sf[["segment", "frame_idx"]].drop_duplicates().shape[0]} '
      f'frames, {sf.segment.nunique()} segments; vlm_frame_scores {len(fs)} rows; expected frames {n_want}')
print(f'  agreement rows {len(ag)}; groupings {sorted(ag.grouping.unique())}; conditions {ag.condition.nunique()}; '
      f'SSMs {ag.ssm.nunique()}; n_conflict values {sorted(ag.n_conflict.unique())}')
cols = ['tau_b_all', 'tau_b_conflict', 'tau_b_conflict_lo', 'tau_b_conflict_hi', 'spearman_conflict',
        'agreement_conflict']
for g, sub in ag.groupby('grouping'):
    cells = sub.groupby(['condition', 'ssm']).size()
    print(f'  {g:11s}: {len(cells)} condition x SSM cells (rows per cell {sorted(cells.unique())}); '
          f'missing values in {", ".join(c for c in cols if sub[c].isna().any()) or "none of"} {cols}')
print(f'  AUC columns: {[c for c in ag.columns if "auc" in c.lower()]}; ssm_differences stats {sorted(dif.stat.unique())}')
for (g, st), sub in dif.groupby(['grouping', 'stat']):
    print(f'  differences {g:11s} {st}: {len(sub)} rows, missing diff/lo/hi {sub[["diff", "diff_lo", "diff_hi"]].isna().any(axis=1).sum()}, '
          f'nboot {sorted(sub.nboot.unique())}')
