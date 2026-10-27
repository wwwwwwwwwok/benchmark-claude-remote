"""model_differences.csv of an analyze_vlm_ssm.py output folder (user, 2026-10-27): rows per model pair (every
condition x grouping x SSM of agreement.csv, each once); value_1 and value_2 are the two models' own tau-b in
agreement.csv (compared as the text written), and diff = value_1 - value_2 exactly; whether tau_b_all and its interval
are the same as tau_b_conflict's there (then the choice between them does not matter); missing values, nboot, point
estimates outside their interval; then per pair and grouping the table diff [lo, hi], * = the interval excludes 0.
usage: python -I check_model_differences.py <analysis subfolder>
"""
import os
import sys

import pandas as pd

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)
d = sys.argv[1]
md = pd.read_csv(os.path.join(d, 'model_differences.csv'), dtype=str, keep_default_na=False)
ag = pd.read_csv(os.path.join(d, 'agreement.csv'), dtype=str, keep_default_na=False)
SSMS = ['TTC', 'THW', 'DRAC', 'MTTC', 'CI', 'PICUD', 'PSD', 'TTC2D', 'ACT']
CONDS = ['grid', 'sep_time', 'sep_camera', 'ctrl_grid', 'ctrl_sep']
key = ['model', 'condition', 'grouping', 'ssm']
agi = ag.set_index(key).sort_index()
if not agi.index.is_unique:
    sys.exit('agreement.csv: a model x condition x grouping x SSM appears more than once')
cells = ag[['condition', 'grouping', 'ssm']].drop_duplicates()


def f(s):
    return float(s) if s != '' else float('nan')


def same(x, y):
    return x == y or (x != x and y != y)                     # NaN = NaN here


print(f'{d}: {len(md)} rows; stats {sorted(md.stat.unique())}; models in agreement.csv {sorted(ag.model.unique())}')
for (m1, m2), sub in md.groupby(['model_1', 'model_2'], sort=False):
    n_cells = len(sub[['condition', 'grouping', 'ssm']].drop_duplicates())
    stat = sub.stat.iloc[0]
    t1 = t2 = exact = from_ag = all_conf = 0
    for r in sub.itertuples():
        a1, a2 = agi.loc[(m1, r.condition, r.grouping, r.ssm)], agi.loc[(m2, r.condition, r.grouping, r.ssm)]
        t1 += r.value_1 == a1[r.stat]
        t2 += r.value_2 == a2[r.stat]
        exact += same(f(r.value_1) - f(r.value_2), f(r.diff))
        from_ag += same(f(a1[r.stat]) - f(a2[r.stat]), f(r.diff))
        all_conf += all(a[c] == a[c.replace('tau_b_all', 'tau_b_conflict')] for a in (a1, a2)
                        for c in ('tau_b_all', 'tau_b_all_lo', 'tau_b_all_hi', 'tau_b_all_nboot'))
    lo, hi, pt = sub.diff_lo.map(f), sub.diff_hi.map(f), sub['diff'].map(f)
    print(f'\n== {m1} - {m2} ({stat}): {len(sub)} rows, {n_cells} distinct condition x grouping x SSM '
          f'(agreement.csv has {len(cells)}); stat in the rows {sorted(sub.stat.unique())}')
    print(f'  value_1 = {m1} in agreement.csv (text): {t1}/{len(sub)}; value_2 = {m2}: {t2}/{len(sub)}')
    print(f'  diff = value_1 - value_2 exactly: {exact}/{len(sub)}; = the two agreement.csv values subtracted: '
          f'{from_ag}/{len(sub)}')
    print(f'  rows where both models have tau_b_all and its interval and nboot equal to tau_b_conflict\'s (text): '
          f'{all_conf}/{len(sub)}')
    print(f'  missing diff / lo / hi: {int(pt.isna().sum())} / {int(lo.isna().sum())} / {int(hi.isna().sum())}; '
          f'nboot {sorted(sub.nboot.unique())}; point estimate outside its interval: {int(((pt < lo) | (pt > hi)).sum())}')
    for g, gs in sub.groupby('grouping', sort=False):
        print(f'  {g}: interval excludes 0 in {int((gs["clear"] == "True").sum())} of {len(gs)} '
              f'(above 0: {int((gs.diff_lo.map(f) > 0).sum())}, below 0: {int((gs.diff_hi.map(f) < 0).sum())})')
for (m1, m2), sub in md.groupby(['model_1', 'model_2'], sort=False):
    for g, gs in sub.groupby('grouping', sort=False):
        print(f'\n=== {m1} - {m2}, grouping {g}: difference in {gs.stat.iloc[0]} [95 % interval], '
              f'* = the interval excludes 0')
        t = gs.assign(cell=gs.apply(lambda r: f'{f(r["diff"]):+.3f} [{f(r.diff_lo):+.3f}, {f(r.diff_hi):+.3f}]'
                                              + ('*' if r['clear'] == 'True' else ' '), axis=1))
        print(t.pivot(index='condition', columns='ssm', values='cell').reindex(index=CONDS, columns=SSMS).to_string())
