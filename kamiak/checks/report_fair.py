"""Read-only (user, 2026-11-06, item 3, descriptive): compare_d_models_fair.py's tables put together: per condition the
21 pairs (Spearman of the scores, disattenuated, Spearman of the additive predictions), their medians and ranges; the
cross-prediction ratios (A's additive function on B's ranks / B's own), median and range over the 210 ordered pairs x
conditions, without Qwen-Drive, and Qwen-Drive as predictor and as target.
usage: python -I report_fair.py <compare_fair folder>"""
import os
import sys

import pandas as pd

C = sys.argv[1]
QD = 'qwen-drive-1.0-4b'
CONDS = ['grid', 'sep_time', 'sep_camera', 'ctrl_grid', 'ctrl_sep']
print(open(os.path.join(C, 'exploratory_d_models_fair_params.txt'), encoding='utf-8').read())
d = pd.read_csv(os.path.join(C, 'exploratory_d_models_disattenuated.csv'))
print('== per condition: median [range] of the 21 pairs: scores Spearman | disattenuated | additive predictions')
for c in CONDS:
    q = d[d.condition == c]
    f = lambda s: f'{s.median():.3f} [{s.min():.3f}, {s.max():.3f}]'  # noqa: E731
    print(f'  {c:10s} {f(q.spearman_rank_scores)} | {f(q.disattenuated)} | {f(q.spearman_additive_predictions)}; '
          f'disattenuated above 1: {int((q.disattenuated > 1).sum())}; disattenuated above the additive: '
          f'{int((q.disattenuated > q.spearman_additive_predictions).sum())} of {len(q)}')
nq = d[(d.model_1 != QD) & (d.model_2 != QD)]
wq = d[(d.model_1 == QD) | (d.model_2 == QD)]
for name, q in (('without Qwen-Drive (15 pairs x 5)', nq), ('with Qwen-Drive (6 pairs x 5)', wq)):
    print(f'  {name}: disattenuated median {q.disattenuated.median():.3f} [{q.disattenuated.min():.3f}, '
          f'{q.disattenuated.max():.3f}]; additive predictions median {q.spearman_additive_predictions.median():.3f} '
          f'[{q.spearman_additive_predictions.min():.3f}, {q.spearman_additive_predictions.max():.3f}]')
print('\n== all pairs (condition, pair: scores | disattenuated | additive)')
for r in d.itertuples():
    print(f'  {r.condition:10s} {r.model_1} ~ {r.model_2}: {r.spearman_rank_scores:.3f} | {r.disattenuated:.3f} | '
          f'{r.spearman_additive_predictions:.3f}')
x = pd.read_csv(os.path.join(C, 'exploratory_d_models_cross.csv'))
print('\n== cross prediction ratio (A\'s additive function on B / B\'s own): median [range]')
g = lambda q: f'{q.ratio.median():.3f} [{q.ratio.min():.3f}, {q.ratio.max():.3f}] (n {len(q)}); cross R2 median ' \
    f'{q.r2_cross.median():.3f}'  # noqa: E731
print(f'  all 42 ordered pairs x 5: {g(x)}')
print(f'  without Qwen-Drive (30 x 5): {g(x[(x.predictor != QD) & (x.target != QD)])}')
print(f'  Qwen-Drive predicting the others (6 x 5): {g(x[x.predictor == QD])}')
print(f'  the others predicting Qwen-Drive (6 x 5): {g(x[x.target == QD])}')
for c in CONDS:
    print(f'  {c:10s} without Qwen-Drive {g(x[(x.condition == c) & (x.predictor != QD) & (x.target != QD)])}')
print('  per predictor (all targets, all conditions): ' + '; '.join(
    f'{m} {q.ratio.median():.3f}' for m, q in x.groupby('predictor')))
print('  per target (all predictors, all conditions): ' + '; '.join(
    f'{m} {q.ratio.median():.3f} (own R2 {q.r2_own_target.min():.3f}-{q.r2_own_target.max():.3f})'
    for m, q in x.groupby('target')))
