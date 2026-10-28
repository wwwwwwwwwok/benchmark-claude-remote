"""Compact tables of an analyze_quantities.py output folder, for the report: answers; tau-b against the truth with 95 %
intervals per model x quantity x condition for each grouping; median absolute error and median error ('all');
acceleration classes; the comparisons fixed in advance (the ones whose interval excludes 0 marked *); the model's SSMs
against the true ones and against its risk score; frames missing for the model's SSMs.
usage: python -I summarize_quantities.py <analysis subfolder>
"""
import os
import sys

import pandas as pd

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 500)
d = sys.argv[1]
rd = lambda name: pd.read_csv(os.path.join(d, name))   # noqa: E731
QUANTS = ['distance', 'ego_speed', 'lead_speed', 'ego_accel', 'lead_accel']
CONDS = ['grid', 'sep_time', 'sep_camera', 'ctrl_grid', 'ctrl_sep']
SSM7 = ['TTC', 'THW', 'DRAC', 'MTTC', 'CI', 'PICUD', 'PSD']


def cell(v, lo, hi, star=False, f='{:+.3f}'):
    return (f + ' [' + f + ', ' + f + ']').format(v, lo, hi) + ('*' if star else '')


ans = rd('quantity_answers.csv')
print('=== answers (read rate; frames without any readable answer; implausible and negative answers)')
print(ans.pivot(index=['model', 'quantity'], columns='condition', values='read_rate').reindex(columns=CONDS)
      .round(4).to_string())
print(f'frames without any readable answer, all cells: {ans.frames_without_answer.sum()}; implausible answers '
      f'{ans.implausible.sum()} ({ans.groupby("model").implausible.sum().to_dict()}); negative distances / speeds '
      f'{ans[~ans.quantity.str.endswith("accel")].negative.sum()}')

tau = rd('quantity_tau_b.csv')
for g in ['all', 'time_of_day', 'segment']:
    for m in sorted(tau.model.unique()):
        sub = tau[(tau.model == m) & (tau.grouping == g)].copy()
        sub['c'] = [cell(r.tau_b, r.tau_b_lo, r.tau_b_hi, f='{:.3f}') for r in sub.itertuples()]
        print(f'\n=== {m}, grouping {g}: Kendall tau-b of the frame estimate (median of the repeats) against the truth')
        print(sub.pivot(index='quantity', columns='condition', values='c').reindex(index=QUANTS, columns=CONDS)
              .to_string())

err = rd('quantity_errors.csv')
for m in sorted(err.model.unique()):
    sub = err[err.model == m].copy()
    sub['mae'] = [cell(r.median_abs_error, r.median_abs_error_lo, r.median_abs_error_hi, f='{:.2f}')
                  for r in sub.itertuples()]
    sub['bias'] = [cell(r.median_error, r.median_error_lo, r.median_error_hi, f='{:+.2f}') for r in sub.itertuples()]
    print(f'\n=== {m}: median absolute error (m, m/s, m/s2)')
    print(sub.pivot(index='quantity', columns='condition', values='mae').reindex(index=QUANTS, columns=CONDS).to_string())
    print(f'=== {m}: median error, estimate - truth')
    print(sub.pivot(index='quantity', columns='condition', values='bias').reindex(index=QUANTS, columns=CONDS)
          .to_string())

acc = rd('accel_classes.csv')
print('\n=== accelerations at +-0.5 m/s2: share right within each truth class (truth frames), all, always "steady"')
acc['up'] = [f'{r.share_right_up:.3f} ({r.truth_up})' for r in acc.itertuples()]
acc['steady'] = [f'{r.share_right_steady:.3f} ({r.truth_steady})' for r in acc.itertuples()]
acc['down'] = [f'{r.share_right_down:.3f} ({r.truth_down})' for r in acc.itertuples()]
acc['answers_up_steady_down'] = [f'{r.n_truth_up_answer_up + r.n_truth_steady_answer_up + r.n_truth_down_answer_up}/'
                                 f'{r.n_truth_up_answer_steady + r.n_truth_steady_answer_steady + r.n_truth_down_answer_steady}/'
                                 f'{r.n_truth_up_answer_down + r.n_truth_steady_answer_down + r.n_truth_down_answer_down}'
                                 for r in acc.itertuples()]
print(acc[['model', 'quantity', 'condition', 'frames', 'up', 'steady', 'down', 'share_right_all', 'share_always_steady',
           'answers_up_steady_down']].round(3).to_string(index=False))

dif = rd('quantity_differences.csv')
for kind in ['time', 'model']:
    sub = dif[dif.comparison == kind].copy()
    if sub.empty:
        continue
    sub['pair'] = (sub.model_1 + ' ' + sub.condition_1 + ' - ' + sub.model_2 + ' ' + sub.condition_2) if kind == 'model' \
        else (sub.model_1 + ': ' + sub.condition_1 + ' - ' + sub.condition_2)
    sub['c'] = [cell(r.diff, r.diff_lo, r.diff_hi, r.clear) for r in sub.itertuples()]
    sub['what'] = sub.stat.str.replace('median_abs_error', 'MAE') + ' ' + sub.grouping
    print(f'\n=== comparisons ({kind}); * = the interval excludes 0; tau-b: positive = the first agrees better; MAE: '
          f'negative = the first errs less')
    print(sub.pivot(index=['pair', 'what'], columns='quantity', values='c').reindex(columns=QUANTS).to_string())
    print(f'  intervals excluding 0: {int(sub.clear.sum())} of {len(sub)}')

pss = rd('perceived_ssm_tau_b.csv')
for comp in ['model_ssm_vs_true_ssm', 'risk_score_vs_model_ssm']:
    for g in ['all', 'segment']:
        sub = pss[(pss.comparison == comp) & (pss.grouping == g)].copy()
        if sub.empty:
            continue
        sub['c'] = [cell(r.tau_b, r.tau_b_lo, r.tau_b_hi, f='{:.3f}') for r in sub.itertuples()]
        print(f'\n=== {comp}, grouping {g}: tau-b [95 % interval] (frames with the model SSM computable; frames where '
              f'it has no value tied at the safe end)')
        print(sub.pivot(index=['model', 'condition'], columns='ssm', values='c').reindex(columns=SSM7).to_string())
sub = pss[(pss.comparison == 'model_ssm_vs_true_ssm') & (pss.grouping == 'all')]
print('\n=== frames where the model SSM has a value (of the frames it can be computed on)')
print(sub.assign(c=sub.frames_model_ssm_with_value.astype(str) + '/' + sub.frames.astype(str))
      .pivot(index=['model', 'condition'], columns='ssm', values='c').reindex(columns=SSM7).to_string())
pmi = rd('perceived_ssm_missing.csv')
print('\n=== frames missing for the model SSMs')
print(pmi[['model', 'condition', 'frames', 'missing', 'quantity_missing', 'gap_negative', 'speed_negative',
           'gap_zero']].to_string(index=False))
