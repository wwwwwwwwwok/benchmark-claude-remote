"""Compact tables of the exploratory outputs of analyze_quantities.py (user, 2026-10-29; all exploratory): E1 (the
frames whose estimated gap is negative, and tau-b of the model's SSMs with their gap at 0.01 m beside the original),
E2 (risk score against the distance and the ego speed), the errors (SD, IQR, the image-free answer, distance within
25 %); and, with a quantities_first_k.py folder, its rule and differences.
usage: python -I summarize_quantities_explore.py <analysis subfolder> [<first_k folder>]
"""
import os
import sys

import pandas as pd

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 500)
d = sys.argv[1]
rd = lambda folder, name: pd.read_csv(os.path.join(folder, name))   # noqa: E731
QUANTS = ['distance', 'ego_speed', 'lead_speed', 'ego_accel', 'lead_accel']
CONDS = ['grid', 'sep_time', 'sep_camera', 'ctrl_grid', 'ctrl_sep']
SSM7 = ['TTC', 'THW', 'DRAC', 'MTTC', 'CI', 'PICUD', 'PSD']


def cell(v, lo, hi, f='{:+.3f}', star=False):
    return (f + ' [' + f + ', ' + f + ']').format(v, lo, hi) + ('*' if star else '')


print(f'=== {d}: exploratory (user, 2026-10-29)')
f = rd(d, 'exploratory_e1_frames.csv')
print('\n=== E1: frames whose estimated distance is below the bumper-to-camera distance (missing in the original)')
print(f.round(3).to_string(index=False))
e1 = rd(d, 'exploratory_e1_perceived_ssm_tau_b.csv')
for comp in ['model_ssm_vs_true_ssm', 'risk_score_vs_model_ssm']:
    for g in ['all', 'segment']:
        sub = e1[(e1.comparison == comp) & (e1.grouping == g)].copy()
        sub['c'] = [f'{r.tau_b_original:+.3f} -> {r.tau_b_e1:+.3f}' for r in sub.itertuples()]
        print(f'\n=== E1 {comp}, grouping {g}: tau-b original -> with the gap of those frames at 0.01 m')
        print(sub.pivot(index=['model', 'condition'], columns='ssm', values='c').reindex(columns=SSM7).to_string())
sub = e1[(e1.comparison == 'model_ssm_vs_true_ssm') & (e1.grouping == 'all')]
print('\n=== E1: frames with a value of the model SSM, original -> E1 (of the frames used)')
print(sub.assign(c=[f'{r.frames_model_ssm_with_value_original}/{r.frames_original} -> '
                    f'{r.frames_model_ssm_with_value_e1}/{r.frames_e1}' for r in sub.itertuples()])
      .pivot(index=['model', 'condition'], columns='ssm', values='c').reindex(columns=SSM7).to_string())
sub = e1[(e1.comparison == 'model_ssm_vs_true_ssm') & (e1.grouping == 'all')]
print('\n=== E1 model_ssm_vs_true_ssm, all: E1 with its interval')
print(sub.assign(c=[cell(r.tau_b_e1, r.tau_b_e1_lo, r.tau_b_e1_hi, '{:.3f}') for r in sub.itertuples()])
      .pivot(index=['model', 'condition'], columns='ssm', values='c').reindex(columns=SSM7).to_string())

e2 = rd(d, 'exploratory_e2_risk_vs_distance_speed.csv')
for g in ['all', 'time_of_day', 'segment']:
    sub = e2[e2.grouping == g].copy()
    sub['c'] = [cell(r.tau_b, r.tau_b_lo, r.tau_b_hi, star=(r.tau_b_lo > 0 or r.tau_b_hi < 0)) for r in sub.itertuples()]
    print(f'\n=== E2, grouping {g}: tau-b of the risk score with ... [95 % interval]; negative = higher risk when '
          f'closer / slower; * = the interval excludes 0')
    print(sub.pivot(index=['variable', 'model'], columns='condition', values='c').reindex(columns=CONDS).to_string())

xe = rd(d, 'exploratory_quantity_errors.csv')
for m in sorted(xe.model.unique()):
    sub = xe[xe.model == m].copy()
    sub['sd'] = [cell(r.error_sd, r.error_sd_lo, r.error_sd_hi, '{:.2f}') for r in sub.itertuples()]
    sub['iqr'] = [cell(r.error_iqr, r.error_iqr_lo, r.error_iqr_hi, '{:.2f}') for r in sub.itertuples()]
    sub['base'] = [f'{r.median_abs_error:.2f} vs {r.baseline_median_abs_error:.2f}: '
                   + cell(r.mae_minus_baseline, r.mae_minus_baseline_lo, r.mae_minus_baseline_hi, '{:+.2f}',
                          r.mae_minus_baseline_clear) for r in sub.itertuples()]
    for col, title in (('sd', 'SD of the error (estimate - truth)'), ('iqr', 'IQR of the error'),
                       ('base', 'median absolute error, model vs answering the median of the truth everywhere: '
                                'model - that [95 %]; * = excludes 0')):
        print(f'\n=== {m}: {title}')
        print(sub.pivot(index='quantity', columns='condition', values=col).reindex(index=QUANTS, columns=CONDS)
              .to_string())
print('\n=== the image-free answers (median of the truth over these frames):',
      xe.drop_duplicates('quantity').set_index('quantity').baseline_answer.round(4).to_dict())
sub = xe[xe.quantity == 'distance'].copy()
sub['c'] = [cell(r.share_within_25pct, r.share_within_25pct_lo, r.share_within_25pct_hi, '{:.3f}')
            for r in sub.itertuples()]
print('\n=== distance within 25 % (larger / smaller below 1.25), share of frames [95 %]')
print(sub.pivot(index='model', columns='condition', values='c').reindex(columns=CONDS).to_string())

if len(sys.argv) > 2:
    k = sys.argv[2]
    print(f'\n=== first k repeats ({k})')
    print(open(os.path.join(k, 'first_k_rule.txt'), encoding='utf-8').read())
    t = rd(k, 'first_k_tau_b.csv')
    for g in ['all', 'segment']:
        for kk in (1, 2, 3, 4):
            sub = t[(t.grouping == g) & (t.k == kk)].copy()
            sub['c'] = [(cell(r.diff, r.diff_lo, r.diff_hi) if g == 'all' else f'{r.diff:+.3f}')
                        + ('!' if not abs(r.diff) < 0.02 else '') for r in sub.itertuples()]
            print(f'\n=== tau-b(k={kk}) - tau-b(5), grouping {g}' + (' [95 %]' if g == 'all' else '')
                  + '; ! = |difference| 0.02 or more')
            print(sub.pivot(index=['model', 'quantity'], columns='condition', values='c').reindex(columns=CONDS)
                  .to_string())
    p = rd(k, 'first_k_perceived_ssm_tau_b.csv')
    p = p[p.k < 5]
    print('\n=== model SSM vs true SSM, all: tau-b(k) - tau-b(5), largest |difference| per k and model, and how many '
          'of the 35 cells have an interval excluding 0')
    p['ex0'] = (p.diff_lo > 0) | (p.diff_hi < 0)
    print(p.groupby(['k', 'model']).agg(largest=('diff', lambda s: s.abs().max()), median_abs=('diff', lambda s: s.abs().median()),
                                        excl_0=('ex0', 'sum')).round(4).to_string())
    e = rd(k, 'first_k_errors_missing.csv')
    e = e[e.k < 5]
    print('\n=== median absolute error, k - 5: largest |difference| per k, model and quantity; frames missing')
    print(e.groupby(['k', 'model', 'quantity'], sort=False).agg(largest=('mae_diff', lambda s: s.abs().max()),
                                                                missing=('frames_missing', 'sum'))
          .unstack('quantity').round(3).to_string())
    pm = rd(k, 'first_k_perceived_ssm_missing.csv')
    print('\n=== frames missing for the model SSMs, per k (sum over conditions)')
    print(pm.groupby(['model', 'k']).missing.sum().unstack('k').to_string())
