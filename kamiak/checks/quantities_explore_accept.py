"""Acceptance of the exploratory tables of code/analyze_quantities.py (user, 2026-10-29; that the tables there before
are unchanged is checked by compare_analysis_versions.py against the run without the additions). Every cell has a
value; the original columns of the E1 table are those of perceived_ssm_tau_b.csv; the E1 frames are those of
perceived_ssm_missing.csv; by other means: tau-b ('all') of E1 (TTC, THW, DRAC by the textbook formulas with the floored
gap, frames without a value tied at the safe end) and of E2 recomputed with scipy; error SD, IQR, the image-free
answer and its median absolute error, the share within 25 % recomputed with numpy; mae_minus_baseline exactly the
difference of the two values.
usage: python -I quantities_explore_accept.py <analysis folder> <truth csv> <frame_ssm csv> <risk analysis subfolder>
                                              [--intervals]
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ap = argparse.ArgumentParser()
ap.add_argument('d')
ap.add_argument('truth')
ap.add_argument('frame_ssm')
ap.add_argument('risk')
ap.add_argument('--intervals', action='store_true')
a = ap.parse_args()
rd = lambda name: pd.read_csv(os.path.join(a.d, name), float_precision='round_trip')   # noqa: E731
e1, e1f, e2, xe = (rd(n) for n in ('exploratory_e1_perceived_ssm_tau_b.csv', 'exploratory_e1_frames.csv',
                                   'exploratory_e2_risk_vs_distance_speed.csv', 'exploratory_quantity_errors.csv'))
pss, pmi, qfr, qer, pfr = (rd(n) for n in ('perceived_ssm_tau_b.csv', 'perceived_ssm_missing.csv',
                                           'quantity_frames.csv', 'quantity_errors.csv', 'perceived_ssm_frames.csv'))
E1_GAP = float(open(os.path.join(a.d, 'exploratory_params.txt'), encoding='utf-8').read().split('a gap of ')[1]
               .split(' m')[0])
models = sorted(qfr.model.unique())
conds = list(dict.fromkeys(qfr.condition))
ng = pss.grouping.nunique()
bad = 0


def report(ok, text):
    global bad
    bad += not ok
    print(f'  [{"ok" if ok else "NOT OK"}] {text}')


def same(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return bool(((x == y) | (np.isnan(x) & np.isnan(y))).all())


def tau_all(x, y):
    return kendalltau(x, y, variant='b').statistic if len(x) >= 3 else np.nan


print(f'{a.d}: models {models}; E1 gap {E1_GAP} m')
# --- E1
report(len(e1) == len(pss), f'E1 table rows {len(e1)} = perceived_ssm_tau_b rows {len(pss)}')
k = ['model', 'condition', 'ssm', 'comparison', 'grouping']
j = e1.merge(pss, on=k, how='outer', validate='one_to_one', indicator=True)
orig = [('frames_original', 'frames'), ('frames_model_ssm_with_value_original', 'frames_model_ssm_with_value'),
        ('tau_b_original', 'tau_b'), ('tau_b_original_lo', 'tau_b_lo'), ('tau_b_original_hi', 'tau_b_hi'),
        ('nboot_original', 'nboot'), ('tau_b_with_value_original', 'tau_b_with_value'),
        ('tau_b_with_value_original_lo', 'tau_b_with_value_lo'), ('tau_b_with_value_original_hi', 'tau_b_with_value_hi')]
report((j._merge == 'both').all() and all(same(j[x], j[y]) for x, y in orig),
       'E1 table: the original columns are those of perceived_ssm_tau_b.csv (frames, values, intervals)')
cols = ['tau_b_e1'] + (['tau_b_e1_lo', 'tau_b_e1_hi'] if a.intervals else [])
risk_rows = e1.comparison == 'risk_score_vs_model_ssm'
cols_r = ['tau_b_with_value_e1'] + (['tau_b_with_value_e1_lo', 'tau_b_with_value_e1_hi'] if a.intervals else [])
report(not e1[cols].isna().any().any() and not e1.loc[risk_rows, cols_r].isna().any().any(),
       f'E1 table: missing values in {cols}: {int(e1[cols].isna().sum().sum())}, in {cols_r} (risk rows): '
       f'{int(e1.loc[risk_rows, cols_r].isna().sum().sum())}')
f1 = e1f.merge(pmi[['model', 'condition', 'gap_negative']], on=['model', 'condition'], validate='one_to_one')
report(len(e1f) == len(pmi) and (f1.frames_gap_negative == f1.gap_negative).all(),
       f'E1 frames: {len(e1f)} rows, frames with a negative gap as in perceived_ssm_missing.csv '
       f'({int(e1f.frames_gap_negative.sum())} in all)')

tr = pd.read_csv(a.truth, float_precision='round_trip').set_index(['segment', 'frame_idx'])
fs = pd.read_csv(a.frame_ssm).set_index(['segment', 'frame_idx'])
est = qfr.set_index(['model', 'condition', 'segment', 'frame_idx', 'quantity']).estimate.unstack('quantity')
#   (unstack keeps the frames as they are; pivot_table with dropna=False would cross every segment with every frame index)
rs = pd.read_csv(os.path.join(a.risk, 'vlm_frame_scores.csv'), float_precision='round_trip') \
    .set_index(['model', 'condition', 'segment', 'frame_idx'])['score_mean']
dt = dw = dr = dz = 0.0
n_rows = n_frames_ok = 0
frame_stats_ok = True
for (m, c), sub in est.groupby(level=[0, 1]):
    fi = sub.index.droplevel([0, 1])
    d, vf, vl = (sub[q].to_numpy(float) for q in ('distance', 'ego_speed', 'lead_speed'))
    lack = sub[['distance', 'ego_speed', 'lead_speed', 'ego_accel', 'lead_accel']].isna().any(axis=1).to_numpy()
    gap = d - tr.loc[fi, 'camera_offset'].to_numpy(float)
    neg = ~lack & (gap < 0)
    ok = ~lack & (vf >= 0) & (vl >= 0)                 # E1: valid whatever the gap
    gap = np.where(neg, E1_GAP, gap)
    closing = vf - vl
    with np.errstate(divide='ignore', invalid='ignore'):
        mine = {'TTC': np.where(closing > 0, -(gap / closing), np.nan),
                'THW': np.where(vf > 0.1, -(gap / vf), np.nan),
                'DRAC': np.where(closing > 0, closing ** 2 / (2 * gap), np.nan)}
    # the frames with a negative gap: true and estimated distance
    r = e1f[(e1f.model == m) & (e1f.condition == c)].iloc[0]
    td, ed = tr.loc[fi, 'distance'].to_numpy(float)[neg], d[neg]
    td = td[~np.isnan(td)]
    frame_stats_ok &= bool(r.frames_with_true_distance == len(td) and r.frames_model_distance_zero == (ed == 0).sum())
    if len(td):
        frame_stats_ok &= bool(r.true_distance_median == np.median(td) and r.true_distance_min == td.min()
                               and r.true_distance_max == td.max())
    if neg.any():
        frame_stats_ok &= bool(r.model_distance_median == np.median(ed) and r.model_distance_min == ed.min()
                               and r.model_distance_max == ed.max())
    rev = (fs.loc[fi, 'reversing_ego'].astype(bool) | fs.loc[fi, 'reversing_lead'].astype(bool)).to_numpy()
    score = rs.reindex(pd.MultiIndex.from_arrays([[m] * len(fi), [c] * len(fi), fi.get_level_values(0),
                                                  fi.get_level_values(1)])).to_numpy(float)
    for s, v in mine.items():
        tv = fs.loc[fi, s].to_numpy(float)
        t_conf = (np.nan_to_num(tv, nan=0.0) > 0) if s == 'DRAC' else np.isfinite(tv)
        t_risk = tv if s == 'DRAC' else -tv
        has = ok & np.isfinite(v) & ((v > 0) if s == 'DRAC' else True)
        x_all = np.where(has, v, (np.nanmin(v[has]) - 1.0) if has.any() else 0.0)
        on = ok & ~rev & t_conf
        row = e1[(e1.model == m) & (e1.condition == c) & (e1.ssm == s) & (e1.grouping == 'all')]
        r1 = row[row.comparison == 'model_ssm_vs_true_ssm'].iloc[0]
        dt = max(dt, abs(tau_all(x_all[on], t_risk[on]) - r1.tau_b_e1))
        n_frames_ok += r1.frames_e1 == int(on.sum())
        n_rows += 1
        r2 = row[row.comparison == 'risk_score_vs_model_ssm']
        if len(r2):
            r2 = r2.iloc[0]
            on2 = ok & ~rev & ~np.isnan(score)
            dr = max(dr, abs(tau_all(score[on2], x_all[on2]) - r2.tau_b_e1))
            onv = on2 & has
            dw = max(dw, abs(tau_all(score[onv], v[onv]) - r2.tau_b_with_value_e1)) if onv.sum() >= 3 else dw
            n_frames_ok += r2.frames_e1 == int(on2.sum())
            n_rows += 1
report(dt < 1e-12 and dr < 1e-12 and dw < 1e-12 and n_frames_ok == n_rows,
       f"E1 by scipy (TTC, THW, DRAC by the formulas with the floored gap, 'all'): largest difference model SSM vs true "
       f"{dt:.2e}, risk score vs model SSM {dr:.2e}, the same over frames with a value {dw:.2e}; frame counts "
       f"{n_frames_ok} of {n_rows}")
report(frame_stats_ok, 'E1 frames: true and estimated distance of the frames with a negative gap recomputed')

# --- E2
n_risk = rs.index.get_level_values(0).nunique()
n_exp = (n_risk * len(conds) * 2 + len(models) * len(conds) * 2) * ng
cols = ['tau_b'] + (['tau_b_lo', 'tau_b_hi'] if a.intervals else [])
report(len(e2) == n_exp and not e2[cols].isna().any().any(),
       f'E2 rows {len(e2)} (expected {n_exp}: {n_risk} risk models, {len(models)} with their own estimates); missing '
       f'values in {cols}: {int(e2[cols].isna().sum().sum())}; frames per cell {e2.frames.min()}-{e2.frames.max()}')
frames = qfr[['segment', 'frame_idx']].drop_duplicates()
fi = pd.MultiIndex.from_frame(frames)
de2 = 0.0
for r in e2[e2.grouping == 'all'].itertuples():
    score = rs.reindex(pd.MultiIndex.from_arrays([[r.model] * len(fi), [r.condition] * len(fi),
                                                  fi.get_level_values(0), fi.get_level_values(1)])).to_numpy(float)
    if r.variable.startswith('true_'):
        v = tr.loc[fi, 'distance' if r.variable == 'true_distance' else 'v_ego'].to_numpy(float)
    else:
        v = est.loc[(r.model, r.condition)].reindex(fi)['distance' if r.variable == 'model_distance'
                                                        else 'ego_speed'].to_numpy(float)
    ok = ~np.isnan(score) & ~np.isnan(v)
    de2 = max(de2, abs(tau_all(score[ok], v[ok]) - r.tau_b))
report(de2 < 1e-12, f"E2 by scipy ('all'): largest difference {de2:.2e}")

# --- errors
nq = qfr.quantity.nunique()
cols = ['error_sd', 'error_iqr', 'median_abs_error', 'baseline_median_abs_error', 'mae_minus_baseline'] + \
    (['error_sd_lo', 'error_sd_hi', 'error_iqr_lo', 'error_iqr_hi', 'baseline_median_abs_error_lo',
      'baseline_median_abs_error_hi', 'mae_minus_baseline_lo', 'mae_minus_baseline_hi'] if a.intervals else [])
wcols = ['share_within_25pct'] + (['share_within_25pct_lo', 'share_within_25pct_hi'] if a.intervals else [])
isd = xe.quantity == 'distance'
report(len(xe) == len(models) * len(conds) * nq and not xe[cols].isna().any().any()
       and not xe.loc[isd, wcols].isna().any().any() and xe.loc[~isd, wcols].isna().all().all(),
       f'errors: {len(xe)} rows; missing values in {cols}: {int(xe[cols].isna().sum().sum())}; the share within 25 % '
       f'for the distance only')
exact = (xe.mae_minus_baseline == xe.median_abs_error - xe.baseline_median_abs_error)
mae_same = same(xe.set_index(['model', 'condition', 'quantity']).median_abs_error,
                qer.set_index(['model', 'condition', 'quantity']).median_abs_error.reindex(
                    pd.MultiIndex.from_frame(xe[['model', 'condition', 'quantity']])))
report(exact.all() and mae_same, f'errors: mae_minus_baseline = median_abs_error - baseline exactly in {int(exact.sum())} '
                                 f'of {len(xe)}; median_abs_error as in quantity_errors.csv: {mae_same}')
mx = {'sd': 0.0, 'iqr': 0.0, 'base': 0.0, 'within': 0.0}
base_ok = True
truth_by_q = qfr.drop_duplicates(['quantity', 'segment', 'frame_idx'])
for r in xe.itertuples():
    t_all = truth_by_q[truth_by_q.quantity == r.quantity].truth.dropna().to_numpy(float)
    base_ok &= bool(r.baseline_answer == np.median(t_all))
    s = qfr[(qfr.model == r.model) & (qfr.condition == r.condition) & (qfr.quantity == r.quantity)]
    s = s.dropna(subset=['estimate', 'truth'])
    x, y = s.estimate.to_numpy(float), s.truth.to_numpy(float)
    e = x - y
    mx['sd'] = max(mx['sd'], abs(np.sqrt(np.mean((e - e.mean()) ** 2)) - r.error_sd))
    mx['iqr'] = max(mx['iqr'], abs(np.quantile(e, 0.75) - np.quantile(e, 0.25) - r.error_iqr))
    mx['base'] = max(mx['base'], abs(np.median(np.abs(np.median(t_all) - y)) - r.baseline_median_abs_error))
    if r.quantity == 'distance':
        hit = (x > 0) & (x < 1.25 * y) & (y < 1.25 * x)
        mx['within'] = max(mx['within'], abs(hit.mean() - r.share_within_25pct))
report(base_ok and mx['sd'] < 1e-9 and mx['iqr'] < 1e-12 and mx['base'] == 0 and mx['within'] == 0,
       f'errors by numpy: image-free answer = median of the truth: {base_ok}; largest difference SD {mx["sd"]:.2e}, '
       f'IQR {mx["iqr"]:.2e}, baseline median absolute error {mx["base"]}, share within 25 % {mx["within"]}')
print(f'\nchecks not passed: {bad}')
