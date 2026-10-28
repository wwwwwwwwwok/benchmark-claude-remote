"""Acceptance of a code/analyze_quantities.py output folder (user, 2026-10-28: as for the risk analysis: frame counts,
every cell has a value, every difference exactly the difference of its two point estimates), plus checks by other
means: tau-b ('all'), median absolute error and median error recomputed from quantity_frames.csv with scipy and numpy;
the model's TTC, THW and DRAC recomputed from the frame estimates with the textbook formulas (distance minus the camera
offset as the gap); the resamples (risk_check.csv all identical); the acceleration class counts adding up.
usage: python -I quantities_accept.py <analysis folder> <expected frames> <truth csv> [--intervals]
       --intervals: also require every interval (not for the pilot: 3 segments)
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ap = argparse.ArgumentParser()
ap.add_argument('d')
ap.add_argument('frames', type=int)
ap.add_argument('truth')
ap.add_argument('--intervals', action='store_true')
a = ap.parse_args()
d = a.d
rd = lambda name, **k: pd.read_csv(os.path.join(d, name), float_precision='round_trip', **k)   # noqa: E731
tau = rd('quantity_tau_b.csv')
err = rd('quantity_errors.csv')
acc = rd('accel_classes.csv')
dif = rd('quantity_differences.csv')
pss = rd('perceived_ssm_tau_b.csv')
pmi = rd('perceived_ssm_missing.csv')
pfr = rd('perceived_ssm_frames.csv')
qfr = rd('quantity_frames.csv')
rc = rd('risk_check.csv')
models = sorted(tau.model.unique())
nm, nq, nc, ng = len(models), tau.quantity.nunique(), tau.condition.nunique(), tau.grouping.nunique()
bad = 0


def report(ok, text):
    global bad
    bad += not ok
    print(f'  [{"ok" if ok else "NOT OK"}] {text}')


print(f'{d}: models {models}; conditions {nc}, quantities {nq}, groupings {sorted(tau.grouping.unique())}')
nf = qfr[['segment', 'frame_idx']].drop_duplicates().shape[0]
report(nf == a.frames, f'frames {nf} (expected {a.frames}); quantity_frames rows {len(qfr)} = models x conditions x '
                       f'quantities x frames {nm * nc * nq * a.frames}: {len(qfr) == nm * nc * nq * a.frames}')
report(len(tau) == nm * nc * nq * ng, f'quantity_tau_b rows {len(tau)} (expected {nm * nc * nq * ng})')
cols = ['tau_b'] + (['tau_b_lo', 'tau_b_hi'] if a.intervals else [])
report(not tau[cols].isna().any().any(), f'quantity_tau_b: missing values in {cols}: {int(tau[cols].isna().sum().sum())}; '
                                         f'nboot {sorted(tau.nboot.unique())}; frames per cell {tau.frames.min()}-{tau.frames.max()}')
report(len(err) == nm * nc * nq, f'quantity_errors rows {len(err)} (expected {nm * nc * nq})')
cols = ['median_abs_error', 'median_error'] + (['median_abs_error_lo', 'median_abs_error_hi', 'median_error_lo',
                                                'median_error_hi'] if a.intervals else [])
report(not err[cols].isna().any().any(), f'quantity_errors: missing values in {cols}: {int(err[cols].isna().sum().sum())}')
report(len(acc) == nm * nc * 2, f'accel_classes rows {len(acc)} (expected {nm * nc * 2})')
sums = acc.truth_up + acc.truth_steady + acc.truth_down
conf = acc[[c for c in acc.columns if c.startswith('n_truth_')]].sum(axis=1)
right = acc.right_up + acc.right_steady + acc.right_down
report((sums == acc.frames).all() and (conf == acc.frames).all() and
       np.allclose(right / acc.frames, acc.share_right_all) and np.allclose(acc.truth_steady / acc.frames,
                                                                             acc.share_always_steady),
       'accel_classes: truth classes and the 3 x 3 counts add up to the frames; shares right and always-steady agree')
n_time = nm * 3 * nq * (ng + 1)
pair = all(m in models for m in ('qwen-drive-1.0-4b', 'qwen3.5-4b'))
n_model = nc * nq * (ng + 1) if pair else 0
report(len(dif) == n_time + n_model, f'quantity_differences rows {len(dif)} (expected time {n_time} + model {n_model})')
exact = (dif['diff'] == dif.value_1 - dif.value_2) | (dif['diff'].isna() & (dif.value_1.isna() | dif.value_2.isna()))
report(exact.all(), f'quantity_differences: diff = value_1 - value_2 exactly in {int(exact.sum())} of {len(dif)}')
cols = ['diff'] + (['diff_lo', 'diff_hi'] if a.intervals else [])
report(not dif[cols].isna().any().any(), f'quantity_differences: missing values in {cols}: '
                                         f'{int(dif[cols].isna().sum().sum())}')
# the point estimates in the differences are the ones in quantity_tau_b / quantity_errors
tk = tau.set_index(['model', 'condition', 'quantity', 'grouping']).tau_b
ek = err.set_index(['model', 'condition', 'quantity']).median_abs_error
same = 0
for r in dif.itertuples():
    if r.stat == 'tau_b':
        v1, v2 = tk[(r.model_1, r.condition_1, r.quantity, r.grouping)], tk[(r.model_2, r.condition_2, r.quantity, r.grouping)]
    else:
        v1, v2 = ek[(r.model_1, r.condition_1, r.quantity)], ek[(r.model_2, r.condition_2, r.quantity)]
    same += (v1 == r.value_1) and (v2 == r.value_2)
report(same == len(dif), f'quantity_differences: value_1 and value_2 are the point estimates of the two cells: {same} '
                         f'of {len(dif)}')
has_risk = pss.comparison.eq('risk_score_vs_model_ssm').any()
n_p = nm * nc * 7 * ng * (2 if has_risk else 1)
report(len(pss) == n_p, f'perceived_ssm_tau_b rows {len(pss)} (expected {n_p}); comparisons {sorted(pss.comparison.unique())}')
cols = ['tau_b'] + (['tau_b_lo', 'tau_b_hi'] if a.intervals else [])
report(not pss[cols].isna().any().any(), f'perceived_ssm_tau_b: missing values in {cols}: '
                                         f'{int(pss[cols].isna().sum().sum())}')
report(len(pmi) == nm * nc and (pmi.missing == pmi.frames - pfr.groupby(['model', 'condition'], sort=False).valid.sum()
                                .reindex(pd.MultiIndex.from_frame(pmi[['model', 'condition']])).to_numpy()).all(),
       f'perceived_ssm_missing: {len(pmi)} rows; missing = frames - valid frames; missing frames per model x condition '
       f'{pmi.missing.min()}-{pmi.missing.max()} (quantity missing {pmi.quantity_missing.sum()}, gap < 0 '
       f'{pmi.gap_negative.sum()}, speed < 0 {pmi.speed_negative.sum()} in all)')
report(len(rc) > 0 and rc.identical.all(), f'risk_check: {int(rc.identical.sum())} of {len(rc)} identical to the risk '
                                           f'analysis agreement.csv')

# --- by other means
tr = pd.read_csv(a.truth, float_precision='round_trip').set_index(['segment', 'frame_idx'])
col = {'distance': 'distance', 'ego_speed': 'v_ego', 'lead_speed': 'v_lead', 'ego_accel': 'a_ego', 'lead_accel': 'a_lead'}
t_all = tau[tau.grouping == 'all'].set_index(['model', 'condition', 'quantity'])
e_all = err.set_index(['model', 'condition', 'quantity'])
dt = dm = db = 0.0
for (m, c, q), sub in qfr.groupby(['model', 'condition', 'quantity']):
    y = tr.loc[pd.MultiIndex.from_frame(sub[['segment', 'frame_idx']]), col[q]].to_numpy(float)
    x = sub.estimate.to_numpy(float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() >= 3:
        dt = max(dt, abs(kendalltau(x[ok], y[ok], variant='b').statistic - t_all.at[(m, c, q), 'tau_b']))
    dm = max(dm, abs(np.median(np.abs(x[ok] - y[ok])) - e_all.at[(m, c, q), 'median_abs_error']))
    db = max(db, abs(np.median(x[ok] - y[ok]) - e_all.at[(m, c, q), 'median_error']))
report(dt < 1e-12 and dm == 0 and db == 0, f'by scipy / numpy from quantity_frames.csv: largest difference in tau-b (all) '
                                            f'{dt:.2e}, median absolute error {dm}, median error {db}')
est = qfr.pivot_table(index=['model', 'condition', 'segment', 'frame_idx'], columns='quantity', values='estimate',
                      dropna=False)
p = pfr.set_index(['model', 'condition', 'segment', 'frame_idx'])
p = p.join(est, how='left')
off = tr['camera_offset'].reindex(pd.MultiIndex.from_arrays([p.index.get_level_values(2), p.index.get_level_values(3)]))
gap = p['distance'].to_numpy(float) - off.to_numpy(float)
vf, vl = p['ego_speed'].to_numpy(float), p['lead_speed'].to_numpy(float)
v = p['valid'].to_numpy(bool)
closing = vf - vl
with np.errstate(divide='ignore', invalid='ignore'):
    ttc = np.where(closing > 0, gap / closing, np.inf)
    thw = np.where(vf > 0.1, gap / vf, np.inf)
    drac = np.where(closing > 0, closing ** 2 / (2 * gap), 0.0)
ok = (np.allclose(gap[v], p['gap'].to_numpy(float)[v], rtol=0, atol=1e-12)
      and np.allclose(ttc[v], p['TTC'].to_numpy(float)[v], rtol=1e-12, atol=0, equal_nan=True)
      and np.allclose(thw[v], p['THW'].to_numpy(float)[v], rtol=1e-12, atol=0, equal_nan=True)
      and np.allclose(drac[v], p['DRAC'].to_numpy(float)[v], rtol=1e-12, atol=0, equal_nan=True))
exp_valid = ~np.isnan(p[list(col)].to_numpy(float)).any(axis=1) & (gap >= 0) & (vf >= 0) & (vl >= 0)
report(ok and (exp_valid == v).all(), f'model SSMs: gap = distance - camera offset, TTC, THW, DRAC by the textbook '
                                      f'formulas on the {int(v.sum())} valid frame x model x condition rows; the valid '
                                      f'frames are those with 5 estimates, gap >= 0 and no negative speed')
print(f'\nchecks not passed: {bad}')
