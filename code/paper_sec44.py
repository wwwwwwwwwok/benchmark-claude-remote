"""Numbers for paper section 4.4, the quantities task (user, 2026-11-12): Qwen3.6-27B with thinking on, sep_camera, the
main result's 1,084 frames (the formal analysis, analysis/quantities_after_27b_think/main_1084). What the analyses wrote
is printed as text with file and line; what they did not write is computed here and marked EXPLORATORY. No formal
output is changed; the outputs go to <out_dir>.

1 settings: the five questions as sent for sep_camera (code/run_vlm.py: QUANTITIES with {now} = '' and
  quantity_question); requests, readable answers, replies at max_tokens, frames without a readable answer
  (quantity_answers.csv); the frame estimate is the median of the readable repeats (analyze_quantities.py).
2 accuracy per quantity: tau-b with the truth and its interval (quantity_tau_b.csv, grouping all); median error
  (estimate - truth) and median absolute error with intervals (quantity_errors.csv); EXPLORATORY: the quartiles of the
  absolute error (numpy percentiles, linear); acceleration classes at +-0.5 m/s2 (accel_classes.csv). The control
  without pictures: every frame answered with the median of the truth over the 1,084 frames (the analysis's
  baseline_answer); its median absolute error with interval is copied (exploratory_quantity_errors.csv); EXPLORATORY:
  its median error (point and interval on the same resamples, the answer fixed, as the analysis does for its median
  absolute error), absolute-error quartiles and acceleration classes; its tau-b is undefined (a constant answer has
  no pair in order).
3 chain table, grouping all, the seven lead SSMs: risk score vs true SSM (the risk analysis's agreement.csv,
  tau_b_conflict = tau_b_all here), the model's SSM vs the true SSM and the risk score vs the model's SSM
  (perceived_ssm_tau_b.csv); EXPLORATORY: per SSM the frames where the model's SSM has no value and why, from the
  model's frame estimates through waymo_ssm_frames.lead_ssms (PARAMS), as analyze_quantities.perceived_ssms does.
Checks (stop marks, exit 1 if one is not met): the drawn segments (sha256 48bc1f41...); recomputed here and the same
text as the analysis: the model's median error and median absolute error with intervals (quantity_errors.csv), the
baseline answer, the baseline's median absolute error with interval and the error IQR (exploratory_quantity_errors.csv),
the always-steady share (accel_classes.csv); the model's gap and SSMs per frame equal perceived_ssm_frames.csv; the
counts with a value equal perceived_ssm_missing.csv; the rows read are at the lines cited.
usage: python -I paper_sec44.py <waymo_v143 folder> <out_dir>
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import csv  # noqa: E402
import hashlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_vlm_ssm as av  # noqa: E402
from run_vlm import NOW, QUANTITIES, quantity_question  # noqa: E402
from waymo_ssm_frames import PARAMS, lead_ssms  # noqa: E402

MODEL, COND = 'qwen3.6-27b', 'sep_camera'
N_BOOT, SEED = 2000, 20261021
HASH = '48bc1f413f9d5b99da57ff4edde7bbb4eb378c5a9343c44ccad4e0165466604c'
QUANTS = list(QUANTITIES)
LEAD7 = ['TTC', 'THW', 'DRAC', 'MTTC', 'CI', 'PICUD', 'PSD']
STEADY = 0.5
L, OUT = sys.argv[1:3]
os.makedirs(OUT, exist_ok=True)
Q = os.path.join(L, 'analysis', 'quantities_after_27b_think', 'main_1084')
RISK = os.path.join(L, 'analysis', 'after_qwen3.5-4b', 'main_1084')
TRUTH = os.path.join(L, 'quantities', 'truth_main.csv')
failed = []


def check(ok, what):
    print(f'  check {"ok" if ok else "NOT MET"}: {what}')
    if not ok:
        failed.append(what)


def rows_of(path, **match):
    """(line number, raw row dict, raw line) of the rows whose columns equal match."""
    with open(path, newline='', encoding='utf-8') as f:
        lines = f.read().splitlines()
    with open(path, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        out = [(rd.line_num, d) for d in rd if all(d.get(k) == v for k, v in match.items())]
    return [(ln, d, lines[ln - 1]) for ln, d in out]


def txt(x):
    return '' if x is None or (isinstance(x, float) and np.isnan(x)) else repr(float(x))


def ci(arr):
    ok = ~np.isnan(arr)
    if ok.sum() < 0.9 * len(arr):
        return np.nan, np.nan, int(ok.sum())
    lo, hi = np.percentile(arr[ok], [2.5, 97.5])
    return float(lo), float(hi), int(ok.sum())


def cite(path, rows, cols=None):
    for ln, d, raw in rows:
        print(f'    {os.path.basename(path)} line {ln}: {raw if cols is None else ", ".join(f"{c}={d[c]}" for c in cols)}')


print(f'paper section 4.4 numbers (user, 2026-11-12), {MODEL} x {COND}, main result; run '
      f'{time.strftime("%Y-%m-%d %H:%M:%S")}\nanalysis folder {Q}')

# ============================================================================================================ 1
print('\n' + '=' * 110 + '\n1. settings')
print(f'  the five questions as sent for {COND} ({{now}} = {NOW[COND]!r}; code/run_vlm.py QUANTITIES and '
      'quantity_question), each closing the same prompt as the risk question (system prompt, pictures, captions):')
for q in QUANTS:
    print(f'  [{q}] ' + repr(quantity_question(q).replace('{now}', NOW[COND])))
p_ans = os.path.join(Q, 'quantity_answers.csv')
ra = rows_of(p_ans, model=MODEL, condition=COND)
check([ln for ln, _, _ in ra] == list(range(52, 57)), f'quantity_answers.csv rows at lines 52-56 ({[x[0] for x in ra]})')
cite(p_ans, ra)
tot = {k: sum(int(d[k]) for _, d, _ in ra) for k in ('requests', 'read', 'at_max_tokens', 'frames_without_answer')}
print(f'  in all: {tot["requests"]} requests, {tot["read"]} read, {tot["requests"] - tot["read"]} not read; '
      f'{tot["at_max_tokens"]} at max_tokens; frames without any readable answer {tot["frames_without_answer"]}')
for ln, d, _ in ra:
    print(f"    {d['quantity']:10s}: not read {int(d['requests']) - int(d['read'])} of {d['requests']}; at max_tokens "
          f"{d['at_max_tokens']}")
with open(os.path.join(Q, 'params.txt'), encoding='utf-8') as f:
    for i, t in enumerate(f.read().splitlines(), 1):
        if t.startswith(('frames asked', 'bootstrap:', 'model qwen3.6-27b:', 'implausible', 'lead_ssms parameters')):
            print(f'    params.txt line {i}: {t}')

# ============================================================================================================ data
qf = pd.read_csv(os.path.join(Q, 'quantity_frames.csv'), float_precision='round_trip')
qf = qf[(qf.model == MODEL) & (qf.condition == COND)]
frames = qf[qf.quantity == 'distance'][['segment', 'frame_idx']].sort_values(['segment', 'frame_idx']) \
    .reset_index(drop=True)
n = len(frames)
mi = pd.MultiIndex.from_frame(frames)
est, tru = {}, {}
for q in QUANTS:
    s = qf[qf.quantity == q].set_index(['segment', 'frame_idx'])
    check(len(s) == n and s.index.isin(mi).all(), f'quantity_frames.csv {q}: {len(s)} frames, the same {n}')
    est[q] = s['estimate'].reindex(mi).to_numpy(float)
    tru[q] = s['truth'].reindex(mi).to_numpy(float)
check(n == 1084 and all(not np.isnan(est[q]).any() and not np.isnan(tru[q]).any() for q in QUANTS),
      f'{n} frames; every frame has an estimate and a truth for every quantity')
seg_names = frames['segment'].to_numpy()
seg_list = np.array(sorted(set(seg_names)))
seg_frames = {s: np.flatnonzero(seg_names == s) for s in seg_list}
rng = np.random.default_rng(SEED)
draws = [rng.choice(seg_list, len(seg_list)) for _ in range(N_BOOT)]
h = hashlib.sha256()
for d in draws:
    h.update(np.searchsorted(seg_list, d).astype(np.int32).tobytes())
check(h.hexdigest() == HASH and len(seg_list) == 238, f"drawn segments' sha256 {h.hexdigest()} (238 segments)")
idxs = [np.concatenate([seg_frames[s] for s in d]) for d in draws]

# ============================================================================================================ 2
print('\n' + '=' * 110 + '\n2. accuracy against the truth (grouping all)')
p_tau = os.path.join(Q, 'quantity_tau_b.csv')
p_err = os.path.join(Q, 'quantity_errors.csv')
p_exp = os.path.join(Q, 'exploratory_quantity_errors.csv')
p_acc = os.path.join(Q, 'accel_classes.csv')
rt = rows_of(p_tau, model=MODEL, condition=COND, grouping='all')
re_ = rows_of(p_err, model=MODEL, condition=COND, grouping='all')
rx = rows_of(p_exp, model=MODEL, condition=COND, grouping='all')
rc = rows_of(p_acc, model=MODEL, condition=COND)
check([x[0] for x in rt] == [152, 155, 158, 161, 164] and [x[0] for x in re_] == list(range(52, 57))
      and [x[0] for x in rx] == list(range(52, 57)) and [x[0] for x in rc] == [22, 23],
      'rows at quantity_tau_b.csv lines 152-164 (every 3rd), quantity_errors.csv and exploratory_quantity_errors.csv '
      'lines 52-56, accel_classes.csv lines 22-23')
print('  copied (tau-b, median error, median absolute error, the control\'s answer and median absolute error):')
cite(p_tau, rt)
cite(p_err, re_)
cite(p_exp, rx, ['quantity', 'median_abs_error', 'baseline_answer', 'baseline_median_abs_error',
                 'baseline_median_abs_error_lo', 'baseline_median_abs_error_hi', 'baseline_median_abs_error_nboot',
                 'error_iqr'])
cite(p_acc, rc)
erow = {d['quantity']: d for _, d, _ in re_}
xrow = {d['quantity']: d for _, d, _ in rx}
out2 = []
print('  EXPLORATORY (computed here): absolute-error quartiles; the control\'s median error, quartiles, classes')
for q in QUANTS:
    e = est[q] - tru[q]
    base = float(np.median(tru[q]))
    eb = base - tru[q]
    stat = {}
    for tag, x in (('model', e), ('control', eb)):
        a = np.abs(x)
        q25, q75 = (float(v) for v in np.percentile(a, [25, 75]))
        q50 = float(np.median(a))                   # the analysis's median absolute error
        bias = float(np.median(x))
        bb = np.array([np.median(x[i]) for i in idxs])
        mb = np.array([np.median(np.abs(x[i])) for i in idxs])
        stat[tag] = dict(bias=bias, bias_ci=ci(bb), mae=float(np.median(a)), mae_ci=ci(mb), q25=q25, q50=q50, q75=q75)
    m, c = stat['model'], stat['control']
    iqr = float(np.subtract(*np.percentile(e, [75, 25])))
    same = [txt(m['bias']) == erow[q]['median_error'], txt(m['bias_ci'][0]) == erow[q]['median_error_lo'],
            txt(m['bias_ci'][1]) == erow[q]['median_error_hi'], txt(m['mae']) == erow[q]['median_abs_error'],
            txt(m['mae_ci'][0]) == erow[q]['median_abs_error_lo'], txt(m['mae_ci'][1]) == erow[q]['median_abs_error_hi'],
            txt(base) == xrow[q]['baseline_answer'], txt(c['mae']) == xrow[q]['baseline_median_abs_error'],
            txt(c['mae_ci'][0]) == xrow[q]['baseline_median_abs_error_lo'],
            txt(c['mae_ci'][1]) == xrow[q]['baseline_median_abs_error_hi'], txt(iqr) == xrow[q]['error_iqr']]
    check(all(same), f'{q}: recomputed = the analysis (median error and median absolute error with intervals, the '
          f"control's answer, its median absolute error with interval, error IQR): {sum(same)} of {len(same)} the same text")
    print(f"    {q:10s} model:   |error| quartiles {m['q25']!r} / {m['q50']!r} / {m['q75']!r}")
    print(f"    {'':10s} control: answer {base!r}; median error {c['bias']!r} [{c['bias_ci'][0]!r}, {c['bias_ci'][1]!r}] "
          f"(nboot {c['bias_ci'][2]}); median |error| {c['mae']!r} [{c['mae_ci'][0]!r}, {c['mae_ci'][1]!r}]; |error| "
          f"quartiles {c['q25']!r} / {c['q50']!r} / {c['q75']!r}; tau-b undefined (one answer for every frame)")
    out2.append(dict(quantity=q, model_abs_error_q25=m['q25'], model_abs_error_median=m['q50'],
                     model_abs_error_q75=m['q75'], control_answer=base, control_median_error=c['bias'],
                     control_median_error_lo=c['bias_ci'][0], control_median_error_hi=c['bias_ci'][1],
                     control_median_error_nboot=c['bias_ci'][2], control_median_abs_error=c['mae'],
                     control_median_abs_error_lo=c['mae_ci'][0], control_median_abs_error_hi=c['mae_ci'][1],
                     control_abs_error_q25=c['q25'], control_abs_error_median=c['q50'], control_abs_error_q75=c['q75']))
pd.DataFrame(out2).to_csv(os.path.join(OUT, 'exploratory_sec44_errors.csv'), index=False)
crow = {d['quantity']: d for _, d, _ in rc}
out_c = []
for q in ('ego_accel', 'lead_accel'):
    tc = np.where(tru[q] > STEADY, 1, np.where(tru[q] < -STEADY, -1, 0))
    base = float(np.median(tru[q]))
    bc = 1 if base > STEADY else -1 if base < -STEADY else 0
    names = {1: 'up', 0: 'steady', -1: 'down'}
    r = dict(quantity=q, control_answer=base, control_class=names[bc])
    for k, nm in names.items():
        r[f'truth_{nm}'] = int((tc == k).sum())
        r[f'control_share_right_{nm}'] = float(bc == k) if (tc == k).any() else np.nan
    r['control_share_right_all'] = float((tc == bc).mean())
    same = txt(r['control_share_right_all']) == crow[q]['share_always_steady'] if bc == 0 else False
    check(same and all(str(r[f'truth_{nm}']) == crow[q][f'truth_{nm}'] for nm in names.values()),
          f"{q}: the control's answer {base!r} is 'steady', so it is right on the steady frames only; its share "
          f"{r['control_share_right_all']!r} = share_always_steady {crow[q]['share_always_steady']} (accel_classes.csv); "
          'truth classes the same')
    out_c.append(r)
pd.DataFrame(out_c).to_csv(os.path.join(OUT, 'exploratory_sec44_control_accel_classes.csv'), index=False)

# ============================================================================================================ 3
print('\n' + '=' * 110 + '\n3. chain table, grouping all (the seven lead SSMs)')
p_agr = os.path.join(RISK, 'agreement.csv')
ragr = [x for x in rows_of(p_agr, model=MODEL, condition=COND, grouping='all') if x[1]['ssm'] in LEAD7]
p_per = os.path.join(Q, 'perceived_ssm_tau_b.csv')
rper = rows_of(p_per, model=MODEL, condition=COND, grouping='all')
check([x[0] for x in ragr] == list(range(866, 885, 3)) and [x[0] for x in rper] == list(range(422, 462, 3)),
      'agreement.csv lines 866-884 (every 3rd), perceived_ssm_tau_b.csv lines 422-461 (every 3rd)')
print('  copied: risk score vs true SSM (agreement.csv tau_b_conflict); model SSM vs true SSM and risk score vs model '
      'SSM (perceived_ssm_tau_b.csv; tau_b over all frames, the model SSM without a value tied at the safe end; '
      'tau_b_with_value over the frames where it has one)')
cite(p_agr, ragr, ['ssm', 'n_conflict', 'tau_b_conflict', 'tau_b_conflict_lo', 'tau_b_conflict_hi',
                   'tau_b_conflict_nboot', 'tau_b_all'])
cite(p_per, rper, ['ssm', 'comparison', 'frames', 'frames_model_ssm_with_value', 'tau_b', 'tau_b_lo', 'tau_b_hi',
                   'nboot', 'tau_b_with_value', 'tau_b_with_value_lo', 'tau_b_with_value_hi', 'nboot_with_value'])
check(all(d['tau_b_conflict'] == d['tau_b_all'] for _, d, _ in ragr), 'agreement.csv: tau_b_conflict = tau_b_all')

print('  EXPLORATORY: the model\'s SSMs recomputed from its frame estimates (gap = estimated distance - the segment\'s '
      f'camera offset; the model\'s ego and lead speeds and accelerations; lead_ssms, PARAMS {PARAMS})')
tr = frames.merge(pd.read_csv(TRUTH, float_precision='round_trip'), on=['segment', 'frame_idx'], how='left',
                  validate='one_to_one')
off = tr['camera_offset'].to_numpy(float)
d_, vf, vl, af, al = (est[q] for q in ('distance', 'ego_speed', 'lead_speed', 'ego_accel', 'lead_accel'))
gap = d_ - off
vals = {s: np.full(n, np.nan) for s in LEAD7}
for i in range(n):
    o = lead_ssms(gap[i], vf[i], vl[i], af[i], al[i])
    for s in LEAD7:
        vals[s][i] = o[s]
pf = pd.read_csv(os.path.join(Q, 'perceived_ssm_frames.csv'), float_precision='round_trip')
pf = pf[(pf.model == MODEL) & (pf.condition == COND)].set_index(['segment', 'frame_idx']).reindex(mi)
check(bool(pf['valid'].all()) and np.array_equal(pf['gap'].to_numpy(float), gap)
      and all(np.array_equal(pf[s].to_numpy(float), vals[s], equal_nan=True) for s in LEAD7),
      'every frame valid; the gap and the seven SSMs per frame equal perceived_ssm_frames.csv')
has = {s: (np.nan_to_num(vals[s], nan=0.0) > 0) if s in av.POSITIVE_IS_CONFLICT else np.isfinite(vals[s])
       for s in LEAD7}
p_mis = os.path.join(Q, 'perceived_ssm_missing.csv')
rm = rows_of(p_mis, model=MODEL, condition=COND)
cite(p_mis, rm)
check(len(rm) == 1 and rm[0][0] == 12 and all(int(rm[0][1][f'with_value_{s}']) == int(has[s].sum()) for s in LEAD7),
      'frames with a value per SSM = perceived_ssm_missing.csv line 12')
closing = vf - vl
dacc = af - al
no_close = closing <= 0
slow = vf <= PARAMS['thw_v_min']
m_inf = ~np.isfinite(vals['MTTC'])
flat = np.abs(dacc) < 1e-6
disc = closing ** 2 + 2 * dacc * gap
reasons = []


def add(ssm, why, mask):
    reasons.append(dict(ssm=ssm, reason=why, frames=int(mask.sum())))
    print(f'    {ssm:5s} {int(mask.sum()):4d} frames: {why}')


print(f'  frames without a value (of {n}):')
add('TTC', 'estimated ego speed <= estimated lead speed (not closing in; TTC = gap / closing is infinite)', no_close)
add('TTC', '  of them ego speed < lead speed', closing < 0)
add('TTC', '  of them ego speed = lead speed', closing == 0)
add('DRAC', 'the same frames (DRAC = closing^2 / (2 gap) only when closing > 0, else 0 = no value)', ~has['DRAC'])
check(np.array_equal(~has['TTC'], no_close) and np.array_equal(~has['DRAC'], no_close),
      'TTC and DRAC: exactly the frames not closing in')
add('THW', f"estimated ego speed <= {PARAMS['thw_v_min']} m/s (THW = gap / v_ego only above it)", slow)
add('THW', '  of them estimated ego speed exactly 0', vf == 0)
add('PSD', 'the same frames (PSD = gap / (v_ego^2 / (2 a_emerg)) only above 0.1 m/s)', ~has['PSD'])
check(np.array_equal(~has['THW'], slow) and np.array_equal(~has['PSD'], slow), 'THW and PSD: exactly the frames with '
      'ego speed <= 0.1 m/s')
add('MTTC', 'no positive time to collision with constant accelerations (gap = closing t + (a_ego - a_lead) t^2 / 2)',
    m_inf)
add('MTTC', '  relative acceleration 0 and not closing in', m_inf & flat)
add('MTTC', '  the relative acceleration stops the closing before contact (closing^2 + 2 (a_ego - a_lead) gap < 0)',
    m_inf & ~flat & (disc < 0))
add('MTTC', '  both roots <= 0 (opening, and the relative acceleration does not bring them back together)',
    m_inf & ~flat & (disc >= 0))
add('MTTC', '  (of the 373: not closing in now)', m_inf & no_close)
check(np.array_equal(m_inf, ~has['MTTC']), 'MTTC: no value = infinite')
ci_le0 = np.isfinite(vals['MTTC']) & (vals['MTTC'] > 0) & ~has['CI']
add('CI', 'MTTC has no value (CI = 0 then)', m_inf & ~has['CI'])
add('CI', 'MTTC has a value but CI <= 0 ((v_ego + a_ego MTTC)^2 <= (v_lead + a_lead MTTC)^2: the ego not faster at '
    'the contact time)', ci_le0)
check(int((~has['CI']).sum()) == int((m_inf & ~has['CI']).sum()) + int(ci_le0.sum()),
      'CI: no value = no MTTC + CI <= 0')
add('PICUD', 'always has a value (vl^2 / (2 a_emerg) + gap - (v_ego tau + v_ego^2 / (2 a_emerg)))', ~has['PICUD'])
pd.DataFrame(reasons).to_csv(os.path.join(OUT, 'exploratory_sec44_model_ssm_without_value.csv'), index=False)
print(f'  the true SSMs: every frame has a value of every lead SSM in the main result (agreement.csv n_conflict '
      f"{sorted({d['n_conflict'] for _, d, _ in ragr})})")
print(f'\nwritten: exploratory_sec44_errors.csv, exploratory_sec44_control_accel_classes.csv, '
      f'exploratory_sec44_model_ssm_without_value.csv in {OUT}')
print(f'checks not met: {len(failed)}')
for w in failed:
    print('  ', w)
sys.exit(1 if failed else 0)
