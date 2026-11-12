"""Paper section 4.3, one more number (user, 2026-11-12). EXPLORATORY; CPU only. Qwen3.6-27B with thinking on,
sep_camera, the main result's 1,084 frames.

The rank regression of the rule analysis (analyze_rules.py) and of paper_sec43.py item 3, with K + time of day +
weather and no location ("K+T+W"): dependent variable the rank of score_mean among the 1,084 frames / 1,084 (ties the
mean rank, analyze_rules.rank01); K = the true distance, ego speed, closing speed (v_ego - v_lead), ego acceleration
and lead acceleration (quantities/truth_main.csv), each as its rank / 1,084 (analyze_rules.K_TRUE); time of day and
weather as the rule analysis's S dummies (references Day and dry; fog and unclear have 0 segments, so their dummies
are all zero and their coefficients undefined, as there); least squares with an intercept and the null-space rule
(analyze_rules.ols). Columns in the rule analysis's order: K, time of day, weather.
Reported: every coefficient (rain and wet first) with its interval from the main analysis's 2,000 resamples of whole
segments (seed 20261021; ranks taken anew and the fit redone in each resample; every drawn copy is its own segment;
2.5 / 97.5 % percentiles, numpy linear, none when fewer than 90 % defined: analyze_rules.ci); the in-sample R2 with its
interval; the leave-one-segment-out R2 as analyze_rules.py --oos (analyze_rules.loso_predict: minimum-norm least
squares on the other segments; R2 = 1 - SSE / sum of squares about the mean over all frames; interval: per resample
the drawn segments' SSE and sums of squares added, no refitting).
Checks: the drawn segments' sha256 is the rule and quantities analyses' (48bc1f41...), seed, number, frames and
segment list the risk analysis's; with location added back (K+S: the rule analysis's S in its order), the 14 rows of
exploratory_rule_coefficients.csv (lines 7470-7483) rebuilt here must be the same text character for character
(user), and the R2 (exploratory_rule_r2.csv line 500) and the out-of-sample R2 (exploratory_oos_r2.csv line 493) within
1e-12 (whether their text is the same is printed).
usage: python -I paper_sec43_k_tw.py <waymo_v143 folder> <scene_labels.csv> <out_dir>
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
import analyze_rules as ar  # noqa: E402

MODEL, COND = 'qwen3.6-27b', 'sep_camera'
N_BOOT, SEED = 2000, 20261021
HASH = '48bc1f413f9d5b99da57ff4edde7bbb4eb378c5a9343c44ccad4e0165466604c'   # rule analysis exploratory_params.txt
TOL = 1e-12
COEF_LINES = list(range(7470, 7484))
L, SL, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
A = os.path.join(L, 'analysis')
RISK = os.path.join(A, 'after_qwen3.5-4b', 'main_1084')
RULE = os.path.join(A, 'rule_explore', 'main_1084')
QUANT = os.path.join(A, 'quantities_after_27b_think', 'main_1084')
TRUTH = os.path.join(L, 'quantities', 'truth_main.csv')
failed = []


def check(ok, what):
    print(f'  check {"ok" if ok else "NOT MET"}: {what}')
    if not ok:
        failed.append(what)


def csv_rows(path):
    """(line number, row as raw strings); line 1 is the header."""
    with open(path, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        for d in rd:
            yield rd.line_num, d


def text_lines(path):
    with open(path, encoding='utf-8') as f:
        return f.read().splitlines()


def txt(x):
    """A value as pandas writes it in the analyses' csv files ('' for NaN)."""
    return '' if x is None or (isinstance(x, float) and np.isnan(x)) else repr(float(x))


def r(x):
    return 'nan' if np.isnan(x) else repr(float(x))


print(f'paper section 4.3, K + time of day + weather (user, 2026-11-12), {MODEL} x {COND}; run '
      f'{time.strftime("%Y-%m-%d %H:%M:%S")}\nEXPLORATORY: everything here is computed')
frames = pd.read_csv(os.path.join(RISK, 'ssm_target_frames.csv'))[['segment', 'frame_idx']] \
    .sort_values(['segment', 'frame_idx']).reset_index(drop=True)
n = len(frames)
check(n == 1084, f'{n} frames (ssm_target_frames.csv of the risk analysis)')
seg_names = frames['segment'].to_numpy()
seg_list = np.array(sorted(set(seg_names)))
mi = pd.MultiIndex.from_frame(frames)
f = pd.read_csv(os.path.join(RISK, 'vlm_frame_scores.csv'), float_precision='round_trip')
s_ = f[(f.model == MODEL) & (f.condition == COND)].set_index(['segment', 'frame_idx'])['score_mean']
check(len(s_) == n and s_.index.isin(mi).all(), f'vlm_frame_scores.csv: {MODEL} {COND} has the {n} frames')
risk = s_.reindex(mi).to_numpy(float)
check(not np.isnan(risk).any(), 'score_mean of every frame known')
seg_frames = {s: np.flatnonzero(seg_names == s) for s in seg_list}
seg_code = np.searchsorted(seg_list, seg_names)
n_boot_r, seed_r, frames_r, segs_r = ar.read_params_risk(os.path.join(RISK, 'params.txt'))
check((n_boot_r, seed_r, frames_r) == (N_BOOT, SEED, n) and list(seg_list) == segs_r,
      f"resamples, seed, frames and segment list as the risk analysis's params.txt ({n_boot_r}, {seed_r}, {frames_r}, "
      f'{len(segs_r)} segments)')
rng = np.random.default_rng(SEED)
draws = [rng.choice(seg_list, len(seg_list)) for _ in range(N_BOOT)]
h = hashlib.sha256()
for d in draws:
    h.update(np.searchsorted(seg_list, d).astype(np.int32).tobytes())
nb_q, seed_q, hash_q = ar.read_hash_quantities(os.path.join(QUANT, 'params.txt'))
check(h.hexdigest() == HASH and (nb_q, seed_q, hash_q) == (N_BOOT, SEED, HASH),
      f"drawn segments' sha256 {h.hexdigest()} = the rule analysis's and the quantities analysis's")
counts = np.stack([np.bincount(np.searchsorted(seg_list, d), minlength=len(seg_list)) for d in draws])

# S, as analyze_rules.main builds it (time of day, location, weather; the reference and the order as there)
lab = pd.read_csv(SL, dtype=str, keep_default_na=False).set_index('segment')
s_terms, s_cols, s_var, s_ref = [], [], [], {}
for col in ('time_of_day', 'location', 'weather'):
    v = lab.loc[seg_names, col].str.strip().to_numpy()
    seg_v = lab.loc[seg_list, col].str.strip()
    if col == 'weather':
        cats = ar.WEATHER
        nseg = {k: int((seg_v == k).sum()) for k in cats}
        ref = max(cats, key=lambda k: nseg[k])
    else:
        cats = sorted(set(v))
        nfr = {k: int((v == k).sum()) for k in cats}
        ref = max(cats, key=lambda k: nfr[k])
    s_ref[col] = ref
    for k in cats:
        if k != ref:
            s_terms.append(f'{col}={k}')
            s_cols.append((v == k).astype(float))
            s_var.append(col)
check(s_ref == {'time_of_day': 'Day', 'location': 'location_sf', 'weather': 'dry'},
      f'reference classes {s_ref} (Day, location_sf, dry)')
tr = frames.merge(pd.read_csv(TRUTH, float_precision='round_trip'), on=['segment', 'frame_idx'], how='left',
                  validate='one_to_one', indicator=True)
check((tr['_merge'] == 'both').all(), 'every frame in the truth csv')
cue = {'true_distance': tr['distance'].to_numpy(float), 'true_ego_speed': tr['v_ego'].to_numpy(float),
       'true_closing_speed': tr['v_ego'].to_numpy(float) - tr['v_lead'].to_numpy(float),
       'true_ego_accel': tr['a_ego'].to_numpy(float), 'true_lead_accel': tr['a_lead'].to_numpy(float)}
FITS = {'K+T+W': [j for j, v in enumerate(s_var) if v in ('time_of_day', 'weather')],
        'K+S': list(range(len(s_var)))}
TERMS = {k: ['intercept'] + ar.K_TRUE + [s_terms[j] for j in js] for k, js in FITS.items()}
print(f"  terms of K+T+W: {TERMS['K+T+W']}")


def fit(idx):
    y = ar.rank01(risk[idx])
    kc = [ar.rank01(cue[k][idx]) for k in ar.K_TRUE]
    out = {}
    for name, js in FITS.items():
        coef, _, r2 = ar.ols(y, kc + [s_cols[j][idx] for j in js])
        out[(name, 'r2')] = r2
        for term, b in zip(TERMS[name], coef):
            out[(name, term)] = b
    return out


ids = np.arange(n)
point = fit(ids)
keys = list(point)
boot = np.full((N_BOOT, len(keys)), np.nan)
tb = time.time()
for b, draw in enumerate(draws):
    rb = fit(np.concatenate([seg_frames[s] for s in draw]))
    boot[b] = [rb.get(k, np.nan) for k in keys]
print(f'  2,000 resamples refitted in {time.time() - tb:.1f} s')
col_of = {k: i for i, k in enumerate(keys)}
y0 = ar.rank01(risk)


def oos(name):
    cols = [ar.rank01(cue[k]) for k in ar.K_TRUE] + [s_cols[j] for j in FITS[name]]
    pred = ar.loso_predict(y0, cols, seg_code)
    sse = np.bincount(seg_code, (y0 - pred) ** 2, minlength=len(seg_list))
    sst = np.bincount(seg_code, (y0 - y0.mean()) ** 2, minlength=len(seg_list))
    lo, hi, nv = ar.ci(1.0 - (counts @ sse) / (counts @ sst))
    return float(1.0 - sse.sum() / sst.sum()), lo, hi, nv


res = {}
for name in FITS:
    lo, hi, nv = ar.ci(boot[:, col_of[(name, 'r2')]])
    res[name] = dict(r2=(point[(name, 'r2')], lo, hi, nv), oos=oos(name),
                     coef={t: (point[(name, t)],) + ar.ci(boot[:, col_of[(name, t)]]) for t in TERMS[name]})

print('\n' + '=' * 110)
print('K + time of day + weather, no location (EXPLORATORY):')
k = res['K+T+W']
for t in ['weather=rain', 'weather=wet']:
    b, lo, hi, nv = k['coef'][t]
    print(f'  {t:13s} {r(b)} [{r(lo)}, {r(hi)}] (nboot {nv}){"  interval without 0" if nv and lo * hi > 0 else ""}')
b, lo, hi, nv = k['r2']
print(f'  R2 in sample {r(b)} [{r(lo)}, {r(hi)}] (nboot {nv})')
b, lo, hi, nv = k['oos']
print(f'  R2 out of sample (leave one segment out) {r(b)} [{r(lo)}, {r(hi)}] (nboot {nv})')
print('  every term:')
for t in TERMS['K+T+W']:
    b, lo, hi, nv = k['coef'][t]
    print(f'    {t:24s} {r(b)} [{r(lo)}, {r(hi)}] (nboot {nv})')
ks = res['K+S']
print('  for comparison, K+S (with location; the rule analysis): rain {} [{}, {}], wet {} [{}, {}]; R2 {} [{}, {}]; '
      'out of sample {} [{}, {}]'.format(*(r(x) for x in ks['coef']['weather=rain'][:3] + ks['coef']['weather=wet'][:3]
                                            + ks['r2'][:3] + ks['oos'][:3])))
rows_c = [dict(fit=nm, term=t, coef=c[0], coef_lo=c[1], coef_hi=c[2], nboot=c[3])
          for nm in FITS for t, c in res[nm]['coef'].items()]
rows_r = [dict(fit=nm, frames=n, segments=len(seg_list), r2_in=res[nm]['r2'][0], r2_in_lo=res[nm]['r2'][1],
               r2_in_hi=res[nm]['r2'][2], r2_in_nboot=res[nm]['r2'][3], r2_oos=res[nm]['oos'][0],
               r2_oos_lo=res[nm]['oos'][1], r2_oos_hi=res[nm]['oos'][2], r2_oos_nboot=res[nm]['oos'][3])
          for nm in FITS]
pd.DataFrame(rows_c).to_csv(os.path.join(OUT, 'exploratory_sec43_k_time_weather_coefficients.csv'), index=False)
pd.DataFrame(rows_r).to_csv(os.path.join(OUT, 'exploratory_sec43_k_time_weather_r2.csv'), index=False)
print(f'  written: exploratory_sec43_k_time_weather_coefficients.csv, exploratory_sec43_k_time_weather_r2.csv in {OUT}')

print('\n' + '=' * 110)
print('check: location added back (K+S) against the rule analysis')
p_c = os.path.join(RULE, 'exploratory_rule_coefficients.csv')
lines_c = text_lines(p_c)
cf = [(ln, d) for ln, d in csv_rows(p_c)
      if d['target'] == 'risk' and d['model'] == MODEL and d['condition'] == COND and d['fit'] == 'K+S']
check([ln for ln, _ in cf] == COEF_LINES, f'the K+S rows of {MODEL} {COND} are lines {cf[0][0]}-{cf[-1][0]} '
      f'({len(cf)} rows; asked: 7470-7483)')
check([d['term'] for _, d in cf] == TERMS['K+S'], f"terms and their order the same as here: {TERMS['K+S']}")
n_same = 0
for ln, d in cf:
    c = ks['coef'].get(d['term'])
    mine = ','.join(['risk', MODEL, COND, '', 'K+S', d['term'], txt(c[0]), txt(c[1]), txt(c[2]), str(c[3])]) \
        if c else None
    same = mine == lines_c[ln - 1]
    n_same += same
    print(f'  line {ln}: {"same text" if same else "DIFFERENT"}: {lines_c[ln - 1]}' + ('' if same else f'\n    here: {mine}'))
check(n_same == len(COEF_LINES), f'{n_same} of {len(COEF_LINES)} rows the same text, character for character')
p_r = os.path.join(RULE, 'exploratory_rule_r2.csv')
rr = [(ln, d) for ln, d in csv_rows(p_r)
      if d['target'] == 'risk' and d['model'] == MODEL and d['condition'] == COND and d['fit'] == 'K+S']
b, lo, hi, nv = ks['r2']
mine = ','.join(['risk', MODEL, COND, '', 'K+S', str(n), txt(b), txt(lo), txt(hi), str(nv)])
dd = max(abs(x - float(t)) for x, t in zip((b, lo, hi), (rr[0][1]['r2'], rr[0][1]['r2_lo'], rr[0][1]['r2_hi'])))
check(len(rr) == 1 and rr[0][0] == 500 and dd <= TOL and str(nv) == rr[0][1]['nboot'],
      f'R2 (exploratory_rule_r2.csv line {rr[0][0]}): largest |difference| {dd!r}; the same text: '
      f'{mine == text_lines(p_r)[rr[0][0] - 1]}')
p_o = os.path.join(RULE, 'exploratory_oos_r2.csv')
ro = [(ln, d) for ln, d in csv_rows(p_o)
      if d['target'] == 'risk' and d['model'] == MODEL and d['condition'] == COND and d['fit'] == 'K+S']
b, lo, hi, nv = ks['oos']
mine = ','.join(['risk', MODEL, COND, 'K+S', str(n), str(len(seg_list)), txt(ks['r2'][0]), txt(b), txt(lo), txt(hi),
                 str(nv)])
dd = max(abs(x - float(t)) for x, t in zip((b, lo, hi), (ro[0][1]['r2_oos'], ro[0][1]['r2_oos_lo'],
                                                        ro[0][1]['r2_oos_hi'])))
check(len(ro) == 1 and ro[0][0] == 493 and dd <= TOL and str(nv) == ro[0][1]['nboot'],
      f'out-of-sample R2 (exploratory_oos_r2.csv line {ro[0][0]}): largest |difference| {dd!r}; the same text: '
      f'{mine == text_lines(p_o)[ro[0][0] - 1]}')
print(f'\nchecks not met: {len(failed)}')
for w in failed:
    print('  ', w)
sys.exit(1 if failed else 0)
