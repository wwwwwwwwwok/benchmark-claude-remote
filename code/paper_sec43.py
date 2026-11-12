"""Numbers for paper section 4.3, scene conditions (user, 2026-11-11): Qwen3.6-27B with thinking on, sep_camera, the
main result's 1,084 frames. Items 1, 5 and the counts of item 2 that the analyses printed are copied as text with file
and line; everything computed here is EXPLORATORY (item 4 descriptive). No formal output is read-modified or written;
the outputs go to <out_dir>.

 1  same-weather tau-b: analysis/after_qwen3.5-4b_weather/main_1084/agreement.csv, the nine SSMs' tau_b_all with lo,
    hi, nboot; the difference to 'all' (the two rows' values subtracted) and the order. Its 'all', 'time_of_day' and
    'segment' rows of this model x condition are checked character for character against
    analysis/after_qwen3.5-4b/main_1084/agreement.csv lines 866-892.
 2  frames and frame pairs (descriptive; computed from ssm_target_frames.csv and the scene labels, and compared with
    the counts the analyses printed, which are cited with file and line): per time of day, weather and location the
    segments, frames and pairs of frames within the class (n (n - 1) / 2); the pairs within the same time of day and
    within the same weather; segments with 2 or more frames, frames per segment (min / median / max), pairs within
    segments.
 3  EXPLORATORY, how much the scene conditions explain the score. Dependent variable: the rank of score_mean among the
    1,084 frames / 1,084, ties the mean rank (analyze_rules.rank01). Dummies built as the rule analysis's S
    (analyze_rules.py: time of day and location, reference the category with the most frames, Day and location_sf;
    weather a dummy for each of the user's five classes but the class with the most segments, dry; fog and unclear
    have 0 segments, so their dummies are all zero and their coefficients undefined, as there). Fits: (a) time of
    day; (b) weather; (c) location; (d) time of day + weather; (e) time of day + weather + location (the rule
    analysis's S); (f) one dummy per segment (238 dummies; with the intercept every coefficient is undefined; the
    in-sample R2 is the between-segment share). analyze_rules.ols (least squares with an intercept, the null-space
    rule). In-sample R2 and coefficients: intervals from the 2,000 resamples of whole segments (seed 20261021), ranks
    taken anew and every fit redone in each resample; every drawn copy is its own segment, so (f) has one dummy per
    drawn copy; a class missing from a resample leaves its coefficient undefined there. Out-of-sample R2 by leaving one
    segment out as analyze_rules.py --oos (analyze_rules.loso_predict: minimum-norm least squares on the other
    segments; R2 = 1 - SSE / sum of squares about the mean over all frames; interval: per resample the drawn segments'
    SSE and sums of squares added, no refitting). For (f) a held-out segment's own dummy is all zero in training, so
    its prediction is the minimum-norm intercept (the other 237 segments' mean ranks summed / 238). Intervals: 2.5 /
    97.5 % percentiles (numpy, linear), none when fewer than 90 % of the resamples are defined.
    Checks: the drawn segments' sha256 is the rule and quantities analyses' (48bc1f41...); seed, number, frames and
    segment list are the risk analysis's (params.txt); (e)'s terms are the rule analysis's S terms in its order, and
    with the rule analysis's K added (K+S) the coefficients, R2, their intervals and the out-of-sample R2 reproduce
    exploratory_rule_coefficients.csv, exploratory_rule_r2.csv and exploratory_oos_r2.csv (within 1e-12); (f)'s
    in-sample R2 equals the rule analysis's between-segment share of qwen3.6-27b sep_camera
    (exploratory_between_segment.csv, within 1e-12), and that share and its interval, recomputed here with
    analyze_rules.between_share, are the same text as there; (f)'s out-of-sample predictions equal the formula above
    (within 1e-12).
 4  descriptive: score_mean (0-100, the mean of the 5 answers) by time of day, weather, time of day x weather:
    frames, segments, mean, SD (ddof 1), min, quartiles (numpy, linear), max. sec43_frames.csv: one row per frame
    (segment, frame_idx, score_mean, rank (1-1,084, ties the mean rank), rank / 1,084, time_of_day, weather,
    location), numbers only, for box plots.
 5  E2 (exploratory, copied): the latest exploratory_e2_risk_vs_distance_speed.csv with the seven risk models
    (analysis/quantities_explore_27b_think_grid/main_1084), the rows of qwen3.6-27b sep_camera for the true distance
    and true ego speed; the same rows checked in the formal one (analysis/quantities_after_27b_think/main_1084).
 6  EXPLORATORY: in every resample, on the same frames, tau-b with analyze_vlm_ssm.all_stats as the risk analysis
    computes it (--resample segment; groupings all, weather, time of day, segment; tau_b_all, every frame has every
    SSM), and the differences time of day - all, weather - all, segment - all (9 SSMs x 3), percentile intervals as
    above. Checks: the point differences equal the two agreement.csv rows subtracted (within 1e-12); every grouping's
    own point estimate and interval recomputed here is the same text as in agreement.csv.
usage: python -I paper_sec43.py <waymo_v143 folder> <scene_labels.csv> <out_dir>
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import ast  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_rules as ar  # noqa: E402
import analyze_vlm_ssm as av  # noqa: E402

MODEL, COND = 'qwen3.6-27b', 'sep_camera'
N_BOOT, SEED = 2000, 20261021
HASH = '48bc1f413f9d5b99da57ff4edde7bbb4eb378c5a9343c44ccad4e0165466604c'   # rule analysis exploratory_params.txt
TOL = 1e-12
GROUPS6 = ['time_of_day', 'weather', 'segment']
FITS = [('a', 'time of day', ['time_of_day']), ('b', 'weather', ['weather']), ('c', 'location', ['location']),
        ('d', 'time of day + weather', ['time_of_day', 'weather']),
        ('e', 'time of day + weather + location', ['time_of_day', 'location', 'weather'])]

L, SL, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
A = os.path.join(L, 'analysis')
RISK = os.path.join(A, 'after_qwen3.5-4b', 'main_1084')
WEA = os.path.join(A, 'after_qwen3.5-4b_weather', 'main_1084')
RULE = os.path.join(A, 'rule_explore', 'main_1084')
QUANT = os.path.join(A, 'quantities_after_27b_think', 'main_1084')
E2_LATEST = os.path.join(A, 'quantities_explore_27b_think_grid', 'main_1084', 'exploratory_e2_risk_vs_distance_speed.csv')
E2_FORMAL = os.path.join(QUANT, 'exploratory_e2_risk_vs_distance_speed.csv')
FSSM = os.path.join(L, 'results_lead_final_excl', 'frame_ssm_validation_training.csv')
TRUTH = os.path.join(L, 'quantities', 'truth_main.csv')
t0 = time.time()
failed = []


def check(ok, what):
    print(f'  check {"ok" if ok else "NOT MET"}: {what}')
    if not ok:
        failed.append(what)


def csv_rows(path):
    """(line number, row as raw strings); line 1 is the header."""
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for d in r:
            yield r.line_num, d


def text_lines(path):
    with open(path, encoding='utf-8') as f:
        return f.read().splitlines()


def txt(x):
    """A value as pandas writes it in the analyses' csv files ('' for NaN)."""
    return '' if x is None or (isinstance(x, float) and np.isnan(x)) else repr(float(x))


def r(x):
    return 'nan' if np.isnan(x) else repr(float(x))


def pairs(k):
    return int(k) * (int(k) - 1) // 2


print(f'paper section 4.3 numbers (user, 2026-11-11), {MODEL} x {COND}; run {time.strftime("%Y-%m-%d %H:%M:%S")}')
print('EXPLORATORY: everything computed here (items 2-4 and 6); items 1, 5 and the printed counts of item 2 copied')

# ====================================================================================================== item 1
print('\n' + '=' * 110)
p_wea = os.path.join(WEA, 'agreement.csv')
p_risk = os.path.join(RISK, 'agreement.csv')
print(f'1. same-weather tau-b, copied as text from {p_wea}\n   (column tau_b_all, = tau_b_conflict: every frame has '
      'every SSM); difference = same weather - all, the two rows\' values subtracted here')
wea = {(d['ssm'], d['grouping']): (ln, d) for ln, d in csv_rows(p_wea) if d['model'] == MODEL and d['condition'] == COND}
check(len(wea) == 36 and {g for _, g in wea} == {'all', 'weather', 'time_of_day', 'segment'},
      f'the weather file has 36 rows for {MODEL} {COND} (9 SSMs x all, weather, time_of_day, segment)')
t1 = []
for s in av.SSMS:
    lw, w = wea[(s, 'weather')]
    la, al = wea[(s, 'all')]
    same_c = all(w[f'tau_b_all{x}'] == w[f'tau_b_conflict{x}'] for x in ('', '_lo', '_hi', '_nboot'))
    t1.append(dict(ssm=s, line=lw, tau=w['tau_b_all'], lo=w['tau_b_all_lo'], hi=w['tau_b_all_hi'],
                   nboot=w['tau_b_all_nboot'], n_groups=w['n_groups'], frames=w['n_conflict'], all_line=la,
                   all_tau=al['tau_b_all'], diff=float(w['tau_b_all']) - float(al['tau_b_all']), same_c=same_c))
for d in t1:
    print(f"  {d['ssm']:6s} line {d['line']}: tau_b_all {d['tau']} [{d['lo']}, {d['hi']}], nboot {d['nboot']}, "
          f"groups {d['n_groups']}, frames {d['frames']}; 'all' line {d['all_line']}: {d['all_tau']}; "
          f"same weather - all {d['diff']!r}")
check(all(d['same_c'] for d in t1), 'tau_b_all and tau_b_conflict (and their lo, hi, nboot) the same text in these rows')
print('  order, same weather (largest first): ' + ' > '.join(d['ssm'] for d in sorted(t1, key=lambda d: -float(d['tau']))))
print('  order, all (largest first):          ' + ' > '.join(
    d['ssm'] for d in sorted(t1, key=lambda d: -float(d['all_tau']))))
print('  intervals without 0 (same weather): ' + (', '.join(
    f"{d['ssm']} ({'above' if float(d['lo']) > 0 else 'below'} 0)" for d in t1
    if d['lo'] != '' and (float(d['lo']) > 0 or float(d['hi']) < 0)) or 'none'))
lr, lw_ = text_lines(p_risk), text_lines(p_wea)
check(lr[0] == lw_[0], 'the two agreement.csv files have the same header')
risk_rows = {(d['ssm'], d['grouping']): ln for ln, d in csv_rows(p_risk) if d['model'] == MODEL and d['condition'] == COND}
check(sorted(risk_rows.values()) == list(range(866, 893)),
      f'{p_risk}: the rows of {MODEL} {COND} are lines 866-892 (27 rows: all, time_of_day, segment)')
n_same = 0
for (s, g), ln in sorted(risk_rows.items(), key=lambda kv: kv[1]):
    same = (s, g) in wea and lr[ln - 1] == lw_[wea[(s, g)][0] - 1]
    n_same += same
    if not same:
        print(f'  differs: {s} {g} (risk analysis line {ln})')
check(n_same == 27, f'{n_same} of 27 rows (all, time_of_day, segment) of the weather file are the same text as lines '
      f'866-892 of the risk analysis (weather file lines {sorted(wea[k][0] for k in risk_rows if k in wea)})')

# ====================================================================================================== item 2
print('\n' + '=' * 110)
frames = pd.read_csv(os.path.join(RISK, 'ssm_target_frames.csv'))[['segment', 'frame_idx']] \
    .sort_values(['segment', 'frame_idx']).reset_index(drop=True)
frames_w = pd.read_csv(os.path.join(WEA, 'ssm_target_frames.csv'))[['segment', 'frame_idx']] \
    .sort_values(['segment', 'frame_idx']).reset_index(drop=True)
n = len(frames)
check(n == 1084 and frames.equals(frames_w), 'ssm_target_frames.csv: 1,084 frames, the same in both risk analyses')
seg_names = frames['segment'].to_numpy()
seg_list = np.array(sorted(set(seg_names)))
lab = pd.read_csv(SL, dtype=str, keep_default_na=False).set_index('segment')
lab_f = {col: lab.loc[seg_names, col].str.strip().to_numpy() for col in ('time_of_day', 'weather', 'location')}
print(f'2. frames and frame pairs (descriptive, computed; scene labels {SL}): {n} frames in {len(seg_list)} segments, '
      f'{pairs(n)} frame pairs in all')
ORDER = {'time_of_day': ['Day', 'Dawn/Dusk', 'Night'], 'weather': ['dry', 'wet', 'rain', 'fog', 'unclear'],
         'location': ['location_sf', 'location_phx', 'location_other']}
cnt2, rows2 = {}, []
for col in ('time_of_day', 'weather', 'location'):
    v = lab_f[col]
    extra = sorted(set(v) - set(ORDER[col]))
    check(not extra, f'{col}: every class one of {ORDER[col]} {extra if extra else ""}')
    within = 0
    for k in ORDER[col]:
        m = v == k
        c = dict(segments=len(set(seg_names[m])), frames=int(m.sum()), pairs=pairs(m.sum()))
        cnt2[(col, k)] = c
        within += c['pairs']
        rows2.append(dict(grouping=col, label=k, **c))
        print(f"  {col:11s} {k:15s} segments {c['segments']:4d}, frames {c['frames']:5d}, pairs within {c['pairs']:7d}")
    cnt2[(col, 'within')] = within
    print(f'  {col:11s} pairs within the same class: {within}; classes with frames: '
          f'{sum(1 for k in ORDER[col] if cnt2[(col, k)]["frames"])}')
    rows2.append(dict(grouping=col, label='pairs within the same class', segments=np.nan, frames=n, pairs=within))
fps = pd.Series(seg_names).value_counts()
within_seg = int(sum(pairs(k) for k in fps))
ge2 = fps[fps >= 2]
print(f'  segment: {len(fps)} segments; with 2 or more frames {len(ge2)}; frames per segment (all {len(fps)}) min '
      f'{fps.min()}, median {float(np.median(fps))!r}, max {fps.max()}; among those with 2 or more: min {ge2.min()}, '
      f'median {float(np.median(ge2))!r}, max {ge2.max()}; pairs within segments {within_seg}')
print(f'  segments by number of frames: {dict(sorted(fps.value_counts().items()))}')
rows2.append(dict(grouping='segment', label='pairs within the same segment', segments=len(fps), frames=n,
                  pairs=within_seg))
check(within_seg == 3566, 'pairs within segments = 3,566')
pd.DataFrame(rows2).to_csv(os.path.join(OUT, 'exploratory_sec43_frame_pairs.csv'), index=False)
# the counts the analyses printed
p_wr = os.path.join(A, 'after_qwen3.5-4b_weather', 'weather_report.txt')
p_rp = os.path.join(RULE, 'exploratory_params.txt')
p_wp = os.path.join(WEA, 'params.txt')
print(f'  printed by the analyses (cited):')
blk, n_cmp, n_ok = None, 0, 0
for i, t in enumerate(text_lines(p_wr), 1):
    if t.startswith('===== '):
        blk = t.split(':')[0][6:]
        if blk == 'main_1084':
            m = re.match(r'===== main_1084: (\d+) frames, (\d+) frame pairs in all$', t)
            print(f'    {os.path.basename(p_wr)} line {i}: {t.strip()}')
            n_cmp += 2
            n_ok += bool(m) and int(m.group(1)) == n
            n_ok += bool(m) and int(m.group(2)) == pairs(n)
        continue
    if blk != 'main_1084':
        continue
    m = re.match(r'^\s+(\S+)\s+(\d+) segments,\s+(\d+) frames,\s+(\d+) frame pairs$', t)
    if m:
        print(f'    {os.path.basename(p_wr)} line {i}: {t.strip()}')
        c = cnt2[('weather', m.group(1))]
        n_cmp += 1
        n_ok += (int(m.group(2)), int(m.group(3)), int(m.group(4))) == (c['segments'], c['frames'], c['pairs'])
    m = re.match(r'^\s+pairs within the same weather: (\d+); groups (\d+)$', t)
    if m:
        print(f'    {os.path.basename(p_wr)} line {i}: {t.strip()}')
        n_cmp += 1
        n_ok += int(m.group(1)) == cnt2[('weather', 'within')]
for i, t in enumerate(text_lines(p_rp), 1):
    m = re.match(r"^S: (\w+), reference '([^']*)' \(the most (?:frames|segments)\); segments per category (\{.*\}), "
                 r"frames (\{.*\})$", t)
    if m:
        print(f'    {os.path.basename(p_rp)} line {i}: {t}')
        segs_, frs_ = ast.literal_eval(m.group(3)), ast.literal_eval(m.group(4))
        for k in segs_:
            n_cmp += 1
            n_ok += (segs_[k], frs_[k]) == (cnt2[(m.group(1), k)]['segments'], cnt2[(m.group(1), k)]['frames'])
for i, t in enumerate(text_lines(p_wp), 1):
    m = re.match(r'^  (weather|time_of_day): (.*) \(frames\)$', t)
    if m:
        print(f'    {os.path.basename(p_wp)} line {i}: {t.strip()}')
        for part in m.group(2).split(', '):
            k, v_ = part.rsplit(' ', 1)
            n_cmp += 1
            n_ok += int(v_) == cnt2[(m.group(1), k)]['frames']
    m = re.match(r'^  segment: (\d+) groups \(frames\)$', t)
    if m:
        print(f'    {os.path.basename(p_wp)} line {i}: {t.strip()}')
        n_cmp += 1
        n_ok += int(m.group(1)) == len(seg_list)
check(n_ok == n_cmp and n_cmp >= 20, f'counts printed by the analyses: {n_ok} of {n_cmp} the same as computed here')

# ====================================================================================================== inputs of 3, 4, 6
mi = pd.MultiIndex.from_frame(frames)


def score(folder):
    f = pd.read_csv(os.path.join(folder, 'vlm_frame_scores.csv'), float_precision='round_trip')
    s = f[(f.model == MODEL) & (f.condition == COND)].set_index(['segment', 'frame_idx'])['score_mean']
    if len(s) != n or not s.index.isin(mi).all():
        sys.exit(f'{folder}: the frames of {MODEL} {COND} are not the {n} frames')
    return s.reindex(mi).to_numpy(float)


risk = score(RISK)
check(not np.isnan(risk).any() and np.array_equal(risk, score(WEA)),
      'score_mean of every frame known, the same in both risk analyses (vlm_frame_scores.csv)')
seg_frames = {s: np.flatnonzero(seg_names == s) for s in seg_list}
seg_code = np.searchsorted(seg_list, seg_names)
ids = np.arange(n)
n_boot_r, seed_r, frames_r, segs_r = ar.read_params_risk(os.path.join(RISK, 'params.txt'))
check((n_boot_r, seed_r, frames_r) == (N_BOOT, SEED, n) and list(seg_list) == segs_r,
      f'resamples, seed, frames and segment list as the risk analysis\'s params.txt ({n_boot_r}, {seed_r}, {frames_r}, '
      f'{len(segs_r)} segments)')
rng = np.random.default_rng(SEED)
draws = [rng.choice(seg_list, len(seg_list)) for _ in range(N_BOOT)]
h = hashlib.sha256()
for d in draws:
    h.update(np.searchsorted(seg_list, d).astype(np.int32).tobytes())
nb_q, seed_q, hash_q = ar.read_hash_quantities(os.path.join(QUANT, 'params.txt'))
check(h.hexdigest() == HASH and (nb_q, seed_q, hash_q) == (N_BOOT, SEED, HASH),
      f'drawn segments\' sha256 {h.hexdigest()} = the rule analysis\'s and the quantities analysis\'s')
counts = np.stack([np.bincount(np.searchsorted(seg_list, d), minlength=len(seg_list)) for d in draws])

# S, as analyze_rules.main builds it
sl = lab
s_terms, s_cols, s_var, s_ref = [], [], [], {}
for col in ('time_of_day', 'location', 'weather'):
    v = sl.loc[seg_names, col].str.strip().to_numpy()
    seg_v = sl.loc[seg_list, col].str.strip()
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


def design(vars_):
    js = [j for j, v in enumerate(s_var) if v in vars_]                  # in the rule analysis's order
    return [s_terms[j] for j in js], js


def one3(idx, g):
    """Item 3 (and its checks) on the frames idx with segment groups g (one code per drawn copy)."""
    y = ar.rank01(risk[idx])
    out = {}
    for key, _, vars_ in FITS:
        terms, js = design(vars_)
        coef, _, r2 = ar.ols(y, [s_cols[j][idx] for j in js])
        out[(key, 'r2')] = r2
        for term, b in zip(['intercept'] + terms, coef):
            out[(key, 'coef', term)] = b
    out[('f', 'r2')] = ar.ols(y, list(np.eye(int(g.max()) + 1)[g].T))[2]
    out[('f', 'between')] = ar.between_share(y, g)
    coef, _, r2 = ar.ols(y, [ar.rank01(cue[k][idx]) for k in ar.K_TRUE] + [c[idx] for c in s_cols])
    out[('K+S', 'r2')] = r2
    for term, b in zip(['intercept'] + ar.K_TRUE + s_terms, coef):
        out[('K+S', 'coef', term)] = b
    return out


# item 6's inputs, as analyze_vlm_ssm.main builds them
ssm_df = av.ssm_table(FSSM, frames)
ssm = {c: ssm_df[c].to_numpy() for c in ssm_df.columns if c.endswith(('_conflict', '_risk', '_use'))}
ssm = {k: (v.astype(float) if k.endswith('_risk') else v.astype(bool)) for k, v in ssm.items()}
check(all(ssm[s + '_conflict'].all() and ssm[s + '_use'].all() for s in av.SSMS),
      'every frame has a value of every SSM and is counted on it')
groupings, _ = av.scene_groupings(frames, SL, ['weather', 'time_of_day'])
gn6 = list(groupings)
check(gn6 == ['all', 'weather', 'time_of_day', 'segment'], f'groupings {gn6}')
point3 = one3(ids, seg_code)
keys3 = list(point3)
point6 = av.all_stats(risk, ssm, ids, groupings)
boot3 = np.full((N_BOOT, len(keys3)), np.nan)
boot6 = np.full((N_BOOT, len(av.SSMS), len(gn6), 2), np.nan)              # tau_b_conflict, tau_b_all
tb = time.time()
for b, draw in enumerate(draws):
    idx = np.concatenate([seg_frames[s] for s in draw])
    gseg = np.concatenate([np.full(len(seg_frames[s]), k) for k, s in enumerate(draw)])
    r3 = one3(idx, gseg)
    boot3[b] = [r3.get(k, np.nan) for k in keys3]
    gb = {gname: v[idx] for gname, v in groupings.items()}
    gb['segment'] = gseg
    st = av.all_stats(risk[idx], {k: v[idx] for k, v in ssm.items()}, ids[idx], gb)
    for si, s in enumerate(av.SSMS):
        for gi, gname in enumerate(gn6):
            boot6[b, si, gi] = (st[s][0][gname][1], st[s][0][gname][4])
    if b in (0, 9) or (b + 1) % 250 == 0:
        print(f'  [resample {b + 1}/{N_BOOT}, {(time.time() - tb) / (b + 1) * (N_BOOT - b - 1):.0f} s left]', flush=True)
col3 = {k: i for i, k in enumerate(keys3)}


def ci3(k):
    return ar.ci(boot3[:, col3[k]])


# ====================================================================================================== item 3
print('\n' + '=' * 110)
print('3. EXPLORATORY: how much the scene conditions explain the score (rank of score_mean / 1,084, ties the mean '
      'rank); least squares with an intercept; intervals from the 2,000 segment resamples')
y0 = ar.rank01(risk)


def oos(cols):
    pred = ar.loso_predict(y0, cols, seg_code)
    sse = np.bincount(seg_code, (y0 - pred) ** 2, minlength=len(seg_list))
    sst = np.bincount(seg_code, (y0 - y0.mean()) ** 2, minlength=len(seg_list))
    return 1.0 - sse.sum() / sst.sum(), 1.0 - (counts @ sse) / (counts @ sst), pred


rows_r2, rows_coef = [], []
f_dummies = list(np.eye(len(seg_list))[seg_code].T)
for key, name, vars_ in FITS + [('f', 'one dummy per segment (238)', None)]:
    terms, js = design(vars_) if vars_ else ([f'segment={s}' for s in seg_list], None)
    cols = [s_cols[j] for j in js] if vars_ else f_dummies
    r2o, rb, pred = oos(cols)
    lo_i, hi_i, nv_i = ci3((key, 'r2'))
    lo_o, hi_o, nv_o = ar.ci(rb)
    rows_r2.append(dict(fit=key, cues=name, terms=len(terms), frames=n, segments=len(seg_list),
                        r2_in=point3[(key, 'r2')], r2_in_lo=lo_i, r2_in_hi=hi_i, r2_in_nboot=nv_i, r2_oos=r2o,
                        r2_oos_lo=lo_o, r2_oos_hi=hi_o, r2_oos_nboot=nv_o))
    if vars_:
        for term in ['intercept'] + terms:
            lo, hi, nv = ci3((key, 'coef', term))
            rows_coef.append(dict(fit=key, cues=name, term=term, coef=point3[(key, 'coef', term)], coef_lo=lo,
                                  coef_hi=hi, nboot=nv))
    else:
        f_pred = pred
pd.DataFrame(rows_r2).to_csv(os.path.join(OUT, 'exploratory_sec43_condition_r2.csv'), index=False)
pd.DataFrame(rows_coef).to_csv(os.path.join(OUT, 'exploratory_sec43_condition_coefficients.csv'), index=False)
print(f'  R2 (exploratory_sec43_condition_r2.csv, line = row + 1):')
for i, d in enumerate(rows_r2):
    print(f"   line {i + 2}: ({d['fit']}) {d['cues']}: in sample {r(d['r2_in'])} [{r(d['r2_in_lo'])}, "
          f"{r(d['r2_in_hi'])}] (nboot {d['r2_in_nboot']}); out of sample {r(d['r2_oos'])} [{r(d['r2_oos_lo'])}, "
          f"{r(d['r2_oos_hi'])}] (nboot {d['r2_oos_nboot']})")
print(f'  coefficients (exploratory_sec43_condition_coefficients.csv, line = row + 1; fog and unclear: 0 segments, '
      'undefined as in the rule analysis):')
for i, d in enumerate(rows_coef):
    print(f"   line {i + 2}: ({d['fit']}) {d['term']}: {r(d['coef'])} [{r(d['coef_lo'])}, {r(d['coef_hi'])}] "
          f"(nboot {d['nboot']})")
print('  (f): the 238 dummies and the intercept are linearly dependent, so every coefficient is undefined (null-space '
      'rule); the fitted values are the segment means')
print(f'  (f) out of sample: a held-out segment\'s prediction is the minimum-norm intercept; here from '
      f'{r(float(f_pred.min()))} to {r(float(f_pred.max()))} (mean rank of all frames {r(float(y0.mean()))})')
seg_mean = np.bincount(seg_code, y0) / np.bincount(seg_code)
d_pred = float(np.abs(f_pred - ((seg_mean.sum() - seg_mean) / len(seg_list))[seg_code]).max())
check(d_pred <= TOL, f'(f) out of sample: every held-out segment\'s prediction = the other {len(seg_list) - 1} '
      f'segments\' mean ranks summed / {len(seg_list)}, within {d_pred!r}')
# checks
cf = [(ln, d) for ln, d in csv_rows(os.path.join(RULE, 'exploratory_rule_coefficients.csv'))
      if d['target'] == 'risk' and d['model'] == MODEL and d['condition'] == COND and d['fit'] == 'K+S']
rule_s = [d['term'] for _, d in cf if d['term'] not in ['intercept'] + ar.K_TRUE]
check(rule_s == s_terms, f"(e)'s terms {s_terms} = the S terms of the rule analysis's K+S fit, in its order "
      f"(exploratory_rule_coefficients.csv lines {cf[0][0] if cf else '?'}-{cf[-1][0] if cf else '?'})")
dmax, n_txt, n_all = 0.0, 0, 0
for ln, d in cf:
    k = ('K+S', 'coef', d['term'])
    lo, hi, nv = ci3(k)
    mine = (point3[k], lo, hi)
    for x, t in zip(mine, (d['coef'], d['coef_lo'], d['coef_hi'])):
        n_all += 1
        n_txt += txt(x) == t
        if t != '' and not np.isnan(x):
            dmax = max(dmax, abs(x - float(t)))
        elif (t == '') != np.isnan(x):
            dmax = np.inf
    dmax = dmax if str(nv) == d['nboot'] else np.inf
rr = [(ln, d) for ln, d in csv_rows(os.path.join(RULE, 'exploratory_rule_r2.csv'))
      if d['target'] == 'risk' and d['model'] == MODEL and d['condition'] == COND and d['fit'] == 'K+S']
lo, hi, nv = ci3(('K+S', 'r2'))
for x, t in zip((point3[('K+S', 'r2')], lo, hi), (rr[0][1]['r2'], rr[0][1]['r2_lo'], rr[0][1]['r2_hi'])):
    n_all += 1
    n_txt += txt(x) == t
    dmax = max(dmax, abs(x - float(t)))
dmax = dmax if str(nv) == rr[0][1]['nboot'] else np.inf
check(len(cf) == 14 and len(rr) == 1 and dmax <= TOL,
      f'K+S with the same S (point, lo, hi of {len(cf)} coefficients and R2; exploratory_rule_coefficients.csv lines '
      f'{cf[0][0]}-{cf[-1][0]}, exploratory_rule_r2.csv line {rr[0][0]}): largest |difference| {dmax!r}; the same '
      f'text {n_txt} of {n_all}; nboot the same')
ro = [(ln, d) for ln, d in csv_rows(os.path.join(RULE, 'exploratory_oos_r2.csv'))
      if d['target'] == 'risk' and d['model'] == MODEL and d['condition'] == COND and d['fit'] == 'K+S']
r2o, rb, _ = oos([ar.rank01(cue[k]) for k in ar.K_TRUE] + s_cols)
r2o = float(r2o)
lo, hi, nv = ar.ci(rb)
dd = max(abs(a_ - float(b_)) for a_, b_ in zip((r2o, lo, hi), (ro[0][1]['r2_oos'], ro[0][1]['r2_oos_lo'],
                                                                ro[0][1]['r2_oos_hi'])))
check(len(ro) == 1 and dd <= TOL and str(nv) == ro[0][1]['nboot'],
      f'K+S out of sample (exploratory_oos_r2.csv line {ro[0][0]}: {ro[0][1]["r2_oos"]} [{ro[0][1]["r2_oos_lo"]}, '
      f'{ro[0][1]["r2_oos_hi"]}]): here {r2o!r} [{lo!r}, {hi!r}], largest |difference| {dd!r}')
bs = [(ln, d) for ln, d in csv_rows(os.path.join(RULE, 'exploratory_between_segment.csv'))
      if d['target'] == 'risk' and d['model'] == MODEL and d['condition'] == COND]
lo_b, hi_b, nv_b = ci3(('f', 'between'))
bl, bd = bs[0]
same_b = (txt(point3[('f', 'between')]), txt(lo_b), txt(hi_b), str(nv_b)) == \
    (bd['between_share'], bd['between_share_lo'], bd['between_share_hi'], bd['nboot'])
check(len(bs) == 1 and same_b, f'between-segment share recomputed (analyze_rules.between_share): '
      f'{r(point3[("f", "between")])} [{r(lo_b)}, {r(hi_b)}], nboot {nv_b}, the same text as '
      f'exploratory_between_segment.csv line {bl}')
d_f = abs(point3[('f', 'r2')] - float(bd['between_share']))
d_fb = float(np.nanmax(np.abs(boot3[:, col3[('f', 'r2')]] - boot3[:, col3[('f', 'between')]])))
check(d_f <= TOL and d_fb <= TOL, f"(f)'s in-sample R2 {r(point3[('f', 'r2')])} = the between-segment share "
      f"{bd['between_share']} (exploratory_between_segment.csv line {bl}) within {d_f!r}; in every resample within "
      f'{d_fb!r}')

# ====================================================================================================== item 4
print('\n' + '=' * 110)
print('4. descriptive (computed): score_mean (0-100, mean of the 5 answers) by time of day, weather, time of day x '
      'weather; SD with ddof 1, quartiles numpy linear')
fr = frames.copy()
fr['score_mean'] = risk
fr['rank'] = rankdata(risk)
fr['rank_div_1084'] = ar.rank01(risk)
for col in ('time_of_day', 'weather', 'location'):
    fr[col] = lab_f[col]
check(np.array_equal(fr['rank_div_1084'].to_numpy(), fr['rank'].to_numpy() / n), 'rank / 1,084 = item 3\'s y')
fr.to_csv(os.path.join(OUT, 'sec43_frames.csv'), index=False)
print(f'  sec43_frames.csv: {len(fr)} rows; columns {list(fr.columns)}')


def desc(m):
    v = risk[m]
    return dict(frames=int(m.sum()), segments=len(set(seg_names[m])), mean=float(np.mean(v)),
                sd=float(np.std(v, ddof=1)) if len(v) > 1 else np.nan, min=float(v.min()),
                q1=float(np.percentile(v, 25)), median=float(np.percentile(v, 50)), q3=float(np.percentile(v, 75)),
                max=float(v.max()))


rows4 = [dict(grouping='all', time_of_day='', weather='', **desc(np.ones(n, bool)))]
for k in ORDER['time_of_day']:
    rows4.append(dict(grouping='time_of_day', time_of_day=k, weather='', **desc(lab_f['time_of_day'] == k)))
for k in ORDER['weather']:
    if (lab_f['weather'] == k).any():
        rows4.append(dict(grouping='weather', time_of_day='', weather=k, **desc(lab_f['weather'] == k)))
empty = []
for t in ORDER['time_of_day']:
    for w in ORDER['weather']:
        m = (lab_f['time_of_day'] == t) & (lab_f['weather'] == w)
        if m.any():
            rows4.append(dict(grouping='time_of_day x weather', time_of_day=t, weather=w, **desc(m)))
        elif w in ('dry', 'wet', 'rain'):
            empty.append(f'{t} x {w}')
pd.DataFrame(rows4).to_csv(os.path.join(OUT, 'exploratory_sec43_score_distribution.csv'), index=False)
for i, d in enumerate(rows4):
    print(f"   line {i + 2}: {d['grouping']:22s} {d['time_of_day']:10s} {d['weather']:5s} frames {d['frames']:4d} "
          f"(segments {d['segments']:3d}): mean {r(d['mean'])}, sd {r(d['sd'])}, min {r(d['min'])}, q1 {r(d['q1'])}, "
          f"median {r(d['median'])}, q3 {r(d['q3'])}, max {r(d['max'])}")
print(f'  combinations of time of day x dry/wet/rain without frames: {empty if empty else "none"}')

# ====================================================================================================== item 5
print('\n' + '=' * 110)
print(f'5. E2 (exploratory), copied as text from {E2_LATEST}')
e2 = [(ln, d) for ln, d in csv_rows(E2_LATEST) if d['model'] == MODEL and d['condition'] == COND
      and d['variable'] in ('true_distance', 'true_ego_speed')]
formal = {(d['variable'], d['grouping']): (ln, d) for ln, d in csv_rows(E2_FORMAL)
          if d['model'] == MODEL and d['condition'] == COND}
for ln, d in e2:
    print(f"  line {ln}: {d['variable']:14s} {d['grouping']:11s} frames {d['frames']}: tau_b {d['tau_b']} "
          f"[{d['tau_b_lo']}, {d['tau_b_hi']}], nboot {d['nboot']}")
print(f"  groupings in the file: {sorted({d['grouping'] for _, d in csv_rows(E2_LATEST)})} (no 'weather')")
same5 = [(d['variable'], d['grouping']) in formal and formal[(d['variable'], d['grouping'])][1] == d for _, d in e2]
check(len(e2) == 6 and all(same5), f'the 6 rows the same text in {E2_FORMAL} (lines '
      f'{[formal[(d["variable"], d["grouping"])][0] for _, d in e2 if (d["variable"], d["grouping"]) in formal]})')

# ====================================================================================================== item 6
print('\n' + '=' * 110)
print('6. EXPLORATORY: grouping - all, tau_b_all, over the same 2,000 resamples (same frames in each resample)')
agw = {(d['ssm'], d['grouping']): (ln, d) for ln, d in csv_rows(p_wea) if d['model'] == MODEL and d['condition'] == COND}
n_own, n_own_ok, dpt = 0, 0, 0.0
rows6 = []
for si, s in enumerate(av.SSMS):
    for gi, gname in enumerate(gn6):
        ln, d = agw[(s, gname)]
        for k, colname in ((0, 'tau_b_conflict'), (1, 'tau_b_all')):
            pe = point6[s][0][gname][1 if k == 0 else 4]
            lo, hi, nv = ar.ci(boot6[:, si, gi, k])
            mine = (txt(pe), txt(lo), txt(hi), str(nv))
            theirs = (d[colname], d[colname + '_lo'], d[colname + '_hi'], d[colname + '_nboot'])
            n_own += 1
            n_own_ok += mine == theirs
            if mine != theirs:
                print(f'  differs: {s} {gname} {colname}: here {mine}, agreement.csv line {ln} {theirs}')
    for gname in GROUPS6:
        gi = gn6.index(gname)
        diff_b = boot6[:, si, gi, 1] - boot6[:, si, 0, 1]
        lo, hi, nv = ar.ci(diff_b)
        pe = point6[s][0][gname][4] - point6[s][0]['all'][4]
        from_csv = float(agw[(s, gname)][1]['tau_b_all']) - float(agw[(s, 'all')][1]['tau_b_all'])
        dpt = max(dpt, abs(pe - from_csv))
        rows6.append(dict(ssm=s, grouping=gname, tau_b_grouping=point6[s][0][gname][4],
                          tau_b_all=point6[s][0]['all'][4], diff=pe, diff_lo=lo, diff_hi=hi, nboot=nv,
                          interval_without_0=bool(not np.isnan(lo) and (lo > 0 or hi < 0)),
                          agreement_lines=f"{agw[(s, gname)][0]} - {agw[(s, 'all')][0]}"))
check(n_own_ok == n_own, f'every grouping\'s own tau-b, interval and nboot recomputed: {n_own_ok} of {n_own} '
      f'(9 SSMs x 4 groupings x tau_b_conflict, tau_b_all) the same text as {p_wea}')
check(dpt <= TOL, f'point differences = the two agreement.csv rows subtracted: largest |difference| {dpt!r}')
pd.DataFrame(rows6).to_csv(os.path.join(OUT, 'exploratory_sec43_grouping_minus_all.csv'), index=False)
print('  exploratory_sec43_grouping_minus_all.csv (line = row + 1):')
for i, d in enumerate(rows6):
    print(f"   line {i + 2}: {d['ssm']:6s} {d['grouping']:11s} - all: {r(d['diff'])} [{r(d['diff_lo'])}, "
          f"{r(d['diff_hi'])}] (nboot {d['nboot']}; agreement.csv lines {d['agreement_lines']}: {r(d['tau_b_grouping'])}"
          f" - {r(d['tau_b_all'])}){' interval without 0' if d['interval_without_0'] else ''}")

print('\n' + '=' * 110)
print(f'done in {time.time() - t0:.0f} s; checks not met: {len(failed)}' + (f': {failed}' if failed else ''))
sys.exit(1 if failed else 0)
