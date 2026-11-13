"""Paper section 4.5, one more analysis (user, 2026-11-12). EXPLORATORY; CPU only; reads existing outputs only.
The acceleration run, Qwen3.6-27B with thinking on, sep_camera, the main result's 1,084 frames. Per frame: the 4.4
quantities run's ego-acceleration estimate (median of the readable repeats) and the true ego acceleration
(sec44_frames.csv), and the score change accel - base (exploratory_accel_frames.csv of analysis/accel_risk_27b/main_1084:
qwen3.6-27b-accel_sep_camera - qwen3.6-27b_sep_camera, both score_mean).
1 least squares with an intercept: score change ~ true acceleration + (true acceleration - estimate) (m/s2;
  analyze_rules.ols); coefficients with intervals.
2 groups: the direction told in the sentence (the number the model saw: code/run_vlm_accel.py accel_text, the truth to
  1 decimal, "-0.0" written "0.0"; positive = told speeding up, 0 = told steady, negative = told slowing down; checked
  against the truth rounded half up with Decimal) x the sign of the model's own estimate (positive = thought speeding
  up, 0 = thought steady, negative = thought slowing down). Per group: frames, segments, mean score change; an interval
  for the groups with 30 frames or more (user). Two differences with intervals: told speeding up: thought slowing down
  - thought steady; told slowing down: thought steady - thought slowing down.
3 check (user, 2026-11-13): the five true-acceleration classes of the acceleration run (cut points as
  analyze_told_quantity.py: numpy quantiles 0.2 / 0.4 / 0.6 / 0.8 of the true acceleration over the 1,084 frames, class
  = 1 + the number of cut points below): per class the range, frames and mean score change, which must be, as text,
  those of exploratory_accel_told_classes.csv (sep_camera rows, lines 12-16: told_min, told_max, frames, diff_mean).
Intervals: the main analysis's 2,000 resamples of whole segments (default_rng(20261021), one choice over the sorted
segments per resample; frames enter as often as their segment is drawn), the statistic recomputed (the regression
refitted) in every resample; 2.5 / 97.5 % percentiles (numpy linear), none when fewer than 90 % defined (analyze_rules.ci).
usage: python -I paper_sec45_est_accel.py <waymo_v143 folder> <sec44_frames.csv> <out_dir>
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import csv  # noqa: E402
import hashlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from decimal import ROUND_HALF_UP, Decimal  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_rules as ar  # noqa: E402
from run_vlm_accel import accel_text  # noqa: E402

N_BOOT, SEED = 2000, 20261021
HASH = '48bc1f413f9d5b99da57ff4edde7bbb4eb378c5a9343c44ccad4e0165466604c'
MIN_FRAMES = 30
L, F44, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
SUB = os.path.join(L, 'analysis', 'accel_risk_27b', 'main_1084')
failed = []


def check(ok, what):
    print(f'  check {"ok" if ok else "NOT MET"}: {what}')
    if not ok:
        failed.append(what)


def r(x):
    return 'nan' if x is None or np.isnan(x) else repr(float(x))


print(f'paper section 4.5: ego-acceleration misjudgement and the acceleration run (user, 2026-11-12); run '
      f'{time.strftime("%Y-%m-%d %H:%M:%S")}\nEXPLORATORY')
af = pd.read_csv(os.path.join(SUB, 'exploratory_accel_frames.csv'), float_precision='round_trip')
q = pd.read_csv(F44, float_precision='round_trip')
m = af.merge(q[['segment', 'frame_idx', 'ego_accel_truth_mps2', 'ego_accel_estimate_mps2']],
             on=['segment', 'frame_idx'], how='outer', validate='one_to_one', indicator=True)
check(len(af) == len(q) == 1084 and (m['_merge'] == 'both').all(),
      f'the same 1,084 frames in exploratory_accel_frames.csv ({len(af)}) and sec44_frames.csv ({len(q)})')
m = m.sort_values(['segment', 'frame_idx']).reset_index(drop=True)
a = m['true_a_ego'].to_numpy(float)
check(np.array_equal(a, m['ego_accel_truth_mps2'].to_numpy(float)), 'the true ego acceleration is the same in both tables')
e = m['ego_accel_estimate_mps2'].to_numpy(float)
d = (m['qwen3.6-27b-accel_sep_camera'] - m['qwen3.6-27b_sep_camera']).to_numpy(float)
told = np.array([float(accel_text(x)) for x in a])
half_up = np.array([float(Decimal(repr(float(x))).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)) for x in a])
check(np.array_equal(told, half_up), f'the told number (accel_text) = the truth rounded half up to 0.1 on '
      f'{int((told == half_up).sum())} of {len(a)} frames')
n = len(m)
seg_names = m['segment'].to_numpy()
seg_list = np.array(sorted(set(seg_names)))
seg_frames = {s: np.flatnonzero(seg_names == s) for s in seg_list}
nb_r, seed_r, frames_r, segs_r = ar.read_params_risk(os.path.join(SUB, 'params.txt'))
check((nb_r, seed_r, frames_r) == (N_BOOT, SEED, n) and list(seg_list) == segs_r,
      f'{SUB}/params.txt: {nb_r} resamples, seed {seed_r}, {frames_r} frames, the same segment list')
rng = np.random.default_rng(SEED)
draws = [rng.choice(seg_list, len(seg_list)) for _ in range(N_BOOT)]
h = hashlib.sha256()
for dr in draws:
    h.update(np.searchsorted(seg_list, dr).astype(np.int32).tobytes())
check(h.hexdigest() == HASH, f"drawn segments' sha256 {h.hexdigest()}")
idxs = [np.concatenate([seg_frames[s] for s in dr]) for dr in draws]
TOLD = {1: 'told speeding up', 0: 'told steady (0.0)', -1: 'told slowing down'}
THOUGHT = {1: 'thought speeding up', 0: 'thought steady (0)', -1: 'thought slowing down'}
ts, es = np.sign(told).astype(int), np.sign(e).astype(int)
GROUPS = [(t_, e_) for t_ in (1, 0, -1) for e_ in (1, 0, -1)]
DIFFS = [('told speeding up: thought slowing down - thought steady', (1, -1), (1, 0)),
         ('told slowing down: thought steady - thought slowing down', (-1, 0), (-1, -1))]


def stats(ix):
    aa, ee, dd, tt, ss = a[ix], e[ix], d[ix], ts[ix], es[ix]
    out = {}
    for g in GROUPS:
        sel = (tt == g[0]) & (ss == g[1])
        out[g] = float(dd[sel].mean()) if sel.any() else np.nan
    for name, g1, g2 in DIFFS:
        out[name] = out[g1] - out[g2]
    coef, _, r2 = ar.ols(dd, [aa, aa - ee])
    out.update(intercept=coef[0], b_true=coef[1], b_true_minus_estimate=coef[2], r2=r2)
    return out


point = stats(np.arange(n))
boot = {k: np.array([stats(ix)[k] for ix in idxs]) for k in point}
ci = {k: ar.ci(boot[k]) for k in point}

print('\n1. least squares: score change ~ true acceleration + (true acceleration - estimate), m/s2, with an intercept')
rows1 = []
for k, label in (('intercept', 'intercept'), ('b_true', 'true acceleration'),
                 ('b_true_minus_estimate', 'true acceleration - estimate'), ('r2', 'R2 (in sample)')):
    lo, hi, nv = ci[k]
    rows1.append(dict(term=label, value=point[k], lo=lo, hi=hi, nboot=nv))
    print(f'  {label:30s} {r(point[k])} [{r(lo)}, {r(hi)}] (nboot {nv})')
pd.DataFrame(rows1).to_csv(os.path.join(OUT, 'exploratory_sec45_est_accel_regression.csv'), index=False)

print(f'\n2. told direction x the model\'s own estimate (interval for groups of {MIN_FRAMES} frames or more)')
rows2 = []
for g in GROUPS:
    sel = (ts == g[0]) & (es == g[1])
    nf = int(sel.sum())
    lo, hi, nv = ci[g] if nf >= MIN_FRAMES else (np.nan, np.nan, 0)
    rows2.append(dict(told=TOLD[g[0]], thought=THOUGHT[g[1]], frames=nf, segments=int(len(set(seg_names[sel]))),
                      mean_change=point[g], lo=lo, hi=hi, nboot=nv if nf >= MIN_FRAMES else np.nan))
    print(f'  {TOLD[g[0]]:18s} x {THOUGHT[g[1]]:20s}: {nf:4d} frames, {len(set(seg_names[sel])):3d} segments, mean change '
          f'{r(point[g])}' + (f' [{r(lo)}, {r(hi)}] (nboot {nv})' if nf >= MIN_FRAMES else ''))
for name, _, _ in DIFFS:
    lo, hi, nv = ci[name]
    rows2.append(dict(told=name, thought='', frames=np.nan, segments=np.nan, mean_change=point[name], lo=lo, hi=hi,
                      nboot=nv))
    print(f'  {name}: {r(point[name])} [{r(lo)}, {r(hi)}] (nboot {nv})')
print(f'  frames told 0.0: {int((ts == 0).sum())}; estimates of exactly 0: {int((es == 0).sum())}; all frames: mean '
      f'change {r(float(d.mean()))}')
pd.DataFrame(rows2).to_csv(os.path.join(OUT, 'exploratory_sec45_est_accel_groups.csv'), index=False)

print('\n3. check: the five true-acceleration classes of the acceleration run (sep_camera)')


def txt(x):
    return '' if x is None or (isinstance(x, float) and np.isnan(x)) else repr(float(x))


qcut = np.quantile(a, [0.2, 0.4, 0.6, 0.8])
cls = 1 + sum((a > q_).astype(int) for q_ in qcut)
p_cls = os.path.join(SUB, 'exploratory_accel_told_classes.csv')
with open(p_cls, newline='', encoding='utf-8') as f:
    rd = csv.DictReader(f)
    tc = {int(x['told_class']): (rd.line_num, x) for x in rd if x['condition'] == 'sep_camera'}
rows3, n_same = [], 0
for c_ in range(1, 6):
    sel = cls == c_
    ln, x = tc[c_]
    mine = (txt(float(a[sel].min())), txt(float(a[sel].max())), str(int(sel.sum())), txt(float(d[sel].mean())))
    same = mine == (x['told_min'], x['told_max'], x['frames'], x['diff_mean'])
    n_same += same
    rows3.append(dict(told_class=c_, true_accel_min=float(a[sel].min()), true_accel_max=float(a[sel].max()),
                      frames=int(sel.sum()), mean_change=float(d[sel].mean()), told_classes_line=ln, same_text=same))
    print(f'  class {c_} (true acceleration {mine[0]} to {mine[1]} m/s2, {mine[2]} frames): mean change {mine[3]}; '
          f'{os.path.basename(p_cls)} line {ln}: {x["told_min"]} to {x["told_max"]}, {x["frames"]} frames, diff_mean '
          f'{x["diff_mean"]}; same text: {same}')
check(n_same == 5, f'classes: range, frames and mean change recomputed = {os.path.basename(p_cls)} (sep_camera rows, '
      f'lines {sorted(ln for ln, _ in tc.values())}): {n_same} of 5 the same text')
print(f'  cut points (m/s2): {[repr(float(x)) for x in qcut]}')
pd.DataFrame(rows3).to_csv(os.path.join(OUT, 'exploratory_sec45_est_accel_classes.csv'), index=False)
print(f'\nwritten: exploratory_sec45_est_accel_regression.csv, _groups.csv, _classes.csv in {OUT}')
print(f'checks not met: {len(failed)}')
for w in failed:
    print('  ', w)
sys.exit(1 if failed else 0)
