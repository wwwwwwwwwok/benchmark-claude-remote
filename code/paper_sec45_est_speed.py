"""Paper section 4.5, one more analysis (user, 2026-11-12). EXPLORATORY; CPU only; reads existing outputs only.
Qwen3.6-27B with thinking on, sep_camera, the main result's 1,084 frames. Per frame: the 4.4 quantities run's ego-speed
estimate (median of the readable repeats) and the true ego speed (sec44_frames.csv, from paper_sec44_frames.py), and
the speed run's score change speed - base (exploratory_speed_frames.csv of analysis/speed_risk_27b/main_1084:
qwen3.6-27b-speed_sep_camera - qwen3.6-27b_sep_camera, both score_mean).
1 two groups: estimate < truth (underestimated) and estimate > truth (overestimated); frames with estimate = truth in
  neither (counted). Frames, segments, the mean score change and its interval, and under - over with its interval.
2 least squares with an intercept: score change ~ true speed + (true speed - estimate) (m/s; analyze_rules.ols);
  coefficients with intervals.
3 the speed run's five true-speed classes (cut points as analyze_told_quantity.py: numpy quantiles 0.2 / 0.4 / 0.6 /
  0.8 of the true speed over the 1,084 frames, class = 1 + the number of cut points below; checked against
  exploratory_speed_told_classes.csv): per class, each group's frames and mean score change.
Intervals: the main analysis's 2,000 resamples of whole segments (default_rng(20261021), one choice over the sorted
segments per resample; frames enter as often as their segment is drawn), the statistic recomputed (the regression
refitted) in every resample; 2.5 / 97.5 % percentiles (numpy linear), none when fewer than 90 % defined (analyze_rules.ci).
usage: python -I paper_sec45_est_speed.py <waymo_v143 folder> <sec44_frames.csv> <out_dir>
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

N_BOOT, SEED = 2000, 20261021
HASH = '48bc1f413f9d5b99da57ff4edde7bbb4eb378c5a9343c44ccad4e0165466604c'
L, F44, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
SUB = os.path.join(L, 'analysis', 'speed_risk_27b', 'main_1084')
failed = []


def check(ok, what):
    print(f'  check {"ok" if ok else "NOT MET"}: {what}')
    if not ok:
        failed.append(what)


def txt(x):
    return '' if x is None or (isinstance(x, float) and np.isnan(x)) else repr(float(x))


def r(x):
    return 'nan' if x is None or np.isnan(x) else repr(float(x))


print(f'paper section 4.5: ego-speed misjudgement and the speed run (user, 2026-11-12); run '
      f'{time.strftime("%Y-%m-%d %H:%M:%S")}\nEXPLORATORY')
sf = pd.read_csv(os.path.join(SUB, 'exploratory_speed_frames.csv'), float_precision='round_trip')
q = pd.read_csv(F44, float_precision='round_trip')
m = sf.merge(q[['segment', 'frame_idx', 'ego_speed_truth_mps', 'ego_speed_estimate_mps']],
             on=['segment', 'frame_idx'], how='outer', validate='one_to_one', indicator=True)
check(len(sf) == len(q) == 1084 and (m['_merge'] == 'both').all(),
      f'the same 1,084 frames in exploratory_speed_frames.csv ({len(sf)}) and sec44_frames.csv ({len(q)})')
m = m.sort_values(['segment', 'frame_idx']).reset_index(drop=True)
check(np.array_equal(m['true_v_ego'].to_numpy(float), m['ego_speed_truth_mps'].to_numpy(float)),
      'the true ego speed is the same in both tables')
v = m['true_v_ego'].to_numpy(float)
e = m['ego_speed_estimate_mps'].to_numpy(float)
d = (m['qwen3.6-27b-speed_sep_camera'] - m['qwen3.6-27b_sep_camera']).to_numpy(float)
under, over, tie = e < v, e > v, e == v
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


def stats(ix):
    vv, ee, dd = v[ix], e[ix], d[ix]
    u, o = ee < vv, ee > vv
    mu = float(dd[u].mean()) if u.any() else np.nan
    mo = float(dd[o].mean()) if o.any() else np.nan
    coef, _, r2 = ar.ols(dd, [vv, vv - ee])
    return dict(mean_under=mu, mean_over=mo, under_minus_over=mu - mo, intercept=coef[0], b_true_speed=coef[1],
                b_true_minus_estimate=coef[2], r2=r2)


point = stats(np.arange(n))
boot = {k: np.array([stats(ix)[k] for ix in idxs]) for k in point}
ci = {k: ar.ci(boot[k]) for k in point}

print('\n1. two groups (score change = speed - base, sep_camera score_mean)')
rows1 = []
for name, mask, k in (('estimate < truth (underestimated)', under, 'mean_under'),
                      ('estimate > truth (overestimated)', over, 'mean_over')):
    lo, hi, nv = ci[k]
    rows1.append(dict(group=name, frames=int(mask.sum()), segments=int(len(set(seg_names[mask]))), mean_change=point[k],
                      lo=lo, hi=hi, nboot=nv))
    print(f'  {name}: {int(mask.sum())} frames in {len(set(seg_names[mask]))} segments; mean change {r(point[k])} '
          f'[{r(lo)}, {r(hi)}] (nboot {nv})')
lo, hi, nv = ci['under_minus_over']
rows1.append(dict(group='under - over', frames=np.nan, segments=np.nan, mean_change=point['under_minus_over'], lo=lo,
                  hi=hi, nboot=nv))
print(f'  under - over: {r(point["under_minus_over"])} [{r(lo)}, {r(hi)}] (nboot {nv}); frames with estimate = truth '
      f'(in neither): {int(tie.sum())}; all frames: mean change {r(float(d.mean()))}')
pd.DataFrame(rows1).to_csv(os.path.join(OUT, 'exploratory_sec45_est_speed_groups.csv'), index=False)

print('\n2. least squares: score change ~ true speed + (true speed - estimate), m/s, with an intercept')
rows2 = []
for k, label in (('intercept', 'intercept'), ('b_true_speed', 'true speed'),
                 ('b_true_minus_estimate', 'true speed - estimate'), ('r2', 'R2 (in sample)')):
    lo, hi, nv = ci[k]
    rows2.append(dict(term=label, value=point[k], lo=lo, hi=hi, nboot=nv))
    print(f'  {label:22s} {r(point[k])} [{r(lo)}, {r(hi)}] (nboot {nv})')
pd.DataFrame(rows2).to_csv(os.path.join(OUT, 'exploratory_sec45_est_speed_regression.csv'), index=False)

print('\n3. the five true-speed classes of the speed run (sep_camera)')
qcut = np.quantile(v, [0.2, 0.4, 0.6, 0.8])
cls = 1 + sum((v > q_).astype(int) for q_ in qcut)
p_cls = os.path.join(SUB, 'exploratory_speed_told_classes.csv')
with open(p_cls, newline='', encoding='utf-8') as f:
    rd = csv.DictReader(f)
    tc = {int(x['told_class']): (rd.line_num, x) for x in rd if x['condition'] == 'sep_camera'}
rows3, n_same = [], 0
for c_ in range(1, 6):
    sel = cls == c_
    ln, x = tc[c_]
    same = (txt(float(v[sel].min())), txt(float(v[sel].max())), str(int(sel.sum())), txt(float(d[sel].mean()))) == \
        (x['told_min'], x['told_max'], x['frames'], x['diff_mean'])
    n_same += same
    row = dict(told_class=c_, true_speed_min=float(v[sel].min()), true_speed_max=float(v[sel].max()),
               frames=int(sel.sum()), mean_change_all=float(d[sel].mean()))
    for name, mask in (('under', under), ('over', over)):
        s2 = sel & mask
        row[f'frames_{name}'] = int(s2.sum())
        row[f'mean_change_{name}'] = float(d[s2].mean()) if s2.any() else np.nan
    row['frames_equal'] = int((sel & tie).sum())
    rows3.append(row)
    print(f"  class {c_} (true speed {r(row['true_speed_min'])} to {r(row['true_speed_max'])} m/s, {row['frames']} frames, "
          f"mean change {r(row['mean_change_all'])}): underestimated {row['frames_under']} frames, mean "
          f"{r(row['mean_change_under'])}; overestimated {row['frames_over']} frames, mean {r(row['mean_change_over'])}"
          f"; equal {row['frames_equal']}")
check(n_same == 5, f'classes: range, frames and mean change recomputed = {os.path.basename(p_cls)} (sep_camera rows, '
      f'lines {sorted(ln for ln, _ in tc.values())}): {n_same} of 5 the same text')
print(f'  cut points (m/s): {[repr(float(x)) for x in qcut]}')
pd.DataFrame(rows3).to_csv(os.path.join(OUT, 'exploratory_sec45_est_speed_classes.csv'), index=False)
print(f'\nwritten: exploratory_sec45_est_speed_groups.csv, _regression.csv, _classes.csv in {OUT}')
print(f'checks not met: {len(failed)}')
for w in failed:
    print('  ', w)
sys.exit(1 if failed else 0)
