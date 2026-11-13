"""Paper section 4.5, one more number (user, 2026-11-12). EXPLORATORY; CPU only; reads existing outputs only.
The speed run (analysis/speed_risk_27b, main result 1,084 frames, sep_camera, grouping all): tau-b of the risk score
(score_mean) with the true ego acceleration (truth_main.csv a_ego) for the speed run and for base, and speed - base,
exactly as analyze_told_quantity.py's item 3 computes it for the acceleration run with --extra_cues a_ego
(analyze_vlm_ssm.stats_by_grouping over the frames where both values exist; the difference taken in every resample;
the 2,000 resamples of whole segments drawn as the risk analysis draws them, seed 20261021; 2.5 / 97.5 % percentiles,
none when fewer than 90 % are defined: analyze_rules.ci). The speed analysis did not hold this cue (its run had no
--extra_cues).
Checks: the drawn segments' sha256 (48bc1f41...); recomputed here, the same text as the analyses: the speed run's rows
for the true ego speed (exploratory_speed_speed_distance_tau_b.csv, sep_camera, all: base, speed, speed - base) and
base's row for the true ego acceleration (accel_risk_27b exploratory_accel_speed_distance_tau_b.csv line 74).
usage: python -I paper_sec45_speed_accel.py <waymo_v143 folder> <scene_labels.csv> <out_dir>
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
import analyze_vlm_ssm as av  # noqa: E402

BASE, NEW, COND = 'qwen3.6-27b', 'qwen3.6-27b-speed', 'sep_camera'
N_BOOT, SEED = 2000, 20261021
HASH = '48bc1f413f9d5b99da57ff4edde7bbb4eb378c5a9343c44ccad4e0165466604c'
L, SL, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
SUB = os.path.join(L, 'analysis', 'speed_risk_27b', 'main_1084')
ACC = os.path.join(L, 'analysis', 'accel_risk_27b', 'main_1084')
TRUTH = os.path.join(L, 'quantities', 'truth_main.csv')
failed = []


def check(ok, what):
    print(f'  check {"ok" if ok else "NOT MET"}: {what}')
    if not ok:
        failed.append(what)


def txt(x):
    return '' if x is None or (isinstance(x, float) and np.isnan(x)) else repr(float(x))


def side(lo, hi):
    if lo is None or np.isnan(lo):
        return 'no interval'
    return 'above 0' if lo > 0 else ('below 0' if hi < 0 else 'contains 0')


def rows_of(path, **match):
    with open(path, encoding='utf-8') as f:
        lines = f.read().splitlines()
    with open(path, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        out = [(rd.line_num, d) for d in rd if all(d.get(k) == v for k, v in match.items())]
    return [(ln, d, lines[ln - 1]) for ln, d in out]


print(f'paper section 4.5: speed run vs the true ego acceleration (user, 2026-11-12), {COND}, all; run '
      f'{time.strftime("%Y-%m-%d %H:%M:%S")}\nEXPLORATORY')
frames = pd.read_csv(os.path.join(SUB, 'ssm_target_frames.csv'))[['segment', 'frame_idx']] \
    .sort_values(['segment', 'frame_idx']).reset_index(drop=True)
n = len(frames)
mi = pd.MultiIndex.from_frame(frames)
seg_names = frames['segment'].to_numpy()
seg_list = np.array(sorted(set(seg_names)))
seg_frames = {s: np.flatnonzero(seg_names == s) for s in seg_list}
seg_code = np.searchsorted(seg_list, seg_names)
ids = np.arange(n)
nb_r, seed_r, frames_r, segs_r = ar.read_params_risk(os.path.join(SUB, 'params.txt'))
check((nb_r, seed_r, frames_r) == (N_BOOT, SEED, n) and list(seg_list) == segs_r and n == 1084,
      f'{SUB}/params.txt: {nb_r} resamples, seed {seed_r}, {frames_r} frames, the same segment list')
rng = np.random.default_rng(SEED)
draws = [rng.choice(seg_list, len(seg_list)) for _ in range(N_BOOT)]
h = hashlib.sha256()
for d in draws:
    h.update(np.searchsorted(seg_list, d).astype(np.int32).tobytes())
check(h.hexdigest() == HASH, f"drawn segments' sha256 {h.hexdigest()}")
fs = pd.read_csv(os.path.join(SUB, 'vlm_frame_scores.csv'), float_precision='round_trip')
score = {}
for m in (BASE, NEW):
    s = fs[(fs.model == m) & (fs.condition == COND)].set_index(['segment', 'frame_idx'])['score_mean']
    check(len(s) == n and s.index.isin(mi).all(), f'vlm_frame_scores.csv: {m} {COND} on the {n} frames')
    score[m] = s.reindex(mi).to_numpy(float)
tr = frames.merge(pd.read_csv(TRUTH, float_precision='round_trip'), on=['segment', 'frame_idx'], how='left',
                  validate='one_to_one')
cue = {'true_ego_speed': tr['v_ego'].to_numpy(float), 'true_ego_accel': tr['a_ego'].to_numpy(float)}
groupings, _ = av.scene_groupings(frames, SL, ['weather', 'time_of_day'])
lab_all = groupings['all']


def tau_all(x, y, idx):
    xs, ys = x[idx], y[idx]
    mask = ~np.isnan(xs) & ~np.isnan(ys)
    return av.stats_by_grouping(xs[mask], np.ones(int(mask.sum()), bool), ys[mask], ids[idx][mask], lab_all[idx][mask],
                                None)[3]


keys = [(m, v) for m in (BASE, NEW) for v in cue]
point = {k: tau_all(score[k[0]], cue[k[1]], ids) for k in keys}
boot = {k: np.empty(N_BOOT) for k in keys}
tb = time.time()
for b, draw in enumerate(draws):
    idx = np.concatenate([seg_frames[s] for s in draw])
    for k in keys:
        boot[k][b] = tau_all(score[k[0]], cue[k[1]], idx)
print(f'  2,000 resamples in {time.time() - tb:.0f} s')
rows = []
for v in cue:
    for m in (BASE, NEW):
        lo, hi, nv = ar.ci(boot[(m, v)])
        nfr = int((~np.isnan(score[m]) & ~np.isnan(cue[v])).sum())
        rows.append(dict(model=m, condition=COND, variable=v, grouping='all', frames=float(nfr), tau_b=point[(m, v)],
                         tau_b_lo=lo, tau_b_hi=hi, nboot=nv, side=side(lo, hi)))
    lo, hi, nv = ar.ci(boot[(NEW, v)] - boot[(BASE, v)])
    rows.append(dict(model=f'{NEW} - {BASE}', condition=COND, variable=v, grouping='all', frames=np.nan,
                     tau_b=point[(NEW, v)] - point[(BASE, v)], tau_b_lo=lo, tau_b_hi=hi, nboot=nv, side=side(lo, hi)))


def line(d):
    return ','.join([d['model'], d['condition'], d['variable'], d['grouping'], txt(d['frames']), txt(d['tau_b']),
                     txt(d['tau_b_lo']), txt(d['tau_b_hi']), str(d['nboot']), d['side']])


print('\nchecks against the analyses')
p_sp = os.path.join(SUB, 'exploratory_speed_speed_distance_tau_b.csv')
rs = [x for x in rows_of(p_sp, condition=COND, grouping='all', variable='true_ego_speed')]
mine = [line(d) for d in rows if d['variable'] == 'true_ego_speed']
check([x[0] for x in rs] == [38, 39, 40] and [x[2] for x in rs] == mine,
      f'true ego speed (base, speed, speed - base) recomputed: the same text as {os.path.basename(p_sp)} lines '
      f'{[x[0] for x in rs]}')
for x, mm in zip(rs, mine):
    print(f'    line {x[0]}: {x[2]}' + ('' if x[2] == mm else f'\n      here: {mm}'))
p_ac = os.path.join(ACC, 'exploratory_accel_speed_distance_tau_b.csv')
ra = rows_of(p_ac, model=BASE, condition=COND, grouping='all', variable='true_ego_accel')
mine_b = line(next(d for d in rows if d['model'] == BASE and d['variable'] == 'true_ego_accel'))
check(len(ra) == 1 and ra[0][0] == 74 and ra[0][2] == mine_b,
      f'base vs the true ego acceleration recomputed: the same text as accel_risk_27b {os.path.basename(p_ac)} line '
      f'{ra[0][0] if ra else "?"}')
print(f'    line {ra[0][0]}: {ra[0][2]}' + ('' if ra[0][2] == mine_b else f'\n      here: {mine_b}'))

print('\nthe speed run vs the true ego acceleration (EXPLORATORY; columns as the analyses\' speed_distance_tau_b tables)')
for d in rows:
    if d['variable'] == 'true_ego_accel':
        print(f'  {line(d)}')
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'exploratory_sec45_speed_ego_accel_tau_b.csv'), index=False)
print(f'  written: exploratory_sec45_speed_ego_accel_tau_b.csv in {OUT} (rows 2-4 true_ego_speed, 5-7 true_ego_accel)')
print(f'\nchecks not met: {len(failed)}')
for w in failed:
    print('  ', w)
sys.exit(1 if failed else 0)
