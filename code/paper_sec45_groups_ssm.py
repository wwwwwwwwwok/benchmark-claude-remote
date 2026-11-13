"""Paper section 4.5, one more analysis (user, 2026-11-13). EXPLORATORY; CPU only; reads existing outputs only; no image
is opened. Qwen3.6-27B with thinking on, sep_camera, the main result's 1,084 frames. The frames are grouped by how well
the model itself estimates the quantity (the 4.4 quantities run, median of the readable repeats: sec44_frames.csv), and
in each group, over the pairs of frames inside the group only, the tau-b of the score with the nine SSMs, the true ego
speed and the true ego acceleration (the 11 rows of tab:told) is taken for images only and for the run told the
quantity, to see how the agreement changes in each group once the speed or the acceleration is told.
Speed groups (as paper_sec45_est_speed.py; images only = the main 27B run, told = speed_risk_27b):
  1 estimate < truth (underestimated), 2 estimate > truth (overestimated).
Acceleration groups (images only vs accel_risk_27b): the told direction = the sign of the number in the sentence
  (run_vlm_accel.accel_text: the truth to 1 decimal, "-0.0" written "0.0", so 0 = told steady) x the sign of the
  estimate (exactly 0 a class of its own): the 9 groups of paper_sec45_est_accel.py, the two same-direction groups split
  by size (|estimate| > |true acceleration| = overestimated, < = underestimated):
  1 told speeding up x thought speeding up, overestimated; 2 the same, underestimated; 3 told speeding up x thought 0;
  4 told speeding up x thought slowing down; 5 told slowing down x thought slowing down, overestimated; 6 the same,
  underestimated; 7 told slowing down x thought 0; 8 told slowing down x thought speeding up; 9 told 0 x thought slowing
  down; 10 told 0 x thought 0; 11 told 0 x thought speeding up.
Version 1: all 11 groups; groups of 10 frames or more with intervals, the others point estimates only ("fewer than 10
  frames"), none where not computable. Version 2: the groups of fewer than 10 frames left out (1, 2, 6, 8, 11), 6 groups
  on 1,055 frames, plus its own "all" on those frames (images only and told the acceleration). The 6 groups' numbers are
  the same in both versions (each group uses its own frames only), so they are written once (column versions). The
  speed groups are the same in both versions.
Per group and row: images only, told, change = told - images only. tau-b: analyze_vlm_ssm.all_stats with one group
  (tau_b_conflict, the agreement.csv column; = tau_b_all here, checked), score = score_mean (the mean of the 5 answers);
  true speed and acceleration (quantities/truth_main.csv v_ego, a_ego): analyze_vlm_ssm.stats_by_grouping as
  analyze_told_quantity.py item 3; fewer than 3 frames = undefined (as paper_sec43_groups.py). Per group also: the mean
  score change (told - images only per frame, as paper_sec45_est_*.py: defined with 1 frame or more) and the median true
  ego speed. Differences between groups (of the changes): speed 1 - 2; acceleration 4 - 3 and 7 - 5.
Intervals: the main analysis's 2,000 resamples of whole segments (default_rng(20261021), one choice over the 238 sorted
  segments per resample; drawn segments' sha256 48bc1f41...), version 2 from the same resamples; in each resample the
  drawn frames are split by group and every group's statistics recomputed, every difference taken within the resample;
  2.5 / 97.5 % percentiles (numpy linear), none when fewer than 90 % defined (analyze_rules.ci), with the number used.
Checks (the program stops when one is not met): 1 frames and segments per group as given by the user, 1,055 = 1,084 -
  29; 2 with the group = all 1,084 frames, the 3 runs x 11 rows' tau-b and intervals and the changes with their
  intervals the same text as the sources of tab:told (agreement.csv of after_qwen3.5-4b lines 866-890, of speed_risk_27b
  and accel_risk_27b lines 74-106; exploratory_*_agreement_differences.csv; exploratory_*_speed_distance_tau_b.csv; the
  speed run's true-acceleration row in exploratory_sec45_speed_ego_accel_tau_b.csv, which the speed analysis did not
  hold); 3 the mean score change of groups 3, 4, 7, 8, 9, 10 = exploratory_sec45_est_accel_groups.csv (the same text),
  groups 1 + 2 = its line 2, 5 + 6 = its line 10. Added by Claude: the speed groups' mean change and under - over =
  exploratory_sec45_est_speed_groups.csv; the acceleration 4 - 3 of the mean change = est_accel_groups.csv line 11.
Outputs in <out_dir>: exploratory_groups_ssm_counts.csv, _tau_b.csv, _mean_change.csv, _group_differences.csv,
  _check.csv; groups_ssm_frames.csv (one row per frame, numbers only) and groups_ssm_frames_columns.txt.
--pairs --ref <the out_dir of the run without --pairs> (user, 2026-11-13, after 6ea1cd9; EXPLORATORY; without the option
  every output is the same as before, byte for byte): the same inputs, groups and checks up to the group counts, then
  tau-b over classes of frame pairs: tau-b = S / sqrt(nx ny), S = concordant - discordant pairs, nx, ny = pairs untied in
  the score, in the row, each summed over the pairs of the class (as the pooled "same group" tau-b of 4.3), same 11 rows.
  Speed (1,084 frames, images only vs + speed): within (pairs inside the underestimated group plus pairs inside the
  overestimated group) and across (one frame of each). Acceleration (version 2, 1,055 frames, images only vs + accel;
  the told direction: groups 3, 4 speeding up, 5, 7 slowing down, 9, 10 zero): within (inside each of the 6 groups),
  across with the same told direction, with different told directions, and among those told speeding up x told slowing
  down. Per class images only, told, change; differences across - within (speed) and speeding up x slowing down - within
  (acceleration) of the change. Among the speeding up x slowing down pairs: the share where the frame told speeding up
  scores higher, lower, the same, for both runs and + accel - images only; for the 9 SSMs and the true speed the share
  where the frame told speeding up is the more dangerous (faster), point estimates only. The same 2,000 resamples, every
  statistic recomputed on the drawn frames (a frame drawn twice: the pair of its two copies is tied in both, so it adds
  nothing), every difference within the resample, analyze_rules.ci. Checks (stop when not met): pairs per class as given
  by the user; the classes' S, nx, ny add up to all pairs' (analyze_vlm_ssm.pair_counts on all frames); all pairs' tau-b
  and each group's own tau-b = the run without --pairs (within 1e-12, and whether the text is the same); the user's
  local point estimates and counts, the same text. Outputs in <out_dir>: exploratory_groups_ssm_pairs_tau_b.csv,
  _differences.csv, _higher.csv, _check.csv (and exploratory_groups_ssm_counts.csv as without the option).
usage: python -I paper_sec45_groups_ssm.py <waymo_v143 folder> <sec44_frames.csv> <sec45 folder> <out_dir> [--workers N]
       [--pairs --ref <out_dir of the run without --pairs>]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from decimal import ROUND_HALF_UP, Decimal  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_rules as ar  # noqa: E402
import analyze_vlm_ssm as av  # noqa: E402
from run_vlm_accel import accel_text  # noqa: E402

N_BOOT, SEED = 2000, 20261021
HASH = '48bc1f413f9d5b99da57ff4edde7bbb4eb378c5a9343c44ccad4e0165466604c'
MIN_CI = 10
BASE, SPEED, ACCEL, COND = 'qwen3.6-27b', 'qwen3.6-27b-speed', 'qwen3.6-27b-accel', 'sep_camera'
CUES = ['true_ego_speed', 'true_ego_accel']
ROWS = av.SSMS + CUES
SHOW = {'TTC2D': '2D-TTC', 'true_ego_speed': 'true speed', 'true_ego_accel': 'true acceleration'}
RUN_NAME = {'base': 'images only', 'speed': '+ speed', 'accel': '+ accel'}
SPEED_GROUPS = {1: ('underestimated (estimate < truth)', 675, 199), 2: ('overestimated (estimate > truth)', 409, 143)}
# group: (told sign, thought sign, size, label, frames, segments), the frames and segments as given by the user
ACCEL_GROUPS = {
    1: (1, 1, 'over', 'told speeding up x thought speeding up, overestimated', 9, 9),
    2: (1, 1, 'under', 'told speeding up x thought speeding up, underestimated', 4, 3),
    3: (1, 0, None, 'told speeding up x thought 0', 199, 97),
    4: (1, -1, None, 'told speeding up x thought slowing down', 153, 74),
    5: (-1, -1, 'over', 'told slowing down x thought slowing down, overestimated', 388, 125),
    6: (-1, -1, 'under', 'told slowing down x thought slowing down, underestimated', 9, 8),
    7: (-1, 0, None, 'told slowing down x thought 0', 206, 84),
    8: (-1, 1, None, 'told slowing down x thought speeding up', 7, 6),
    9: (0, -1, None, 'told 0 x thought slowing down', 59, 40),
    10: (0, 0, None, 'told 0 x thought 0', 50, 36),
    11: (0, 1, None, 'told 0 x thought speeding up', 0, 0)}
V2_GROUPS, V2_FRAMES, V2_SEGMENTS = [3, 4, 5, 7, 9, 10], 1055, 235
EST_ROW = {3: ('told speeding up', 'thought steady (0)'), 4: ('told speeding up', 'thought slowing down'),
           7: ('told slowing down', 'thought steady (0)'), 8: ('told slowing down', 'thought speeding up'),
           9: ('told steady (0.0)', 'thought slowing down'), 10: ('told steady (0.0)', 'thought steady (0)')}
MERGED = {(1, 2): (('told speeding up', 'thought speeding up'), 2), (5, 6): (('told slowing down', 'thought slowing down'), 10)}
DIFFS = [('speed: underestimated (1) - overestimated (2)', ('speed', 1), ('speed', 2), 'speed'),
         ('acceleration, told speeding up: thought slowing down (4) - thought 0 (3)', ('accel', 4), ('accel', 3), 'accel'),
         ('acceleration, told slowing down: thought 0 (7) - thought slowing down, overestimated (5)', ('accel', 7),
          ('accel', 5), 'accel')]
G = {}                                                            # data for the worker processes (fork)


def txt(x):
    return '' if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)) else repr(float(x))


def r(x):
    return txt(x) or 'none'


def side(lo, hi):
    if lo is None or np.isnan(lo):
        return 'no interval'
    return 'above 0' if lo > 0 else ('below 0' if hi < 0 else 'contains 0')


def csv_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        for d in rd:
            yield rd.line_num, d


def taus(x, idx, with_all=False):
    """The 11 rows' tau-b of the score x on the frames idx (a frame may come more than once); NaN with fewer than 3."""
    if len(idx) < 3:
        out = {k: np.nan for k in ROWS}
        if with_all:
            out.update({('tau_b_all', s): np.nan for s in av.SSMS})
        return out
    out = {}
    ids, one = G['ids'][idx], np.zeros(len(idx), int)
    xs = x[idx]
    st = av.all_stats(xs, {k: v[idx] for k, v in G['ssm'].items()}, ids, {'all': one})
    for s in av.SSMS:
        out[s] = st[s][0]['all'][1]                                # tau_b_conflict
        if with_all:
            out[('tau_b_all', s)] = st[s][0]['all'][4]
    every = np.ones(len(idx), bool)
    for var in CUES:                                               # analyze_told_quantity.py item 3, grouping all
        out[var] = av.stats_by_grouping(xs, every, G['cue'][var][idx], ids, one)[3]
    return out


def stats(idx, point=False):
    """Every statistic on the frames idx (all frames, or one resample)."""
    res = {}
    sg, ag, v2 = G['speed_group'][idx], G['accel_group'][idx], G['v2'][idx]

    def put(tab, runs, ii):
        res[('n', tab)] = len(ii)
        for run in runs:
            for k, v in taus(G['score'][run], ii, point).items():
                res[(tab, run, k)] = v
            if run != 'base':
                res[('mean', tab, run)] = float(G['d'][run][ii].mean()) if len(ii) else np.nan

    put('all_1084', ('base', 'speed', 'accel'), idx)
    for g in SPEED_GROUPS:
        put(('speed', g), ('base', 'speed'), idx[sg == g])
    for g, spec in ACCEL_GROUPS.items():
        ii = idx[ag == g]
        if point or spec[4] >= MIN_CI:
            put(('accel', g), ('base', 'accel'), ii)
        else:
            res[('n', ('accel', g))] = len(ii)
    put('all_1055', ('base', 'accel'), idx[v2])
    if point:
        for gs in MERGED:
            res[('mean_merged', gs)] = float(G['d']['accel'][idx[np.isin(ag, gs)]].mean())
    return res


def one(b):
    idx = np.concatenate([G['seg_frames'][s] for s in G['draws'][b]])
    rr = stats(idx)
    return np.array([rr.get(k, np.nan) for k in G['keys']], float)


# ---------------------------------------------------------------------------------------------------- --pairs
P = {}                                                            # data for the --pairs worker processes (fork)
# pair codes: speed 0 within, 1 across; acceleration 0 within, 1 across with the same told direction, 2 told speeding
# up x told slowing down, 3 the other pairs with different told directions (one frame told 0)
CLS = {'speed': {'within': [0], 'across': [1], 'all': [0, 1]},
       'accel': {'within': [0], 'across_same_told': [1], 'different_told': [2, 3], 'up_x_down': [2],
                 'all': [0, 1, 2, 3]}}
CLS_NAME = {'within': 'within groups', 'across': 'across the two groups', 'all': 'all pairs',
            'across_same_told': 'across groups, the same told direction', 'different_told': 'different told directions',
            'up_x_down': 'told speeding up x told slowing down'}
PAIR_N = {('speed', 'within'): 310911, ('speed', 'across'): 276075, ('accel', 'within'): 130458,
          ('accel', 'across_same_told'): 113325, ('accel', 'different_told'): 312202, ('accel', 'up_x_down'): 209088}
PAIR_DIFFS = [('speed', 'across', 'within'), ('accel', 'up_x_down', 'within')]
# the user's local point estimates: the same text; the two marked 15 were given to 15 significant digits (the first run
# of --pairs stopped there, 31704812: rounded to 15 significant digits they are the same; user, 2026-11-13: accept them
# as matching when rounded to 15 significant digits, the others the same text)
USER_PT = [(('speed', 'within', 'THW'), '0.08340830014358655', 0), (('speed', 'across', 'THW'), '0.08395416093171847', 0),
           (('speed', 'within', 'PSD'), '0.15396294060766805', 0), (('speed', 'across', 'PSD'), '0.15325468083277374', 0),
           (('accel', 'within', 'CI'), '0.0003495514201248', 15), (('accel', 'up_x_down', 'CI'), '0.11959168676598457', 0),
           (('accel', 'within', 'TTC'), '0.00396175379099819', 15), (('accel', 'up_x_down', 'TTC'), '-0.1174788620244554', 0)]
USER_HIGHER = {'base': (108660, 96526, 3902), 'accel': (130710, 75008, 3370)}
USER_DANGER = {'CI': 158912, 'TTC': 49754}
HIGHER = ['up_higher', 'down_higher', 'same']
DANGER_ROWS = av.SSMS + ['true_ego_speed']
REF_ALL = {'speed': ('all 1,084 frames (= tab:told)', '+ speed'), 'accel': ('version 2: all 1,055 frames', '+ accel')}
REF_GROUP = {'speed': ('speed group', '+ speed'), 'accel': ('acceleration group', '+ accel')}


def pair_part(ii, part, point=False):
    """S, nx, ny per pair code (exact integers in float64) of the frames ii (a frame may come more than once), for
    images only and the told run, every row; for the acceleration part also who scores higher among the speeding up x
    slowing down pairs."""
    g = (P['speed_group'] if part == 'speed' else P['accel_group'])[ii]
    p, q = np.triu_indices(len(ii), 1)
    same_g = g[p] == g[q]
    if part == 'speed':
        code, K = (~same_g).astype(np.int64), 2
    else:
        dg = P['told_dir'][ii]
        code = np.where(same_g, 0, np.where(dg[p] == dg[q], 1, np.where(dg[p] * dg[q] == -1, 2, 3))).astype(np.int64)
        K = 4
        m2 = code == 2
        o = dg[p][m2]                                            # +1: the first frame of the pair is the one told speeding up
    res = {'npairs': np.bincount(code, minlength=K).astype(float)}
    sy, ny = {}, {}
    for row in ROWS:
        v = P['row'][row][ii]
        sy[row] = np.sign(v[p] - v[q])
        ny[row] = np.bincount(code, weights=np.abs(sy[row]), minlength=K)
    if point:
        gcode = np.where(same_g, g[p], 0).astype(np.int64)
    for run in ('base', part):
        x = P['score'][run][ii]
        sx = np.sign(x[p] - x[q])
        nx = np.bincount(code, weights=np.abs(sx), minlength=K)
        for row in ROWS:
            res[(run, row)] = (np.bincount(code, weights=sx * sy[row], minlength=K), nx, ny[row])
            if point:
                res[('group', run, row)] = (np.bincount(gcode, weights=sx * sy[row], minlength=12),
                                            np.bincount(gcode, weights=np.abs(sx), minlength=12),
                                            np.bincount(gcode, weights=np.abs(sy[row]), minlength=12))
        if part == 'accel':
            w = sx[m2] * o
            res[('higher', run)] = (int((w > 0).sum()), int((w < 0).sum()), int((w == 0).sum()))
    if part == 'accel' and point:
        for row in DANGER_ROWS:
            w = sy[row][m2] * o
            res[('danger', row)] = (int((w > 0).sum()), int((w < 0).sum()), int((w == 0).sum()))
    return res


def tau_of(S, nx, ny):
    return float(S / np.sqrt(nx * ny)) if nx * ny > 0 else np.nan


def pair_values(res):
    out = {}
    for part, rr in res.items():
        for cl, codes in CLS[part].items():
            out[('npairs', part, cl)] = float(rr['npairs'][codes].sum())
            for run in ('base', part):
                for row in ROWS:
                    S, nx, ny = rr[(run, row)]
                    out[('tau', part, cl, run, row)] = tau_of(S[codes].sum(), nx[codes].sum(), ny[codes].sum())
        if part == 'accel':
            n2 = rr['npairs'][2]
            for run in ('base', 'accel'):
                for k, c in zip(HIGHER, rr[('higher', run)]):
                    out[('share', run, k)] = c / n2 if n2 > 0 else np.nan
    return out


def one_pairs(b):
    idx = np.concatenate([P['seg_frames'][s] for s in P['draws'][b]])
    rr = pair_values({'speed': pair_part(idx, 'speed'), 'accel': pair_part(idx[P['v2'][idx]], 'accel')})
    return np.array([rr.get(k, np.nan) for k in P['keys']], float)


def run_pairs(a, O, D, check, stop_if_failed, failed, t0):
    """--pairs: tau-b over classes of frame pairs (see the docstring)."""
    P.update(D)
    P['row'] = {**{s: D['ssm'][s + '_risk'] for s in av.SSMS}, 'true_ego_speed': D['v'], 'true_ego_accel': D['at']}
    P['score'] = {'base': D['base'], 'speed': D['spd'], 'accel': D['acc']}
    n = len(D['ids'])
    ii = {'speed': np.arange(n), 'accel': np.arange(n)[D['v2']]}
    print('\n' + '=' * 110 + f'\n--pairs (user, 2026-11-13; EXPLORATORY): tau-b = S / sqrt(nx ny) over classes of frame '
          f'pairs; reference run {a.ref}\n\n2p. point estimates and checks')
    pt = {part: pair_part(ii[part], part, point=True) for part in ('speed', 'accel')}
    point = pair_values(pt)
    for (part, cl), want in PAIR_N.items():
        got = int(point[('npairs', part, cl)])
        check(got == want, f'{part} {CLS_NAME[cl]}: {got} frame pairs; expected {want}')
    n_ok = n_all = 0
    dmax = 0.0
    for part in ('speed', 'accel'):
        for run in ('base', part):
            for row in ROWS:
                S, nx, ny = pt[part][(run, row)]
                s0, tx0, ty0, _, _ = av.pair_counts(P['score'][run][ii[part]], P['row'][row][ii[part]], D['ids'][ii[part]])
                n_all += 1
                n_ok += (tx0 == nx.sum() and ty0 == ny.sum() and abs(s0 - S.sum()) < 1e-6
                         and all(float(v_).is_integer() for v_ in np.concatenate([S, nx, ny])))
                dmax = max(dmax, abs(s0 - S.sum()))
    check(n_ok == n_all, f'the classes\' S, nx, ny (whole numbers) add up to all pairs\' (analyze_vlm_ssm.pair_counts on all '
          f'frames: nx, ny the same, S within 1e-6) in {n_ok} of {n_all} (2 parts x 2 runs x 11 rows); largest |S '
          f'difference| {dmax!r}')
    ref = {(d['table'], d['group_no'], d['told_run'], d['row']): (ln, d)
           for ln, d in csv_rows(os.path.join(a.ref, 'exploratory_groups_ssm_tau_b.csv'))}
    rows_k = []

    def compare(what, mine, ln, d, col):
        rows_k.append(dict(what=what, mine=txt(mine), reference=d[col], difference=mine - float(d[col]),
                           same_text=txt(mine) == d[col], source=f'{a.ref}/exploratory_groups_ssm_tau_b.csv', line=ln,
                           column=col))

    for part in ('speed', 'accel'):
        tab, told = REF_ALL[part]
        for row in ROWS:
            ln, d = ref[(tab, '', told, SHOW.get(row, row))]
            b_, t_ = point[('tau', part, 'all', 'base', row)], point[('tau', part, 'all', part, row)]
            compare(f'{part} all pairs, images only, {row}', b_, ln, d, 'images_only')
            compare(f'{part} all pairs, {told}, {row}', t_, ln, d, 'told')
            compare(f'{part} all pairs, change, {row}', t_ - b_, ln, d, 'change')
        tab, told = REF_GROUP[part]
        groups = list(SPEED_GROUPS) if part == 'speed' else V2_GROUPS
        for g in groups:
            for row in ROWS:
                ln, d = ref[(tab, str(g), told, SHOW.get(row, row))]
                for run, col in (('base', 'images_only'), (part, 'told')):
                    S, nx, ny = pt[part][('group', run, row)]
                    compare(f'{part} group {g} inside, {col}, {row}', tau_of(S[g], nx[g], ny[g]), ln, d, col)
    dm = max(abs(x['difference']) for x in rows_k)
    check(dm <= 1e-12, f'all pairs\' tau-b (and change) and each group\'s own tau-b = the run without --pairs: largest '
          f'|difference| {dm!r} over {len(rows_k)} values; the same text {sum(x["same_text"] for x in rows_k)}')
    for (part, cl, row), want, digits in USER_PT:
        b_, t_ = point[('tau', part, cl, 'base', row)], point[('tau', part, cl, part, row)]
        mine = txt(t_ - b_) if digits == 0 else format(t_ - b_, f'.{digits}g')
        how = 'the same text' if digits == 0 else f'{txt(t_ - b_)} rounded to {digits} significant digits'
        alt = []
        if mine != want:
            for run in ('base', part):
                S, nx, ny = [v_[CLS[part][cl]].sum() for v_ in pt[part][(run, row)]]
                alt.append(f'{run}: S {S:g}, nx {nx:g}, ny {ny:g}, S/sqrt(nx)/sqrt(ny) {S / np.sqrt(nx) / np.sqrt(ny)!r}')
        check(mine == want, f'user\'s local estimate: {part} {CLS_NAME[cl]} {row} change {mine} = {want} ({how})'
              + (f' ({"; ".join(alt)})' if alt else ''))
    for run in ('base', 'accel'):
        got = pt['accel'][('higher', run)]
        check(got == USER_HIGHER[run], f'user\'s local counts: told speeding up x told slowing down, {RUN_NAME[run]}: the '
              f'frame told speeding up scores higher / lower / the same {got} = {USER_HIGHER[run]}')
    for row, want in USER_DANGER.items():
        got = pt['accel'][('danger', row)][0]
        check(got == want, f'user\'s local counts: the frame told speeding up is the more dangerous in {row}: {got} = {want}')
    stop_if_failed('before the resamples (--pairs)')

    keys = list(point)
    P['keys'] = keys
    col = {k: i for i, k in enumerate(keys)}
    print(f'\n3p. resamples: {N_BOOT}, {a.workers} worker(s)', flush=True)
    tb = time.time()
    out = []
    with mp.get_context('fork').Pool(max(a.workers, 1)) as pool:
        for i, row_ in enumerate(pool.imap(one_pairs, range(N_BOOT), chunksize=8)):
            out.append(row_)
            if (i + 1) % 250 == 0:
                print(f'  [resample {i + 1}/{N_BOOT}, {(time.time() - tb) / (i + 1) * (N_BOOT - i - 1):.0f} s left]',
                      flush=True)
    boot = np.stack(out)
    print(f'  done in {time.time() - tb:.0f} s')

    def B(k):
        return boot[:, col[k]]

    rows_t, rows_d, rows_h = [], [], []
    for part in ('speed', 'accel'):
        for cl in CLS[part]:
            for row in ROWS:
                kb, kt = ('tau', part, cl, 'base', row), ('tau', part, cl, part, row)
                lb, hb, nb = ar.ci(B(kb))
                lt, ht, nt = ar.ci(B(kt))
                lc, hc, nc = ar.ci(B(kt) - B(kb))
                Sb, nxb, nyb = [v_[CLS[part][cl]].sum() for v_ in pt[part][('base', row)]]
                St, nxt, nyt = [v_[CLS[part][cl]].sum() for v_ in pt[part][(part, row)]]
                rows_t.append(dict(part=part, pair_class=CLS_NAME[cl], frame_pairs=int(point[('npairs', part, cl)]),
                                   told_run=RUN_NAME[part], row=SHOW.get(row, row),
                                   images_only=point[kb], images_only_lo=lb, images_only_hi=hb, images_only_nboot=nb,
                                   images_only_side=side(lb, hb), told=point[kt], told_lo=lt, told_hi=ht, told_nboot=nt,
                                   told_side=side(lt, ht), change=point[kt] - point[kb], change_lo=lc, change_hi=hc,
                                   change_nboot=nc, change_side=side(lc, hc), S_images_only=int(Sb), nx_images_only=int(nxb),
                                   ny_images_only=int(nyb), S_told=int(St), nx_told=int(nxt), ny_told=int(nyt)))
    for part, c1, c0 in PAIR_DIFFS:
        for row in ROWS:
            k1b, k1t = ('tau', part, c1, 'base', row), ('tau', part, c1, part, row)
            k0b, k0t = ('tau', part, c0, 'base', row), ('tau', part, c0, part, row)
            ch1, ch0 = point[k1t] - point[k1b], point[k0t] - point[k0b]
            lo, hi, nv = ar.ci((B(k1t) - B(k1b)) - (B(k0t) - B(k0b)))
            rows_d.append(dict(comparison=f'{part}: change {CLS_NAME[c1]} - change {CLS_NAME[c0]}', row=SHOW.get(row, row),
                               change_1=ch1, change_0=ch0, diff=ch1 - ch0, lo=lo, hi=hi, nboot=nv, side=side(lo, hi)))
    n2 = int(point[('npairs', 'accel', 'up_x_down')])
    for run in ('base', 'accel'):
        for k, c in zip(HIGHER, pt['accel'][('higher', run)]):
            lo, hi, nv = ar.ci(B(('share', run, k)))
            rows_h.append(dict(what=f'score, {RUN_NAME[run]}', measure=k, pairs=c, of_pairs=n2, share=point[('share', run, k)],
                               lo=lo, hi=hi, nboot=nv, side=''))
    for k in HIGHER:
        lo, hi, nv = ar.ci(B(('share', 'accel', k)) - B(('share', 'base', k)))
        rows_h.append(dict(what='score, + accel - images only', measure=k, pairs=np.nan, of_pairs=n2,
                           share=point[('share', 'accel', k)] - point[('share', 'base', k)], lo=lo, hi=hi, nboot=nv,
                           side=side(lo, hi)))
    for row in DANGER_ROWS:
        up, down, same = pt['accel'][('danger', row)]
        meas = 'up_faster' if row == 'true_ego_speed' else 'up_more_dangerous'
        rows_h.append(dict(what=SHOW.get(row, row), measure=meas, pairs=up, of_pairs=n2, share=up / n2, lo=np.nan, hi=np.nan,
                           nboot=np.nan, side='', down_count=down, same_count=same))

    print('\n4p. checks with the intervals')
    n_ci = n_ci_same = 0
    dm = 0.0
    for part in ('speed', 'accel'):
        tab, told = REF_ALL[part]
        for x in rows_t:
            if x['part'] != part or x['pair_class'] != CLS_NAME['all']:
                continue
            ln, d = ref[(tab, '', told, x['row'])]
            for c_ in ('images_only', 'told', 'change'):
                for suf in ('_lo', '_hi'):
                    n_ci += 1
                    n_ci_same += txt(x[c_ + suf]) == d[c_ + suf]
                    dm = max(dm, abs(x[c_ + suf] - float(d[c_ + suf])))
                    rows_k.append(dict(what=f'{part} all pairs, {c_}{suf}, {x["row"]}', mine=txt(x[c_ + suf]),
                                       reference=d[c_ + suf], difference=x[c_ + suf] - float(d[c_ + suf]),
                                       same_text=txt(x[c_ + suf]) == d[c_ + suf],
                                       source=f'{a.ref}/exploratory_groups_ssm_tau_b.csv', line=ln, column=c_ + suf))
                n_ci += 1
                n_ci_same += str(float(x[c_ + '_nboot'])) == d[c_ + '_nboot']
    check(dm <= 1e-12, f'all pairs\' intervals (images only, told, change; and resamples used) = the run without --pairs: '
          f'largest |difference| {dm!r}; the same text {n_ci_same} of {n_ci}')

    pd.DataFrame(rows_t).to_csv(os.path.join(O, 'exploratory_groups_ssm_pairs_tau_b.csv'), index=False)
    pd.DataFrame(rows_d).to_csv(os.path.join(O, 'exploratory_groups_ssm_pairs_differences.csv'), index=False)
    pd.DataFrame(rows_h).to_csv(os.path.join(O, 'exploratory_groups_ssm_pairs_higher.csv'), index=False)
    pd.DataFrame(rows_k).to_csv(os.path.join(O, 'exploratory_groups_ssm_pairs_check.csv'), index=False)

    def star(lo, hi):
        return '*' if side(lo, hi) in ('above 0', 'below 0') else ' '

    print('\n' + '=' * 110 + '\ntau-b over classes of frame pairs (exploratory_groups_ssm_pairs_tau_b.csv); * = interval '
          'leaves out 0')
    last = None
    for i, x in enumerate(rows_t):
        if (x['part'], x['pair_class']) != last:
            print(f"\n  {x['part']}: {x['pair_class']} ({x['frame_pairs']} frame pairs; images only vs {x['told_run']})")
            last = (x['part'], x['pair_class'])
        print(f"    line {i + 2}: {x['row']:17s} images only {star(x['images_only_lo'], x['images_only_hi'])}"
              f"{r(x['images_only'])} [{r(x['images_only_lo'])}, {r(x['images_only_hi'])}] (n {x['images_only_nboot']}); "
              f"told {star(x['told_lo'], x['told_hi'])}{r(x['told'])} [{r(x['told_lo'])}, {r(x['told_hi'])}] "
              f"(n {x['told_nboot']}); change {star(x['change_lo'], x['change_hi'])}{r(x['change'])} "
              f"[{r(x['change_lo'])}, {r(x['change_hi'])}] (n {x['change_nboot']})")
    print('\ndifferences of the change between classes (exploratory_groups_ssm_pairs_differences.csv)')
    for i, x in enumerate(rows_d):
        print(f"  line {i + 2}: {x['comparison']}: {x['row']:17s} {star(x['lo'], x['hi'])}{r(x['diff'])} "
              f"[{r(x['lo'])}, {r(x['hi'])}] (n {x['nboot']}; changes {r(x['change_1'])} and {r(x['change_0'])})")
    print('\ntold speeding up x told slowing down: who is higher (exploratory_groups_ssm_pairs_higher.csv)')
    for i, x in enumerate(rows_h):
        print(f"  line {i + 2}: {x['what']}: {x['measure']}: {x['pairs']} of {x['of_pairs']} pairs, share "
              f"{star(x['lo'], x['hi']) if x['side'] else ' '}{r(x['share'])} [{r(x['lo'])}, {r(x['hi'])}] (n {x['nboot']})"
              + (f"; lower {x['down_count']}, the same {x['same_count']}" if 'down_count' in x else ''))
    print(f'\nwritten in {O}: exploratory_groups_ssm_pairs_tau_b.csv, _differences.csv, _higher.csv, _check.csv')
    print(f'done in {time.time() - t0:.0f} s; checks not met: {len(failed)}')
    for w_ in failed:
        print('  ', w_)
    sys.exit(1 if failed else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('waymo')
    ap.add_argument('sec44')
    ap.add_argument('sec45')
    ap.add_argument('out')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--pairs', action='store_true', help='tau-b over classes of frame pairs (see the docstring)')
    ap.add_argument('--ref', default=None, help='with --pairs: the out_dir of the run without --pairs')
    a = ap.parse_args()
    A = os.path.join(a.waymo, 'analysis')
    RISK = os.path.join(A, 'after_qwen3.5-4b', 'main_1084')
    SP = os.path.join(A, 'speed_risk_27b', 'main_1084')
    AC = os.path.join(A, 'accel_risk_27b', 'main_1084')
    TRUTH = os.path.join(a.waymo, 'quantities', 'truth_main.csv')
    FSSM = os.path.join(a.waymo, 'results_lead_final_excl', 'frame_ssm_validation_training.csv')
    EST_A = os.path.join(a.sec45, 'est_accel_with_classes', 'exploratory_sec45_est_accel_groups.csv')
    EST_S = os.path.join(a.sec45, 'exploratory_sec45_est_speed_groups.csv')
    SPA = os.path.join(a.sec45, 'exploratory_sec45_speed_ego_accel_tau_b.csv')
    O = a.out
    if a.pairs:
        if not a.ref or not os.path.exists(os.path.join(a.ref, 'exploratory_groups_ssm_tau_b.csv')):
            sys.exit('--pairs needs --ref <the out_dir of the run without --pairs>')
        if os.path.exists(os.path.join(O, 'exploratory_groups_ssm_pairs_tau_b.csv')):
            sys.exit(f'{O}/exploratory_groups_ssm_pairs_tau_b.csv exists already: give a new folder')
    elif os.path.exists(os.path.join(O, 'exploratory_groups_ssm_tau_b.csv')):
        sys.exit(f'{O}/exploratory_groups_ssm_tau_b.csv exists already: give a new folder')
    os.makedirs(O, exist_ok=True)
    t0 = time.time()
    failed = []

    def check(ok, what):
        print(f'  check {"ok" if ok else "NOT MET"}: {what}', flush=True)
        if not ok:
            failed.append(what)

    def stop_if_failed(stage):
        if failed:
            print(f'\nstopped {stage}: checks not met: {len(failed)}')
            for w_ in failed:
                print('  ', w_)
            sys.exit(1)

    print(f'paper section 4.5: tau-b by groups of the model\'s own estimate (user, 2026-11-13); EXPLORATORY; '
          f'{COND}, main_1084; run {time.strftime("%Y-%m-%d %H:%M:%S")}\n\n0. inputs')
    frames = pd.read_csv(os.path.join(RISK, 'ssm_target_frames.csv'))[['segment', 'frame_idx']] \
        .sort_values(['segment', 'frame_idx']).reset_index(drop=True)
    for f_ in (SP, AC):
        o_ = pd.read_csv(os.path.join(f_, 'ssm_target_frames.csv'))[['segment', 'frame_idx']] \
            .sort_values(['segment', 'frame_idx']).reset_index(drop=True)
        check(frames.equals(o_), f'{f_}/ssm_target_frames.csv: the same {len(frames)} frames as {RISK}')
    n = len(frames)
    mi = pd.MultiIndex.from_frame(frames)
    seg_names = frames['segment'].to_numpy()
    seg_list = np.array(sorted(set(seg_names)))
    seg_frames = {s: np.flatnonzero(seg_names == s) for s in seg_list}
    ids = np.arange(n)
    for f_ in (RISK, SP, AC):
        nb_r, seed_r, frames_r, segs_r = ar.read_params_risk(os.path.join(f_, 'params.txt'))
        check((nb_r, seed_r, frames_r) == (N_BOOT, SEED, n) and list(seg_list) == segs_r,
              f'{f_}/params.txt: {nb_r} resamples, seed {seed_r}, {frames_r} frames, the same {len(seg_list)} segments')
    rng = np.random.default_rng(SEED)
    draws = [rng.choice(seg_list, len(seg_list)) for _ in range(N_BOOT)]
    h = hashlib.sha256()
    for dr in draws:
        h.update(np.searchsorted(seg_list, dr).astype(np.int32).tobytes())
    check(h.hexdigest() == HASH, f"drawn segments' sha256 {h.hexdigest()}")

    def scores(folder, model):
        fs = pd.read_csv(os.path.join(folder, 'vlm_frame_scores.csv'), float_precision='round_trip')
        s = fs[(fs.model == model) & (fs.condition == COND)].set_index(['segment', 'frame_idx'])['score_mean']
        check(len(s) == n and s.index.isin(mi).all(), f'{folder}/vlm_frame_scores.csv: {model} {COND} on the {n} frames')
        return s.reindex(mi).to_numpy(float)

    base, spd, acc = scores(RISK, BASE), scores(SP, SPEED), scores(AC, ACCEL)
    check(np.array_equal(base, scores(SP, BASE)) and np.array_equal(base, scores(AC, BASE)),
          f'{BASE} {COND} score_mean the same in the three analysis folders')
    check(not (np.isnan(base).any() or np.isnan(spd).any() or np.isnan(acc).any()), 'every frame has the three scores')
    truth = pd.read_csv(TRUTH, float_precision='round_trip')
    tr = frames.merge(truth[['segment', 'frame_idx', 'v_ego', 'a_ego']], on=['segment', 'frame_idx'], how='left',
                      validate='one_to_one')
    v, at = tr['v_ego'].to_numpy(float), tr['a_ego'].to_numpy(float)
    check(not (np.isnan(v).any() or np.isnan(at).any()), f'{TRUTH}: v_ego and a_ego of every frame')
    for folder, name, model, x, tcol, tv in ((SP, 'exploratory_speed_frames.csv', SPEED, spd, 'true_v_ego', v),
                                              (AC, 'exploratory_accel_frames.csv', ACCEL, acc, 'true_a_ego', at)):
        t_ = pd.read_csv(os.path.join(folder, name), float_precision='round_trip')
        ok_ = len(t_) == n
        t_ = t_.set_index(['segment', 'frame_idx']).reindex(mi)
        check(ok_ and np.array_equal(t_[f'{BASE}_{COND}'].to_numpy(float), base)
              and np.array_equal(t_[f'{model}_{COND}'].to_numpy(float), x) and np.array_equal(t_[tcol].to_numpy(float), tv),
              f'{folder}/{name} (the score change of paper_sec45_est_*.py): both {COND} columns = score_mean, {tcol} = '
              f'truth_main.csv')
    d = {'speed': spd - base, 'accel': acc - base}
    q_raw = pd.read_csv(a.sec44, float_precision='round_trip')
    q = q_raw.set_index(['segment', 'frame_idx']).reindex(mi)
    ev, ea = q['ego_speed_estimate_mps'].to_numpy(float), q['ego_accel_estimate_mps2'].to_numpy(float)
    check(len(q_raw) == n and np.array_equal(q['ego_speed_truth_mps'].to_numpy(float), v)
          and np.array_equal(q['ego_accel_truth_mps2'].to_numpy(float), at) and not (np.isnan(ev).any() or np.isnan(ea).any()),
          f'{a.sec44}: {len(q_raw)} frames, truths = truth_main.csv, an estimate of the ego speed and acceleration on '
          f'every frame')
    ssm_df = av.ssm_table(FSSM, frames)
    ssm = {c: ssm_df[c].to_numpy() for c in ssm_df.columns if c.endswith(('_conflict', '_risk', '_use'))}
    ssm = {k: (v_.astype(float) if k.endswith('_risk') else v_.astype(bool)) for k, v_ in ssm.items()}
    check(all(ssm[s + '_conflict'].all() and ssm[s + '_use'].all() for s in av.SSMS),
          'every frame has a value of every SSM and counts for it')
    st_ = pd.read_csv(os.path.join(RISK, 'ssm_target_frames.csv'), float_precision='round_trip') \
        .set_index(['segment', 'frame_idx']).reindex(mi)
    check(all(np.array_equal(st_[s + '_risk'].to_numpy(float), ssm[s + '_risk']) for s in av.SSMS),
          f'the SSM values used (larger = riskier) = {RISK}/ssm_target_frames.csv')
    stop_if_failed('at the inputs')

    # ---------------------------------------------------------------------------------------------- groups
    print('\n1. groups (check 1)')
    speed_group = np.where(ev < v, 1, np.where(ev > v, 2, 0))
    told = np.array([float(accel_text(x)) for x in at])
    half_up = np.array([float(Decimal(repr(float(x))).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)) for x in at])
    check(np.array_equal(told, half_up), 'the told number (accel_text) = the true acceleration rounded half up to 0.1')
    ts, es = np.sign(told).astype(int), np.sign(ea).astype(int)
    accel_group = np.zeros(n, int)
    for g, (t_s, e_s, size, *_) in ACCEL_GROUPS.items():
        sel = (ts == t_s) & (es == e_s)
        if size == 'over':
            sel &= np.abs(ea) > np.abs(at)
        elif size == 'under':
            sel &= np.abs(ea) < np.abs(at)
        accel_group[sel] = g
    rows_c = []
    for grouping, gdict, lab in (('speed', SPEED_GROUPS, speed_group), ('accel', ACCEL_GROUPS, accel_group)):
        for g, spec in gdict.items():
            m_ = lab == g
            nf, ns = int(m_.sum()), len(set(seg_names[m_]))
            want_f, want_s = spec[-2], spec[-1]
            vers = '1,2' if grouping == 'speed' or want_f >= MIN_CI else '1'
            rows_c.append(dict(grouping=grouping, group_no=g, group=spec[-3] if grouping == 'accel' else spec[0],
                               frames=nf, segments=ns, expected_frames=want_f, expected_segments=want_s,
                               same=(nf, ns) == (want_f, want_s), versions=vers))
            check((nf, ns) == (want_f, want_s), f'{grouping} group {g} ({rows_c[-1]["group"]}): {nf} frames / {ns} '
                  f'segments; expected {want_f} / {want_s}')
    print(f'  frames in no speed group (estimate = truth): {int((speed_group == 0).sum())}; in no acceleration group: '
          f'{int((accel_group == 0).sum())} (same direction with |estimate| = |truth|: '
          f'{int(((ts == es) & (ts != 0) & (np.abs(ea) == np.abs(at))).sum())})')
    alt = {}
    for sgn in (1, -1):
        sel = (ts == sgn) & (es == sgn)
        alt[sgn] = (int((sel & (np.abs(ea) > np.abs(told))).sum()), int((sel & (np.abs(ea) < np.abs(told))).sum()),
                    int((sel & (np.abs(ea) == np.abs(told))).sum()))
    print(f'  for reference only, the split by |told number| instead of |truth|: told speeding up {alt[1]}, told slowing '
          f'down {alt[-1]} (over, under, equal)')
    v2 = np.isin(accel_group, V2_GROUPS)
    small = [g for g, spec in ACCEL_GROUPS.items() if int((accel_group == g).sum()) < MIN_CI]
    n_small = int(np.isin(accel_group, small).sum())
    check(sorted(set(range(1, 12)) - set(small)) == V2_GROUPS and int(v2.sum()) == V2_FRAMES == n - n_small
          and len(set(seg_names[v2])) == V2_SEGMENTS,
          f'version 2: groups of fewer than {MIN_CI} frames {small} ({n_small} frames) left out: {int(v2.sum())} frames / '
          f'{len(set(seg_names[v2]))} segments = {n} - {n_small}; expected {V2_FRAMES} / {V2_SEGMENTS}')
    pd.DataFrame(rows_c).to_csv(os.path.join(O, 'exploratory_groups_ssm_counts.csv'), index=False)
    stop_if_failed('at the group counts')
    if a.pairs:
        told_dir = np.zeros(n, int)
        for g, spec in ACCEL_GROUPS.items():
            told_dir[accel_group == g] = spec[0]
        run_pairs(a, O, dict(ids=ids, ssm=ssm, v=v, at=at, base=base, spd=spd, acc=acc, speed_group=speed_group,
                             accel_group=accel_group, v2=v2, told_dir=told_dir, seg_frames=seg_frames, draws=draws),
                  check, stop_if_failed, failed, t0)

    # ---------------------------------------------------------------------------------------------- point estimates
    G.update(ids=ids, ssm=ssm, cue={'true_ego_speed': v, 'true_ego_accel': at}, score={'base': base, 'speed': spd,
             'accel': acc}, d=d, speed_group=speed_group, accel_group=accel_group, v2=v2, seg_frames=seg_frames,
             draws=draws)
    point = stats(ids, point=True)
    dmax = max(abs(point[k] - point[(k[0], k[1], k[2][1])]) for k in point
               if len(k) == 3 and isinstance(k[2], tuple) and not np.isnan(point[k]))
    nan_mis = sum(np.isnan(point[k]) != np.isnan(point[(k[0], k[1], k[2][1])]) for k in point
                  if len(k) == 3 and isinstance(k[2], tuple))
    check(dmax == 0 and nan_mis == 0, f'tau_b_all = tau_b_conflict in every group, run and SSM (largest |difference| '
          f'{dmax!r})')

    # references of tab:told (check 2)
    refs = []                                    # (key, kind, path, line, (value, lo, hi, nboot) as text)
    for folder, model, run, first, step in ((RISK, BASE, 'base', 866, 3), (SP, SPEED, 'speed', 74, 4),
                                            (AC, ACCEL, 'accel', 74, 4)):
        p_ = os.path.join(folder, 'agreement.csv')
        got = {dd['ssm']: (ln, dd) for ln, dd in csv_rows(p_)
               if dd['model'] == model and dd['condition'] == COND and dd['grouping'] == 'all'}
        lines = [got[s][0] for s in av.SSMS]
        check(lines == list(range(first, first + 9 * step, step)), f'{p_}: {model} {COND} all at lines {lines}')
        for s in av.SSMS:
            ln, dd = got[s]
            refs.append((('all_1084', run, s), 'run', p_, ln,
                         (dd['tau_b_conflict'], dd['tau_b_conflict_lo'], dd['tau_b_conflict_hi'],
                          dd['tau_b_conflict_nboot'])))
    for folder, run, pre in ((SP, 'speed', 'exploratory_speed_'), (AC, 'accel', 'exploratory_accel_')):
        p_ = os.path.join(folder, pre + 'agreement_differences.csv')
        for ln, dd in csv_rows(p_):
            if dd['condition'] == COND and dd['grouping'] == 'all' and dd['stat'] == 'tau_b_all':
                refs.append((('all_1084', run, dd['ssm']), 'change', p_, ln,
                             (dd['diff'], dd['diff_lo'], dd['diff_hi'], dd['nboot'])))
    for p_, models_ in ((os.path.join(SP, 'exploratory_speed_speed_distance_tau_b.csv'), {BASE: 'base', SPEED: 'speed'}),
                        (os.path.join(AC, 'exploratory_accel_speed_distance_tau_b.csv'), {BASE: 'base', ACCEL: 'accel'}),
                        (SPA, {BASE: 'base', SPEED: 'speed'})):
        for ln, dd in csv_rows(p_):
            if dd['condition'] != COND or dd['grouping'] != 'all' or dd['variable'] not in CUES:
                continue
            vals = (dd['tau_b'], dd['tau_b_lo'], dd['tau_b_hi'], dd['nboot'])
            if dd['model'] in models_:
                refs.append((('all_1084', models_[dd['model']], dd['variable']), 'run', p_, ln, vals))
            else:
                new = [m_ for m_ in models_ if m_ != BASE][0]
                if dd['model'] == f'{new} - {BASE}':
                    refs.append((('all_1084', models_[new], dd['variable']), 'change', p_, ln, vals))
    covered = {(k, kind) for k, kind, *_ in refs}
    need = {(('all_1084', run, row), 'run') for run in ('base', 'speed', 'accel') for row in ROWS} | \
           {(('all_1084', run, row), 'change') for run in ('speed', 'accel') for row in ROWS}
    check(need <= covered, f'tab:told sources: {len(refs)} reference rows cover the 3 runs x 11 rows and the 2 changes x '
          f'11 rows (missing: {sorted(need - covered)})')

    def mine_pt(key, kind):
        if kind == 'run':
            return point[key]
        return point[key] - point[(key[0], 'base', key[2])]

    n_pt = sum(txt(mine_pt(k, kind)) == ref[0] for k, kind, _, _, ref in refs)
    check(n_pt == len(refs), f'check 2, point estimates on all 1,084 frames: {n_pt} of {len(refs)} the same text')
    for k, kind, p_, ln, ref in refs:
        if txt(mine_pt(k, kind)) != ref[0]:
            print(f'    differs: {k} {kind}: {txt(mine_pt(k, kind))} vs {os.path.basename(p_)} line {ln} {ref[0]}')

    # mean score change (check 3)
    est = {(dd['told'], dd['thought']): (ln, dd) for ln, dd in csv_rows(EST_A)}
    for g, key in EST_ROW.items():
        ln, dd = est[key]
        mine = (int((accel_group == g).sum()), len(set(seg_names[accel_group == g])), txt(point[('mean', ('accel', g), 'accel')]))
        check(mine == (int(float(dd['frames'])), int(float(dd['segments'])), dd['mean_change']),
              f'check 3: group {g} mean change {mine[2]} ({mine[0]} frames, {mine[1]} segments) = est_accel_groups.csv '
              f'line {ln} {dd["mean_change"]} ({dd["frames"]}, {dd["segments"]})')
    for gs, (key, want_ln) in MERGED.items():
        ln, dd = est[key]
        sel = np.isin(accel_group, gs)
        mine = (int(sel.sum()), len(set(seg_names[sel])), txt(point[('mean_merged', gs)]))
        check(ln == want_ln and mine == (int(float(dd['frames'])), int(float(dd['segments'])), dd['mean_change']),
              f'check 3: groups {gs[0]} + {gs[1]} together: mean change {mine[2]} ({mine[0]} frames, {mine[1]} segments) '
              f'= est_accel_groups.csv line {ln} {dd["mean_change"]} ({dd["frames"]}, {dd["segments"]})')
    ests = {dd['group']: (ln, dd) for ln, dd in csv_rows(EST_S)}
    for g, key in ((1, 'estimate < truth (underestimated)'), (2, 'estimate > truth (overestimated)')):
        ln, dd = ests[key]
        check(txt(point[('mean', ('speed', g), 'speed')]) == dd['mean_change'],
              f'(added by Claude) speed group {g} mean change {txt(point[("mean", ("speed", g), "speed")])} = '
              f'est_speed_groups.csv line {ln} {dd["mean_change"]}')
    stop_if_failed('before the resamples (point estimates)')

    # ---------------------------------------------------------------------------------------------- resamples
    keys = list(point)
    G['keys'] = keys
    col = {k: i for i, k in enumerate(keys)}
    print(f'\n2. resamples: {N_BOOT}, {a.workers} worker(s)', flush=True)
    tb = time.time()
    out = []
    if a.workers > 1:
        with mp.get_context('fork').Pool(a.workers) as pool:
            for i, row in enumerate(pool.imap(one, range(N_BOOT), chunksize=8)):
                out.append(row)
                if (i + 1) % 250 == 0:
                    print(f'  [resample {i + 1}/{N_BOOT}, {(time.time() - tb) / (i + 1) * (N_BOOT - i - 1):.0f} s left]',
                          flush=True)
    else:
        for b in range(N_BOOT):
            out.append(one(b))
            if (b + 1) % 250 == 0:
                print(f'  [resample {b + 1}/{N_BOOT}, {(time.time() - tb) / (b + 1) * (N_BOOT - b - 1):.0f} s left]',
                      flush=True)
    boot = np.stack(out)
    print(f'  done in {time.time() - tb:.0f} s')

    def B(k):
        return boot[:, col[k]]

    # ---------------------------------------------------------------------------------------------- tables
    tables = [('all_1084', 'all 1,084 frames (= tab:told)', 'check', '', 'all 1,084 frames', run,
               np.ones(n, bool), True) for run in ('speed', 'accel')]
    tables += [(('speed', g), 'speed group', '1,2', g, spec[0], 'speed', speed_group == g, True)
               for g, spec in SPEED_GROUPS.items()]
    tables += [(('accel', g), 'acceleration group', '1,2' if spec[4] >= MIN_CI else '1', g, spec[3], 'accel',
                accel_group == g, spec[4] >= MIN_CI) for g, spec in ACCEL_GROUPS.items()]
    tables += [('all_1055', 'version 2: all 1,055 frames', '2', '', 'the 6 groups of 10 frames or more', 'accel', v2,
                True)]
    rows_t, rows_m = [], []
    for tab, tname, vers, gno, glab, run, mask, with_ci in tables:
        nf, ns = int(mask.sum()), len(set(seg_names[mask]))
        note = '' if with_ci else ('no frames' if nf == 0 else f'fewer than {MIN_CI} frames: point estimate only')
        for row in ROWS:
            kb, kt = (tab, 'base', row), (tab, run, row)
            pb, pt = point[kb], point[kt]
            if with_ci:
                lb, hb, nb = ar.ci(B(kb))
                lt, ht, nt = ar.ci(B(kt))
                lc, hc, nc = ar.ci(B(kt) - B(kb))
            else:
                lb = hb = lt = ht = lc = hc = np.nan
                nb = nt = nc = np.nan
            rows_t.append(dict(table=tname, versions=vers, group_no=gno, group=glab, frames=nf, segments=ns,
                               told_run=RUN_NAME[run], row=SHOW.get(row, row),
                               images_only=pb, images_only_lo=lb, images_only_hi=hb, images_only_nboot=nb,
                               images_only_side=side(lb, hb), told=pt, told_lo=lt, told_hi=ht, told_nboot=nt,
                               told_side=side(lt, ht), change=pt - pb, change_lo=lc, change_hi=hc, change_nboot=nc,
                               change_side=side(lc, hc), note=note))
        km = ('mean', tab, run)
        lo, hi, nv = ar.ci(B(km)) if with_ci else (np.nan, np.nan, np.nan)
        nfb = B(('n', tab))
        rows_m.append(dict(table=tname, versions=vers, group_no=gno, group=glab, frames=nf, segments=ns,
                           told_run=RUN_NAME[run], mean_change=point[km], lo=lo, hi=hi, nboot=nv, side=side(lo, hi),
                           resamples_0_frames=int((nfb == 0).sum()), resamples_1_2_frames=int(((nfb >= 1) & (nfb <= 2)).sum()),
                           true_speed_median=float(np.median(v[mask])) if nf else np.nan, note=note))
    rows_d = []
    for name, t1, t2, run in DIFFS:
        for row in ROWS + ['mean score change']:
            if row == 'mean score change':
                c1, c2 = point[('mean', t1, run)], point[('mean', t2, run)]
                a1, a2 = B(('mean', t1, run)), B(('mean', t2, run))
            else:
                c1 = point[(t1, run, row)] - point[(t1, 'base', row)]
                c2 = point[(t2, run, row)] - point[(t2, 'base', row)]
                a1, a2 = B((t1, run, row)) - B((t1, 'base', row)), B((t2, run, row)) - B((t2, 'base', row))
            lo, hi, nv = ar.ci(a1 - a2)
            rows_d.append(dict(comparison=name, row=SHOW.get(row, row), change_1=c1, change_2=c2, diff=c1 - c2, lo=lo,
                               hi=hi, nboot=nv, side=side(lo, hi)))

    # ---------------------------------------------------------------------------------------------- checks with intervals
    print('\n3. checks with the intervals')
    rows_k = []
    for k, kind, p_, ln, ref in refs:
        if kind == 'run':
            lo, hi, nv = ar.ci(B(k))
            val = point[k]
        else:
            kb = (k[0], 'base', k[2])
            lo, hi, nv = ar.ci(B(k) - B(kb))
            val = point[k] - point[kb]
        mine = (txt(val), txt(lo), txt(hi), str(nv))
        rows_k.append(dict(what=f'all 1,084: {RUN_NAME[k[1]]} {SHOW.get(k[2], k[2])} ({kind})', mine_value=mine[0],
                           mine_lo=mine[1], mine_hi=mine[2], mine_nboot=mine[3], ref_value=ref[0], ref_lo=ref[1],
                           ref_hi=ref[2], ref_nboot=ref[3], source=p_, line=ln, same_text=mine == tuple(ref)))
    n_same = sum(x['same_text'] for x in rows_k)
    check(n_same == len(rows_k), f'check 2: tau-b, interval and resamples used on all 1,084 frames (3 runs x 11 rows and '
          f'2 changes x 11 rows; the true speed of images only from 2 files, its true acceleration from 2): {n_same} of '
          f'{len(rows_k)} the same text')
    for x in rows_k:
        if not x['same_text']:
            print(f"    differs: {x['what']}: {x['mine_value']} [{x['mine_lo']}, {x['mine_hi']}] {x['mine_nboot']} vs "
                  f"{os.path.basename(x['source'])} line {x['line']}: {x['ref_value']} [{x['ref_lo']}, {x['ref_hi']}] "
                  f"{x['ref_nboot']}")
    mrow = {(x['table'], x['group_no']): x for x in rows_m}
    for g, key in EST_ROW.items():
        ln, dd = est[key]
        x = mrow[('acceleration group', g)]
        nv = '' if np.isnan(x['nboot']) else str(float(x['nboot']))
        mine = (txt(x['mean_change']), txt(x['lo']), txt(x['hi']), nv)
        rows_k.append(dict(what=f'mean change, acceleration group {g}', mine_value=mine[0], mine_lo=mine[1], mine_hi=mine[2],
                           mine_nboot=mine[3], ref_value=dd['mean_change'], ref_lo=dd['lo'], ref_hi=dd['hi'],
                           ref_nboot=dd['nboot'], source=EST_A, line=ln,
                           same_text=mine == (dd['mean_change'], dd['lo'], dd['hi'], dd['nboot'])))
        check(rows_k[-1]['same_text'], f'check 3: group {g} mean change {mine[0]} [{mine[1]}, {mine[2]}] nboot {mine[3]} = '
              f'est_accel_groups.csv line {ln} {dd["mean_change"]} [{dd["lo"]}, {dd["hi"]}] nboot {dd["nboot"]}')
    for g, key in ((1, 'estimate < truth (underestimated)'), (2, 'estimate > truth (overestimated)')):
        ln, dd = ests[key]
        x = mrow[('speed group', g)]
        mine = (txt(x['mean_change']), txt(x['lo']), txt(x['hi']), str(x['nboot']))
        same = mine == (dd['mean_change'], dd['lo'], dd['hi'], dd['nboot'])
        rows_k.append(dict(what=f'(added by Claude) mean change, speed group {g}', mine_value=mine[0], mine_lo=mine[1],
                           mine_hi=mine[2], mine_nboot=mine[3], ref_value=dd['mean_change'], ref_lo=dd['lo'],
                           ref_hi=dd['hi'], ref_nboot=dd['nboot'], source=EST_S, line=ln, same_text=same))
        check(same, f'(added by Claude) speed group {g} mean change {mine} = est_speed_groups.csv line {ln}')
    for name, key, path_, label in ((DIFFS[0][0], 'under - over', EST_S, 'group'),
                                    (DIFFS[1][0], 'told speeding up: thought slowing down - thought steady', EST_A, 'told')):
        ln, dd = [(ln_, dd_) for ln_, dd_ in csv_rows(path_) if dd_[label] == key][0]
        x = [y for y in rows_d if y['comparison'] == name and y['row'] == 'mean score change'][0]
        nv = str(x['nboot']) if path_ == EST_S else str(float(x['nboot']))
        mine = (txt(x['diff']), txt(x['lo']), txt(x['hi']), nv)
        same = mine == (dd['mean_change'], dd['lo'], dd['hi'], dd['nboot'])
        rows_k.append(dict(what=f'(added by Claude) mean change, {name}', mine_value=mine[0], mine_lo=mine[1],
                           mine_hi=mine[2], mine_nboot=mine[3], ref_value=dd['mean_change'], ref_lo=dd['lo'],
                           ref_hi=dd['hi'], ref_nboot=dd['nboot'], source=path_, line=ln, same_text=same))
        check(same, f'(added by Claude) {name}, mean change {mine} = {os.path.basename(path_)} line {ln}')

    # ---------------------------------------------------------------------------------------------- write
    pd.DataFrame(rows_t).to_csv(os.path.join(O, 'exploratory_groups_ssm_tau_b.csv'), index=False)
    pd.DataFrame(rows_m).to_csv(os.path.join(O, 'exploratory_groups_ssm_mean_change.csv'), index=False)
    pd.DataFrame(rows_d).to_csv(os.path.join(O, 'exploratory_groups_ssm_group_differences.csv'), index=False)
    pd.DataFrame(rows_k).to_csv(os.path.join(O, 'exploratory_groups_ssm_check.csv'), index=False)
    ft = frames.copy()
    for s in av.SSMS:
        ft[s + '_risk'] = ssm[s + '_risk']
    ft['score_images_only'], ft['score_told_speed'], ft['score_told_accel'] = base, spd, acc
    ft['true_ego_speed_mps'], ft['true_ego_accel_mps2'] = v, at
    ft['speed_group'], ft['accel_group'] = speed_group, accel_group
    ft.to_csv(os.path.join(O, 'groups_ssm_frames.csv'), index=False)
    with open(os.path.join(O, 'groups_ssm_frames_columns.txt'), 'w', encoding='utf-8') as f:
        f.write('groups_ssm_frames.csv (paper_sec45_groups_ssm.py, user 2026-11-13): one row per frame of the main '
                'result (1,084), sorted by segment and frame_idx; numbers only.\n')
        f.write('segment, frame_idx: the frame.\n')
        for s in av.SSMS:
            how = f'-{s} (a smaller {s} is more dangerous)' if s in av.SMALLER_RISKIER else \
                f'{s} itself (a larger {s} is more dangerous)'
            f.write(f'{s}_risk: the value the tau-b uses, larger = more dangerous: {how}'
                    + (' (2D-TTC)' if s == 'TTC2D' else '') + '; frame_ssm_validation_training.csv via '
                    'analyze_vlm_ssm.ssm_table, the same as ssm_target_frames.csv of the main analysis.\n')
        f.write(f'score_images_only, score_told_speed, score_told_accel: score_mean (mean of the 5 answers), {COND}, of '
                f'{BASE} (after_qwen3.5-4b), {SPEED} (speed_risk_27b), {ACCEL} (accel_risk_27b); larger = riskier.\n')
        f.write('true_ego_speed_mps, true_ego_accel_mps2: quantities/truth_main.csv v_ego, a_ego (positive = speeding '
                'up).\n')
        f.write('speed_group: 1 = estimate < true speed (underestimated), 2 = estimate > true speed (overestimated); the '
                'estimate is sec44_frames.csv ego_speed_estimate_mps.\n')
        f.write('accel_group: 1-11 as numbered by the user (estimate sec44_frames.csv ego_accel_estimate_mps2; told = '
                'the truth to 1 decimal):\n')
        for g, spec in ACCEL_GROUPS.items():
            f.write(f'  {g}: {spec[3]}\n')
        f.write(f'  0: in none ({int((accel_group == 0).sum())} frames)\n')

    # ---------------------------------------------------------------------------------------------- report
    def star(lo, hi):
        return '*' if side(lo, hi) in ('above 0', 'below 0') else ' '

    print('\n' + '=' * 110 + '\nfiles: exploratory_groups_ssm_*.csv (line = row + 2 - 1, header = line 1)')
    print('\ncounts (exploratory_groups_ssm_counts.csv)')
    for i, x in enumerate(rows_c):
        print(f"  line {i + 2}: {x['grouping']} {x['group_no']}: {x['group']}: {x['frames']} frames / {x['segments']} "
              f"segments (versions {x['versions']})")
    print('\ntau-b (exploratory_groups_ssm_tau_b.csv); * = interval leaves out 0')
    last = None
    for i, x in enumerate(rows_t):
        head = (x['table'], x['group_no'], x['told_run'])
        if head != last:
            print(f"\n  {x['table']} {x['group_no']} {x['group']} ({x['frames']} frames, {x['segments']} segments; "
                  f"images only vs {x['told_run']}; versions {x['versions']}){'; ' + x['note'] if x['note'] else ''}")
            last = head
        print(f"    line {i + 2}: {x['row']:17s} images only {star(x['images_only_lo'], x['images_only_hi'])}"
              f"{r(x['images_only'])} [{r(x['images_only_lo'])}, {r(x['images_only_hi'])}] (n {x['images_only_nboot']}); "
              f"told {star(x['told_lo'], x['told_hi'])}{r(x['told'])} [{r(x['told_lo'])}, {r(x['told_hi'])}] "
              f"(n {x['told_nboot']}); change {star(x['change_lo'], x['change_hi'])}{r(x['change'])} "
              f"[{r(x['change_lo'])}, {r(x['change_hi'])}] (n {x['change_nboot']})")
    print('\nmean score change, told - images only (exploratory_groups_ssm_mean_change.csv)')
    for i, x in enumerate(rows_m):
        print(f"  line {i + 2}: {x['table']} {x['group_no']} {x['group']} ({x['told_run']}): "
              f"{star(x['lo'], x['hi'])}{r(x['mean_change'])} [{r(x['lo'])}, {r(x['hi'])}] (n {x['nboot']}; resamples "
              f"with 0 frames {x['resamples_0_frames']}, 1-2 frames {x['resamples_1_2_frames']}); median true speed "
              f"{r(x['true_speed_median'])}{'; ' + x['note'] if x['note'] else ''}")
    print('\ndifferences between groups of the change (exploratory_groups_ssm_group_differences.csv)')
    for i, x in enumerate(rows_d):
        print(f"  line {i + 2}: {x['comparison']}: {x['row']:17s} {star(x['lo'], x['hi'])}{r(x['diff'])} "
              f"[{r(x['lo'])}, {r(x['hi'])}] (n {x['nboot']}; changes {r(x['change_1'])} and {r(x['change_2'])})")
    print('\nchanges whose interval leaves out 0 (tau-b file)')
    for i, x in enumerate(rows_t):
        if x['change_side'] in ('above 0', 'below 0'):
            print(f"  line {i + 2}: {x['table']} {x['group_no']} {x['group']} ({x['told_run']}): {x['row']} "
                  f"{r(x['change'])} [{r(x['change_lo'])}, {r(x['change_hi'])}] ({x['change_side']})")
    print(f'\nwritten in {O}: exploratory_groups_ssm_counts.csv, _tau_b.csv, _mean_change.csv, _group_differences.csv, '
          f'_check.csv, groups_ssm_frames.csv, groups_ssm_frames_columns.txt')
    print(f'done in {time.time() - t0:.0f} s; checks not met: {len(failed)}')
    for w_ in failed:
        print('  ', w_)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
