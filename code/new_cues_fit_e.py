"""EXPLORATORY; round E's cues were chosen after the residuals of D's additive model on these frames were seen (user,
2026-11-06; CLAUDE.md "我定（2026-11-06…）"), so E - D on these frames may be optimistic. How much more of Qwen3.6-27B's
risk-score ranks round E's cues (code/new_cues_e.py) reproduce than D (code/new_cues_fit.py), with D's models, frames
and conventions.

Frames: the risk analysis subfolder's frames with every cue of D and of E known (checked: the same as D's fit frames).
Cue sets. E is D with the pedestrian + cyclist counts split into on / off the road and the stationary-vehicle counts into
in a lane / elsewhere (the split counts in place of the totals, user 2026-11-06), plus NEW_E:
 Linear (ranks / n, ties the mean rank; S dummies as they are; analyze_rules.ols, loso_predict), R = 10, 20, 30, 50
 (20 the main reading):
   D  K + S + ped_cyc_R, moving_R, stationary_R + NEW9                                            (= new_cues_fit.py's D)
   E  K + S + ped_cyc_road_R, ped_cyc_offroad_R, moving_R, stationary_inlane_R, stationary_other_R + NEW9 + NEW_E
 Trees and boosting (the cues in their own units): D = new_cues_fit.py's D; E = A, S, the eight pedestrian + cyclist
   counts (road / off road x 4 radii), the four moving-vehicle counts, the eight stationary counts (in a lane / elsewhere
   x 4), NEW9, NEW_E.
NEW_E: n_side_vehicle, n_side_vehicle_toward (non-zero on few frames: it says little), dist_merge,
n_ped_cyc_in_crosswalk_50m, max_vehicle_length_30m. Not used (user, 2026-11-06): the heading change over the next 3 s,
large-vehicle counts with a threshold, traffic lights.
Models, out of sample and intervals as new_cues_fit.py (its nonlin_model and leave-one-segment-out workers): linear,
the depth-3 tree, conservative boosting in full and its additive version; leave one segment out, intervals from the
2,000 resamples' segment sums, nothing refitted. Differences from the same resamples: E - D per model; on E boosting
- tree, boosting - linear R20, full - additive. Proxy test (user: as for D, E against D): the ego-speed coefficient in
D and E (linear, every R, refitted in every resample, ranks anew) and E - D; the resampled depth-3 trees' first cut
(D and E). Single-cue tau-b of the split counts and NEW_E with the risk score (as new_cues_fit.py, groupings all and
segment). E's additive boosting on all the fit frames: partial dependence at the cue's 10, 30, 50, 70, 90 % quantiles;
for a cue taking only the values 0 and 1 on these frames at 0 and 1 only (user, 2026-11-06); for max_vehicle_length_30m
also at 5, 6, 7, 9 and 12 m. The ceiling on the fit frames (score_ceiling.py's method) for the shares.
Check (real data, --d_fit): every D result of the out-of-sample table and every D prediction equals new_cues_fit.py's
run on these frames within 1e-12.
Outputs (out_dir): exploratory_e_*.csv, exploratory_e_params.txt; exit code 1 when a check is not met.
usage: python -I new_cues_fit_e.py <risk subfolder> <quantities subfolder> <frame_ssm.csv> <truth.csv> <scene_labels.csv>
       <nearby_counts.csv> <D's new cues csv (new_cues_min30.csv)> <new_cues_e.csv (with max_vehicle_length_30m)>
       <rule analysis subfolder> <out_dir> [--d_fit <new_cues_fit.py out_dir run with --predictions>] [--workers 8]
       [--no_ceiling] [--no_linear_check]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import hashlib  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_rules as ar  # noqa: E402
import analyze_vlm_ssm as av  # noqa: E402
import new_cues_fit as ncf  # noqa: E402
import rule_tree as rt  # noqa: E402
from score_ceiling import K_REP, pair_corr, spearman_brown  # noqa: E402

MODEL = rt.MODEL
RADII = ar.RADII
NEW9, SPLIT, NONLIN, PD_Q = ncf.NEW9, ncf.SPLIT, ncf.NONLIN, ncf.PD_Q
NEW_E = ['n_side_vehicle', 'n_side_vehicle_toward', 'dist_merge', 'n_ped_cyc_in_crosswalk_50m', 'max_vehicle_length_30m']
PED_SPLIT = [f'n_ped_cyc_{w}_{r}m' for r in RADII for w in ('road', 'offroad')]
MOVING_C = [f'n_vehicle_moving_{r}m' for r in RADII]
STAT_SPLIT = [f'n_vehicle_stationary_{w}_{r}m' for r in RADII for w in ('inlane', 'other')]
E_CUES = PED_SPLIT + STAT_SPLIT + NEW_E
SETS = ['D', 'E']
LONG_CUE, LONG_POINTS = 'max_vehicle_length_30m', [5.0, 6.0, 7.0, 9.0, 12.0]
TOL = 1e-12
NOTE = ('E\'s cues were chosen after the residuals of D\'s additive model on these frames were seen (user, 2026-11-06): '
        'E - D on these frames may be optimistic')


def pd_points(values, name):
    """The points of the partial dependence of one cue: [(kind, quantile, value)]."""
    u = np.unique(values)
    if set(u.tolist()) <= {0.0, 1.0}:
        return [('value', np.nan, 0.0), ('value', np.nan, 1.0)]
    pts = [('quantile', q, float(v)) for q, v in zip(PD_Q, np.quantile(values, PD_Q))]
    if name == LONG_CUE:
        pts += [('value', np.nan, v) for v in LONG_POINTS]
    return pts


def main():
    ap = argparse.ArgumentParser()
    for name in ('risk', 'quantities', 'frame_ssm', 'truth', 'scene_labels', 'nearby', 'd_cues', 'e_cues', 'rules',
                 'out_dir'):
        ap.add_argument(name)
    ap.add_argument('--d_fit', default=None)
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261021)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--no_ceiling', action='store_true', help='fake scores: no ceiling (their repeats are not made up)')
    ap.add_argument('--no_linear_check', action='store_true', help='fake scores: the rule analysis is not of them')
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out_dir, 'exploratory_e_params.txt')):
        sys.exit(f'{a.out_dir} has exploratory_e_params.txt already: give a new folder')
    t0 = time.time()
    failed, checks = [], []

    # --- frames, risk scores, cues: as new_cues_fit.py
    st = pd.read_csv(os.path.join(a.risk, 'ssm_target_frames.csv'))
    frames = st[['segment', 'frame_idx']].sort_values(['segment', 'frame_idx']).reset_index(drop=True)
    n = len(frames)
    mi = pd.MultiIndex.from_frame(frames)
    seg_names = frames['segment'].to_numpy()
    seg_list = np.array(sorted(set(seg_names)))
    nseg = len(seg_list)
    seg_code = np.searchsorted(seg_list, seg_names)
    fsc = pd.read_csv(os.path.join(a.risk, 'vlm_frame_scores.csv'), float_precision='round_trip')
    conds = av.CONDITIONS
    score = {}
    for c in conds:
        s = fsc[(fsc.model == MODEL) & (fsc.condition == c)].set_index(['segment', 'frame_idx'])['score_mean']
        score[c] = s.reindex(mi).to_numpy(float)
        if np.isnan(score[c]).any():
            sys.exit(f'{c}: frames without a risk score')
    tr = frames.merge(pd.read_csv(a.truth, float_precision='round_trip'), on=['segment', 'frame_idx'], how='left',
                      validate='one_to_one')
    cue = {'true_distance': tr['distance'].to_numpy(float), 'true_ego_speed': tr['v_ego'].to_numpy(float),
           'true_lead_speed': tr['v_lead'].to_numpy(float),
           'true_closing_speed': tr['v_ego'].to_numpy(float) - tr['v_lead'].to_numpy(float),
           'true_ego_accel': tr['a_ego'].to_numpy(float), 'true_lead_accel': tr['a_lead'].to_numpy(float)}
    nb = frames.merge(pd.read_csv(a.nearby), on=['segment', 'frame_idx'], how='left', validate='one_to_one')
    for r in RADII:
        cue[f'ped_cyc_{r}m'] = nb[f'n_ped_cyc_{r}m'].to_numpy(float)
        cue[f'vehicle_{r}m'] = nb[f'n_vehicle_{r}m'].to_numpy(float)
    nw = frames.merge(pd.read_csv(a.d_cues), on=['segment', 'frame_idx'], how='left', validate='one_to_one')
    for k in NEW9 + SPLIT:
        cue[k] = nw[k].to_numpy(float)
    ne_ = frames.merge(pd.read_csv(a.e_cues), on=['segment', 'frame_idx'], how='left', validate='one_to_one')
    for k in E_CUES:
        cue[k] = ne_[k].to_numpy(float)
    sl = pd.read_csv(a.scene_labels, dtype=str, keep_default_na=False).set_index('segment')
    s_terms, s_cols = [], []
    for col in ('time_of_day', 'location', 'weather'):
        v = sl.loc[seg_names, col].str.strip().to_numpy()
        seg_v = sl.loc[seg_list, col].str.strip()
        if col == 'weather':
            cats = ar.WEATHER
            ref = max(cats, key=lambda k: int((seg_v == k).sum()))
        else:
            cats = sorted(set(v))
            ref = max(cats, key=lambda k: int((v == k).sum()))
        for k in cats:
            if k != ref:
                s_terms.append(f'{col}={k}')
                s_cols.append((v == k).astype(float))
    dfit = ~np.isnan(np.column_stack([cue[k] for k in NEW9 + SPLIT])).any(1)
    fit = dfit & ~np.isnan(np.column_stack([cue[k] for k in E_CUES])).any(1)
    nf = int(fit.sum())
    checks.append(f'fit frames (every cue of D and E known) {nf} of {n}; the same as D\'s fit frames: '
                  f'{bool((fit == dfit).all())}')
    if not (fit == dfit).all():
        failed.append('fit frames')
    idf = np.flatnonzero(fit)
    u, g = np.unique(seg_code[fit], return_inverse=True)
    stat_sum = max(int(np.abs(cue[f'n_vehicle_stationary_inlane_{r}m'][fit] + cue[f'n_vehicle_stationary_other_{r}m'][fit]
                              - cue[f'n_vehicle_stationary_{r}m'][fit]).max()) for r in RADII)
    ped_sum = max(int(np.abs(cue[f'n_ped_cyc_road_{r}m'][fit] + cue[f'n_ped_cyc_offroad_{r}m'][fit]
                             - cue[f'ped_cyc_{r}m'][fit]).max()) for r in RADII)
    checks.append(f'split counts add up to D\'s totals on the fit frames: largest difference pedestrians + cyclists '
                  f'{ped_sum}, stationary vehicles {stat_sum}')
    if ped_sum or stat_sum:
        failed.append('split totals')

    # --- resamples (as analyze_rules.py)
    n_boot_r, seed_r, frames_r, segs_r = ar.read_params_risk(os.path.join(a.risk, 'params.txt'))
    if (n_boot_r, seed_r, frames_r) != (a.n_boot, a.seed, n) or list(seg_list) != segs_r:
        sys.exit('stopped before the resamples are used: resamples, seed, frames or segments not the risk analysis\'s')
    rng = np.random.default_rng(a.seed)
    draws = [rng.choice(seg_list, nseg) for _ in range(a.n_boot)]
    h = hashlib.sha256()
    for d in draws:
        h.update(np.searchsorted(seg_list, d).astype(np.int32).tobytes())
    nb_q, seed_q, hash_q = ar.read_hash_quantities(os.path.join(a.quantities, 'params.txt'))
    if (nb_q, seed_q, hash_q) != (a.n_boot, a.seed, h.hexdigest()):
        sys.exit('stopped before the resamples are used: the drawn segments are not the quantities analysis\'s')
    counts = np.stack([np.bincount(np.searchsorted(seg_list, d), minlength=nseg) for d in draws])
    os.makedirs(a.out_dir, exist_ok=True)

    # --- the linear fits on all the frames against the rule analysis (the inputs are its own)
    if not a.no_linear_check:
        oo = pd.read_csv(os.path.join(a.rules, 'exploratory_oos_r2.csv'), float_precision='round_trip')
        oo = oo[(oo.target == 'risk') & (oo.model == MODEL)].set_index(['condition', 'fit'])
        u_all, g_all = np.unique(seg_code, return_inverse=True)
        dev = 0.0
        for c in conds:
            y_ = ar.rank01(score[c])
            K_ = [ar.rank01(cue[k]) for k in ar.K_TRUE]
            for fname, cols in (('K', K_), ('K+S+N20', K_ + s_cols + [ar.rank01(cue['ped_cyc_20m']),
                                                                      ar.rank01(cue['vehicle_20m'])])):
                ref = oo.loc[(c, fname)]
                sse, sst = ncf.parts(y_, ar.loso_predict(y_, cols, g_all), g_all, u_all, nseg)
                p, lo, hi, nv, _ = ncf.r2_of(sse, sst, counts)
                dev = max(dev, abs(ar.ols(y_, cols)[2] - ref.r2_in), abs(p - ref.r2_oos), abs(lo - ref.r2_oos_lo),
                          abs(hi - ref.r2_oos_hi))
        checks.append(f'linear K and K+S+N20 on all {n} frames against exploratory_oos_r2.csv: largest difference '
                      f'{dev:.3g} (tolerance {TOL:g})')
        if not dev <= TOL:
            failed.append('linear check')
    print(f'inputs ready ({time.time() - t0:.0f} s); fit frames {nf}', flush=True)

    # --- the designs on the fit frames
    rk = {k: ar.rank01(v[fit]) for k, v in cue.items() if not np.isnan(v[fit]).any()}
    S = [c_[fit] for c_ in s_cols]
    K = [rk[k] for k in ar.K_TRUE]

    def lin_from(rk_, S_, s, r):
        K_ = [rk_[k] for k in ar.K_TRUE]
        if s == 'D':
            return K_ + S_ + [rk_[f'ped_cyc_{r}m'], rk_[f'n_vehicle_moving_{r}m'], rk_[f'n_vehicle_stationary_{r}m']] + \
                [rk_[k] for k in NEW9]
        return K_ + S_ + [rk_[f'n_ped_cyc_road_{r}m'], rk_[f'n_ped_cyc_offroad_{r}m'], rk_[f'n_vehicle_moving_{r}m'],
                          rk_[f'n_vehicle_stationary_inlane_{r}m'], rk_[f'n_vehicle_stationary_other_{r}m']] + \
            [rk_[k] for k in NEW9] + [rk_[k] for k in NEW_E]

    lin = {(s, r): lin_from(rk, S, s, r) for s in SETS for r in RADII}
    names = {'D': list(ar.TRUE_CUES) + s_terms + [f'ped_cyc_{r}m' for r in RADII] + SPLIT + NEW9,
             'E': list(ar.TRUE_CUES) + s_terms + PED_SPLIT + MOVING_C + STAT_SPLIT + NEW9 + NEW_E}
    full_s = {k: c_ for k, c_ in zip(s_terms, s_cols)}
    Xraw = {s: np.column_stack([full_s[k] if k in full_s else cue[k] for k in names[s]]) for s in SETS}
    X = {s: Xraw[s][fit] for s in SETS}
    y = {c: ar.rank01(score[c][fit]) for c in conds}

    # --- out of sample
    res, preds = {}, {}
    for c in conds:
        for (s, r), cols in lin.items():
            pred = ar.loso_predict(y[c], cols, g)
            sse, sst = ncf.parts(y[c], pred, g, u, nseg)
            res[(c, f'linear_R{r}', s)] = (sse, sst, ar.ols(y[c], cols)[2])
            preds[(c, f'linear_R{r}', s)] = pred
    print(f'linear done ({time.time() - t0:.0f} s)', flush=True)
    seg_frames_fit = {s_: np.flatnonzero(fit & (seg_names == s_)) for s_ in seg_list if (fit & (seg_names == s_)).any()}
    ncf._W.update(X=X, y=y, g=g, Xraw=Xraw, score=score, names=names, draws=draws, seg_frames_fit=seg_frames_fit)
    tasks = [(c, s, kind) for c in conds for s in SETS for kind in NONLIN]
    ft_tasks = [(c, s) for c in conds for s in SETS]
    if a.workers > 1:
        with mp.get_context('fork').Pool(a.workers) as pool:
            outs = pool.map(ncf._loso_task, tasks, chunksize=1)
            fts = pool.map(ncf._firstcut_task, ft_tasks, chunksize=1)
    else:
        outs = [ncf._loso_task(t) for t in tasks]
        fts = [ncf._firstcut_task(t) for t in ft_tasks]
    unpred = 0
    for (c, s, kind), pred, r2_in in outs:
        unpred += int(np.isnan(pred).sum())
        sse, sst = ncf.parts(y[c], pred, g, u, nseg)
        res[(c, kind, s)] = (sse, sst, r2_in)
        preds[(c, kind, s)] = pred
    checks.append(f'frames not predicted in the leave-one-segment-out runs: {unpred}')
    if unpred:
        failed.append('unpredicted')
    print(f'trees and boosting done ({time.time() - t0:.0f} s)', flush=True)
    rows, boot = [], {}
    for (c, m, s), (sse, sst, r2_in) in res.items():
        p, lo, hi, nv, rb = ncf.r2_of(sse, sst, counts)
        boot[(c, m, s)] = (p, rb)
        rows.append(dict(model=MODEL, condition=c, fit=m, cues=s, frames=nf, r2_in=r2_in, r2_oos=p, r2_oos_lo=lo,
                         r2_oos_hi=hi, nboot=nv))
    oos = pd.DataFrame(rows)
    oos.to_csv(os.path.join(a.out_dir, 'exploratory_e_oos.csv'), index=False)
    fits_ = [f'linear_R{r}' for r in RADII] + NONLIN
    prow = []
    for c in conds:
        tab = {'model': MODEL, 'condition': c, 'segment': seg_names[idf],
               'frame_idx': frames['frame_idx'].to_numpy()[idf], 'y_rank': y[c]}
        for m_ in fits_:
            for s in SETS:
                tab[f'pred_{m_}_{s}'] = preds[(c, m_, s)]
        prow.append(pd.DataFrame(tab))
    ptab = pd.concat(prow, ignore_index=True)
    ptab.to_csv(os.path.join(a.out_dir, 'exploratory_e_predictions.csv'), index=False)
    comps = [(f'E - D, {m}', (m, 'E'), (m, 'D')) for m in fits_] + \
        [(f'boost_full - {x}, E', ('boost_full', 'E'), (x, 'E')) for x in ('tree', 'linear_R20', 'boost_add')]
    drows = []
    for c in conds:
        for name, (m1, s1), (m2, s2) in comps:
            p1, b1 = boot[(c, m1, s1)]
            p2, b2 = boot[(c, m2, s2)]
            lo, hi, nv = ar.ci(b1 - b2)
            drows.append(dict(model=MODEL, condition=c, comparison=name, diff=p1 - p2, lo=lo, hi=hi, nboot=nv,
                              above_0=bool(not np.isnan(lo) and lo > 0), below_0=bool(not np.isnan(hi) and hi < 0)))
    pd.DataFrame(drows).to_csv(os.path.join(a.out_dir, 'exploratory_e_differences.csv'), index=False)

    # --- D against new_cues_fit.py's run
    if a.d_fit:
        do = pd.read_csv(os.path.join(a.d_fit, 'exploratory_d_oos.csv'), float_precision='round_trip')
        do = do[do.cues == 'D'].set_index(['condition', 'fit'])
        mine = oos[oos.cues == 'D'].set_index(['condition', 'fit'])
        dev = max(float(np.abs(mine.loc[do.index, col].to_numpy(float) - do[col].to_numpy(float)).max())
                  for col in ('r2_in', 'r2_oos', 'r2_oos_lo', 'r2_oos_hi'))
        dp = pd.read_csv(os.path.join(a.d_fit, 'exploratory_d_predictions.csv'), float_precision='round_trip')
        same_rows = dp[['condition', 'segment', 'frame_idx']].equals(ptab[['condition', 'segment', 'frame_idx']])
        devp = max(float(np.abs(dp[col].to_numpy(float) - ptab[col].to_numpy(float)).max())
                   for col in ['y_rank'] + [f'pred_{m_}_D' for m_ in fits_]) if same_rows else np.inf
        checks.append(f'D against new_cues_fit.py\'s run ({a.d_fit}): out-of-sample table rows {len(do)}, largest '
                      f'difference {dev:.3g}; predictions on the same frames {same_rows}, largest difference '
                      f'{devp:.3g} (tolerance {TOL:g})')
        if not (len(do) == len(fits_) * len(conds) and dev <= TOL and devp <= TOL):
            failed.append('D against new_cues_fit.py')

    # --- first cuts of the resampled trees
    frows = []
    for (c, s), cuts_ in fts:
        vc = pd.Series(cuts_).value_counts()
        for k in sorted(vc.index, key=lambda k: (-vc[k], k)):
            frows.append(dict(model=MODEL, condition=c, cues=s, first_cue=k, resamples=int(vc[k]),
                              share=vc[k] / len(cuts_)))
    pd.DataFrame(frows).to_csv(os.path.join(a.out_dir, 'exploratory_e_first_cuts.csv'), index=False)
    print(f'resampled trees done ({time.time() - t0:.0f} s)', flush=True)

    # --- ego-speed coefficient (linear, every R) in D and E, refitted per resample
    ego = 1 + ar.K_TRUE.index('true_ego_speed')

    def cols_at(idx, s, r):
        rk_ = {k: ar.rank01(cue[k][idx]) for k in ar.K_TRUE + [f'ped_cyc_{r}m', f'n_vehicle_moving_{r}m',
                                                               f'n_vehicle_stationary_{r}m', f'n_ped_cyc_road_{r}m',
                                                               f'n_ped_cyc_offroad_{r}m',
                                                               f'n_vehicle_stationary_inlane_{r}m',
                                                               f'n_vehicle_stationary_other_{r}m'] + NEW9 + NEW_E}
        return lin_from(rk_, [c_[idx] for c_ in s_cols], s, r)

    eb = np.full((a.n_boot, len(conds), len(RADII), 2), np.nan)
    ept = np.full((len(conds), len(RADII), 2), np.nan)
    for ci_, c in enumerate(conds):
        for ri, r in enumerate(RADII):
            for si, s in enumerate(SETS):
                ept[ci_, ri, si] = ar.ols(ar.rank01(score[c][idf]), cols_at(idf, s, r))[0][ego]
    for b, d in enumerate(draws):
        idx = np.concatenate([seg_frames_fit[x] for x in d if x in seg_frames_fit])
        ybs = [ar.rank01(score[c][idx]) for c in conds]
        for ri, r in enumerate(RADII):
            for si, s in enumerate(SETS):
                cols = cols_at(idx, s, r)                 # the cue ranks do not depend on the condition
                for ci_ in range(len(conds)):
                    eb[b, ci_, ri, si] = ar.ols(ybs[ci_], cols)[0][ego]
    erows = []
    for ci_, c in enumerate(conds):
        for ri, r in enumerate(RADII):
            row = dict(model=MODEL, condition=c, R=r)
            for si, s in enumerate(SETS):
                lo, hi, nv = ar.ci(eb[:, ci_, ri, si])
                row.update({f'ego_{s}': ept[ci_, ri, si], f'ego_{s}_lo': lo, f'ego_{s}_hi': hi})
            lo, hi, nv = ar.ci(eb[:, ci_, ri, 1] - eb[:, ci_, ri, 0])
            row.update(diff_E_minus_D=ept[ci_, ri, 1] - ept[ci_, ri, 0], diff_lo=lo, diff_hi=hi, nboot=nv)
            erows.append(row)
    pd.DataFrame(erows).to_csv(os.path.join(a.out_dir, 'exploratory_e_ego_coefficient.csv'), index=False)
    print(f'ego-speed coefficients done ({time.time() - t0:.0f} s)', flush=True)

    # --- single-cue tau-b of E's quantities (as new_cues_fit.py)
    ids = np.arange(n)
    seg_frames_all = {s_: np.flatnonzero(seg_names == s_) for s_ in seg_list}

    def tau_stats(idx, gb_seg):
        gb = {'all': np.zeros(len(idx), int), 'segment': gb_seg}
        cache, out = {}, {}
        for c in conds:
            x = score[c][idx]
            for k in E_CUES:
                v = cue[k][idx]
                mask = ~np.isnan(v)
                key = mask.tobytes()
                if key not in cache:
                    labs = {gn: lv[mask] for gn, lv in gb.items()}
                    cache[key] = labs, {gn: av.within_pairs(lv) for gn, lv in labs.items()
                                        if gn != 'all' and len(np.unique(lv)) > 20}
                labs, pairs = cache[key]
                cs = np.ones(int(mask.sum()), bool)
                for gn in ar.GROUPINGS:
                    out[(c, k, gn)] = av.stats_by_grouping(x[mask], cs, v[mask], ids[idx][mask], labs[gn],
                                                           pairs.get(gn))[3]
        return out

    tpt = tau_stats(ids, seg_names)
    keys = list(tpt)

    def one(b):
        d = draws[b]
        idx = np.concatenate([seg_frames_all[s_] for s_ in d])
        gseg = np.concatenate([np.full(len(seg_frames_all[s_]), k) for k, s_ in enumerate(d)])
        r_ = tau_stats(idx, gseg)
        return np.array([r_[k] for k in keys])

    ncf._W['tau'] = one
    if a.workers > 1:
        chunks = [list(c_) for c_ in np.array_split(np.arange(a.n_boot), a.workers * 4) if len(c_)]
        with mp.get_context('fork').Pool(a.workers) as pool:
            tb = np.array([v for part in pool.map(ncf._tau_chunk, chunks) for v in part])
    else:
        tb = np.array([one(b) for b in range(a.n_boot)])
    trows = []
    for j, (c, k, gn) in enumerate(keys):
        lo, hi, nv = ar.ci(tb[:, j])
        trows.append(dict(model=MODEL, condition=c, cue=k, grouping=gn, frames=int((~np.isnan(cue[k])).sum()),
                          tau_b=tpt[(c, k, gn)], tau_b_lo=lo, tau_b_hi=hi, nboot=nv))
    pd.DataFrame(trows).to_csv(os.path.join(a.out_dir, 'exploratory_e_tau_b.csv'), index=False)
    print(f'tau-b done ({time.time() - t0:.0f} s)', flush=True)

    # --- E's additive boosting: partial dependence
    prow = []
    for c in conds:
        mdl = ncf.nonlin_model('boost_add').fit(X['E'], y[c])
        for j, k in enumerate(names['E']):
            for kind, q, v in pd_points(X['E'][:, j], k):
                Xp = X['E'].copy()
                Xp[:, j] = v
                prow.append(dict(model=MODEL, condition=c, cue=k, point=kind, quantile=q, value=float(v),
                                 partial_dependence=float(mdl.predict(Xp).mean())))
    pd.DataFrame(prow).to_csv(os.path.join(a.out_dir, 'exploratory_e_partial_dependence.csv'), index=False)

    # --- the ceiling on the fit frames
    vs = []
    if not a.no_ceiling:
        rq = pd.read_csv(os.path.join(a.risk, 'vlm_requests_tidy.csv'), low_memory=False,
                         usecols=['model', 'condition', 'segment', 'frame_idx', 'repeat', 'risk'])
        rq = rq[rq.model == MODEL]
        for c in conds:
            gq = rq[rq.condition == c]
            W = gq.pivot(index=['segment', 'frame_idx'], columns='repeat', values='risk').reindex(index=mi).to_numpy(float)
            full5 = ~np.isnan(W).any(1) & fit

            def ceil_of(idx):
                i = idx[full5[idx]]
                return spearman_brown(pair_corr(np.column_stack([ar.rank01(W[i, k]) for k in range(K_REP)]))[1])

            p = ceil_of(np.arange(n))
            cb = np.array([ceil_of(np.concatenate([seg_frames_all[s_] for s_ in d])) for d in draws])
            lo, hi, nv = ar.ci(cb)
            vs.append(dict(condition=c, measure=f'ceiling on the {nf} fit frames ({int(full5.sum())} with 5 scores)',
                           value=p, lo=lo, hi=hi))
    pd.DataFrame(vs, columns=['condition', 'measure', 'value', 'lo', 'hi']).to_csv(
        os.path.join(a.out_dir, 'exploratory_e_ceiling.csv'), index=False)
    n_long = int((cue[LONG_CUE][fit] >= 7.0).sum())
    n_toward = int((cue['n_side_vehicle_toward'][fit] != 0).sum())
    with open(os.path.join(a.out_dir, 'exploratory_e_params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'EXPLORATORY ({NOTE}): new_cues_fit_e.py run {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        for k_, v_ in vars(a).items():
            f.write(f'{k_} = {v_}\n')
        f.write(f'versions: python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, scipy '
                f'{scipy.__version__}, scikit-learn {sklearn.__version__}\n')
        f.write(f'frames {n} in {nseg} segments; fit frames {nf}\n')
        for s in SETS:
            f.write(f'tree / boosting cues {s}: {names[s]}\n')
        f.write('linear: D = K+S+(ped_cyc, moving, stationary)_R+NEW9, E = K+S+(ped_cyc road, off road, moving, '
                f'stationary in a lane, elsewhere)_R+NEW9+NEW_E; NEW_E {NEW_E}\n')
        f.write(f'boosting: {rt.BOOST["conservative"]}; additive version interaction_cst no_interactions\n')
        f.write(f'partial dependence (E additive): 0/1 cues at 0 and 1; the others at the {PD_Q} quantiles; {LONG_CUE} '
                f'also at {LONG_POINTS} m\n')
        f.write(f'described only: fit frames with a vehicle of {7.0} m or more within 30 m {n_long}; with a side '
                f'vehicle moving toward the lane {n_toward} (it says little)\n')
        f.write(f'resamples {a.n_boot}, seed {a.seed}, sha256 of the drawn segment indices {h.hexdigest()}\n')
        for c_ in checks:
            f.write(f'check: {c_}\n')
        f.write(f'checks not met: {failed if failed else "none"}\n')
    print(f'done in {time.time() - t0:.0f} s; checks not met: {failed if failed else "none"}')
    for c_ in checks:
        print('  ' + c_)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
