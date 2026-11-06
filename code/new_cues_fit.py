"""EXPLORATORY (user, 2026-11-05, decided before D's results were seen; CLAUDE.md "补线索" and "我定（2026-11-05，看过新量
的分布以后…）"): how much more of Qwen3.6-27B's risk-score ranks the new cues (code/new_cues.py) reproduce than B, and
whether "slow ego -> high score" stands in for them.

Frames: the risk analysis subfolder's (as analyze_rules.py), of them the frames with every new cue (FIT_FRAMES: the
ego on a mapped lane); B and D are fitted on these, the ranks taken among them; the ceiling (score_ceiling.py's
method) recomputed on them. B's earlier results on all the frames beside (copied).
Cue sets. Linear (ranks / n, ties the mean rank; S dummies as they are; analyze_rules.ols, loso_predict), R = 10, 20,
30, 50 (20 the main reading):
  B   K + S + N_R (pedestrians + cyclists, vehicles within R)                      = the rule analysis's K+S+N_R
  D   K + S + pedestrians + cyclists, moving vehicles, stationary vehicles within R + NEW9
  Dn  K + S + N_R + NEW9 (the vehicle count not split; the check)
Trees and boosting (the cues in their own units, as rule_tree.py): B = rule_tree.py's B (A, S, the eight counts); D = A,
S, the four pedestrian + cyclist counts, the eight moving / stationary vehicle counts, NEW9; Dn = B + NEW9.
Models: linear; the depth-3 tree (rule_tree.grow); boosting, the conservative setting (rule_tree.BOOST) in full and its
additive version (interaction_cst 'no_interactions', the rest the same).
Out of sample: leave one segment out as analyze_rules.py --oos; intervals from the 2,000 resamples' segment sums,
nothing refitted; the differences (D - B per model; on B and D: boosting - tree, boosting - linear R20, full - additive)
from the same resamples. Ego-speed coefficient (linear, every R) in B and D, and D - B: refitted in every resample, the
ranks anew (as analyze_rules.py's coefficients). Resampled trees (B, D): the share of resamples whose first cut is on
the true ego speed (rule_tree.py --resampled_trees's way). Single-cue tau-b of the new cues with the risk score,
groupings all and segment (as analyze_rules.py item 2: the frames with a value). Within segments (descriptive; ranks
first, then minus the segment mean, as analyze_rules.py): linear K+N20, and K + pedestrians + cyclists, moving and
stationary vehicles within 20 m + WITHIN_NEW; in-sample R2 and coefficients with intervals (refitted per resample).
D's additive boosting on all the fit frames: partial dependence (every frame's prediction with one cue set to its 10,
30, 50, 70, 90 % quantile, averaged).
Outputs (out_dir): exploratory_d_*.csv, exploratory_d_params.txt; exit code 1 when a check is not met.
Options added later (user, 2026-11-05, after D's results; without them every output is the same byte for byte, and
they are written into the params only when given): --model <name> fits another model's risk scores (default
qwen3.6-27b; the earlier tree B and boosting B results beside exist only for the models in those tables, so for the
others only the two linear rows); --predictions also writes exploratory_d_predictions.csv, every fit frame's
leave-one-segment-out prediction of every model and cue set with the rank score it predicts.
usage: python -I new_cues_fit.py <risk subfolder> <quantities subfolder> <frame_ssm.csv> <truth.csv> <scene_labels.csv>
       <nearby_counts.csv> <new_cues.csv> <rule analysis subfolder> <tree subfolder> <boosting subfolder> <out_dir>
       [--control_crossing <new_cues csv of another definition>] [--workers 8] [--no_ceiling] [--no_linear_check]
       [--model <name>] [--predictions]
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
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_rules as ar  # noqa: E402
import analyze_vlm_ssm as av  # noqa: E402
import rule_tree as rt  # noqa: E402
from score_ceiling import K_REP, pair_corr, spearman_brown  # noqa: E402

MODEL = rt.MODEL
RADII = ar.RADII
NEW9 = ['speed_limit_mps', 'dist_crosswalk', 'dist_stop_sign', 'dist_lane_crossing', 'n_crossing_ped_cyc',
        'n_crossing_vehicle', 'n_oncoming_vehicle', 'n_ped_cyc_on_road', 'heading_change_deg']
WITHIN_NEW = ['n_crossing_ped_cyc', 'n_crossing_vehicle', 'n_oncoming_vehicle', 'n_ped_cyc_on_road',
              'heading_change_deg', 'dist_crosswalk', 'dist_stop_sign', 'dist_lane_crossing']
SPLIT = [f'n_vehicle_{m}_{r}m' for r in RADII for m in ('moving', 'stationary')]
SETS = ['B', 'D', 'Dn']
NONLIN = ['tree', 'boost_full', 'boost_add']
PD_Q = [0.1, 0.3, 0.5, 0.7, 0.9]
LIN_TOL = 1e-12
NEW_OPTS = {'model': rt.MODEL, 'predictions': False}   # added later: written into the params only when not these
_W = {}


def nonlin_model(kind):
    if kind == 'tree':
        return rt.DecisionTreeRegressor(max_depth=rt.MAX_DEPTH, min_samples_leaf=rt.MIN_LEAF, random_state=rt.TREE_SEED)
    kw = dict(rt.BOOST['conservative'])
    if kind == 'boost_add':
        kw['interaction_cst'] = 'no_interactions'
    return HistGradientBoostingRegressor(**kw)


def _loso_task(task):
    """(condition, set, kind) -> leave-one-segment-out predictions and the in-sample R2 (fit on all the fit frames)."""
    c, s, kind = task
    X, y, g = _W['X'][s], _W['y'][c], _W['g']
    pred = np.full(len(y), np.nan)
    for k in range(int(g.max()) + 1):
        te = g == k
        pred[te] = nonlin_model(kind).fit(X[~te], y[~te]).predict(X[te])
    full = nonlin_model(kind).fit(X, y)
    r2_in = 1.0 - float(((y - full.predict(X)) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())
    return task, pred, r2_in


def _firstcut_task(task):
    """(condition, set) -> the cue of the first cut of the depth-3 tree in every resample (ranks anew)."""
    c, s = task
    X, sc, sf = _W['Xraw'][s], _W['score'][c], _W['seg_frames_fit']
    out = []
    for d in _W['draws']:
        idx = np.concatenate([sf[x] for x in d if x in sf])
        t = rt.grow(X[idx], ar.rank01(sc[idx]))
        out.append(rt.first_cuts(t, _W['names'][s])[0][0])
    return task, out


def _tau_chunk(bs):
    return [_W['tau'](b) for b in bs]


def parts(y, pred, g, u, nseg, demeaned=False):
    sse, sst = np.zeros(nseg), np.zeros(nseg)
    sse[u] = np.bincount(g, (y - pred) ** 2)
    sst[u] = np.bincount(g, (y - (0.0 if demeaned else y.mean())) ** 2)
    return sse, sst


def r2_of(sse, sst, counts):
    rb = 1.0 - (counts @ sse) / (counts @ sst)
    lo, hi, nv = ar.ci(rb)
    return 1.0 - sse.sum() / sst.sum(), lo, hi, nv, rb


def main():
    ap = argparse.ArgumentParser()
    for name in ('risk', 'quantities', 'frame_ssm', 'truth', 'scene_labels', 'nearby', 'new_cues', 'rules', 'trees',
                 'boosting', 'out_dir'):
        ap.add_argument(name)
    ap.add_argument('--control_crossing', default=None)
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261021)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--no_ceiling', action='store_true', help='fake scores: no ceiling (their repeats are not made up)')
    ap.add_argument('--no_linear_check', action='store_true', help='fake scores: the rule analysis is not of them')
    ap.add_argument('--model', default=NEW_OPTS['model'], help='the model whose risk scores are fitted')
    ap.add_argument('--predictions', action='store_true', help='also write exploratory_d_predictions.csv')
    a = ap.parse_args()
    global MODEL
    MODEL = a.model                                # every use below (scores, rule analysis rows, ceiling, output rows)
    if os.path.exists(os.path.join(a.out_dir, 'exploratory_d_params.txt')):
        sys.exit(f'{a.out_dir} has exploratory_d_params.txt already: give a new folder')
    t0 = time.time()
    failed, checks = [], []

    # --- frames, risk scores, cues: as rule_tree.py / analyze_rules.py
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
    nw = frames.merge(pd.read_csv(a.new_cues), on=['segment', 'frame_idx'], how='left', validate='one_to_one')
    for k in NEW9 + SPLIT:
        cue[k] = nw[k].to_numpy(float)
    if a.control_crossing:
        cc = frames.merge(pd.read_csv(a.control_crossing), on=['segment', 'frame_idx'], how='left', validate='one_to_one')
        cue['dist_lane_crossing_control'] = cc['dist_lane_crossing'].to_numpy(float)
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
    fit = ~np.isnan(np.column_stack([cue[k] for k in NEW9 + SPLIT])).any(1)
    nf = int(fit.sum())
    miss_segs = sorted(set(seg_names[~fit]))
    checks.append(f'fit frames (every new cue known) {nf} of {n}; the {n - nf} others are in {len(miss_segs)} segments '
                  f'({sum(1 for s in miss_segs if (seg_names[fit] == s).any())} of them keep other frames)')
    idf = np.flatnonzero(fit)
    u, g = np.unique(seg_code[fit], return_inverse=True)

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
    oo = pd.read_csv(os.path.join(a.rules, 'exploratory_oos_r2.csv'), float_precision='round_trip')
    oo = oo[(oo.target == 'risk') & (oo.model == MODEL)].set_index(['condition', 'fit'])
    if not a.no_linear_check:
        u_all, g_all = np.unique(seg_code, return_inverse=True)
        dev = 0.0
        for c in conds:
            y = ar.rank01(score[c])
            K = [ar.rank01(cue[k]) for k in ar.K_TRUE]
            for fname, cols in (('K', K), ('K+S+N20', K + s_cols + [ar.rank01(cue['ped_cyc_20m']),
                                                                    ar.rank01(cue['vehicle_20m'])])):
                ref = oo.loc[(c, fname)]
                sse, sst = parts(y, ar.loso_predict(y, cols, g_all), g_all, u_all, nseg)
                p, lo, hi, nv, _ = r2_of(sse, sst, counts)
                dev = max(dev, abs(ar.ols(y, cols)[2] - ref.r2_in), abs(p - ref.r2_oos), abs(lo - ref.r2_oos_lo),
                          abs(hi - ref.r2_oos_hi))
        checks.append(f'linear K and K+S+N20 on all {n} frames against exploratory_oos_r2.csv: largest difference '
                      f'{dev:.3g} (tolerance {LIN_TOL:g})')
        if not dev <= LIN_TOL:
            failed.append('linear check')
    print(f'inputs ready ({time.time() - t0:.0f} s); fit frames {nf}', flush=True)

    # --- the designs on the fit frames
    rk = {k: ar.rank01(v[fit]) for k, v in cue.items() if not np.isnan(v[fit]).any()}
    S = [c_[fit] for c_ in s_cols]
    K = [rk[k] for k in ar.K_TRUE]
    lin = {}
    for r in RADII:
        lin[('B', r)] = K + S + [rk[f'ped_cyc_{r}m'], rk[f'vehicle_{r}m']]
        lin[('D', r)] = K + S + [rk[f'ped_cyc_{r}m'], rk[f'n_vehicle_moving_{r}m'], rk[f'n_vehicle_stationary_{r}m']] + \
            [rk[k] for k in NEW9]
        lin[('Dn', r)] = K + S + [rk[f'ped_cyc_{r}m'], rk[f'vehicle_{r}m']] + [rk[k] for k in NEW9]
    names = {'B': list(ar.TRUE_CUES) + s_terms + list(ar.NEAR_CUES),
             'D': list(ar.TRUE_CUES) + s_terms + [f'ped_cyc_{r}m' for r in RADII] + SPLIT + NEW9,
             'Dn': list(ar.TRUE_CUES) + s_terms + list(ar.NEAR_CUES) + NEW9}
    full_s = {k: c_ for k, c_ in zip(s_terms, s_cols)}
    Xraw = {s: np.column_stack([full_s[k] if k in full_s else cue[k] for k in names[s]]) for s in SETS}
    X = {s: Xraw[s][fit] for s in SETS}
    y = {c: ar.rank01(score[c][fit]) for c in conds}

    # --- out of sample
    res = {}                                       # (condition, model, set) -> (sse, sst, r2_in)
    preds = {}                                     # (condition, model, set) -> the leave-one-segment-out predictions
    for c in conds:
        for (s, r), cols in lin.items():
            pred = ar.loso_predict(y[c], cols, g)
            sse, sst = parts(y[c], pred, g, u, nseg)
            res[(c, f'linear_R{r}', s)] = (sse, sst, ar.ols(y[c], cols)[2])
            preds[(c, f'linear_R{r}', s)] = pred
    print(f'linear done ({time.time() - t0:.0f} s)', flush=True)
    _W.update(X=X, y=y, g=g, Xraw=Xraw, score=score, names=names, draws=draws,
              seg_frames_fit={s_: np.flatnonzero(fit & (seg_names == s_)) for s_ in seg_list
                              if (fit & (seg_names == s_)).any()})
    tasks = [(c, s, kind) for c in conds for s in SETS for kind in NONLIN]
    ft_tasks = [(c, s) for c in conds for s in ('B', 'D')]
    if a.workers > 1:
        with mp.get_context('fork').Pool(a.workers) as pool:
            outs = pool.map(_loso_task, tasks, chunksize=1)
            fts = pool.map(_firstcut_task, ft_tasks, chunksize=1)
    else:
        outs = [_loso_task(t) for t in tasks]
        fts = [_firstcut_task(t) for t in ft_tasks]
    unpred = 0
    for (c, s, kind), pred, r2_in in outs:
        unpred += int(np.isnan(pred).sum())
        sse, sst = parts(y[c], pred, g, u, nseg)
        res[(c, kind, s)] = (sse, sst, r2_in)
        preds[(c, kind, s)] = pred
    checks.append(f'frames not predicted in the leave-one-segment-out runs: {unpred}')
    if unpred:
        failed.append('unpredicted')
    print(f'trees and boosting done ({time.time() - t0:.0f} s)', flush=True)
    rows, boot = [], {}
    for (c, m, s), (sse, sst, r2_in) in res.items():
        p, lo, hi, nv, rb = r2_of(sse, sst, counts)
        boot[(c, m, s)] = (p, rb)
        rows.append(dict(model=MODEL, condition=c, fit=m, cues=s, frames=nf, r2_in=r2_in, r2_oos=p, r2_oos_lo=lo,
                         r2_oos_hi=hi, nboot=nv))
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, 'exploratory_d_oos.csv'), index=False)
    if a.predictions:
        fits_ = [f'linear_R{r}' for r in RADII] + NONLIN
        prow = []
        for c in conds:
            tab = {'model': MODEL, 'condition': c, 'segment': seg_names[idf],
                   'frame_idx': frames['frame_idx'].to_numpy()[idf], 'y_rank': y[c]}
            for m_ in fits_:
                for s in SETS:
                    tab[f'pred_{m_}_{s}'] = preds[(c, m_, s)]
            prow.append(pd.DataFrame(tab))
        pd.concat(prow, ignore_index=True).to_csv(os.path.join(a.out_dir, 'exploratory_d_predictions.csv'), index=False)
    comps = []
    for m in [f'linear_R{r}' for r in RADII] + NONLIN:
        comps += [(f'D - B, {m}', (m, 'D'), (m, 'B')), (f'Dn - B, {m}', (m, 'Dn'), (m, 'B')),
                  (f'D - Dn, {m}', (m, 'D'), (m, 'Dn'))]
    for s in ('B', 'D'):
        comps += [(f'boost_full - tree, {s}', ('boost_full', s), ('tree', s)),
                  (f'boost_full - linear_R20, {s}', ('boost_full', s), ('linear_R20', s)),
                  (f'boost_full - boost_add, {s}', ('boost_full', s), ('boost_add', s))]
    drows = []
    for c in conds:
        for name, (m1, s1), (m2, s2) in comps:
            p1, b1 = boot[(c, m1, s1)]
            p2, b2 = boot[(c, m2, s2)]
            lo, hi, nv = ar.ci(b1 - b2)
            drows.append(dict(model=MODEL, condition=c, comparison=name, diff=p1 - p2, lo=lo, hi=hi, nboot=nv,
                              above_0=bool(not np.isnan(lo) and lo > 0), below_0=bool(not np.isnan(hi) and hi < 0)))
    pd.DataFrame(drows).to_csv(os.path.join(a.out_dir, 'exploratory_d_differences.csv'), index=False)

    # --- first cuts of the resampled trees
    frows = []
    for (c, s), cuts_ in fts:
        vc = pd.Series(cuts_).value_counts()
        for k in sorted(vc.index, key=lambda k: (-vc[k], k)):
            frows.append(dict(model=MODEL, condition=c, cues=s, first_cue=k, resamples=int(vc[k]),
                              share=vc[k] / len(cuts_)))
    pd.DataFrame(frows).to_csv(os.path.join(a.out_dir, 'exploratory_d_first_cuts.csv'), index=False)
    print(f'resampled trees done ({time.time() - t0:.0f} s)', flush=True)

    # --- ego-speed coefficient (linear, every R) in B and D, refitted per resample
    ego = 1 + ar.K_TRUE.index('true_ego_speed')
    seg_fit = {s_: v for s_, v in _W['seg_frames_fit'].items()}

    def lin_cols(idx, s, r):
        K_ = [ar.rank01(cue[k][idx]) for k in ar.K_TRUE]
        S_ = [c_[idx] for c_ in s_cols]
        if s == 'B':
            return K_ + S_ + [ar.rank01(cue[f'ped_cyc_{r}m'][idx]), ar.rank01(cue[f'vehicle_{r}m'][idx])]
        return K_ + S_ + [ar.rank01(cue[f'ped_cyc_{r}m'][idx]), ar.rank01(cue[f'n_vehicle_moving_{r}m'][idx]),
                          ar.rank01(cue[f'n_vehicle_stationary_{r}m'][idx])] + [ar.rank01(cue[k][idx]) for k in NEW9]

    erows = []
    eb = np.full((a.n_boot, len(conds), len(RADII), 2), np.nan)
    ept = np.full((len(conds), len(RADII), 2), np.nan)
    for ci_, c in enumerate(conds):
        for ri, r in enumerate(RADII):
            for si, s in enumerate(('B', 'D')):
                ept[ci_, ri, si] = ar.ols(ar.rank01(score[c][idf]), lin_cols(idf, s, r))[0][ego]
    for b, d in enumerate(draws):
        idx = np.concatenate([seg_fit[x] for x in d if x in seg_fit])
        for ci_, c in enumerate(conds):
            yb = ar.rank01(score[c][idx])
            for ri, r in enumerate(RADII):
                for si, s in enumerate(('B', 'D')):
                    eb[b, ci_, ri, si] = ar.ols(yb, lin_cols(idx, s, r))[0][ego]
    for ci_, c in enumerate(conds):
        for ri, r in enumerate(RADII):
            row = dict(model=MODEL, condition=c, R=r)
            for si, s in enumerate(('B', 'D')):
                lo, hi, nv = ar.ci(eb[:, ci_, ri, si])
                row.update({f'ego_{s}': ept[ci_, ri, si], f'ego_{s}_lo': lo, f'ego_{s}_hi': hi})
            lo, hi, nv = ar.ci(eb[:, ci_, ri, 1] - eb[:, ci_, ri, 0])
            row.update(diff_D_minus_B=ept[ci_, ri, 1] - ept[ci_, ri, 0], diff_lo=lo, diff_hi=hi, nboot=nv)
            erows.append(row)
    pd.DataFrame(erows).to_csv(os.path.join(a.out_dir, 'exploratory_d_ego_coefficient.csv'), index=False)
    print(f'ego-speed coefficients done ({time.time() - t0:.0f} s)', flush=True)

    # --- within segments (descriptive), refitted per resample
    def within_cols(idx, gg, which):
        K_ = [ar.demean(ar.rank01(cue[k][idx]), gg) for k in ar.K_TRUE]
        if which == 'K+N20':
            return K_ + [ar.demean(ar.rank01(cue[k][idx]), gg) for k in ('ped_cyc_20m', 'vehicle_20m')]
        return K_ + [ar.demean(ar.rank01(cue[k][idx]), gg) for k in
                     ('ped_cyc_20m', 'n_vehicle_moving_20m', 'n_vehicle_stationary_20m') + tuple(WITHIN_NEW)]

    wterms = {'K+N20': list(ar.K_TRUE) + ['ped_cyc_20m', 'vehicle_20m'],
              'K+N20 split + new': list(ar.K_TRUE) + ['ped_cyc_20m', 'n_vehicle_moving_20m',
                                                      'n_vehicle_stationary_20m'] + WITHIN_NEW}
    wpt, wb = {}, {}
    for c in conds:
        for wname in wterms:
            gg = g
            co, _, r2 = ar.ols(ar.demean(y[c], gg), within_cols(idf, gg, wname))
            wpt[(c, wname)] = (co, r2)
            wb[(c, wname)] = []
    for d in draws:
        idx = np.concatenate([seg_fit[x] for x in d if x in seg_fit])
        gseg = np.concatenate([np.full(len(seg_fit[x]), k) for k, x in enumerate(d) if x in seg_fit])
        gg = np.unique(gseg, return_inverse=True)[1]
        for c in conds:
            yb = ar.demean(ar.rank01(score[c][idx]), gg)
            for wname in wterms:
                co, _, r2 = ar.ols(yb, within_cols(idx, gg, wname))
                wb[(c, wname)].append(np.concatenate([co, [r2]]))
    wrows = []
    for (c, wname), (co, r2) in wpt.items():
        bb = np.array(wb[(c, wname)])
        lo, hi, nv = ar.ci(bb[:, -1])
        wrows.append(dict(model=MODEL, condition=c, fit=f'within: {wname}', term='R2 in sample', value=r2, lo=lo, hi=hi,
                          nboot=nv))
        for j, term in enumerate(['intercept'] + wterms[wname]):
            lo, hi, nv = ar.ci(bb[:, j])
            wrows.append(dict(model=MODEL, condition=c, fit=f'within: {wname}', term=term, value=co[j], lo=lo, hi=hi,
                              nboot=nv))
    pd.DataFrame(wrows).to_csv(os.path.join(a.out_dir, 'exploratory_d_within.csv'), index=False)
    print(f'within segments done ({time.time() - t0:.0f} s)', flush=True)

    # --- single-cue tau-b of the new cues (as analyze_rules.py item 2)
    tau_cues = NEW9 + SPLIT + (['dist_lane_crossing_control'] if a.control_crossing else [])
    ids = np.arange(n)
    seg_frames_all = {s_: np.flatnonzero(seg_names == s_) for s_ in seg_list}

    def tau_stats(idx, gseg, gb_seg):
        gb = {'all': np.zeros(len(idx), int), 'segment': gb_seg}
        cache, out = {}, {}
        for c in conds:
            x = score[c][idx]
            for k in tau_cues:
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

    tpt = tau_stats(ids, seg_code, seg_names)
    keys = list(tpt)

    def one(b):
        d = draws[b]
        idx = np.concatenate([seg_frames_all[s_] for s_ in d])
        gseg = np.concatenate([np.full(len(seg_frames_all[s_]), k) for k, s_ in enumerate(d)])
        r_ = tau_stats(idx, gseg, gseg)
        return np.array([r_[k] for k in keys])

    _W['tau'] = one
    if a.workers > 1:
        chunks = [list(c_) for c_ in np.array_split(np.arange(a.n_boot), a.workers * 4) if len(c_)]
        with mp.get_context('fork').Pool(a.workers) as pool:
            tb = np.array([v for part in pool.map(_tau_chunk, chunks) for v in part])
    else:
        tb = np.array([one(b) for b in range(a.n_boot)])
    trows = []
    for j, (c, k, gn) in enumerate(keys):
        lo, hi, nv = ar.ci(tb[:, j])
        trows.append(dict(model=MODEL, condition=c, cue=k, grouping=gn, frames=int((~np.isnan(cue[k])).sum()),
                          tau_b=tpt[(c, k, gn)], tau_b_lo=lo, tau_b_hi=hi, nboot=nv))
    pd.DataFrame(trows).to_csv(os.path.join(a.out_dir, 'exploratory_d_tau_b.csv'), index=False)
    print(f'tau-b done ({time.time() - t0:.0f} s)', flush=True)

    # --- D's additive boosting: partial dependence at the cues' quantiles
    prow = []
    for c in conds:
        mdl = nonlin_model('boost_add').fit(X['D'], y[c])
        for j, k in enumerate(names['D']):
            qv = np.quantile(X['D'][:, j], PD_Q)
            for q, v in zip(PD_Q, qv):
                Xp = X['D'].copy()
                Xp[:, j] = v
                prow.append(dict(model=MODEL, condition=c, cue=k, quantile=q, value=float(v),
                                 partial_dependence=float(mdl.predict(Xp).mean())))
    pd.DataFrame(prow).to_csv(os.path.join(a.out_dir, 'exploratory_d_partial_dependence.csv'), index=False)

    # --- the ceiling on the fit frames, and the earlier results beside
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
    to = pd.read_csv(os.path.join(a.trees, 'exploratory_tree_oos.csv'), float_precision='round_trip')
    bo = pd.read_csv(os.path.join(a.boosting, 'exploratory_boosting_oos.csv'), float_precision='round_trip')
    for c in conds:
        for f_ in ('K+S+N20', 'K+S+N20+H'):
            q = oo.loc[(c, f_)]
            vs.append(dict(condition=c, measure=f'earlier, all {n} frames: {f_} linear' +
                           (' (H from the same reply)' if f_.endswith('H') else ''),
                           value=q.r2_oos, lo=q.r2_oos_lo, hi=q.r2_oos_hi))
        q = to[(to.target == 'risk') & (to.model == MODEL) & (to.condition == c) & (to.cues == 'B') &
               (to.max_depth == 3)]
        if len(q):                                 # the tree tables have Qwen3.6-27B only
            q = q.iloc[0]
            vs.append(dict(condition=c, measure=f'earlier, all {n} frames: tree B', value=q.r2_oos, lo=q.r2_oos_lo,
                           hi=q.r2_oos_hi))
        q = bo[(bo.model == MODEL) & (bo.condition == c) & (bo.cues == 'B') & (bo.setting == 'conservative')]
        if len(q):
            q = q.iloc[0]
            vs.append(dict(condition=c, measure=f'earlier, all {n} frames: boosting B (conservative)', value=q.r2_oos,
                           lo=q.r2_oos_lo, hi=q.r2_oos_hi))
    pd.DataFrame(vs).to_csv(os.path.join(a.out_dir, 'exploratory_d_beside.csv'), index=False)

    with open(os.path.join(a.out_dir, 'exploratory_d_params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'EXPLORATORY (user, 2026-11-05, decided before D\'s results were seen): new_cues_fit.py run '
                f'{time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        if MODEL != NEW_OPTS['model']:
            f.write(f'model {MODEL}: fitted the same way on the user\'s request of 2026-11-05 (item 3), made after '
                    f'{NEW_OPTS["model"]}\'s D results were seen\n')
        for k_, v_ in vars(a).items():
            if k_ in NEW_OPTS and v_ == NEW_OPTS[k_]:
                continue
            f.write(f'{k_} = {v_}\n')
        f.write(f'versions: python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, scipy '
                f'{scipy.__version__}, scikit-learn {sklearn.__version__}\n')
        f.write(f'frames {n} in {nseg} segments; fit frames {nf}; frames left out in segments {miss_segs}\n')
        for s in SETS:
            f.write(f'tree / boosting cues {s}: {names[s]}\n')
        f.write(f'linear: B = K+S+N_R, D = K+S+(ped_cyc, vehicle moving, vehicle stationary)_R+NEW9, Dn = K+S+N_R+NEW9; '
                f'NEW9 {NEW9}\n')
        f.write(f'boosting: {rt.BOOST["conservative"]}; additive version interaction_cst no_interactions\n')
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
