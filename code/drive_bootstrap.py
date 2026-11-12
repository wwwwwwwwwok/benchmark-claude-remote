"""EXPLORATORY sensitivity analysis (user, 2026-11-12; the reading below was fixed before the run): resampling whole
drives instead of segments. Qwen3.6-27B with thinking on, sep_camera, the main result's 1,084 frames. CPU only; no image
is opened (only each segment's frames.parquet for the timestamps). Tables <out_dir>/exploratory_drive_bootstrap*.csv,
the report on stdout.

Units: the chains of kamiak/checks/segment_chains.py at T = 60 s (a segment whose first frame is less than 60 s after the
latest last frame of the chain so far joins it; 217 chains for the 238 segments), recomputed here from frames.parquet and
checked against <chains csv>. Inside a chain its segments in name order; the chains in the order of their first segment
name. 2,000 resamples, numpy default_rng(20261021), one choice of as many chains as there are over that list per resample;
every drawn copy of a segment is its own segment group (as in the risk analysis). The point estimates do not change;
only the intervals do (2.5 / 97.5 % percentiles, numpy linear, none when fewer than 90 % of the resamples are defined).
The out-of-sample R2 keeps its leave-one-segment-out predictions; its interval adds, per resample, the drawn chains' error
and total sums of squares (a chain = the sum over its segments).
Items (each as computed before, by the same functions, ranks taken anew and fits redone in every resample):
  4.1  tau-b 'all' of the nine SSMs (agreement.csv); ACT minus each other SSM (ssm_differences.csv, stored as X - ACT);
  4.3  same time of day / same weather / same segment minus all (27; exploratory_sec43_grouping_minus_all.csv); the 54
       differences between groups of tau-b within each group (exploratory_sec43_groups_differences.csv); the time of
       day, weather and location coefficients of fits (a)-(e) (exploratory_sec43_condition_coefficients.csv; fog and
       unclear have no value) and the out-of-sample R2 of (d) and (e) (exploratory_sec43_condition_r2.csv);
  4.5  the speed run minus base: the nine SSMs' tau-b 'all' (exploratory_speed_agreement_differences.csv), D
       (exploratory_speed_ranking.csv), the ego-speed tau-b in all, time_of_day, segment
       (exploratory_speed_speed_distance_tau_b.csv), K's ego-speed coefficient (exploratory_speed_k_coefficients.csv).
Checks: every point estimate the same text as in those tables; the whole run again with every segment its own chain:
the drawn segments' sha256 the risk analysis's (48bc1f41...) and every interval and nboot the same text as there.
Report: old and new intervals side by side, the width ratio (new / old), its median and largest value, and every item
whose interval changes between containing 0 and leaving it out.
Reading (fixed before the run): no item changes "contains 0 or not" = the paper says "resampling drives leaves the
conclusions unchanged", with the width ratios; otherwise the changed items are listed as they are and the user decides.
usage: python -I drive_bootstrap.py <waymo_v143 folder> <scene_labels.csv> <chains csv> <sec43 folder> <out_dir>
       [--workers N]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_rules as ar  # noqa: E402
import analyze_vlm_ssm as av  # noqa: E402

BASE, SPEED, COND = 'qwen3.6-27b', 'qwen3.6-27b-speed', 'sep_camera'
N_BOOT, SEED, T_CHAIN = 2000, 20261021, 60
HASH = '48bc1f413f9d5b99da57ff4edde7bbb4eb378c5a9343c44ccad4e0165466604c'
G4 = ['all', 'weather', 'time_of_day', 'segment']
G3 = ['all', 'time_of_day', 'segment']
GROUPS = {'time_of_day': ['Day', 'Dawn/Dusk', 'Night'], 'weather': ['dry', 'wet', 'rain'],
          'location': ['location_sf', 'location_phx', 'location_other']}
GDIFFS = [('time_of_day', 'Dawn/Dusk', 'Day'), ('time_of_day', 'Night', 'Day'), ('weather', 'rain', 'dry'),
          ('weather', 'wet', 'dry'), ('location', 'location_phx', 'location_sf'),
          ('location', 'location_other', 'location_sf')]
FITS = [('a', ['time_of_day']), ('b', ['weather']), ('c', ['location']), ('d', ['time_of_day', 'weather']),
        ('e', ['time_of_day', 'location', 'weather'])]
_WORK = {}


def _chunk(bs):
    return np.array([_WORK['one'](b) for b in bs])


def txt(x):
    return '' if x is None or (isinstance(x, float) and np.isnan(x)) else repr(float(x))


def side(lo, hi):
    if lo is None or np.isnan(lo):
        return 'no interval'
    return 'above 0' if lo > 0 else ('below 0' if hi < 0 else 'contains 0')


def csv_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        for d in rd:
            yield rd.line_num, d


def main():
    ap = argparse.ArgumentParser()
    for nm in ('waymo', 'scene_labels', 'chains_csv', 'sec43', 'out_dir'):
        ap.add_argument(nm)
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    A = os.path.join(a.waymo, 'analysis')
    RISK = os.path.join(A, 'after_qwen3.5-4b', 'main_1084')
    QUANT = os.path.join(A, 'quantities_after_27b_think', 'main_1084')
    SP = os.path.join(A, 'speed_risk_27b', 'main_1084')
    FSSM = os.path.join(a.waymo, 'results_lead_final_excl', 'frame_ssm_validation_training.csv')
    os.makedirs(a.out_dir, exist_ok=True)
    if os.path.exists(os.path.join(a.out_dir, 'exploratory_drive_bootstrap.csv')):
        sys.exit(f'{a.out_dir} has results already: give a new folder')
    t0 = time.time()
    failed = []

    def check(ok, what):
        print(f'  check {"ok" if ok else "NOT MET"}: {what}', flush=True)
        if not ok:
            failed.append(what)

    print(f'EXPLORATORY drive resampling (user, 2026-11-12); run {time.strftime("%Y-%m-%d %H:%M:%S")}')
    frames = pd.read_csv(os.path.join(RISK, 'ssm_target_frames.csv'))[['segment', 'frame_idx']] \
        .sort_values(['segment', 'frame_idx']).reset_index(drop=True)
    n = len(frames)
    mi = pd.MultiIndex.from_frame(frames)
    seg_names = frames['segment'].to_numpy()
    seg_list = np.array(sorted(set(seg_names)))
    seg_frames = {s: np.flatnonzero(seg_names == s) for s in seg_list}
    seg_code = np.searchsorted(seg_list, seg_names)
    ids = np.arange(n)
    nb_r, seed_r, frames_r, segs_r = ar.read_params_risk(os.path.join(RISK, 'params.txt'))
    check((nb_r, seed_r, frames_r) == (N_BOOT, SEED, n) and list(seg_list) == segs_r,
          f'{n} frames in {len(seg_list)} segments, as the risk analysis\'s params.txt')

    # --- chains at T = 60 s, recomputed from frames.parquet and checked against the chains csv
    se = []
    for s in seg_list:
        fr = pd.read_parquet(os.path.join(a.waymo, 'vlm_inputs', 'segments', s, 'frames.parquet'),
                             columns=['frame_idx', 'timestamp_micros']).sort_values('frame_idx')
        se.append((int(fr['timestamp_micros'].iloc[0]), s, int(fr['timestamp_micros'].iloc[-1])))
    chain_of, end, k = {}, None, -1
    for first, s, last in sorted(se):
        if end is None or first - end >= T_CHAIN * 1_000_000:
            k += 1
            end = last
        else:
            end = max(end, last)
        chain_of[s] = k
    members = {}
    for s, c in chain_of.items():
        members.setdefault(c, []).append(s)
    drives = sorted(sorted(v) for v in members.values())
    cc = pd.read_csv(a.chains_csv)
    theirs = sorted(sorted(g.segment) for _, g in cc.groupby('chain_60s'))
    check(drives == theirs and len(drives) == 217, f'{len(drives)} chains at T = {T_CHAIN} s, the same partition as '
          f'{a.chains_csv}; sizes {dict(sorted(pd.Series([len(d) for d in drives]).value_counts().items()))}')
    singles = [[s] for s in seg_list]

    # --- data
    def score(folder, model):
        f = pd.read_csv(os.path.join(folder, 'vlm_frame_scores.csv'), float_precision='round_trip')
        s_ = f[(f.model == model) & (f.condition == COND)].set_index(['segment', 'frame_idx'])['score_mean']
        if len(s_) != n or not s_.index.isin(mi).all():
            sys.exit(f'{folder} {model}: not the {n} frames')
        return s_.reindex(mi).to_numpy(float)

    xb, xs = score(RISK, BASE), score(SP, SPEED)
    check(np.array_equal(xb, score(SP, BASE)) and not np.isnan(xb).any() and not np.isnan(xs).any(),
          'score_mean: base the same in both analyses; every frame has a score in both runs')
    tidy = pd.read_csv(os.path.join(SP, 'vlm_requests_tidy.csv'), low_memory=False,
                       usecols=['model', 'condition', 'segment', 'frame_idx', 'repeat', 'risk'])
    ans = {}
    for m in (BASE, SPEED):
        g = tidy[(tidy.model == m) & (tidy.condition == COND)]
        ans[m] = g.set_index(['segment', 'frame_idx', 'repeat'])['risk'].unstack('repeat') \
            .reindex(index=mi, columns=range(5)).to_numpy(float)
    truth = pd.read_csv(os.path.join(a.waymo, 'quantities', 'truth_main.csv'), float_precision='round_trip')
    tr = frames.merge(truth, on=['segment', 'frame_idx'], how='left', validate='one_to_one')
    cue = {'true_distance': tr['distance'].to_numpy(float), 'true_ego_speed': tr['v_ego'].to_numpy(float),
           'true_closing_speed': tr['v_ego'].to_numpy(float) - tr['v_lead'].to_numpy(float),
           'true_ego_accel': tr['a_ego'].to_numpy(float), 'true_lead_accel': tr['a_lead'].to_numpy(float)}
    ssm_df = av.ssm_table(FSSM, frames)
    ssm = {c: ssm_df[c].to_numpy() for c in ssm_df.columns if c.endswith(('_conflict', '_risk', '_use'))}
    ssm = {k_: (v.astype(float) if k_.endswith('_risk') else v.astype(bool)) for k_, v in ssm.items()}
    groupings, _ = av.scene_groupings(frames, a.scene_labels, ['weather', 'time_of_day'])
    check(list(groupings) == G4, f'groupings {list(groupings)}')
    lab = pd.read_csv(a.scene_labels, dtype=str, keep_default_na=False).set_index('segment')
    labels = {g: lab.loc[seg_names, g].str.strip().to_numpy() for g in GROUPS}
    s_terms, s_cols, s_var = [], [], []                         # S as analyze_rules.main builds it
    for col in ('time_of_day', 'location', 'weather'):
        v = lab.loc[seg_names, col].str.strip().to_numpy()
        seg_v = lab.loc[seg_list, col].str.strip()
        if col == 'weather':
            cats = ar.WEATHER
            ref = max(cats, key=lambda c: int((seg_v == c).sum()))
        else:
            cats = sorted(set(v))
            ref = max(cats, key=lambda c: int((v == c).sum()))
        for c in cats:
            if c != ref:
                s_terms.append(f'{col}={c}')
                s_cols.append((v == c).astype(float))
                s_var.append(col)
    fit_js = {f: [j for j, v in enumerate(s_var) if v in vars_] for f, vars_ in FITS}
    pairs2 = [('b', i, 'b', j) for i, j in itertools.combinations(range(5), 2)] + \
             [('s', i, 's', j) for i, j in itertools.combinations(range(5), 2)] + \
             [('b', i, 's', j) for i in range(5) for j in range(5)]
    am = {'b': ans[BASE], 's': ans[SPEED]}

    # --- every statistic on the frames idx (gseg: segment group of each position, one per drawn copy)
    def stats(idx, gseg):
        res = {}
        gb = {g: v[idx] for g, v in groupings.items()}
        gb['segment'] = gseg
        sb = {k_: v[idx] for k_, v in ssm.items()}
        ids_b = ids[idx]
        st = av.all_stats(xb[idx], sb, ids_b, gb)
        for s in av.SSMS:
            for g in G4:
                res[('t', g, s)] = st[s][0][g][4]
        st2 = av.all_stats(xs[idx], sb, ids_b, {'all': np.zeros(len(idx), int)})
        for s in av.SSMS:
            res[('ts', s)] = st2[s][0]['all'][4]
        for g, classes in GROUPS.items():                       # tau-b within each group
            for c in classes:
                sub = idx[labels[g][idx] == c]
                if len(sub) < 3:
                    for s in av.SSMS:
                        res[('grp', g, c, s)] = np.nan
                    continue
                stg = av.all_stats(xb[sub], {k_: v[sub] for k_, v in ssm.items()}, ids[sub],
                                   {'all': np.zeros(len(sub), int)})
                for s in av.SSMS:
                    res[('grp', g, c, s)] = stg[s][0]['all'][4]
        y = ar.rank01(xb[idx])                                  # condition fits (a)-(e)
        for f, js in fit_js.items():
            coef = ar.ols(y, [s_cols[j][idx] for j in js])[0]
            for term, b_ in zip(['intercept'] + [s_terms[j] for j in js], coef):
                res[('coef', f, term)] = b_
        for tag, x in (('b', xb), ('s', xs)):                   # the speed run's items
            xi = x[idx]
            yk = ar.rank01(xi)
            coef = ar.ols(yk, [ar.rank01(cue[k_][idx]) for k_ in ar.K_TRUE])[0]
            res[('k', tag)] = coef[1 + ar.K_TRUE.index('true_ego_speed')]
            v = cue['true_ego_speed'][idx]
            mask = ~np.isnan(xi) & ~np.isnan(v)
            labs = {gn: gb[gn][mask] for gn in G3}
            prs = {gn: av.within_pairs(lv) for gn, lv in labs.items() if gn != 'all' and len(np.unique(lv)) > 20}
            cs = np.ones(int(mask.sum()), bool)
            for gn in G3:
                res[('e', tag, gn)] = av.stats_by_grouping(xi[mask], cs, v[mask], ids_b[mask], labs[gn],
                                                           prs.get(gn))[3]
        for p_ in pairs2:
            x, y_ = am[p_[0]][idx, p_[1]], am[p_[2]][idx, p_[3]]
            ok = ~np.isnan(x) & ~np.isnan(y_)
            res[('rt', *p_)] = kendalltau(x[ok], y_[ok], variant='b').statistic if ok.sum() >= 3 else np.nan
        return res

    point = stats(ids, seg_code)
    keys = list(point)
    col = {k_: i for i, k_ in enumerate(keys)}

    # out-of-sample (d), (e): per-segment sums of squares of the leave-one-segment-out predictions (unchanged)
    y0 = ar.rank01(xb)
    oos = {}
    for f in ('d', 'e'):
        pred = ar.loso_predict(y0, [s_cols[j] for j in fit_js[f]], seg_code)
        sse = np.bincount(seg_code, (y0 - pred) ** 2, minlength=len(seg_list))
        sst = np.bincount(seg_code, (y0 - y0.mean()) ** 2, minlength=len(seg_list))
        oos[f] = (float(1.0 - sse.sum() / sst.sum()), sse, sst)

    def run(units, label):
        rng = np.random.default_rng(SEED)
        draws = [rng.choice(len(units), len(units)) for _ in range(N_BOOT)]
        h = hashlib.sha256()
        for d in draws:
            h.update(np.asarray(d).astype(np.int32).tobytes())

        def one(b):
            parts, gparts, code = [], [], 0
            for ui in draws[b]:
                for s in units[ui]:
                    parts.append(seg_frames[s])
                    gparts.append(np.full(len(seg_frames[s]), code))
                    code += 1
            r_ = stats(np.concatenate(parts), np.concatenate(gparts))
            return np.array([r_.get(k_, np.nan) for k_ in keys], float)

        boot = np.full((N_BOOT, len(keys)), np.nan)
        tb = time.time()
        if a.workers > 1:
            _WORK['one'] = one
            chunks = [c_ for c_ in np.array_split(np.arange(N_BOOT), a.workers * 8) if len(c_)]
            with mp.get_context('fork').Pool(a.workers) as pool:
                for bs, vals in zip(chunks, pool.imap(_chunk, [c_.tolist() for c_ in chunks])):
                    boot[bs] = vals
        else:
            for b in range(N_BOOT):
                boot[b] = one(b)
        counts_u = np.stack([np.bincount(d, minlength=len(units)) for d in draws])
        member = np.zeros((len(units), len(seg_list)))
        for ui, u in enumerate(units):
            for s in u:
                member[ui, np.searchsorted(seg_list, s)] = 1
        print(f'  {label}: {len(units)} units, {N_BOOT} resamples in {time.time() - tb:.0f} s; sha256 of the drawn unit '
              f'indices {h.hexdigest()}', flush=True)
        return boot, counts_u, member, h.hexdigest()

    # --- the items and their old values
    ag = {d['ssm']: (ln, d) for ln, d in csv_rows(os.path.join(RISK, 'agreement.csv'))
          if d['model'] == BASE and d['condition'] == COND and d['grouping'] == 'all'}
    sd = {d['ssm_1']: (ln, d) for ln, d in csv_rows(os.path.join(RISK, 'ssm_differences.csv'))
          if d['model'] == BASE and d['condition'] == COND and d['grouping'] == 'all' and d['stat'] == 'tau_b_all'
          and d['ssm_2'] == 'ACT'}
    gm = {(d['grouping'], d['ssm']): (ln, d) for ln, d in
          csv_rows(os.path.join(a.sec43, 'exploratory_sec43_grouping_minus_all.csv'))}
    gd = {(d['comparison'], d['ssm']): (ln, d) for ln, d in
          csv_rows(os.path.join(a.sec43, 'exploratory_sec43_groups_differences.csv'))}
    cf = {(d['fit'], d['term']): (ln, d) for ln, d in
          csv_rows(os.path.join(a.sec43, 'exploratory_sec43_condition_coefficients.csv'))}
    r2 = {d['fit']: (ln, d) for ln, d in csv_rows(os.path.join(a.sec43, 'exploratory_sec43_condition_r2.csv'))}
    spd = {d['ssm']: (ln, d) for ln, d in csv_rows(os.path.join(SP, 'exploratory_speed_agreement_differences.csv'))
           if d['condition'] == COND and d['grouping'] == 'all'}
    rk = {d['condition']: (ln, d) for ln, d in csv_rows(os.path.join(SP, 'exploratory_speed_ranking.csv'))}
    sdt = {d['grouping']: (ln, d) for ln, d in csv_rows(os.path.join(SP, 'exploratory_speed_speed_distance_tau_b.csv'))
           if d['condition'] == COND and d['variable'] == 'true_ego_speed' and d['model'] == f'{SPEED} - {BASE}'}
    kc = [(ln, d) for ln, d in csv_rows(os.path.join(SP, 'exploratory_speed_k_coefficients.csv'))
          if d['version'] == 'K' and d['condition'] == COND and d['term'] == 'true_ego_speed'
          and d['model'] == f'{SPEED} - {BASE}'][0]

    def items(boot, counts_u, member):
        """(section, item, point, resampled values, old source (line, point text, lo, hi, nboot texts), flip)."""
        B = lambda k_: boot[:, col[k_]]  # noqa: E731
        out = []
        for s in av.SSMS:
            ln, d = ag[s]
            out.append(('4.1', f'tau-b all {s}', point[('t', 'all', s)], B(('t', 'all', s)),
                        (f'agreement.csv line {ln}', d['tau_b_all'], d['tau_b_all_lo'], d['tau_b_all_hi'],
                         d['tau_b_all_nboot']), False))
        for s in av.SSMS[:-1]:
            ln, d = sd[s]
            out.append(('4.1', f'ACT - {s} (stored as {s} - ACT)', point[('t', 'all', s)] - point[('t', 'all', 'ACT')],
                        B(('t', 'all', s)) - B(('t', 'all', 'ACT')),
                        (f'ssm_differences.csv line {ln}', d['diff'], d['diff_lo'], d['diff_hi'], d['nboot']), True))
        for g in ('time_of_day', 'weather', 'segment'):
            for s in av.SSMS:
                ln, d = gm[(g, s)]
                out.append(('4.3', f'{g} - all {s}', point[('t', g, s)] - point[('t', 'all', s)],
                            B(('t', g, s)) - B(('t', 'all', s)),
                            (f'exploratory_sec43_grouping_minus_all.csv line {ln}', d['diff'], d['diff_lo'],
                             d['diff_hi'], d['nboot']), False))
        for g, c1, c0 in GDIFFS:
            for s in av.SSMS:
                ln, d = gd[(f'{c1} - {c0}', s)]
                out.append(('4.3', f'{c1} - {c0} {s}', point[('grp', g, c1, s)] - point[('grp', g, c0, s)],
                            B(('grp', g, c1, s)) - B(('grp', g, c0, s)),
                            (f'exploratory_sec43_groups_differences.csv line {ln}', d['diff'], d['diff_lo'],
                             d['diff_hi'], d['nboot']), False))
        for f, js in fit_js.items():
            for j in js:
                term = s_terms[j]
                if term in ('weather=fog', 'weather=unclear'):
                    continue
                ln, d = cf[(f, term)]
                out.append(('4.3', f'({f}) {term}', point[('coef', f, term)], B(('coef', f, term)),
                            (f'exploratory_sec43_condition_coefficients.csv line {ln}', d['coef'], d['coef_lo'],
                             d['coef_hi'], d['nboot']), False))
        for f in ('d', 'e'):
            r2o, sse, sst = oos[f]
            su, tu = member @ sse, member @ sst
            ln, d = r2[f]
            out.append(('4.3', f'({f}) out-of-sample R2', r2o, 1.0 - (counts_u @ su) / (counts_u @ tu),
                        (f'exploratory_sec43_condition_r2.csv line {ln}', d['r2_oos'], d['r2_oos_lo'], d['r2_oos_hi'],
                         d['r2_oos_nboot']), False))
        for s in av.SSMS:
            ln, d = spd[s]
            out.append(('4.5', f'speed - base {s}', point[('ts', s)] - point[('t', 'all', s)],
                        B(('ts', s)) - B(('t', 'all', s)),
                        (f'exploratory_speed_agreement_differences.csv line {ln}', d['diff'], d['diff_lo'],
                         d['diff_hi'], d['nboot']), False))
        btw = [p_ for p_ in pairs2 if p_[0] != p_[2]]
        wit = [p_ for p_ in pairs2 if p_[0] == p_[2] == 'b'] + [p_ for p_ in pairs2 if p_[0] == p_[2] == 's']
        dpt = float(np.mean([point[('rt', *p_)] for p_ in btw]) - np.mean([point[('rt', *p_)] for p_ in wit]))
        dbt = np.mean(np.stack([B(('rt', *p_)) for p_ in btw]), axis=0) - \
            np.mean(np.stack([B(('rt', *p_)) for p_ in wit]), axis=0)
        ln, d = rk[COND]
        out.append(('4.5', 'D', dpt, dbt, (f'exploratory_speed_ranking.csv line {ln}', d['D'], d['D_lo'], d['D_hi'],
                                           d['D_nboot']), False))
        for gn in G3:
            ln, d = sdt[gn]
            out.append(('4.5', f'ego-speed tau-b difference ({gn})', point[('e', 's', gn)] - point[('e', 'b', gn)],
                        B(('e', 's', gn)) - B(('e', 'b', gn)),
                        (f'exploratory_speed_speed_distance_tau_b.csv line {ln}', d['tau_b'], d['tau_b_lo'],
                         d['tau_b_hi'], d['nboot']), False))
        ln, d = kc
        out.append(('4.5', 'K ego-speed coefficient difference', point[('k', 's')] - point[('k', 'b')],
                    B(('k', 's')) - B(('k', 'b')),
                    (f'exploratory_speed_k_coefficients.csv line {ln}', d['coef'], d['coef_lo'], d['coef_hi'],
                     d['nboot']), False))
        return out

    # --- 1: every segment its own chain (the check); 2: drives
    print('\nresampling')
    b1, c1_, m1, h1 = run(singles, 'every segment its own chain')
    check(h1 == HASH, f'every segment its own chain: drawn indices\' sha256 {h1} = the risk analysis\'s')
    b2, c2_, m2, h2 = run(drives, f'drives (chains at T = {T_CHAIN} s)')
    it1, it2 = items(b1, c1_, m1), items(b2, c2_, m2)
    n_pt = n_old = 0
    rows = []
    for (sec, name, pt, arr1, old, flip), (_, _, _, arr2, _, _) in zip(it1, it2):
        src, opt, olo, ohi, onb = old
        n_pt += txt(pt) == opt
        lo1, hi1, nv1 = ar.ci(arr1)
        n_old += (txt(lo1), txt(hi1), str(nv1)) == (olo, ohi, onb)
        lo2, hi2, nv2 = ar.ci(arr2)
        if flip:                                                # reported as ACT - X
            pt_r, (olo_r, ohi_r), (nlo_r, nhi_r) = -pt, (-hi1, -lo1), (-hi2 if not np.isnan(hi2) else np.nan,
                                                                     -lo2 if not np.isnan(lo2) else np.nan)
        else:
            pt_r, (olo_r, ohi_r), (nlo_r, nhi_r) = pt, (lo1, hi1), (lo2, hi2)
        w1, w2 = ohi_r - olo_r, nhi_r - nlo_r
        rows.append(dict(section=sec, item=name, point=pt_r, old_lo=olo_r, old_hi=ohi_r, old_nboot=nv1, old_width=w1,
                         new_lo=nlo_r, new_hi=nhi_r, new_nboot=nv2, new_width=w2,
                         width_ratio=w2 / w1 if w1 > 0 else np.nan, old_side=side(olo_r, ohi_r),
                         new_side=side(nlo_r, nhi_r), changed=side(olo_r, ohi_r) != side(nlo_r, nhi_r), source=src))
    check(n_pt == len(it1), f'point estimates: {n_pt} of {len(it1)} the same text as the tables')
    check(n_old == len(it1), f'every segment its own chain: {n_old} of {len(it1)} intervals and nboot the same text as '
          'the tables')
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(a.out_dir, 'exploratory_drive_bootstrap.csv'), index=False)
    ratio = t.width_ratio.dropna()
    print(f'\nexploratory_drive_bootstrap.csv: {len(t)} items (line = row + 1); width ratio new / old: median '
          f'{float(ratio.median())!r}, largest {float(ratio.max())!r} ({t.loc[ratio.idxmax(), "item"]}), smallest '
          f'{float(ratio.min())!r}')
    for i, d in t.iterrows():
        print(f"  line {i + 2}: {d['section']} {d['item']:42s} {d['point']!r}: old [{d['old_lo']!r}, {d['old_hi']!r}] "
              f"(nboot {d['old_nboot']}; {d['old_side']}), new [{d['new_lo']!r}, {d['new_hi']!r}] (nboot {d['new_nboot']}; "
              f"{d['new_side']}); width ratio {d['width_ratio']!r}{'  CHANGED' if d['changed'] else ''}")
    ch = t[t.changed]
    print(f'\nREADING (fixed before the run): items whose interval changes between containing 0 and leaving it out: '
          f'{len(ch)}' + ('' if len(ch) else ' -> "resampling drives leaves the conclusions unchanged"'))
    for i, d in ch.iterrows():
        print(f"  line {i + 2}: {d['section']} {d['item']}: {d['old_side']} -> {d['new_side']}")
    print(f'\ndone in {time.time() - t0:.0f} s; checks not met: {len(failed)}' + (f': {failed}' if failed else ''))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
