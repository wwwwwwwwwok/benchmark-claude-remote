"""More numbers for paper section 4.3, scene conditions (user, 2026-11-11, second request; EXPLORATORY): Qwen3.6-27B with
thinking on. CPU only; no image is opened. Tables <out_dir>/exploratory_sec43_more_*.csv, the report on stdout.

 1  "same location" as a grouping (labels/scene_labels.csv location: location_sf, location_phx, location_other, the
    rule analysis's S variable), as paper_sec43.py item 6: in each of the 2,000 resamples of whole segments (seed
    20261021, drawn as the risk analysis draws them, every drawn copy its own segment) analyze_vlm_ssm.all_stats with
    the groupings all, weather, time_of_day, location, segment; tau_b_all (= tau_b_conflict, every frame has every SSM)
    with 2.5 / 97.5 % percentile intervals (numpy, linear; none under 90 % defined), and grouping - all over the same
    resamples. Checks as there: every grouping the risk analysis has (all, weather, time_of_day, segment) recomputed
    here, point, interval and nboot, the same text as after_qwen3.5-4b_weather/<frames>/agreement.csv; the location
    tau-b recomputed pair by pair without scipy (per class: S = sum of sign(dx) sign(dy), untied pairs in x and in y;
    tau = sum S / sqrt(sum tx * sum ty)) equal to the program's within 1e-12. Copied besides, with their lines: the rule
    analysis's K+S coefficients of location_phx and location_other (exploratory_rule_coefficients.csv).
 2  the model's own words (describe only): sep_camera, its 5,420 answers on the 1,084 frames (the answer the analysis
    uses: the reply's own or, without one, the forced answer; <reasons csv> from code/extract_reasons.py), the answer's
    fields main_hazard and reason (not the thinking); word classes WORDS, matched case-insensitively as whole words or
    whole phrases (no letter or digit right before or after). Per weather, time of day and location class (and all):
    answers; per word class the answers whose main_hazard, whose reason, whose either field has one (count, share);
    the mean score (the answer's own risk) of the answers with and without a weather word in either field, and with
    and without a light word. For each word class 5 answers with a word in either field, drawn with numpy
    default_rng(20261112) (a new generator per class) from them ordered by segment, frame, repeat: main_hazard and
    reason printed as text, with the words found.
 3  (optional, done) every condition x frame set (main_1084, sensitivity_1123): paper_sec43.py items 3 (fits (d) time
    of day + weather and (e) time of day + weather + location, dummies as the rule analysis's S, rank of score_mean /
    frames; coefficients and in-sample R2 refitted in every resample, leave-one-segment-out R2 as analyze_rules --oos)
    and 6 (grouping - all) and item 1 above; one summary row per condition x frame set: the (e) coefficients of time of
    day, weather and location with intervals, (d) and (e) R2 in and out of sample with intervals, and per grouping - all
    the SSMs whose interval leaves out 0 (with the sign) and the largest absolute difference. Checks per frame set: seed,
    number and segment list as the risk analysis's params.txt, the drawn segments' sha256 the quantities analysis's;
    reference classes Day, location_sf, dry; with the rule analysis's K added, (e) reproduces its K+S coefficients, R2,
    intervals (exploratory_rule_coefficients.csv, exploratory_rule_r2.csv) and out-of-sample R2 (exploratory_oos_r2.csv),
    the same text; sep_camera x main_1084 the same text as paper_sec43.py's exploratory_sec43_condition_r2.csv,
    exploratory_sec43_condition_coefficients.csv and exploratory_sec43_grouping_minus_all.csv in <out_dir>.
usage: python -I paper_sec43_more.py <waymo_v143 folder> <scene_labels.csv> <reasons csv of qwen3.6-27b> <out_dir>
       [--workers N]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import multiprocessing as mp  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_rules as ar  # noqa: E402
import analyze_vlm_ssm as av  # noqa: E402

MODEL = 'qwen3.6-27b'
CONDS = av.CONDITIONS
SUBS = ['main_1084', 'sensitivity_1123']
G = ['all', 'weather', 'time_of_day', 'location', 'segment']
G_RISK = ['all', 'weather', 'time_of_day', 'segment']                  # the groupings agreement.csv has
DIFF_G = ['time_of_day', 'weather', 'location', 'segment']
FITS = {'d': ('time of day + weather', ['time_of_day', 'weather']),
        'e': ('time of day + weather + location', ['time_of_day', 'location', 'weather'])}
SUMMARY_TERMS = ['time_of_day=Dawn/Dusk', 'time_of_day=Night', 'weather=wet', 'weather=rain',
                 'location=location_other', 'location=location_phx']
ORDER = {'all': ['all'], 'weather': ['dry', 'wet', 'rain'], 'time_of_day': ['Day', 'Dawn/Dusk', 'Night'],
         'location': ['location_sf', 'location_phx', 'location_other']}
WORDS = {'weather': ['rain', 'rainy', 'raining', 'wet', 'slippery', 'slick', 'puddle', 'puddles', 'drizzle',
                     'precipitation', 'fog', 'foggy', 'snow', 'weather'],
         'light': ['night', 'nighttime', 'dark', 'darkness', 'dim', 'dusk', 'dawn', 'twilight', 'headlight',
                   'headlights', 'glare', 'sunlight', 'low light', 'lighting'],
         'visibility': ['visibility']}
TOL = 1e-12
_WORK = {}


def _chunk(bs):
    return np.array([_WORK['one'](b) for b in bs])


def txt(x):
    return '' if x is None or (isinstance(x, float) and np.isnan(x)) else repr(float(x))


def r(x):
    return 'nan' if x is None or (isinstance(x, float) and np.isnan(x)) else repr(float(x))


def side(lo, hi):
    if lo is None or np.isnan(lo):
        return 'no interval'
    return 'above 0' if lo > 0 else ('below 0' if hi < 0 else 'contains 0')


def csv_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        for d in rd:
            yield rd.line_num, d


def brute_tau(x, y, labels):
    """tau-b over the pairs within each label, pair by pair (no scipy)."""
    s = tx = ty = 0.0
    for g in np.unique(labels):
        m = labels == g
        xs, ys = x[m], y[m]
        if len(xs) < 2:
            continue
        iu = np.triu_indices(len(xs), 1)
        dx = np.sign(xs[:, None] - xs[None, :])[iu]
        dy = np.sign(ys[:, None] - ys[None, :])[iu]
        s += float((dx * dy).sum())
        tx += float((dx != 0).sum())
        ty += float((dy != 0).sum())
    return s / np.sqrt(tx * ty)


def main():
    ap = argparse.ArgumentParser()
    for nm in ('waymo', 'scene_labels', 'reasons', 'out_dir'):
        ap.add_argument(nm)
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261021)
    ap.add_argument('--sample_seed', type=int, default=20261112)
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    A = os.path.join(a.waymo, 'analysis')
    O = a.out_dir
    for f_ in ('exploratory_sec43_more_summary.csv', 'exploratory_sec43_reason_words.csv'):
        if os.path.exists(os.path.join(O, f_)):
            sys.exit(f'{O}/{f_} exists already: give a new folder')
    t0 = time.time()
    failed = []

    def check(ok, what):
        print(f'  check {"ok" if ok else "NOT MET"}: {what}', flush=True)
        if not ok:
            failed.append(what)

    def write(name, rows):
        pd.DataFrame(rows).to_csv(os.path.join(O, name), index=False)
        return name

    print(f'paper section 4.3, more numbers (user, 2026-11-11, second request); EXPLORATORY; {MODEL}; run '
          f'{time.strftime("%Y-%m-%d %H:%M:%S")}')
    truth = pd.read_csv(os.path.join(a.waymo, 'quantities', 'truth_main.csv'), float_precision='round_trip')
    FSSM = os.path.join(a.waymo, 'results_lead_final_excl', 'frame_ssm_validation_training.csv')
    lab = pd.read_csv(a.scene_labels, dtype=str, keep_default_na=False).set_index('segment')
    rows_tau, rows_diff, rows_fit, rows_coef, rows_ks, rows_sum = [], [], [], [], [], []
    main_labels = None
    for sub in SUBS:
        print('\n' + '=' * 110 + f'\n{sub}')
        RISK = os.path.join(A, 'after_qwen3.5-4b', sub)
        WEA = os.path.join(A, 'after_qwen3.5-4b_weather', sub)
        RULE = os.path.join(A, 'rule_explore', sub)
        QUANT = os.path.join(A, 'quantities_after_27b_think', sub)
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
        check((nb_r, seed_r, frames_r) == (a.n_boot, a.seed, n) and list(seg_list) == segs_r,
              f'{n} frames in {len(seg_list)} segments; resamples, seed, frames, segment list as the risk analysis\'s '
              f'params.txt ({nb_r}, {seed_r}, {frames_r})')
        rng = np.random.default_rng(a.seed)
        draws = [rng.choice(seg_list, len(seg_list)) for _ in range(a.n_boot)]
        h = hashlib.sha256()
        for d in draws:
            h.update(np.searchsorted(seg_list, d).astype(np.int32).tobytes())
        nb_q, seed_q, hash_q = ar.read_hash_quantities(os.path.join(QUANT, 'params.txt'))
        check((nb_q, seed_q, hash_q) == (a.n_boot, a.seed, h.hexdigest()),
              f'drawn segments\' sha256 {h.hexdigest()} = the quantities analysis\'s ({QUANT}/params.txt)')
        counts = np.stack([np.bincount(np.searchsorted(seg_list, d), minlength=len(seg_list)) for d in draws])
        fs = pd.read_csv(os.path.join(RISK, 'vlm_frame_scores.csv'), float_precision='round_trip')
        risk = {}
        for c in CONDS:
            s = fs[(fs.model == MODEL) & (fs.condition == c)].set_index(['segment', 'frame_idx'])['score_mean']
            if len(s) != n or not s.index.isin(mi).all():
                sys.exit(f'{sub} {c}: the frames of {MODEL} are not the {n} frames')
            risk[c] = s.reindex(mi).to_numpy(float)
            if np.isnan(risk[c]).any():
                sys.exit(f'{sub} {c}: frames without a score; this program assumes every frame has one')
        # S as analyze_rules.main builds it
        s_terms, s_cols, s_var, s_ref = [], [], [], {}
        for col in ('time_of_day', 'location', 'weather'):
            v = lab.loc[seg_names, col].str.strip().to_numpy()
            seg_v = lab.loc[seg_list, col].str.strip()
            if col == 'weather':
                cats = ar.WEATHER
                ref = max(cats, key=lambda k: int((seg_v == k).sum()))
            else:
                cats = sorted(set(v))
                ref = max(cats, key=lambda k: int((v == k).sum()))
            s_ref[col] = ref
            for k in cats:
                if k != ref:
                    s_terms.append(f'{col}={k}')
                    s_cols.append((v == k).astype(float))
                    s_var.append(col)
        check(s_ref == {'time_of_day': 'Day', 'location': 'location_sf', 'weather': 'dry'}, f'reference classes {s_ref}')
        tr = frames.merge(truth, on=['segment', 'frame_idx'], how='left', validate='one_to_one', indicator=True)
        check((tr['_merge'] == 'both').all(), 'every frame in truth_main.csv')
        cue = {'true_distance': tr['distance'].to_numpy(float), 'true_ego_speed': tr['v_ego'].to_numpy(float),
               'true_closing_speed': tr['v_ego'].to_numpy(float) - tr['v_lead'].to_numpy(float),
               'true_ego_accel': tr['a_ego'].to_numpy(float), 'true_lead_accel': tr['a_lead'].to_numpy(float)}
        ssm_df = av.ssm_table(FSSM, frames)
        ssm = {c: ssm_df[c].to_numpy() for c in ssm_df.columns if c.endswith(('_conflict', '_risk', '_use'))}
        ssm = {k: (v_.astype(float) if k.endswith('_risk') else v_.astype(bool)) for k, v_ in ssm.items()}
        check(all(ssm[s + '_conflict'].all() and ssm[s + '_use'].all() for s in av.SSMS),
              'every frame has a value of every SSM and is counted on it')
        groupings, _ = av.scene_groupings(frames, a.scene_labels, ['weather', 'time_of_day', 'location'])
        check(list(groupings) == G, f'groupings {list(groupings)}')
        if sub == 'main_1084':
            main_labels = (frames.copy(), {g: groupings[g] for g in ('weather', 'time_of_day', 'location')})
        fit_js = {f: [j for j, v in enumerate(s_var) if v in vars_] for f, (_, vars_) in FITS.items()}

        def stats(idx, gseg):
            res = {}
            gb = {g: v_[idx] for g, v_ in groupings.items()}
            gb['segment'] = gseg
            sb = {k: v_[idx] for k, v_ in ssm.items()}
            for c in CONDS:
                st = av.all_stats(risk[c][idx], sb, ids[idx], gb)
                for s in av.SSMS:
                    for g in G:
                        res[('t', c, s, g, 'tau_b_conflict')] = st[s][0][g][1]
                        res[('t', c, s, g, 'tau_b_all')] = st[s][0][g][4]
                y = ar.rank01(risk[c][idx])
                for f, js in fit_js.items():
                    coef, _, r2 = ar.ols(y, [s_cols[j][idx] for j in js])
                    res[('r2', c, f)] = r2
                    for term, b_ in zip(['intercept'] + [s_terms[j] for j in js], coef):
                        res[('coef', c, f, term)] = b_
                coef, _, r2 = ar.ols(y, [ar.rank01(cue[k][idx]) for k in ar.K_TRUE] + [c_[idx] for c_ in s_cols])
                res[('r2', c, 'K+S')] = r2
                for term, b_ in zip(['intercept'] + ar.K_TRUE + s_terms, coef):
                    res[('coef', c, 'K+S', term)] = b_
            return res

        point = stats(ids, seg_code)
        keys = list(point)
        col = {k: i for i, k in enumerate(keys)}

        def one(b):
            draw = draws[b]
            idx = np.concatenate([seg_frames[s] for s in draw])
            gseg = np.concatenate([np.full(len(seg_frames[s]), k) for k, s in enumerate(draw)])
            rr = stats(idx, gseg)
            return np.array([rr.get(k, np.nan) for k in keys], float)

        boot = np.full((a.n_boot, len(keys)), np.nan)
        tb = time.time()
        if a.workers > 1:
            _WORK['one'] = one
            chunks = [c_ for c_ in np.array_split(np.arange(a.n_boot), a.workers * 8) if len(c_)]
            done = 0
            with mp.get_context('fork').Pool(a.workers) as pool:
                for bs, vals in zip(chunks, pool.imap(_chunk, [c_.tolist() for c_ in chunks])):
                    boot[bs] = vals
                    done += len(bs)
                    if done == len(bs) or done % 500 < len(bs) or done == a.n_boot:
                        print(f'  [resample {done}/{a.n_boot}, {(time.time() - tb) / done * (a.n_boot - done):.0f} s '
                              f'left; {a.workers} workers]', flush=True)
        else:
            for b in range(a.n_boot):
                boot[b] = one(b)
                if b in (0, 9) or (b + 1) % 250 == 0:
                    print(f'  [resample {b + 1}/{a.n_boot}]', flush=True)

        def ci(k):
            return ar.ci(boot[:, col[k]])

        # checks: the risk analysis's groupings, location pair by pair, K+S
        ag = {(d['condition'], d['ssm'], d['grouping']): (ln, d) for ln, d in csv_rows(os.path.join(WEA, 'agreement.csv'))
              if d['model'] == MODEL}
        n_own = n_ok = 0
        dmax_loc = 0.0
        for c in CONDS:
            for s in av.SSMS:
                for g in G_RISK:
                    ln, d = ag[(c, s, g)]
                    for stat in ('tau_b_conflict', 'tau_b_all'):
                        lo, hi, nv = ci(('t', c, s, g, stat))
                        n_own += 1
                        n_ok += (txt(point[('t', c, s, g, stat)]), txt(lo), txt(hi), str(nv)) == \
                            (d[stat], d[stat + '_lo'], d[stat + '_hi'], d[stat + '_nboot'])
                bt = brute_tau(risk[c], ssm[s + '_risk'], groupings['location'])
                dmax_loc = max(dmax_loc, abs(bt - point[('t', c, s, 'location', 'tau_b_all')]),
                               abs(bt - point[('t', c, s, 'location', 'tau_b_conflict')]))
        check(n_ok == n_own, f'all, weather, time_of_day, segment recomputed: {n_ok} of {n_own} (5 conditions x 9 SSMs x '
              f'4 groupings x tau_b_conflict, tau_b_all; point, interval, nboot) the same text as {WEA}/agreement.csv')
        check(dmax_loc <= TOL, f'location tau-b pair by pair (no scipy) = the program\'s: largest |difference| '
              f'{dmax_loc!r} (45 values x 2 columns)')
        rc = [(ln, d) for ln, d in csv_rows(os.path.join(RULE, 'exploratory_rule_coefficients.csv'))
              if d['target'] == 'risk' and d['model'] == MODEL and d['fit'] == 'K+S']
        rr2 = {d['condition']: (ln, d) for ln, d in csv_rows(os.path.join(RULE, 'exploratory_rule_r2.csv'))
               if d['target'] == 'risk' and d['model'] == MODEL and d['fit'] == 'K+S'}
        roo = {d['condition']: (ln, d) for ln, d in csv_rows(os.path.join(RULE, 'exploratory_oos_r2.csv'))
               if d['target'] == 'risk' and d['model'] == MODEL and d['fit'] == 'K+S'}
        n_ks = n_ks_ok = 0
        dmax_ks = 0.0
        for ln, d in rc:
            k = ('coef', d['condition'], 'K+S', d['term'])
            lo, hi, nv = ci(k)
            mine = (txt(point[k]), txt(lo), txt(hi), str(nv))
            n_ks += 1
            n_ks_ok += mine == (d['coef'], d['coef_lo'], d['coef_hi'], d['nboot'])
            for x_, t_ in zip((point[k], lo, hi), (d['coef'], d['coef_lo'], d['coef_hi'])):
                if t_ != '' and not np.isnan(x_):
                    dmax_ks = max(dmax_ks, abs(x_ - float(t_)))
            if d['term'] in ('location=location_phx', 'location=location_other'):
                rows_ks.append(dict(frames=sub, condition=d['condition'], term=d['term'], coef=d['coef'],
                                    coef_lo=d['coef_lo'], coef_hi=d['coef_hi'], nboot=d['nboot'],
                                    source=f'{RULE}/exploratory_rule_coefficients.csv line {ln}'))
        for c in CONDS:
            k = ('r2', c, 'K+S')
            lo, hi, nv = ci(k)
            ln, d = rr2[c]
            n_ks += 1
            n_ks_ok += (txt(point[k]), txt(lo), txt(hi), str(nv)) == (d['r2'], d['r2_lo'], d['r2_hi'], d['nboot'])
        check(n_ks_ok == n_ks == 5 * 15, f'with K added, (e) = the rule analysis\'s K+S: {n_ks_ok} of {n_ks} (5 conditions x '
              f'14 coefficients and R2; point, interval, nboot) the same text (largest |difference| {dmax_ks!r})')

        def oos(c, cols):
            y = ar.rank01(risk[c])
            pred = ar.loso_predict(y, cols, seg_code)
            sse = np.bincount(seg_code, (y - pred) ** 2, minlength=len(seg_list))
            sst = np.bincount(seg_code, (y - y.mean()) ** 2, minlength=len(seg_list))
            lo_, hi_, nv_ = ar.ci(1.0 - (counts @ sse) / (counts @ sst))
            return float(1.0 - sse.sum() / sst.sum()), lo_, hi_, nv_

        n_o = n_o_ok = 0
        for c in CONDS:
            ro_ = oos(c, [ar.rank01(cue[k]) for k in ar.K_TRUE] + s_cols)
            ln, d = roo[c]
            n_o += 1
            n_o_ok += (txt(point[('r2', c, 'K+S')]), *map(txt, ro_[:3]), str(ro_[3])) == \
                (d['r2_in'], d['r2_oos'], d['r2_oos_lo'], d['r2_oos_hi'], d['nboot'])
        check(n_o_ok == n_o == 5, f'K+S out of sample: {n_o_ok} of {n_o} conditions the same text as '
              f'{RULE}/exploratory_oos_r2.csv')

        # results
        for c in CONDS:
            for s in av.SSMS:
                for g in G:
                    k = ('t', c, s, g, 'tau_b_all')
                    lo, hi, nv = ci(k)
                    rows_tau.append(dict(frames=sub, condition=c, ssm=s, grouping=g, tau_b=point[k], tau_b_lo=lo,
                                         tau_b_hi=hi, nboot=nv, side=side(lo, hi)))
                for g in DIFF_G:
                    kg, ka = ('t', c, s, g, 'tau_b_all'), ('t', c, s, 'all', 'tau_b_all')
                    lo, hi, nv = ar.ci(boot[:, col[kg]] - boot[:, col[ka]])
                    rows_diff.append(dict(frames=sub, condition=c, ssm=s, grouping=g, tau_b_grouping=point[kg],
                                          tau_b_all=point[ka], diff=point[kg] - point[ka], diff_lo=lo, diff_hi=hi,
                                          nboot=nv, side=side(lo, hi)))
            for f, (name, _) in FITS.items():
                k = ('r2', c, f)
                lo, hi, nv = ci(k)
                ro_ = oos(c, [s_cols[j] for j in fit_js[f]])
                rows_fit.append(dict(frames=sub, condition=c, fit=f, cues=name, r2_in=point[k], r2_in_lo=lo,
                                     r2_in_hi=hi, r2_in_nboot=nv, r2_oos=ro_[0], r2_oos_lo=ro_[1], r2_oos_hi=ro_[2],
                                     r2_oos_nboot=ro_[3]))
                for term in ['intercept'] + [s_terms[j] for j in fit_js[f]]:
                    kc = ('coef', c, f, term)
                    lo, hi, nv = ci(kc)
                    rows_coef.append(dict(frames=sub, condition=c, fit=f, term=term, coef=point[kc], coef_lo=lo,
                                          coef_hi=hi, nboot=nv, side=side(lo, hi)))
        print(f'  {sub} done ({time.time() - t0:.0f} s)', flush=True)

    # ============================================================================ tables, the check against paper_sec43
    f_tau = write('exploratory_sec43_more_tau_b.csv', rows_tau)
    f_diff = write('exploratory_sec43_more_grouping_minus_all.csv', rows_diff)
    f_fit = write('exploratory_sec43_more_condition_r2.csv', rows_fit)
    f_coef = write('exploratory_sec43_more_condition_coefficients.csv', rows_coef)
    f_ks = write('exploratory_sec43_more_rule_location.csv', rows_ks)
    old_r2 = {d['fit']: d for _, d in csv_rows(os.path.join(O, 'exploratory_sec43_condition_r2.csv'))}
    old_coef = {(d['fit'], d['term']): d for _, d in csv_rows(os.path.join(O, 'exploratory_sec43_condition_coefficients.csv'))}
    old_diff = {(d['ssm'], d['grouping']): d for _, d in csv_rows(os.path.join(O, 'exploratory_sec43_grouping_minus_all.csv'))}
    n_c = n_c_ok = 0
    for d in rows_fit:
        if d['frames'] == 'main_1084' and d['condition'] == 'sep_camera':
            o = old_r2[d['fit']]
            n_c += 1
            n_c_ok += tuple(txt(d[k]) if not k.endswith('nboot') else str(d[k]) for k in
                            ('r2_in', 'r2_in_lo', 'r2_in_hi', 'r2_in_nboot', 'r2_oos', 'r2_oos_lo', 'r2_oos_hi',
                             'r2_oos_nboot')) == tuple(o[k] for k in ('r2_in', 'r2_in_lo', 'r2_in_hi', 'r2_in_nboot',
                                                                      'r2_oos', 'r2_oos_lo', 'r2_oos_hi', 'r2_oos_nboot'))
    for d in rows_coef:
        if d['frames'] == 'main_1084' and d['condition'] == 'sep_camera':
            o = old_coef[(d['fit'], d['term'])]
            n_c += 1
            n_c_ok += (txt(d['coef']), txt(d['coef_lo']), txt(d['coef_hi']), str(d['nboot'])) == \
                (o['coef'], o['coef_lo'], o['coef_hi'], o['nboot'])
    for d in rows_diff:
        if d['frames'] == 'main_1084' and d['condition'] == 'sep_camera' and d['grouping'] != 'location':
            o = old_diff[(d['ssm'], d['grouping'])]
            n_c += 1
            n_c_ok += (txt(d['diff']), txt(d['diff_lo']), txt(d['diff_hi']), str(d['nboot'])) == \
                (o['diff'], o['diff_lo'], o['diff_hi'], o['nboot'])
    check(n_c_ok == n_c == 2 + 7 + 9 + 27, f'sep_camera x main_1084: {n_c_ok} of {n_c} rows ((d), (e) R2; their '
          'coefficients; time of day, weather, segment - all) the same text as paper_sec43.py\'s tables')

    # ============================================================================ 1, report
    print('\n' + '=' * 110 + '\n1. same location (sep_camera, main_1084)')
    for i, d in enumerate(rows_tau):
        if d['frames'] == 'main_1084' and d['condition'] == 'sep_camera' and d['grouping'] in ('all', 'location'):
            print(f"  {f_tau} line {i + 2}: {d['ssm']:6s} {d['grouping']:9s} tau_b {r(d['tau_b'])} [{r(d['tau_b_lo'])}, "
                  f"{r(d['tau_b_hi'])}] (nboot {d['nboot']}; {d['side']})")
    for i, d in enumerate(rows_diff):
        if d['frames'] == 'main_1084' and d['condition'] == 'sep_camera' and d['grouping'] == 'location':
            print(f"  {f_diff} line {i + 2}: {d['ssm']:6s} location - all {r(d['diff'])} [{r(d['diff_lo'])}, "
                  f"{r(d['diff_hi'])}] (nboot {d['nboot']}; {d['side']})")
    for i, d in enumerate(rows_ks):
        print(f"  {f_ks} line {i + 2}: {d['frames']} {d['condition']:10s} K+S {d['term']}: {d['coef']} [{d['coef_lo']}, "
              f"{d['coef_hi']}] nboot {d['nboot']} ({d['source']})")

    # ============================================================================ 2
    print('\n' + '=' * 110 + '\n2. its own words: sep_camera, main_1084, main_hazard and reason (not the thinking)')
    frames_m, lab_m = main_labels
    mi_m = pd.MultiIndex.from_frame(frames_m)
    rs = pd.read_csv(a.reasons, dtype={'main_hazard': str, 'reason': str}, keep_default_na=False)
    rs = rs[(rs.model == MODEL) & (rs.condition == 'sep_camera')].copy()
    rs['frame_idx'] = rs['frame_idx'].astype(int)
    rs = rs[pd.MultiIndex.from_frame(rs[['segment', 'frame_idx']]).isin(mi_m)]
    rs = rs[rs['risk'] != ''].copy()
    rs['score'] = rs['risk'].astype(float)
    pos = mi_m.get_indexer(pd.MultiIndex.from_frame(rs[['segment', 'frame_idx']]))
    for g in ('weather', 'time_of_day', 'location'):
        rs[g] = lab_m[g][pos]
    rs['all'] = 'all'
    check(len(rs) == 5420, f'answers of {MODEL} sep_camera on the 1,084 frames: {len(rs)}')
    pats = {wc: re.compile(r'(?<![A-Za-z0-9])(?:' + '|'.join(re.escape(w_) for w_ in ws) + r')(?![A-Za-z0-9])',
                           re.IGNORECASE) for wc, ws in WORDS.items()}
    for wc, p_ in pats.items():
        for fld in ('main_hazard', 'reason'):
            rs[f'{wc}_{fld}'] = rs[fld].map(lambda t_: bool(p_.search(t_)))
        rs[f'{wc}_either'] = rs[f'{wc}_main_hazard'] | rs[f'{wc}_reason']
    rows_w = []
    for g, classes in ORDER.items():
        for k in classes:
            q = rs[rs[g] == k]
            row = dict(grouping=g, label=k, answers=len(q))
            for wc in WORDS:
                for fld in ('main_hazard', 'reason', 'either'):
                    cnt = int(q[f'{wc}_{fld}'].sum())
                    row.update({f'{wc}_{fld}': cnt, f'{wc}_{fld}_share': cnt / len(q) if len(q) else np.nan})
            for wc in ('weather', 'light'):
                for has, nm in ((True, 'with'), (False, 'without')):
                    qq = q[q[f'{wc}_either'] == has]
                    row.update({f'mean_score_{nm}_{wc}_word': float(qq.score.mean()) if len(qq) else np.nan,
                                f'answers_{nm}_{wc}_word': len(qq)})
            rows_w.append(row)
    f_w = write('exploratory_sec43_reason_words.csv', rows_w)
    print(f'  words: {WORDS}')
    for i, d in enumerate(rows_w):
        print(f"  {f_w} line {i + 2}: {d['grouping']:11s} {d['label']:14s} answers {d['answers']}; " + '; '.join(
            f"{wc}: main_hazard {d[f'{wc}_main_hazard']} ({r(d[f'{wc}_main_hazard_share'])}), reason {d[f'{wc}_reason']} "
            f"({r(d[f'{wc}_reason_share'])}), either {d[f'{wc}_either']} ({r(d[f'{wc}_either_share'])})" for wc in WORDS))
        print('       mean score: ' + '; '.join(
            f"{nm} a {wc} word {r(d[f'mean_score_{nm}_{wc}_word'])} ({d[f'answers_{nm}_{wc}_word']} answers)"
            for wc in ('weather', 'light') for nm in ('with', 'without')))
    samples = []
    for wc, p_ in pats.items():
        pool = rs[rs[f'{wc}_either']].sort_values(['segment', 'frame_idx', 'repeat']).reset_index(drop=True)
        pick = np.random.default_rng(a.sample_seed).choice(len(pool), size=min(5, len(pool)), replace=False)
        for j_, i_ in enumerate(pick, 1):
            p2 = pool.iloc[int(i_)]
            samples.append(dict(word_class=wc, draw=j_, segment=p2['segment'], frame_idx=int(p2['frame_idx']),
                                repeat=int(p2['repeat']), score=float(p2['score']), weather=p2['weather'],
                                time_of_day=p2['time_of_day'], words_in_main_hazard='; '.join(p_.findall(p2['main_hazard'])),
                                words_in_reason='; '.join(p_.findall(p2['reason'])), main_hazard=p2['main_hazard'],
                                reason=p2['reason']))
    f_s = write('exploratory_sec43_reason_samples.csv', samples)
    for i, d in enumerate(samples):
        print(f"  {f_s} line {i + 2}: {d['word_class']} draw {d['draw']}: {d['segment']} frame {d['frame_idx']} repeat "
              f"{d['repeat']} ({d['weather']}, {d['time_of_day']}; score {d['score']:g}); words: main_hazard "
              f"[{d['words_in_main_hazard']}], reason [{d['words_in_reason']}]\n      main_hazard: {d['main_hazard']!r}\n"
              f"      reason: {d['reason']!r}")

    # ============================================================================ 3 summary
    print('\n' + '=' * 110 + '\n3. summary per condition x frame set')
    for sub in SUBS:
        for c in CONDS:
            row = dict(frames=sub, condition=c)
            for term in SUMMARY_TERMS:
                d = next(x for x in rows_coef if x['frames'] == sub and x['condition'] == c and x['fit'] == 'e'
                         and x['term'] == term)
                row.update({f'e_{term}': d['coef'], f'e_{term}_lo': d['coef_lo'], f'e_{term}_hi': d['coef_hi'],
                            f'e_{term}_nboot': d['nboot']})
            for f in FITS:
                d = next(x for x in rows_fit if x['frames'] == sub and x['condition'] == c and x['fit'] == f)
                row.update({f'{f}_{k}': d[k] for k in ('r2_in', 'r2_in_lo', 'r2_in_hi', 'r2_oos', 'r2_oos_lo',
                                                         'r2_oos_hi')})
            for g in DIFF_G:
                ds = [x for x in rows_diff if x['frames'] == sub and x['condition'] == c and x['grouping'] == g]
                clear = [f"{x['ssm']}{'+' if x['side'] == 'above 0' else '-'}" for x in ds
                         if x['side'] in ('above 0', 'below 0')]
                big = max(ds, key=lambda x: abs(x['diff']))
                row.update({f'{g}_minus_all_without_0': ' '.join(clear) if clear else 'none',
                            f'{g}_minus_all_largest_ssm': big['ssm'], f'{g}_minus_all_largest': big['diff']})
            rows_sum.append(row)
    f_sum = write('exploratory_sec43_more_summary.csv', rows_sum)
    for i, d in enumerate(rows_sum):
        print(f"  {f_sum} line {i + 2}: {d['frames']} {d['condition']}")
        print('     (e) ' + '; '.join(f"{t} {r(d[f'e_{t}'])} [{r(d[f'e_{t}_lo'])}, {r(d[f'e_{t}_hi'])}] (nboot "
                                      f"{d[f'e_{t}_nboot']})" for t in SUMMARY_TERMS))
        print('     ' + '; '.join(f"({f}) R2 in {r(d[f'{f}_r2_in'])} [{r(d[f'{f}_r2_in_lo'])}, {r(d[f'{f}_r2_in_hi'])}], "
                                  f"out {r(d[f'{f}_r2_oos'])} [{r(d[f'{f}_r2_oos_lo'])}, {r(d[f'{f}_r2_oos_hi'])}]"
                                  for f in FITS))
        print('     ' + '; '.join(f"{g} - all: without 0 {d[f'{g}_minus_all_without_0']}, largest |diff| "
                                  f"{d[f'{g}_minus_all_largest_ssm']} {r(d[f'{g}_minus_all_largest'])}" for g in DIFF_G))
    print(f'\ndone in {time.time() - t0:.0f} s; checks not met: {len(failed)}' + (f': {failed}' if failed else ''))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
