"""EXPLORATORY (user, 2026-11-11; the readings below were fixed before the run and are not changed after it): Qwen3.6-27B's
risk run with one true quantity told in the question (the speed run 31628444: "Your car's speed is <v> meters per
second."; the acceleration run 31681661 alike) against the main 27B run with thinking on (base). Input: one subfolder
of analyze_vlm_ssm.py's output for the two models (main_1084 or sensitivity_1123, run with analyze_weather.sh's
arguments; kamiak/told_risk_analysis.sbatch). Only numbers are read: no image is opened (the requests are rebuilt as
text; run_vlm.build keeps the image paths and does not read them). New tables: <prefix>*.csv in the subfolder, the
report <prefix>report.out beside them. The other tables of the subfolder are analyze_vlm_ssm.py's, unchanged.

 0  checks. Frames: the same as after_qwen3.5-4b/<sub>/ssm_target_frames.csv. Base: every row of the base model in
    agreement.csv, ssm_differences.csv, condition_summary.csv, vlm_frame_scores.csv, agreement_by_segment.csv and
    vlm_requests_tidy.csv the same text, in the same order, as in after_qwen3.5-4b_weather/<sub>. Resamples: seed,
    number and segment list as this subfolder's params.txt, the drawn segments' sha256 the quantities analysis's. The
    told number: the csv the job read has the truth_main.csv value of every frame; every request of the 1,123 frames x
    5 conditions is rebuilt with the job's own wrapper (its make_build and text function) from vlm_inputs' target lists,
    and the number in its sentence must equal the truth value rounded half up to 1 decimal (Decimal; "-0.0" written
    "0.0" as the acceleration wrapper does); every reply's prompt tokens minus base's for the condition must be one
    value for all numbers of the same length (the sentence; one token per character of the number).
 1  SSM agreement: in each of the 2,000 resamples of whole segments, drawn as the risk analysis draws them
    (default_rng(seed), one choice over the sorted segments per resample, every drawn copy its own segment),
    analyze_vlm_ssm.all_stats for both models in the 5 conditions (groupings all, weather, time_of_day, segment); new -
    base of tau_b_all (= tau_b_conflict: every frame has every SSM) per condition x grouping x SSM, 2.5 / 97.5 %
    percentiles (numpy, linear), none when fewer than 90 % are defined; as the model differences of 2026-10-27
    (analyze_vlm_ssm.py's MODEL_PAIRS does not hold this pair; the program is not changed). Checks: both models' own
    tau-b, intervals and nboot the same text as agreement.csv; the point differences = agreement.csv's values subtracted.
 2  ranking: single answers (vlm_requests_tidy.csv risk, forced answers included as there); tau-b 'all' between two
    repeats over the frames where both have an answer (scipy kendalltau variant b, as analyze_vlm_ssm.repeat_stability):
    base x base 10 pairs, new x new 10, base x new 25; D = mean of the 25 - mean of the 20 within; the tau-b of the two
    models' 5-repeat mean scores (score_mean); per frame new - base of score_mean: mean, mean |d|, quartiles (numpy,
    linear). Intervals of D, of the three means and of the mean-score tau-b from the same resamples, every tau-b taken
    anew on the drawn frames. Check: the mean of each model's 10 within pairs = condition_summary.csv retest_tau_b_mean.
 3  score_mean against the true ego speed and the true distance (truth_main.csv v_ego, distance): tau-b all,
    time_of_day, segment with analyze_vlm_ssm.stats_by_grouping as E2 (analyze_quantities.py), both models, intervals,
    new - base. Check: base = the quantities analysis's exploratory_e2_risk_vs_distance_speed.csv, the same text.
 4  the rule analysis's K (analyze_rules.py: ranks / frames of the true distance, ego speed, closing speed = ego - lead,
    ego and lead acceleration; dependent the rank of score_mean; least squares with an intercept; the null-space rule)
    per model x condition: coefficients and in-sample R2 (ranks taken anew and refitted in every resample),
    leave-one-segment-out R2 (analyze_rules --oos; its interval adds the drawn segments' sums of squares, no
    refitting), new - base of every coefficient (both refitted on the same resample). Check: base = the rule analysis's
    exploratory_rule_coefficients.csv, exploratory_rule_r2.csv and exploratory_oos_r2.csv (fit K), the same text.
    Beside it, because the user's list read "lead speed" (Claude's addition, not used by the readings): K_lead, the same
    with the lead speed in place of the closing speed, coefficients, in-sample R2 and differences.
 5  description per model x condition: single answers (mean, SD ddof 1, distinct values, the 5 most frequent with
    counts and shares); repeat stability copied from condition_summary.csv with the range of the 10 retest tau-b;
    answers equal to the told number rounded half up to an integer (base on the same frames for reference); answers
    whose reason holds one of --words (case-insensitive, whole words: no letter or digit right before or after), and 5
    of them per model drawn with numpy default_rng(--sample_seed) from all conditions (ordered by condition, segment,
    frame, repeat), printed as text; the frames in 5 classes of the told quantity (np.quantile 0.2, 0.4, 0.6, 0.8;
    x <= q1, q1 < x <= q2, ...): its range, frames, segments, per model the mean score and mean rank (score_mean's rank
    / frames, ranks taken anew in each resample) with intervals, and the mean per-frame difference with its interval;
    one row per frame in <prefix>frames.csv (segment, frame_idx, the told quantity's truth, score_mean of both models in
    the 5 conditions).
Readings (main: sep_camera, main_1084, 'all'; a conclusion is said for every condition only when at least 4 of the 5
agree; the sensitivity subfolder is reported alike):
  2: D's interval wholly below 0 = telling the number changed the ranking beyond its own repeat-to-repeat
     differences; containing 0 = no change beyond those differences seen.
  3: the ego-speed tau-b difference wholly above 0 = moved toward "faster = riskier"; the new run's own tau-b wholly
     above 0 = became "faster = riskier"; wholly below 0 = still "slower = higher score".
  1: per SSM, the difference wholly above 0 = agreement better, wholly below 0 = worse, containing 0 = no change seen.
     Expected beforehand if the speed is used as a kinematic quantity: THW, PSD, PICUD, CI better, TTC and DRAC little
     changed; all nine reported.
  4: the ego-speed coefficient difference (K) wholly above 0 = with the other quantities fixed, the speed's direction
     in the score turned positive.
usage: python -I analyze_told_quantity.py <subfolder> <new model> <base model> <reasons csv of new> <reasons csv of
       base> <wrapper .py the job ran> <told csv the job read> <told column> --words w1,w2 --prefix exploratory_speed_
       [--workers N]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import itertools  # noqa: E402
import multiprocessing as mp  # noqa: E402
import re  # noqa: E402
import string  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from decimal import ROUND_HALF_UP, Decimal  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_rules as ar  # noqa: E402
import analyze_vlm_ssm as av  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDS = av.CONDITIONS
G1 = ['all', 'weather', 'time_of_day', 'segment']
G3 = ['all', 'time_of_day', 'segment']
E2_VARS = {'true_ego_speed': 'v_ego', 'true_distance': 'distance'}
K_LEAD = ['true_distance', 'true_ego_speed', 'true_lead_speed', 'true_ego_accel', 'true_lead_accel']
EXPECT_BETTER = ['THW', 'PSD', 'PICUD', 'CI']
EXPECT_SAME = ['TTC', 'DRAC']
TOL = 1e-12
_WORK = {}


def _chunk(bs):
    return np.array([_WORK['one'](b) for b in bs])


def txt(x):
    """A value as pandas writes it in the analyses' csv files ('' for NaN)."""
    return '' if x is None or (isinstance(x, float) and np.isnan(x)) else repr(float(x))


def r(x):
    return 'nan' if x is None or (isinstance(x, float) and np.isnan(x)) else repr(float(x))


def side(lo, hi):
    if lo is None or np.isnan(lo):
        return 'no interval'
    return 'above 0' if lo > 0 else ('below 0' if hi < 0 else 'contains 0')


def half_up(v, places):
    s = str(Decimal(float(v)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))
    return '0.0' if s == '-0.0' else s


def csv_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        for d in rd:
            yield rd.line_num, d


def main():
    ap = argparse.ArgumentParser()
    for nm in ('sub', 'new', 'base', 'reasons_new', 'reasons_base', 'wrapper', 'told_csv', 'told_column'):
        ap.add_argument(nm)
    ap.add_argument('--words', required=True, help='comma-separated words or phrases looked for in the reasons')
    ap.add_argument('--prefix', required=True, help='e.g. exploratory_speed_')
    ap.add_argument('--waymo', default='/data/lab/cejli/sy/waymo_v143')
    ap.add_argument('--scene_labels', default=os.path.join(REPO, 'labels', 'scene_labels.csv'))
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261021)
    ap.add_argument('--sample_seed', type=int, default=20261112)
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    sub = a.sub.rstrip('/')
    sname = os.path.basename(sub)
    A = os.path.join(a.waymo, 'analysis')
    REF_F = os.path.join(A, 'after_qwen3.5-4b', sname)
    REF_W = os.path.join(A, 'after_qwen3.5-4b_weather', sname)
    QUANT = os.path.join(A, 'quantities_after_27b_think', sname)
    RULE = os.path.join(A, 'rule_explore', sname)
    TRUTH = os.path.join(a.waymo, 'quantities', 'truth_main.csv')
    FSSM = os.path.join(a.waymo, 'results_lead_final_excl', 'frame_ssm_validation_training.csv')
    V = os.path.join(a.waymo, 'vlm_inputs')
    rep_path = os.path.join(sub, a.prefix + 'report.out')
    if os.path.exists(rep_path):
        sys.exit(f'{rep_path} exists already: give a new prefix or folder')
    rep = open(rep_path, 'w', encoding='utf-8')
    t0 = time.time()
    failed = []
    models = [a.base, a.new]

    def say(s=''):
        print(s, flush=True)
        rep.write(s + '\n')
        rep.flush()

    def check(ok, what):
        say(f'  check {"ok" if ok else "NOT MET"}: {what}')
        if not ok:
            failed.append(what)

    def write(name, rows):
        p = os.path.join(sub, a.prefix + name)
        pd.DataFrame(rows).to_csv(p, index=False)
        return a.prefix + name

    say(f'EXPLORATORY (user, 2026-11-11; readings fixed before the run): {a.new} vs {a.base} in {sub}; '
        f'analyze_told_quantity.py run {time.strftime("%Y-%m-%d %H:%M:%S")}')

    # ================================================================================================ 0 checks
    say('\n0. checks')
    frames = pd.read_csv(os.path.join(sub, 'ssm_target_frames.csv'))[['segment', 'frame_idx']] \
        .sort_values(['segment', 'frame_idx']).reset_index(drop=True)
    ref = pd.read_csv(os.path.join(REF_F, 'ssm_target_frames.csv'))[['segment', 'frame_idx']] \
        .sort_values(['segment', 'frame_idx']).reset_index(drop=True)
    n = len(frames)
    check(frames.equals(ref), f'frames: {n} in {frames.segment.nunique()} segments, the same as {REF_F}/ssm_target_frames.csv')
    mi = pd.MultiIndex.from_frame(frames)
    seg_names = frames['segment'].to_numpy()
    seg_list = np.array(sorted(set(seg_names)))
    seg_frames = {s: np.flatnonzero(seg_names == s) for s in seg_list}
    seg_code = np.searchsorted(seg_list, seg_names)
    ids = np.arange(n)
    for t in ('agreement.csv', 'ssm_differences.csv', 'condition_summary.csv', 'vlm_frame_scores.csv',
              'agreement_by_segment.csv', 'vlm_requests_tidy.csv'):
        with open(os.path.join(sub, t), encoding='utf-8') as f:
            mine = f.read().splitlines()
        with open(os.path.join(REF_W, t), encoding='utf-8') as f:
            theirs = f.read().splitlines()
        rm = [x for x in mine[1:] if x.startswith(a.base + ',')]
        rt = [x for x in theirs[1:] if x.startswith(a.base + ',')]
        check(mine[0] == theirs[0] and rm == rt and len(rm) > 0,
              f'{t}: the {len(rm)} rows of {a.base} the same text and order as the {len(rt)} in {REF_W}; same header: '
              f'{mine[0] == theirs[0]}')
    nb_r, seed_r, frames_r, segs_r = ar.read_params_risk(os.path.join(sub, 'params.txt'))
    check((nb_r, seed_r, frames_r) == (a.n_boot, a.seed, n) and list(seg_list) == segs_r,
          f'params.txt: {nb_r} resamples, seed {seed_r}, {frames_r} frames, the same segment list: {list(seg_list) == segs_r}')
    rng = np.random.default_rng(a.seed)
    draws = [rng.choice(seg_list, len(seg_list)) for _ in range(a.n_boot)]
    h = hashlib.sha256()
    for d in draws:
        h.update(np.searchsorted(seg_list, d).astype(np.int32).tobytes())
    nb_q, seed_q, hash_q = ar.read_hash_quantities(os.path.join(QUANT, 'params.txt'))
    check((nb_q, seed_q, hash_q) == (a.n_boot, a.seed, h.hexdigest()),
          f'drawn segments\' sha256 {h.hexdigest()} = {QUANT}/params.txt')
    counts = np.stack([np.bincount(np.searchsorted(seg_list, d), minlength=len(seg_list)) for d in draws])

    # the told number
    truth = pd.read_csv(TRUTH, float_precision='round_trip')
    tcsv = pd.read_csv(a.told_csv, float_precision='round_trip')
    j = tcsv[['segment', 'frame_idx', a.told_column]].merge(
        truth[['segment', 'frame_idx', a.told_column]], on=['segment', 'frame_idx'], how='left', suffixes=('', '_t'),
        validate='one_to_one', indicator=True)
    same_csv = bool((j['_merge'] == 'both').all()) and np.array_equal(j[a.told_column].to_numpy(float),
                                                                      j[a.told_column + '_t'].to_numpy(float))
    check(same_csv and len(tcsv) == 1123, f'{a.told_csv}: {len(tcsv)} frames, every {a.told_column} the same as '
          f'truth_main.csv: {same_csv}')
    spec = importlib.util.spec_from_file_location('told_wrapper', a.wrapper)
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    fmt = getattr(w, 'speed_text', None) or getattr(w, 'accel_text', None)
    field = [f_ for _, f_, _, _ in string.Formatter().parse(w.SENTENCE) if f_][0]
    pat = re.compile('^' + re.escape(w.SENTENCE).replace(re.escape('{' + field + '}'), r'(-?\d+\.\d)') + '\n')
    told_text = {(s, int(k)): fmt(v) for s, k, v in zip(tcsv.segment, tcsv.frame_idx, tcsv[a.told_column])}
    build = w.make_build(told_text)
    tg = pd.read_csv(os.path.join(V, 'grid', 'targets_main.csv'))
    ts = pd.read_csv(os.path.join(V, 'separate', 'targets_main.csv'))
    tt = tg.merge(ts[['segment', 'frame_idx', 'images_by_time']], on=['segment', 'frame_idx'], validate='one_to_one')
    tru = truth.set_index(['segment', 'frame_idx'])[a.told_column]
    n_req = n_ok = n_tie = 0
    bad = []
    for row in tt.itertuples():
        grid_imgs = [os.path.join(V, 'grid', p) for p in row.images.split(';')]
        by_time = [os.path.join(V, 'separate', p) for p in row.images_by_time.split(';')]
        sep_imgs = {cam: [by_time[i * 5 + c] for i in range(5)] for c, cam in enumerate(w.rv.CAMERAS)}
        v = float(tru.loc[(row.segment, int(row.frame_idx))])
        want = half_up(v, 1)
        n_tie += f'{v:.1f}'.replace('-0.0', '0.0') != want
        for cond in CONDS:
            m_ = pat.match(build(cond, grid_imgs, sep_imgs)[-1][1])
            n_req += 1
            ok_ = bool(m_) and m_.group(1) == want
            n_ok += ok_
            if not ok_ and len(bad) < 5:
                bad.append((row.segment, row.frame_idx, cond, m_.group(1) if m_ else None, want))
    check(n_req == 1123 * 5 and n_ok == n_req,
          f'the told number in {n_ok} of {n_req} rebuilt requests (1,123 frames x 5 conditions; the job\'s wrapper '
          f'{a.wrapper}) = truth {a.told_column} rounded half up to 1 decimal{"; e.g. not: " + str(bad) if bad else ""}; '
          f'frames where Python\'s rounding (half to even) would differ: {n_tie}')
    tidy = pd.read_csv(os.path.join(sub, 'vlm_requests_tidy.csv'), low_memory=False,
                       usecols=['model', 'condition', 'segment', 'frame_idx', 'repeat', 'risk', 'forced', 'prompt_tokens'])
    tidy = tidy[tidy.model.isin(models)]
    for c in CONDS:
        pb = sorted(float(x) for x in tidy[(tidy.model == a.base) & (tidy.condition == c)].prompt_tokens.unique())
        q = tidy[(tidy.model == a.new) & (tidy.condition == c)]
        lens = np.array([len(told_text[(s, int(k))]) for s, k in zip(q.segment, q.frame_idx)])
        delta = q.prompt_tokens.to_numpy(float) - (pb[0] if len(pb) == 1 else np.nan)
        by_len = {int(L_): sorted({float(x) for x in delta[lens == L_]}) for L_ in sorted(set(lens))}
        n_len = {int(L_): int((lens == L_).sum()) for L_ in sorted(set(lens))}
        check(len(pb) == 1 and all(len(v_) == 1 for v_ in by_len.values()),
              f'{c}: base prompt tokens {pb}; {a.new} minus base by the length of the number: '
              + ', '.join(f'{L_} characters {by_len[L_]} ({n_len[L_]} replies)' for L_ in by_len))

    # ================================================================================================ inputs
    fs = pd.read_csv(os.path.join(sub, 'vlm_frame_scores.csv'), float_precision='round_trip')
    score, ans = {}, {}
    for m in models:
        for c in CONDS:
            s = fs[(fs.model == m) & (fs.condition == c)].set_index(['segment', 'frame_idx'])['score_mean']
            if len(s) != n or not s.index.isin(mi).all():
                sys.exit(f'{m} {c}: the frames in vlm_frame_scores.csv are not the {n} frames')
            score[(m, c)] = s.reindex(mi).to_numpy(float)
            g = tidy[(tidy.model == m) & (tidy.condition == c)]
            ans[(m, c)] = g.set_index(['segment', 'frame_idx', 'repeat'])['risk'].unstack('repeat') \
                .reindex(index=mi, columns=range(5)).to_numpy(float)
    tr = frames.merge(truth, on=['segment', 'frame_idx'], how='left', validate='one_to_one', indicator=True)
    check((tr['_merge'] == 'both').all(), 'every frame in truth_main.csv')
    cue = {'true_distance': tr['distance'].to_numpy(float), 'true_ego_speed': tr['v_ego'].to_numpy(float),
           'true_lead_speed': tr['v_lead'].to_numpy(float),
           'true_closing_speed': tr['v_ego'].to_numpy(float) - tr['v_lead'].to_numpy(float),
           'true_ego_accel': tr['a_ego'].to_numpy(float), 'true_lead_accel': tr['a_lead'].to_numpy(float)}
    told_true = tr[a.told_column].to_numpy(float)
    ssm_df = av.ssm_table(FSSM, frames)
    ssm = {c: ssm_df[c].to_numpy() for c in ssm_df.columns if c.endswith(('_conflict', '_risk', '_use'))}
    ssm = {k: (v_.astype(float) if k.endswith('_risk') else v_.astype(bool)) for k, v_ in ssm.items()}
    groupings, _ = av.scene_groupings(frames, a.scene_labels, ['weather', 'time_of_day'])
    check(list(groupings) == G1, f'groupings {list(groupings)}')
    qcut = np.quantile(told_true, [0.2, 0.4, 0.6, 0.8])
    cls = 1 + sum((told_true > q_).astype(int) for q_ in qcut)
    pairs2 = [('base', i, 'base', j) for i, j in itertools.combinations(range(5), 2)] + \
             [('new', i, 'new', j) for i, j in itertools.combinations(range(5), 2)] + \
             [('base', i, 'new', j) for i in range(5) for j in range(5)]
    mk = {'base': a.base, 'new': a.new}

    # ================================================================================================ one resample
    def stats(idx, gseg):
        res = {}
        gb = {gn: v_[idx] for gn, v_ in groupings.items()}
        gb['segment'] = gseg
        sb = {k: v_[idx] for k, v_ in ssm.items()}
        ids_b = ids[idx]
        for m in models:                                                   # 1
            for c in CONDS:
                st = av.all_stats(score[(m, c)][idx], sb, ids_b, gb)
                for s in av.SSMS:
                    for gn in G1:
                        res[('t1', m, c, s, gn, 'tau_b_conflict')] = st[s][0][gn][1]
                        res[('t1', m, c, s, gn, 'tau_b_all')] = st[s][0][gn][4]
        cache = {}

        def tau3(x, y, mask):                                              # E2's way (analyze_quantities.py)
            key = mask.tobytes()
            if key not in cache:
                labs = {gn: gb[gn][mask] for gn in G3}
                cache[key] = labs, {gn: av.within_pairs(v_) for gn, v_ in labs.items()
                                    if gn != 'all' and len(np.unique(v_)) > 20}
            labs, prs = cache[key]
            xs, ys, ids_s = x[mask], y[mask], ids_b[mask]
            cs = np.ones(len(xs), bool)
            return {gn: (av.stats_by_grouping(xs, cs, ys, ids_s, labs[gn], prs.get(gn))[3] if len(xs) >= 3 else np.nan)
                    for gn in G3}

        for m in models:                                                   # 3
            for c in CONDS:
                x = score[(m, c)][idx]
                for var in E2_VARS:
                    y = cue[var][idx]
                    for gn, t in tau3(x, y, ~np.isnan(x) & ~np.isnan(y)).items():
                        res[('t3', m, c, var, gn)] = t
        for c in CONDS:                                                    # 2
            X = {'base': ans[(a.base, c)][idx], 'new': ans[(a.new, c)][idx]}
            for m1, i, m2, j_ in pairs2:
                x, y = X[m1][:, i], X[m2][:, j_]
                ok = ~np.isnan(x) & ~np.isnan(y)
                res[('t2', c, m1, i, m2, j_)] = kendalltau(x[ok], y[ok], variant='b').statistic if ok.sum() >= 3 \
                    else np.nan
            xb, xn = score[(a.base, c)][idx], score[(a.new, c)][idx]
            ok = ~np.isnan(xb) & ~np.isnan(xn)
            res[('t2mean', c)] = kendalltau(xb[ok], xn[ok], variant='b').statistic
        for m in models:                                                   # 4
            for c in CONDS:
                x = score[(m, c)][idx]
                sel = ~np.isnan(x)
                y = ar.rank01(x[sel])
                for ver, terms in (('K', ar.K_TRUE), ('K_lead', K_LEAD)):
                    coef, _, r2 = ar.ols(y, [ar.rank01(cue[k][idx][sel]) for k in terms])
                    res[('k_r2', ver, m, c)] = r2
                    for term, b_ in zip(['intercept'] + terms, coef):
                        res[('k', ver, m, c, term)] = b_
        cls_b = cls[idx]                                                   # 5, classes of the told quantity
        for c in CONDS:
            for m in models:
                x = score[(m, c)][idx]
                ok = ~np.isnan(x)
                rk = np.full(len(x), np.nan)
                rk[ok] = ar.rank01(x[ok])
                for q_ in range(1, 6):
                    sel = (cls_b == q_) & ok
                    res[('q_mean', m, c, q_)] = float(x[sel].mean()) if sel.any() else np.nan
                    res[('q_rank', m, c, q_)] = float(rk[sel].mean()) if sel.any() else np.nan
            d_ = score[(a.new, c)][idx] - score[(a.base, c)][idx]
            for q_ in range(1, 6):
                sel = (cls_b == q_) & ~np.isnan(d_)
                res[('q_diff', c, q_)] = float(d_[sel].mean()) if sel.any() else np.nan
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
                if done == len(bs) or done % 250 < len(bs) or done == a.n_boot:
                    print(f'  [resample {done}/{a.n_boot}, {(time.time() - tb) / done * (a.n_boot - done):.0f} s left; '
                          f'{a.workers} workers]', flush=True)
    else:
        for b in range(a.n_boot):
            boot[b] = one(b)
            if b in (0, 9) or (b + 1) % 250 == 0:
                print(f'  [resample {b + 1}/{a.n_boot}, {(time.time() - tb) / (b + 1) * (a.n_boot - b - 1):.0f} s left]',
                      flush=True)

    def ci(k):
        return ar.ci(boot[:, col[k]])

    def ci_arr(arr):
        return ar.ci(arr)

    # ================================================================================================ 1
    say('\n1. SSM agreement: new - base of tau_b_all, per condition x grouping x SSM')
    ag = {(d['model'], d['condition'], d['ssm'], d['grouping']): (ln, d)
          for ln, d in csv_rows(os.path.join(sub, 'agreement.csv')) if d['model'] in models}
    n_own = n_own_ok = n_own_pt = 0
    dpt = 0.0
    rows1 = []
    for m in models:
        for c in CONDS:
            for s in av.SSMS:
                for gn in G1:
                    ln, d = ag[(m, c, s, gn)]
                    for stat in ('tau_b_conflict', 'tau_b_all'):
                        lo, hi, nv = ci(('t1', m, c, s, gn, stat))
                        mine = (txt(point[('t1', m, c, s, gn, stat)]), txt(lo), txt(hi), str(nv))
                        n_own += 1
                        n_own_ok += mine == (d[stat], d[stat + '_lo'], d[stat + '_hi'], d[stat + '_nboot'])
                        n_own_pt += mine[0] == d[stat]
    check(n_own_ok == n_own, f'both models\' own tau-b, intervals and nboot recomputed: {n_own_ok} of {n_own} the same '
          f'text as agreement.csv (point estimates alone: {n_own_pt})')
    for c in CONDS:
        for gn in G1:
            for s in av.SSMS:
                kb, kn = ('t1', a.base, c, s, gn, 'tau_b_all'), ('t1', a.new, c, s, gn, 'tau_b_all')
                lo, hi, nv = ci_arr(boot[:, col[kn]] - boot[:, col[kb]])
                pe = point[kn] - point[kb]
                lb, db = ag[(a.base, c, s, gn)]
                ln_, dn = ag[(a.new, c, s, gn)]
                dpt = max(dpt, abs(pe - (float(dn['tau_b_all']) - float(db['tau_b_all']))))
                rows1.append(dict(condition=c, grouping=gn, ssm=s, stat='tau_b_all', value_base=point[kb],
                                  value_new=point[kn], diff=pe, diff_lo=lo, diff_hi=hi, nboot=nv,
                                  side=side(lo, hi), agreement_lines=f'{ln_} - {lb}'))
    check(dpt <= TOL, f'point differences = agreement.csv\'s two values subtracted: largest |difference| {dpt!r}')
    f1 = write('agreement_differences.csv', rows1)
    for i, d in enumerate(rows1):
        if d['grouping'] == 'all' or d['condition'] == 'sep_camera':
            say(f"  {f1} line {i + 2}: {d['condition']:10s} {d['grouping']:11s} {d['ssm']:6s} base {r(d['value_base'])}, "
                f"new {r(d['value_new'])}; new - base {r(d['diff'])} [{r(d['diff_lo'])}, {r(d['diff_hi'])}] "
                f"(nboot {d['nboot']}; {d['side']}; agreement.csv lines {d['agreement_lines']})")

    # ================================================================================================ 2
    say('\n2. ranking: single answers\' tau-b (all; frames where both repeats have an answer)')
    cs_rows = {(d['model'], d['condition']): (ln, d) for ln, d in csv_rows(os.path.join(sub, 'condition_summary.csv'))
               if d['model'] in models}
    rows2, rows2p = [], []
    for c in CONDS:
        grp = {'base x base': [p_ for p_ in pairs2 if p_[0] == p_[2] == 'base'],
               'new x new': [p_ for p_ in pairs2 if p_[0] == p_[2] == 'new'],
               'base x new': [p_ for p_ in pairs2 if p_[0] != p_[2]]}
        for p_ in pairs2:
            x, y = ans[(mk[p_[0]], c)][:, p_[1]], ans[(mk[p_[2]], c)][:, p_[3]]
            rows2p.append(dict(condition=c, model_1=mk[p_[0]], repeat_1=p_[1], model_2=mk[p_[2]], repeat_2=p_[3],
                               frames=int((~np.isnan(x) & ~np.isnan(y)).sum()), tau_b=point[('t2', c, *p_)]))
        row = dict(condition=c)
        bmean = {}
        for gname, ps in grp.items():
            vals = np.array([point[('t2', c, *p_)] for p_ in ps])
            bm = np.mean(np.stack([boot[:, col[('t2', c, *p_)]] for p_ in ps]), axis=0)
            bmean[gname] = bm
            lo, hi, nv = ci_arr(bm)
            tag = gname.replace(' x ', '_')
            row.update({f'{tag}_pairs': len(ps), f'{tag}_mean': float(np.mean(vals)), f'{tag}_mean_lo': lo,
                        f'{tag}_mean_hi': hi, f'{tag}_min': float(vals.min()), f'{tag}_max': float(vals.max())})
        within = [point[('t2', c, *p_)] for p_ in grp['base x base'] + grp['new x new']]
        between = [point[('t2', c, *p_)] for p_ in grp['base x new']]
        dd = float(np.mean(between) - np.mean(within))
        bd = np.mean(np.stack([boot[:, col[('t2', c, *p_)]] for p_ in grp['base x new']]), axis=0) - \
            np.mean(np.stack([boot[:, col[('t2', c, *p_)]] for p_ in grp['base x base'] + grp['new x new']]), axis=0)
        lo, hi, nv = ci_arr(bd)
        row.update(D=dd, D_lo=lo, D_hi=hi, D_nboot=nv, D_side=side(lo, hi))
        lo, hi, nv = ci(('t2mean', c))
        row.update(mean_score_tau_b=point[('t2mean', c)], mean_score_tau_b_lo=lo, mean_score_tau_b_hi=hi,
                   mean_score_tau_b_nboot=nv)
        dfr = score[(a.new, c)] - score[(a.base, c)]
        dfr = dfr[~np.isnan(dfr)]
        row.update(frame_diff_frames=len(dfr), frame_diff_mean=float(dfr.mean()),
                   frame_diff_abs_mean=float(np.abs(dfr).mean()), frame_diff_q1=float(np.percentile(dfr, 25)),
                   frame_diff_median=float(np.percentile(dfr, 50)), frame_diff_q3=float(np.percentile(dfr, 75)))
        for m, tag in ((a.base, 'base_base'), (a.new, 'new_new')):
            ln, d = cs_rows[(m, c)]
            check(txt(row[f'{tag}_mean']) == d['retest_tau_b_mean'],
                  f'{c} {m}: mean of the 10 within pairs {r(row[f"{tag}_mean"])} = condition_summary.csv line {ln} '
                  f'retest_tau_b_mean {d["retest_tau_b_mean"]}')
        rows2.append(row)
    write('retest_tau_b_pairs.csv', rows2p)
    f2 = write('ranking.csv', rows2)
    for i, d in enumerate(rows2):
        say(f"  {f2} line {i + 2}: {d['condition']}: base x base mean {r(d['base_base_mean'])} [{r(d['base_base_mean_lo'])}, "
            f"{r(d['base_base_mean_hi'])}] range {r(d['base_base_min'])}-{r(d['base_base_max'])}; new x new mean "
            f"{r(d['new_new_mean'])} [{r(d['new_new_mean_lo'])}, {r(d['new_new_mean_hi'])}] range {r(d['new_new_min'])}-"
            f"{r(d['new_new_max'])}; base x new mean {r(d['base_new_mean'])} [{r(d['base_new_mean_lo'])}, "
            f"{r(d['base_new_mean_hi'])}] range {r(d['base_new_min'])}-{r(d['base_new_max'])}")
        say(f"     D {r(d['D'])} [{r(d['D_lo'])}, {r(d['D_hi'])}] (nboot {d['D_nboot']}; {d['D_side']}); tau-b of the mean "
            f"scores {r(d['mean_score_tau_b'])} [{r(d['mean_score_tau_b_lo'])}, {r(d['mean_score_tau_b_hi'])}]; per frame "
            f"new - base: mean {r(d['frame_diff_mean'])}, mean |d| {r(d['frame_diff_abs_mean'])}, q1 {r(d['frame_diff_q1'])}"
            f", median {r(d['frame_diff_median'])}, q3 {r(d['frame_diff_q3'])} ({d['frame_diff_frames']} frames)")

    # ================================================================================================ 3
    say('\n3. score against the true ego speed and distance (E2\'s tau-b), both models and new - base')
    e2 = {(d['condition'], d['variable'], d['grouping']): (ln, d) for ln, d in
          csv_rows(os.path.join(QUANT, 'exploratory_e2_risk_vs_distance_speed.csv')) if d['model'] == a.base}
    rows3, n_e2, n_e2_ok, n_e2_pt = [], 0, 0, 0
    for c in CONDS:
        for var in E2_VARS:
            for gn in G3:
                kb, kn = ('t3', a.base, c, var, gn), ('t3', a.new, c, var, gn)
                vals = {}
                for m, k in ((a.base, kb), (a.new, kn)):
                    lo, hi, nv = ci(k)
                    vals[m] = (point[k], lo, hi, nv)
                    nfr = int((~np.isnan(score[(m, c)]) & ~np.isnan(cue[var])).sum())
                    rows3.append(dict(model=m, condition=c, variable=var, grouping=gn, frames=nfr, tau_b=point[k],
                                      tau_b_lo=lo, tau_b_hi=hi, nboot=nv, side=side(lo, hi)))
                    if m == a.base:
                        ln, d = e2[(c, var, gn)]
                        n_e2 += 1
                        n_e2_ok += (str(nfr), txt(point[k]), txt(lo), txt(hi), str(nv)) == \
                            (d['frames'], d['tau_b'], d['tau_b_lo'], d['tau_b_hi'], d['nboot'])
                        n_e2_pt += (str(nfr), txt(point[k])) == (d['frames'], d['tau_b'])
                lo, hi, nv = ci_arr(boot[:, col[kn]] - boot[:, col[kb]])
                rows3.append(dict(model=f'{a.new} - {a.base}', condition=c, variable=var, grouping=gn, frames=np.nan,
                                  tau_b=point[kn] - point[kb], tau_b_lo=lo, tau_b_hi=hi, nboot=nv, side=side(lo, hi)))
    check(n_e2_ok == n_e2 == 30, f'base: {n_e2_ok} of {n_e2} rows the same text as {QUANT}/'
          f'exploratory_e2_risk_vs_distance_speed.csv (frames and point estimates alone: {n_e2_pt})')
    f3 = write('speed_distance_tau_b.csv', rows3)
    for i, d in enumerate(rows3):
        say(f"  {f3} line {i + 2}: {d['condition']:10s} {d['variable']:14s} {d['grouping']:11s} {d['model']:30s} "
            f"{r(d['tau_b'])} [{r(d['tau_b_lo'])}, {r(d['tau_b_hi'])}] (nboot {d['nboot']}; {d['side']})")

    # ================================================================================================ 4
    say('\n4. the rule analysis\'s K (ranks), both models; new - base of every coefficient')
    rows4c, rows4r = [], []
    rc = {(d['condition'], d['term']): (ln, d) for ln, d in
          csv_rows(os.path.join(RULE, 'exploratory_rule_coefficients.csv'))
          if d['target'] == 'risk' and d['model'] == a.base and d['fit'] == 'K'}
    rr2 = {d['condition']: (ln, d) for ln, d in csv_rows(os.path.join(RULE, 'exploratory_rule_r2.csv'))
           if d['target'] == 'risk' and d['model'] == a.base and d['fit'] == 'K'}
    ro = {d['condition']: (ln, d) for ln, d in csv_rows(os.path.join(RULE, 'exploratory_oos_r2.csv'))
          if d['target'] == 'risk' and d['model'] == a.base and d['fit'] == 'K'}
    n_k = n_k_ok = n_k_pt = 0
    for ver, terms in (('K', ar.K_TRUE), ('K_lead', K_LEAD)):
        for c in CONDS:
            for m in models:
                for term in ['intercept'] + terms:
                    k = ('k', ver, m, c, term)
                    lo, hi, nv = ci(k)
                    rows4c.append(dict(version=ver, model=m, condition=c, term=term, coef=point[k], coef_lo=lo,
                                       coef_hi=hi, nboot=nv, side=side(lo, hi)))
                    if ver == 'K' and m == a.base:
                        ln, d = rc[(c, term)]
                        n_k += 1
                        n_k_ok += (txt(point[k]), txt(lo), txt(hi), str(nv)) == \
                            (d['coef'], d['coef_lo'], d['coef_hi'], d['nboot'])
                        n_k_pt += txt(point[k]) == d['coef']
                kr = ('k_r2', ver, m, c)
                lo, hi, nv = ci(kr)
                row = dict(version=ver, model=m, condition=c, frames=int((~np.isnan(score[(m, c)])).sum()),
                           r2_in=point[kr], r2_in_lo=lo, r2_in_hi=hi, r2_in_nboot=nv)
                if ver == 'K':
                    sel = ~np.isnan(score[(m, c)])
                    y0 = ar.rank01(score[(m, c)][sel])
                    g0 = np.unique(seg_code[sel], return_inverse=True)[1]
                    u0 = np.unique(seg_code[sel])
                    pred = ar.loso_predict(y0, [ar.rank01(cue[k_][sel]) for k_ in terms], g0)
                    sse, sst = np.zeros(len(seg_list)), np.zeros(len(seg_list))
                    sse[u0] = np.bincount(g0, (y0 - pred) ** 2)
                    sst[u0] = np.bincount(g0, (y0 - y0.mean()) ** 2)
                    rb = 1.0 - (counts @ sse) / (counts @ sst)
                    olo, ohi, onv = ar.ci(rb)
                    row.update(r2_oos=float(1.0 - sse.sum() / sst.sum()), r2_oos_lo=olo, r2_oos_hi=ohi, r2_oos_nboot=onv)
                    if m == a.base:
                        ln, d = rr2[c]
                        n_k += 1
                        n_k_ok += (txt(point[kr]), txt(lo), txt(hi), str(nv)) == (d['r2'], d['r2_lo'], d['r2_hi'],
                                                                                  d['nboot'])
                        n_k_pt += txt(point[kr]) == d['r2']
                        ln, d = ro[c]
                        n_k += 1
                        n_k_ok += (txt(point[kr]), txt(row['r2_oos']), txt(olo), txt(ohi), str(onv)) == \
                            (d['r2_in'], d['r2_oos'], d['r2_oos_lo'], d['r2_oos_hi'], d['nboot'])
                        n_k_pt += (txt(point[kr]), txt(row['r2_oos'])) == (d['r2_in'], d['r2_oos'])
                rows4r.append(row)
            for term in ['intercept'] + terms:
                kb, kn = ('k', ver, a.base, c, term), ('k', ver, a.new, c, term)
                lo, hi, nv = ci_arr(boot[:, col[kn]] - boot[:, col[kb]])
                rows4c.append(dict(version=ver, model=f'{a.new} - {a.base}', condition=c, term=term,
                                   coef=point[kn] - point[kb], coef_lo=lo, coef_hi=hi, nboot=nv, side=side(lo, hi)))
    check(n_k_ok == n_k == 5 * 8, f'base K: {n_k_ok} of {n_k} (5 conditions x 6 coefficients, R2, out-of-sample R2) the '
          f'same text as {RULE}/exploratory_rule_coefficients.csv, exploratory_rule_r2.csv, exploratory_oos_r2.csv '
          f'(point estimates alone: {n_k_pt})')
    f4c = write('k_coefficients.csv', rows4c)
    f4r = write('k_r2.csv', rows4r)
    for i, d in enumerate(rows4c):
        if d['version'] == 'K' or d['term'] == 'true_ego_speed':
            say(f"  {f4c} line {i + 2}: {d['version']:6s} {d['condition']:10s} {d['model']:30s} {d['term']:18s} "
                f"{r(d['coef'])} [{r(d['coef_lo'])}, {r(d['coef_hi'])}] (nboot {d['nboot']}; {d['side']})")
    for i, d in enumerate(rows4r):
        say(f"  {f4r} line {i + 2}: {d['version']:6s} {d['condition']:10s} {d['model']:20s} R2 in sample {r(d['r2_in'])} "
            f"[{r(d['r2_in_lo'])}, {r(d['r2_in_hi'])}]" + (f"; out of sample {r(d['r2_oos'])} [{r(d['r2_oos_lo'])}, "
                                                          f"{r(d['r2_oos_hi'])}]" if 'r2_oos' in d else ''))

    # ================================================================================================ 5
    say('\n5. description')
    rows5a, rows5s, rows5n, rows5w, rows5q = [], [], [], [], []
    told_int = {k: int(Decimal(v_).quantize(Decimal(1), rounding=ROUND_HALF_UP)) for k, v_ in told_text.items()}
    ti = np.array([told_int[(s, int(k))] for s, k in zip(frames.segment, frames.frame_idx)])
    for m in models:
        for c in CONDS:
            x = ans[(m, c)]
            v_ = x[~np.isnan(x)]
            vc = pd.Series(v_).value_counts()
            top = sorted(vc.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
            row = dict(model=m, condition=c, answers=len(v_), mean=float(v_.mean()), sd=float(v_.std(ddof=1)),
                       distinct=int(len(vc)))
            for i, (val, cnt) in enumerate(top, 1):
                row.update({f'top{i}_value': float(val), f'top{i}_count': int(cnt), f'top{i}_share': cnt / len(v_)})
            rows5a.append(row)
            ln, d = cs_rows[(m, c)]
            wv = [point[('t2', c, *p_)] for p_ in pairs2 if p_[0] == p_[2] == ('base' if m == a.base else 'new')]
            rows5s.append(dict(model=m, condition=c, condition_summary_line=ln, sd_repeats_mean=d['sd_repeats_mean'],
                               icc_mean_of_k=d['icc_mean_of_k'], icc_single=d['icc_single'],
                               retest_tau_b_mean=d['retest_tau_b_mean'], retest_tau_b_min=float(min(wv)),
                               retest_tau_b_max=float(max(wv))))
            ok = ~np.isnan(x)
            eq = (x == ti[:, None]) & ok
            rows5n.append(dict(model=m, condition=c, answers=int(ok.sum()), equal_to_told_integer=int(eq.sum()),
                               share=float(eq.sum() / ok.sum())))
    fa_ = write('answers.csv', rows5a)
    fs_ = write('stability.csv', rows5s)
    fn_ = write('told_number_answers.csv', rows5n)
    for i, d in enumerate(rows5a):
        say(f"  {fa_} line {i + 2}: {d['model']:20s} {d['condition']:10s} answers {d['answers']}, mean {r(d['mean'])}, sd "
            f"{r(d['sd'])}, distinct {d['distinct']}; most frequent " + ', '.join(
                f"{d[f'top{k}_value']:g} {d[f'top{k}_count']} ({r(d[f'top{k}_share'])})" for k in range(1, 6)
                if f'top{k}_value' in d))
    for i, d in enumerate(rows5s):
        say(f"  {fs_} line {i + 2}: {d['model']:20s} {d['condition']:10s} (condition_summary.csv line "
            f"{d['condition_summary_line']}) sd over repeats {d['sd_repeats_mean']}, ICC(1,5) {d['icc_mean_of_k']}, "
            f"ICC(1) {d['icc_single']}, retest tau-b mean {d['retest_tau_b_mean']}, range {r(d['retest_tau_b_min'])}-"
            f"{r(d['retest_tau_b_max'])}")
    for i, d in enumerate(rows5n):
        say(f"  {fn_} line {i + 2}: {d['model']:20s} {d['condition']:10s} answers equal to the told number rounded to an "
            f"integer: {d['equal_to_told_integer']} of {d['answers']} ({r(d['share'])})")
    words = [w_.strip() for w_ in a.words.split(',') if w_.strip()]
    wpat = re.compile(r'(?<![A-Za-z0-9])(?:' + '|'.join(re.escape(w_) for w_ in words) + r')(?![A-Za-z0-9])',
                      re.IGNORECASE)
    say(f'  reasons: words {words} (case-insensitive, whole words)')
    samples = []
    for m, path in ((a.base, a.reasons_base), (a.new, a.reasons_new)):
        rs = pd.read_csv(path, dtype={'reason': str}, keep_default_na=False)
        rs = rs[rs.model == m]
        rs['frame_idx'] = rs['frame_idx'].astype(int)
        rs = rs[pd.MultiIndex.from_frame(rs[['segment', 'frame_idx']]).isin(mi)].copy()
        rs['hit'] = rs['reason'].map(lambda t_: bool(wpat.search(t_)))
        rs['has_reason'] = rs['reason'] != ''
        rs['answered'] = rs['risk'] != ''
        for c in CONDS + ['all conditions']:
            g = rs if c == 'all conditions' else rs[rs.condition == c]
            rows5w.append(dict(model=m, condition=c, answers=int(g.answered.sum()), with_reason=int(g.has_reason.sum()),
                               with_a_word=int((g.hit & g.has_reason).sum()),
                               share=float((g.hit & g.has_reason).sum() / g.has_reason.sum())))
        pool = rs[rs.hit & rs.has_reason].assign(ci=lambda t_: t_.condition.map(CONDS.index)) \
            .sort_values(['ci', 'segment', 'frame_idx', 'repeat']).reset_index(drop=True)
        pick = np.random.default_rng(a.sample_seed).choice(len(pool), size=min(5, len(pool)), replace=False)
        for k_, i_ in enumerate(pick, 1):
            p_ = pool.iloc[int(i_)]
            samples.append(dict(model=m, draw=k_, segment=p_['segment'], frame_idx=int(p_['frame_idx']),
                                condition=p_['condition'], repeat=int(p_['repeat']), forced=p_['forced'],
                                reason=p_['reason']))
    fw_ = write('reason_words.csv', rows5w)
    fsm = write('reason_samples.csv', samples)
    for i, d in enumerate(rows5w):
        say(f"  {fw_} line {i + 2}: {d['model']:20s} {d['condition']:14s} reasons with a word {d['with_a_word']} of "
            f"{d['with_reason']} ({r(d['share'])}); answers {d['answers']}")
    for i, d in enumerate(samples):
        say(f"  {fsm} line {i + 2}: {d['model']} draw {d['draw']}: {d['segment']} frame {d['frame_idx']} {d['condition']} "
            f"repeat {d['repeat']}: {d['reason']!r}")
    for c in CONDS:
        for q_ in range(1, 6):
            sel = cls == q_
            row = dict(condition=c, told_class=q_, told_min=float(told_true[sel].min()), told_max=float(told_true[sel].max()),
                       frames=int(sel.sum()), segments=len(set(seg_names[sel])))
            for m, tag in ((a.base, 'base'), (a.new, 'new')):
                for what in ('mean', 'rank'):
                    k = (f'q_{what}', m, c, q_)
                    lo, hi, nv = ci(k)
                    row.update({f'{tag}_{what}': point[k], f'{tag}_{what}_lo': lo, f'{tag}_{what}_hi': hi})
            k = ('q_diff', c, q_)
            lo, hi, nv = ci(k)
            row.update(diff_mean=point[k], diff_mean_lo=lo, diff_mean_hi=hi, nboot=nv)
            rows5q.append(row)
    fq_ = write('told_classes.csv', rows5q)
    say(f'  classes of the true {a.told_column}: np.quantile 0.2-0.8 = {[float(q_) for q_ in qcut]}')
    for i, d in enumerate(rows5q):
        say(f"  {fq_} line {i + 2}: {d['condition']:10s} class {d['told_class']} ({r(d['told_min'])}-{r(d['told_max'])}; "
            f"{d['frames']} frames, {d['segments']} segments): base mean {r(d['base_mean'])} [{r(d['base_mean_lo'])}, "
            f"{r(d['base_mean_hi'])}] rank {r(d['base_rank'])} [{r(d['base_rank_lo'])}, {r(d['base_rank_hi'])}]; new mean "
            f"{r(d['new_mean'])} [{r(d['new_mean_lo'])}, {r(d['new_mean_hi'])}] rank {r(d['new_rank'])} "
            f"[{r(d['new_rank_lo'])}, {r(d['new_rank_hi'])}]; new - base {r(d['diff_mean'])} [{r(d['diff_mean_lo'])}, "
            f"{r(d['diff_mean_hi'])}]")
    fr = frames.copy()
    fr[f'true_{a.told_column}'] = told_true
    for c in CONDS:
        for m in models:
            fr[f'{m}_{c}'] = score[(m, c)]
    fr.to_csv(os.path.join(sub, a.prefix + 'frames.csv'), index=False)
    say(f'  {a.prefix}frames.csv: {len(fr)} rows, columns {list(fr.columns)}')

    # ================================================================================================ readings
    say('\nREADINGS (fixed before the run; main: sep_camera, "all"; "every condition" only when at least 4 of 5 agree)')

    def agree(get):
        main_ = get('sep_camera')
        same = [c for c in CONDS if get(c) == main_]
        return main_, same

    def r1(s):
        def get(c):
            d = next(x for x in rows1 if x['condition'] == c and x['grouping'] == 'all' and x['ssm'] == s)
            return {'above 0': 'better', 'below 0': 'worse', 'contains 0': 'no change seen'}.get(d['side'], d['side'])
        return get
    for s in av.SSMS:
        main_, same = agree(r1(s))
        exp = 'expected better' if s in EXPECT_BETTER else ('expected little change' if s in EXPECT_SAME else 'no expectation')
        say(f'  1 {s:6s}: sep_camera {main_} ({exp}); the same in {len(same)} of 5 conditions {same}'
            + ('; every condition' if len(same) >= 4 else ''))

    def r2(c):
        d = next(x for x in rows2 if x['condition'] == c)
        return {'below 0': 'changed the ranking beyond its own repeat-to-repeat differences',
                'contains 0': 'no change beyond the repeat-to-repeat differences seen',
                'above 0': 'between runs closer than within (no reading fixed)'}.get(d['D_side'], d['D_side'])
    main_, same = agree(r2)
    say(f'  2 D: sep_camera {main_}; the same in {len(same)} of 5 {same}' + ('; every condition' if len(same) >= 4 else ''))

    def r3d(c):
        d = next(x for x in rows3 if x['condition'] == c and x['variable'] == 'true_ego_speed' and x['grouping'] == 'all'
                 and x['model'] == f'{a.new} - {a.base}')
        return 'moved toward faster = riskier' if d['side'] == 'above 0' else f'not moved toward it ({d["side"]})'

    def r3o(c):
        d = next(x for x in rows3 if x['condition'] == c and x['variable'] == 'true_ego_speed' and x['grouping'] == 'all'
                 and x['model'] == a.new)
        return {'above 0': 'became faster = riskier', 'below 0': 'still slower = higher score'}.get(
            d['side'], f'neither ({d["side"]})')
    for nm, fn in (('3 ego-speed tau-b difference', r3d), ('3 the new run\'s own ego-speed tau-b', r3o)):
        main_, same = agree(fn)
        say(f'  {nm}: sep_camera {main_}; the same in {len(same)} of 5 {same}' + ('; every condition' if len(same) >= 4 else ''))

    def r4(c):
        d = next(x for x in rows4c if x['version'] == 'K' and x['condition'] == c and x['term'] == 'true_ego_speed'
                 and x['model'] == f'{a.new} - {a.base}')
        return 'turned positive' if d['side'] == 'above 0' else f'not turned positive ({d["side"]})'
    main_, same = agree(r4)
    say(f'  4 ego-speed coefficient difference (K): sep_camera {main_}; the same in {len(same)} of 5 {same}'
        + ('; every condition' if len(same) >= 4 else ''))
    say(f'\ndone in {time.time() - t0:.0f} s; checks not met: {len(failed)}' + (f': {failed}' if failed else ''))
    rep.close()
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
