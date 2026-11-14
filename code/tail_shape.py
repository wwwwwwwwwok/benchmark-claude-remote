"""EXPLORATORY (user, 2026-11-14; the plan is in CLAUDE.md "尾部形状", recorded and committed before this program was
written; it is not changed after the results): the shape of the upper (risky) tail of Qwen3.6-27B's risk score and of
each of the 9 SSMs over the main result's 1,084 frames (238 segments), sep_camera. Only the shape of each distribution
is described, not whether the risky frames are the same ones. CPU only; no image is opened.
Data: 27B = score_mean of the risk analysis (after_gemma-4-31b-it/main_1084/vlm_frame_scores.csv, the 4.1 score); its
5 single scores (vlm_requests_tidy.csv risk, forced answers included, repeats 0-4) are each a reference quantity. SSMs:
the raw values in the main analysis's ssm_target_frames.csv (the same values as its *_risk columns).
Orientations (every quantity larger = riskier; 27B unchanged in both): negated (main) = the *_risk columns: TTC, THW,
MTTC, PICUD, PSD, 2D-TTC (TTC2D), ACT negated, DRAC and CI unchanged; reciprocal (sensitivity): TTC, THW, MTTC, PSD,
TTC2D, ACT -> 1/x, PICUD negated (it can be negative), DRAC and CI unchanged. The 6 reciprocal ones must be > 0 on every
frame: otherwise the program stops (exit 1) before any fit and writes the minima.
Tail: the top 10 % (main), 5 % and 20 % (sensitivity): threshold u = numpy's linear percentile 90 / 95 / 80 of the
quantity; exceedances y = x - u for x > u (the threshold itself not counted), sorted ascending; generalised Pareto with
location 0 fitted by maximum likelihood (scipy.stats.genpareto.fit(y, floc=0); shape c = xi) and by probability-weighted
moments (Hosking & Wallis 1987: a_r = mean((1 - p_j)^r y_(j)), plotting positions p_j = (j - 0.35) / n; k = a0 / (a0 -
2 a1) - 2, scale = 2 a0 a1 / (a0 - 2 a1), xi = -k; failed when a0 - 2 a1 <= 0). A fit fails when it raises or gives a
non-finite shape or a non-positive scale; maximum-likelihood fits that raised a warning are counted (and kept). Quantile
ratio (no fit): (P95 - P50) / (P75 - P50), numpy linear percentiles; undefined when P75 = P50. References: normal
2.4387 (z95 / z75), exponential 3.3219 (ln 10 / ln 2; the GPD with xi = 0).
Differences: 27B (the 5-score mean) minus each SSM: xi (both methods, each threshold) and the quantile ratio.
Intervals: the main analysis's 2,000 resamples of whole segments (default_rng(20261021), drawn segments' sha256
48bc1f41...); in each, every frame of the drawn segments (as often as drawn), thresholds and fits redone; 2.5 / 97.5 %
percentiles (numpy linear) over the resamples where the value is defined; failed fits counted apart (missing), fits with
xi < -1 counted apart and kept in the interval; a difference is missing when either side is.
Readings (the plan's, fixed before the run): xi interval wholly above 0 = long tail, below 0 = a ceiling, containing 0 =
cannot tell (maybe exponential); a 27B - SSM interval without 0 = different tail shapes; a maximum-likelihood xi below
-0.5 is reported as a number only (conclusions from the PWM xi and the quantile ratio). Claude's reading of the quantile
ratio for comparing it with xi (not in the plan; flagged in the outputs): its interval against the exponential's 3.3219
(the xi = 0 reference), and against the normal's 2.4387.
Checks (exit 1 when one is not met): 1,084 frames, 238 segments; the mean of the 5 single scores = score_mean frame by
frame; every frame has every SSM; the 9 negated columns = ssm_target_frames.csv *_risk frame by frame; the resamples as
the main analysis; the user's local maximum-likelihood top-10 % xi (negated) to 3 decimals. The run script compares a
second run byte for byte.
usage: python -I tail_shape.py <waymo_v143 folder> <out_dir> [--workers N]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import hashlib  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
from scipy.stats import genpareto, norm  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_rules as ar  # noqa: E402
import analyze_vlm_ssm as av  # noqa: E402

MODEL, COND = 'qwen3.6-27b', 'sep_camera'
N_BOOT, BOOT_SEED = 2000, 20261021
HASH = '48bc1f413f9d5b99da57ff4edde7bbb4eb378c5a9343c44ccad4e0165466604c'
TOPS = [(10, 90.0), (5, 95.0), (20, 80.0)]                       # (top %, percentile); 10 % is the main threshold
RECIP = ['TTC', 'THW', 'MTTC', 'PSD', 'TTC2D', 'ACT']
PWM_A = 0.35
QR_NORMAL = float(norm.ppf(0.95) / norm.ppf(0.75))
QR_EXP = float(np.log(10) / np.log(2))
LOCAL = {'27B mean': -0.524, 'TTC': -0.300, 'THW': -1.043, 'DRAC': -0.363, 'MTTC': -0.168, 'CI': 0.423,
         'PICUD': -0.158, 'PSD': -0.984, 'TTC2D': -0.276, 'ACT': -0.420}
LOCAL_TOL = 0.0005 + 1e-12
B27 = ['27B mean'] + [f'27B repeat {r}' for r in range(5)]
G = {}                                                            # data for the worker processes (fork)


# ---------------------------------------------------------------------------------------------------- fits
def ml_fit(y):
    """Maximum likelihood, location 0: (xi, scale, failed, warned)."""
    if len(y) == 0:
        return np.nan, np.nan, True, False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try:
            c, _, sc = genpareto.fit(y, floc=0)
        except Exception:                                         # noqa: BLE001 -- any failure is counted
            return np.nan, np.nan, True, len(w) > 0
    if not (np.isfinite(c) and np.isfinite(sc) and sc > 0):
        return np.nan, np.nan, True, len(w) > 0
    return float(c), float(sc), False, len(w) > 0


def pwm_fit(y):
    """Probability-weighted moments (Hosking & Wallis 1987), location 0, y sorted ascending: (xi, scale, failed)."""
    n = len(y)
    if n < 2:
        return np.nan, np.nan, True
    p = (np.arange(1, n + 1) - PWM_A) / n
    a0 = float(y.mean())
    a1 = float(((1 - p) * y).mean())
    d = a0 - 2 * a1
    if not d > 0:
        return np.nan, np.nan, True
    k = a0 / d - 2
    sc = 2 * a0 * a1 / d
    if not (np.isfinite(k) and np.isfinite(sc) and sc > 0):
        return np.nan, np.nan, True
    return float(-k), float(sc), False


def qratio(x):
    q50, q75, q95 = np.percentile(x, [50, 75, 95])
    return float((q95 - q50) / (q75 - q50)) if q75 > q50 else np.nan


def quant_stats(x):
    out = {}
    for top, q in TOPS:
        u = float(np.percentile(x, q))
        y = np.sort(x[x > u] - u)
        xi, sc, _, warned = ml_fit(y)
        pxi, psc, _ = pwm_fit(y)
        out.update({(top, 'u'): u, (top, 'n'): float(len(y)), (top, 'ml_xi'): xi, (top, 'ml_scale'): sc,
                    (top, 'ml_warn'): float(warned), (top, 'pwm_xi'): pxi, (top, 'pwm_scale'): psc})
    out[('qr',)] = qratio(x)
    return out


def all_stats(idx):
    res = {}
    for name, v in G['vals'].items():
        for k, x in quant_stats(v[idx]).items():
            res[(name,) + k] = x
    return res


def one(b):
    idx = np.concatenate([G['seg_frames'][s] for s in G['draws'][b]])
    rr = all_stats(idx)
    return np.array([rr[k] for k in G['keys']], float)


def ci(v):
    ok = ~np.isnan(v)
    if not ok.any():
        return np.nan, np.nan, 0, int((~ok).sum())
    lo, hi = np.percentile(v[ok], [2.5, 97.5])
    return float(lo), float(hi), int(ok.sum()), int((~ok).sum())


def side(lo, hi, ref=0.0, fmt='g'):
    if np.isnan(lo):
        return 'no interval'
    r = format(ref, fmt)
    return f'above {r}' if lo > ref else (f'below {r}' if hi < ref else f'contains {r}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('waymo')
    ap.add_argument('out')
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    L, O = a.waymo, a.out
    A = os.path.join(L, 'analysis', 'after_gemma-4-31b-it', 'main_1084')
    if os.path.exists(os.path.join(O, 'exploratory_tail_params.txt')):
        sys.exit(f'{O}/exploratory_tail_params.txt exists already: give a new folder')
    os.makedirs(O, exist_ok=True)
    t0 = time.time()
    failed, rows_ck = [], []

    def check(ok, what):
        print(f'  check {"ok" if ok else "NOT MET"}: {what}', flush=True)
        rows_ck.append(dict(check=what, ok=bool(ok)))
        if not ok:
            failed.append(what)

    def W(name, rows):
        pd.DataFrame(rows).to_csv(os.path.join(O, name), index=False)

    with open(os.path.abspath(__file__), 'rb') as f:
        prog_sha = hashlib.sha256(f.read()).hexdigest()
    print(f'tail shape, {MODEL} {COND} vs the 9 SSMs (user, 2026-11-14; EXPLORATORY); run {time.strftime("%Y-%m-%d %H:%M:%S")}; '
          f'program sha256 {prog_sha}; numpy {np.__version__}, scipy {scipy.__version__}\n\n0. inputs', flush=True)

    # ------------------------------------------------------------------------------------------------- inputs
    p_t = os.path.join(A, 'ssm_target_frames.csv')
    tf = pd.read_csv(p_t, float_precision='round_trip').sort_values(['segment', 'frame_idx']).reset_index(drop=True)
    frames = tf[['segment', 'frame_idx']]
    n = len(frames)
    mi = pd.MultiIndex.from_frame(frames)
    check(n == 1084 and frames.segment.nunique() == 238 and not frames.duplicated().any(),
          f'{p_t}: {n} frames, {frames.segment.nunique()} segments')
    p_fs = os.path.join(A, 'vlm_frame_scores.csv')
    fs = pd.read_csv(p_fs, float_precision='round_trip')
    score = fs[(fs.model == MODEL) & (fs.condition == COND)].set_index(['segment', 'frame_idx'])['score_mean'] \
        .reindex(mi).to_numpy(float)
    p_td = os.path.join(A, 'vlm_requests_tidy.csv')
    td = pd.read_csv(p_td, low_memory=False, usecols=['model', 'condition', 'segment', 'frame_idx', 'repeat', 'risk'],
                     float_precision='round_trip')
    td = td[(td.model == MODEL) & (td.condition == COND)]
    rep = td.pivot(index=['segment', 'frame_idx'], columns='repeat', values='risk').reindex(mi)
    reps = np.stack([rep[r].to_numpy(float) for r in range(5)], axis=1)
    fm = td.groupby(['segment', 'frame_idx'], sort=True)['risk'].mean().reindex(mi).to_numpy(float)
    check(len(td) == 5 * n and list(rep.columns) == [0, 1, 2, 3, 4] and not np.isnan(reps).any()
          and not np.isnan(score).any() and np.array_equal(fm, score),
          f'{p_td}: 5 single scores (repeats 0-4) for every frame; their mean = {p_fs} score_mean frame by frame')
    raw = {s: tf[s].to_numpy(float) for s in av.SSMS}
    check(all(tf[s + '_conflict'].astype(str).str.lower().eq('true').all()
              and tf[s + '_use'].astype(str).str.lower().eq('true').all()
              and np.isfinite(raw[s]).all() for s in av.SSMS), 'every frame has a finite value of every SSM')
    neg = {s: (-raw[s] if s in av.SMALLER_RISKIER else raw[s].copy()) for s in av.SSMS}
    check(all(np.array_equal(neg[s], tf[s + '_risk'].to_numpy(float)) for s in av.SSMS),
          f'the 9 negated columns (TTC, THW, MTTC, PICUD, PSD, TTC2D, ACT negated; DRAC, CI unchanged) = {p_t} *_risk '
          f'frame by frame')
    mins = {s: float(raw[s].min()) for s in RECIP}
    rows_min = [dict(ssm=s, minimum=mins[s], argmin_segment=frames.segment[int(np.argmin(raw[s]))],
                     argmin_frame_idx=int(frames.frame_idx[int(np.argmin(raw[s]))]), all_above_0=bool(mins[s] > 0))
                for s in RECIP]
    print('  minima of the 6 SSMs to be inverted: ' + ', '.join(f'{s} {mins[s]!r}' for s in RECIP), flush=True)
    ok_rec = all(mins[s] > 0 for s in RECIP)
    check(ok_rec, 'the 6 SSMs to be inverted (TTC, THW, MTTC, PSD, TTC2D, ACT) are > 0 on every frame')
    W('exploratory_tail_reciprocal_minima.csv', rows_min)
    if not ok_rec:
        W('exploratory_tail_check.csv', rows_ck)
        sys.exit('stopped: a quantity to be inverted is not > 0 on every frame (see exploratory_tail_reciprocal_minima.csv)')
    rec = {s: 1.0 / raw[s] for s in RECIP}

    vals = {'27B mean': score}
    for r in range(5):
        vals[f'27B repeat {r}'] = reps[:, r]
    for s in av.SSMS:
        vals[f'{s} negated'] = neg[s]
    for s in RECIP:
        vals[f'{s} reciprocal'] = rec[s]
    TRANS = {s: ('negated' if s in av.SMALLER_RISKIER else 'unchanged') for s in av.SSMS}
    ORIENT = {'negated': [(q, q, 'unchanged') for q in B27] + [(s, f'{s} negated', TRANS[s]) for s in av.SSMS],
              'reciprocal': [(q, q, 'unchanged') for q in B27] +
                            [(s, f'{s} reciprocal', 'reciprocal') if s in RECIP else
                             (s, f'{s} negated', 'negated (a reciprocal is not meaningful: PICUD can be negative)'
                              if s == 'PICUD' else 'unchanged') for s in av.SSMS]}

    # ------------------------------------------------------------------------------------------------- resamples
    seg_names = frames.segment.to_numpy()
    seg_list = np.array(sorted(set(seg_names)))
    nb_r, seed_r, frames_r, segs_r = ar.read_params_risk(os.path.join(A, 'params.txt'))
    check((nb_r, seed_r, frames_r) == (N_BOOT, BOOT_SEED, n) and list(seg_list) == segs_r,
          f'{A}/params.txt: {nb_r} resamples, seed {seed_r}, {frames_r} frames, the same {len(seg_list)} segments')
    rng = np.random.default_rng(BOOT_SEED)
    draws = [rng.choice(seg_list, len(seg_list)) for _ in range(N_BOOT)]
    hh = hashlib.sha256()
    for dr in draws:
        hh.update(np.searchsorted(seg_list, dr).astype(np.int32).tobytes())
    check(hh.hexdigest() == HASH, f"drawn segments' sha256 {hh.hexdigest()}")
    G.update(vals=vals, draws=draws, seg_frames={s: np.flatnonzero(seg_names == s) for s in seg_list})
    point = all_stats(np.arange(n))
    rows_local = []
    for q, v in LOCAL.items():
        name = q if q == '27B mean' else f'{q} negated'
        ours = point[(name, 10, 'ml_xi')]
        rows_local.append(dict(quantity=q, user_local_ml_xi_top10=v, this_program=ours, abs_difference=abs(ours - v),
                               same_to_3_decimals=bool(abs(ours - v) <= LOCAL_TOL)))
    n_same = sum(r['same_to_3_decimals'] for r in rows_local)
    check(n_same == len(LOCAL), f"the user's local maximum-likelihood top-10 % xi (negated, no resampling) to 3 decimals: "
                                f"{n_same} of {len(LOCAL)} the same (exploratory_tail_local_check.csv)")
    keys = list(point)
    G['keys'] = keys
    col = {k: i for i, k in enumerate(keys)}
    print(f'\n1. resamples: {N_BOOT}, {a.workers} worker(s), {len(keys)} numbers each', flush=True)
    tb = time.time()
    with mp.get_context('fork').Pool(max(a.workers, 1)) as pool:
        boot = np.stack(list(pool.imap(one, range(N_BOOT), chunksize=4)))
    print(f'  done in {time.time() - tb:.0f} s', flush=True)

    def bv(k):
        return boot[:, col[k]]

    # ------------------------------------------------------------------------------------------------- orientation tables
    def xi_side(lo, hi):
        s_ = side(lo, hi)
        return {'above 0': 'above 0 (long tail)', 'below 0': 'below 0 (ceiling)',
                'contains 0': 'contains 0 (cannot tell, maybe exponential)'}.get(s_, s_)

    def qr_side(lo, hi):
        return side(lo, hi, QR_EXP, '.4f'), side(lo, hi, QR_NORMAL, '.4f')

    tables, sides = {}, {}
    for o, items in ORIENT.items():
        rows = []
        for label, name, how in items:
            qv = bv((name, 'qr'))
            qlo, qhi, qnb, qmiss = ci(qv)
            qe, qn = qr_side(qlo, qhi)
            sides[(o, label, 'qr')] = qe
            for top, _ in TOPS:
                mv = bv((name, top, 'ml_xi'))
                pv = bv((name, top, 'pwm_xi'))
                mlo, mhi, mnb, mmiss = ci(mv)
                plo, phi, pnb, pmiss = ci(pv)
                mxi, pxi = point[(name, top, 'ml_xi')], point[(name, top, 'pwm_xi')]
                sides[(o, label, top, 'ml')] = xi_side(mlo, mhi)
                sides[(o, label, top, 'pwm')] = xi_side(plo, phi)
                gap = abs(mxi - pxi) if np.isfinite(mxi) and np.isfinite(pxi) else np.nan
                rows.append(dict(
                    orientation=o, quantity=label, transform=how, top_percent=top,
                    threshold=point[(name, top, 'u')], exceedances=int(point[(name, top, 'n')]),
                    ml_xi=mxi, ml_xi_lo=mlo, ml_xi_hi=mhi, ml_nboot=mnb, ml_failed=mmiss,
                    ml_xi_below_minus1=int((mv[~np.isnan(mv)] < -1).sum()),
                    ml_warned_resamples=int(np.nansum(bv((name, top, 'ml_warn')))),
                    ml_warned_point=bool(point[(name, top, 'ml_warn')]), ml_scale=point[(name, top, 'ml_scale')],
                    ml_side=sides[(o, label, top, 'ml')],
                    ml_xi_below_minus_half=bool(np.isfinite(mxi) and mxi < -0.5),
                    pwm_xi=pxi, pwm_xi_lo=plo, pwm_xi_hi=phi, pwm_nboot=pnb, pwm_failed=pmiss,
                    pwm_xi_below_minus1=int((pv[~np.isnan(pv)] < -1).sum()), pwm_scale=point[(name, top, 'pwm_scale')],
                    pwm_side=sides[(o, label, top, 'pwm')], ml_minus_pwm=mxi - pxi, ml_pwm_gap_over_0_1=bool(gap > 0.1),
                    quantile_ratio=point[(name, 'qr')], qr_lo=qlo, qr_hi=qhi, qr_nboot=qnb, qr_failed=qmiss,
                    qr_vs_exponential=qe, qr_vs_normal=qn))
        tables[o] = rows

    # ------------------------------------------------------------------------------------------------- differences
    rows_d = []
    dsides = {}
    for o, items in ORIENT.items():
        for label, name, how in items:
            if label in B27:
                continue
            for top, _ in TOPS + [(None, None)]:
                for meas in (('ml_xi', 'pwm_xi') if top is not None else ('qr',)):
                    k27 = ('27B mean', top, meas) if top is not None else ('27B mean', 'qr')
                    ks = (name, top, meas) if top is not None else (name, 'qr')
                    dv = bv(k27) - bv(ks)
                    lo, hi, nb, miss = ci(dv)
                    sd = side(lo, hi)
                    mname = {'ml_xi': 'xi (maximum likelihood)', 'pwm_xi': 'xi (PWM)', 'qr': 'quantile ratio'}[meas]
                    dsides[(o, label, top, meas)] = (point[k27] - point[ks], sd)
                    rows_d.append(dict(orientation=o, ssm=label, transform=how, top_percent=top if top else '',
                                       measure=mname, b27=point[k27], ssm_value=point[ks],
                                       b27_minus_ssm=point[k27] - point[ks], lo=lo, hi=hi, nboot=nb, missing=miss,
                                       side=sd, ml_xi_below_minus_half_either=bool(
                                           meas == 'ml_xi' and (point[k27] < -0.5 or point[ks] < -0.5))))

    # ------------------------------------------------------------------------------------------------- two orientations
    def sgn(x):
        return '' if not np.isfinite(x) else ('+' if x > 0 else ('-' if x < 0 else '0'))

    rows_o = []
    for (label, nn, hn), (_, nr, hr) in zip(ORIENT['negated'][6:], ORIENT['reciprocal'][6:]):
        for top, _ in TOPS + [(None, None)]:
            for meas in (('ml_xi', 'pwm_xi') if top is not None else ('qr',)):
                kn = (nn, top, meas) if top is not None else (nn, 'qr')
                kr = (nr, top, meas) if top is not None else (nr, 'qr')
                sn = sides[('negated', label, top, meas[:-3] if meas != 'qr' else 'qr')] if top is not None else \
                    sides[('negated', label, 'qr')]
                sr = sides[('reciprocal', label, top, meas[:-3] if meas != 'qr' else 'qr')] if top is not None else \
                    sides[('reciprocal', label, 'qr')]
                dn, dsn = dsides[('negated', label, top, meas)]
                dr, dsr = dsides[('reciprocal', label, top, meas)]
                rows_o.append(dict(
                    ssm=label, transform_negated=hn, transform_reciprocal=hr, same_values=nn == nr,
                    top_percent=top if top else '',
                    measure={'ml_xi': 'xi (maximum likelihood)', 'pwm_xi': 'xi (PWM)', 'qr': 'quantile ratio'}[meas],
                    negated=point[kn], reciprocal=point[kr], reciprocal_minus_negated=point[kr] - point[kn],
                    side_negated=sn, side_reciprocal=sr, side_changed=sn != sr,
                    b27_minus_ssm_negated=dn, b27_minus_ssm_reciprocal=dr, sign_negated=sgn(dn), sign_reciprocal=sgn(dr),
                    sign_changed=sgn(dn) != sgn(dr), diff_side_negated=dsn, diff_side_reciprocal=dsr,
                    diff_side_changed=dsn != dsr))

    # ------------------------------------------------------------------------------------------------- readings
    def direction(s_):
        return 'above' if s_.startswith('above') else ('below' if s_.startswith('below') else
                                                         ('contains' if s_.startswith('contains') else 'none'))

    rows_r, rows_th = [], []
    for o, items in ORIENT.items():
        for label, name, how in items:
            qe = sides[(o, label, 'qr')]
            for top, _ in TOPS:
                mxi = point[(name, top, 'ml_xi')]
                ms, ps = sides[(o, label, top, 'ml')], sides[(o, label, top, 'pwm')]
                poor = bool(np.isfinite(mxi) and mxi < -0.5)
                used = [ps, qe] if poor else [ms, ps, qe]
                rows_r.append(dict(orientation=o, quantity=label, top_percent=top, ml_xi=mxi, ml_side=ms,
                                   ml_xi_below_minus_half=poor, pwm_side=ps, qr_vs_exponential=qe,
                                   measures_read='PWM and quantile ratio (ML below -0.5: number only)' if poor
                                   else 'ML, PWM and quantile ratio',
                                   agree=len({direction(x) for x in used}) == 1))
            for meas, nm in (('ml', 'xi (maximum likelihood)'), ('pwm', 'xi (PWM)')):
                ss = {top: sides[(o, label, top, meas)] for top, _ in TOPS}
                main_s = ss[10]
                rows_th.append(dict(orientation=o, quantity=label, measure=nm, side_top10=ss[10], side_top5=ss[5],
                                    side_top20=ss[20], same_at_all_three=len(set(ss.values())) == 1,
                                    thresholds_differing_from_top10=';'.join(f'top {t} %' for t in (5, 20)
                                                                             if ss[t] != main_s)))
            if label not in B27:
                for meas, nm in (('ml_xi', 'difference 27B - SSM, xi (maximum likelihood)'),
                                 ('pwm_xi', 'difference 27B - SSM, xi (PWM)')):
                    ss = {top: dsides[(o, label, top, meas)][1] for top, _ in TOPS}
                    rows_th.append(dict(orientation=o, quantity=label, measure=nm, side_top10=ss[10], side_top5=ss[5],
                                        side_top20=ss[20], same_at_all_three=len(set(ss.values())) == 1,
                                        thresholds_differing_from_top10=';'.join(f'top {t} %' for t in (5, 20)
                                                                                 if ss[t] != ss[10])))

    # ------------------------------------------------------------------------------------------------- frames, write
    fr = frames.copy()
    fr['score_mean'] = score
    for r in range(5):
        fr[f'score_repeat{r}'] = reps[:, r]
    for s in av.SSMS:
        fr[f'{s}_negated'] = neg[s]
    for s in RECIP:
        fr[f'{s}_reciprocal'] = rec[s]
    W('exploratory_tail_negated.csv', tables['negated'])
    W('exploratory_tail_reciprocal.csv', tables['reciprocal'])
    W('exploratory_tail_differences.csv', rows_d)
    W('exploratory_tail_orientations.csv', rows_o)
    W('exploratory_tail_readings.csv', rows_r)
    W('exploratory_tail_thresholds.csv', rows_th)
    W('exploratory_tail_local_check.csv', rows_local)
    fr.to_csv(os.path.join(O, 'exploratory_tail_frames.csv'), index=False)
    W('exploratory_tail_check.csv', rows_ck)
    with open(os.path.join(O, 'exploratory_tail_params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'EXPLORATORY: tail shape, {MODEL} {COND} vs the 9 SSMs, main result 1,084 frames (user, 2026-11-14; plan in '
                f'CLAUDE.md "尾部形状")\nrun {time.strftime("%Y-%m-%d %H:%M:%S")}; out_dir {os.path.abspath(O)}\n'
                f'program {os.path.abspath(__file__)} sha256 {prog_sha}; numpy {np.__version__}, scipy {scipy.__version__}\n')
        f.write(f'27B: {p_fs} score_mean; single scores: {p_td} risk (repeats 0-4, forced answers included); SSMs: {p_t} '
                f'(raw values; negated = the *_risk columns)\n')
        f.write('orientations: negated (main): TTC, THW, MTTC, PICUD, PSD, TTC2D, ACT negated, DRAC and CI unchanged; '
                'reciprocal (sensitivity): TTC, THW, MTTC, PSD, TTC2D, ACT -> 1/x, PICUD negated, DRAC and CI unchanged; '
                '27B unchanged in both\nminima of the 6 inverted SSMs: ' +
                ', '.join(f'{s} {mins[s]!r}' for s in RECIP) + '\n')
        f.write('tail: thresholds numpy linear percentiles 90 (top 10 %, main), 95 (top 5 %), 80 (top 20 %); exceedances x - '
                'u for x > u; maximum likelihood scipy.stats.genpareto.fit(y, floc=0), failed when it raises or gives a '
                'non-finite shape or a non-positive scale (warnings counted, fits kept); PWM (Hosking & Wallis 1987) '
                f'with plotting positions (j - {PWM_A}) / n, failed when a0 - 2 a1 <= 0; quantile ratio (P95 - P50) / '
                f'(P75 - P50), undefined when P75 = P50; references normal {QR_NORMAL!r}, exponential {QR_EXP!r}\n')
        f.write(f'intervals: {N_BOOT} resamples of whole segments, default_rng({BOOT_SEED}), drawn segments\' sha256 '
                f'{hh.hexdigest()}; percentiles 2.5 / 97.5 (numpy linear) over the resamples where the value is defined; '
                'failed fits = missing (counted); xi < -1 counted and kept\n')
        f.write("readings: the plan's (section 4); Claude's reading of the quantile ratio for comparing it with xi (not in the "
                f"plan): its interval against the exponential's {QR_EXP:.4f} (xi = 0) and the normal's {QR_NORMAL:.4f}\n")
        f.write(f"user's local check (maximum likelihood, top 10 %, negated): {LOCAL}; tolerance {LOCAL_TOL}\n")

    print('\n' + '=' * 110)
    for r_ in tables['negated']:
        if r_['top_percent'] == 10:
            print(f"  {r_['quantity']:14s} ML xi {r_['ml_xi']:+.4f} [{r_['ml_xi_lo']:+.4f}, {r_['ml_xi_hi']:+.4f}] "
                  f"PWM {r_['pwm_xi']:+.4f} [{r_['pwm_xi_lo']:+.4f}, {r_['pwm_xi_hi']:+.4f}] QR {r_['quantile_ratio']:.4f}")
    print(f'\ndone in {time.time() - t0:.0f} s; checks not met: {len(failed)}')
    for w_ in failed:
        print('  ', w_)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
