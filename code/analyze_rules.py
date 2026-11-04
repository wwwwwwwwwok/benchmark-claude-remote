"""EXPLORATORY: the risk scores' own rule (user, 2026-11-03, decided after seeing the risk and quantities results;
CLAUDE.md "找模型自己的规则", with the user's changes of the evening of 2026-11-03). Aim: to tell "the model cannot read
the quantities" from "the model scores by a rule of its own". Every output is exploratory (exploratory_*); the formal
analyses' tables are only read, none is written.

Frames and resamples: the frames of the risk analysis subfolder (ssm_target_frames.csv: 1,084 main, 1,123 sensitivity),
sorted by segment and frame as there, and its resamples of whole segments (numpy default_rng(--seed), one rng.choice
over the sorted segments per resample, every drawn copy its own segment group). Checked before the resamples are used:
seed, number and segment list against the risk analysis's params.txt (it lists the segments, not the drawn ones), the
sha256 of the drawn segment indices against the quantities analysis's params.txt; after: the risk analysis's tau-b and
intervals of every model x condition x lead-vehicle SSM ('all', 'segment') recomputed from these resamples must equal
its agreement.csv. 95 % percentile intervals, none when fewer than 90 % of the resamples are defined. Groupings 'all'
and 'segment' (pairs of frames of the same segment), tau-b as in analyze_vlm_ssm.py.

Inputs
 - risk scores: <risk>/vlm_frame_scores.csv score_mean (every model x condition);
 - hazard type and camera: <risk>/vlm_requests_tidy.csv; per frame the share of each category among the replies whose
   hazard_type (camera) is one of the prompt's (av.HAZARD_TYPES, av.CAMERAS); other answers ('traffic', ...) and
   replies without one are left out of the shares and of the most frequent category;
 - SSMs: av.ssm_table of <frame_ssm.csv>, "larger = riskier" (the seven lead-vehicle SSMs; every frame has a value);
 - truth: <truth.csv> (quantity_truth.py): distance, ego and lead speed, ego and lead acceleration; closing speed =
   ego speed - lead speed;
 - own estimates (the models with both risk scores and quantity estimates: the three Qwen, Qwen3.6-27B with thinking on
   sep_camera only): <quantities>/quantity_frames.csv estimate (the median of the repeats); own closing speed = own ego
   speed - own lead speed;
 - scene: <scene_labels.csv> (labels/scene_labels.csv): time_of_day and location (Waymo's), weather (the user's, one of
   WEATHER, every segment labelled);
 - road users around: <nearby.csv> of count_nearby.py (pedestrians + cyclists, vehicles within 10, 20, 30, 50 m).

Analyses, per model x condition
 1 exploratory_chain.csv: the models with own estimates x the seven SSMs x 'all', 'segment', copied as text: risk score
   vs true SSM (agreement.csv tau_b_conflict), the model's SSM vs the true SSM and risk score vs the model's SSM
   (perceived_ssm_tau_b.csv tau_b); checked character for character against the sources.
 2 exploratory_cue_tau_b.csv: tau-b of the risk score with each cue: the true six, the counts (pedestrians + cyclists,
   vehicles; four radii), the own six. The rows E2 has (true distance and ego speed, own distance and ego speed) must
   equal exploratory_e2_risk_vs_distance_speed.csv character for character.
 3 exploratory_two_way.csv (no intervals): true distance, ego speed and closing speed cut at their tertiles over these
   frames (np.quantile 1/3 and 2/3: low <= q1 < middle <= q2 < high); distance x ego speed and distance x closing
   speed: frames per cell and the mean rank (rank / frames) of the risk score, and of each SSM.
 4 rule: every variable is replaced by its rank among the frames used divided by their number (ties: mean rank),
   except the 0/1 scene dummies and the hazard type and camera shares, used as they are; the dependent variable is the
   rank of the risk score; least squares with an intercept.
     K = true distance, ego speed, closing speed, ego and lead acceleration; K_own = the same five, own estimates
     S = time of day, location (dummies; reference = the category with the most frames) and weather (a dummy for each
         of the user's five classes but the reference, the class with the most segments; none merged)
     N_R = pedestrians + cyclists and vehicles within R (one R at a time)
     H = the shares of the hazard types and of the cameras (references rear-end, front)
   Fits K, K+S, K+S+N_R, K+S+N_R+H (exploratory_rule_coefficients.csv, exploratory_rule_r2.csv); the R2 each group adds,
   the change of the ego-speed coefficient from K to K+S+N_R, R2(K_own) - R2(K) (exploratory_rule_differences.csv); the
   seven SSMs' ranks on K ("the SSM's rule", in the same tables); within segments (every variable minus its segment
   mean; S drops out) the R2 of K and K+N_R. Lens model (K only), exploratory_lens.csv, per model x condition x SSM:
   Rs = corr(risk, its fit), Re = corr(SSM, its fit), G = corr(the two fits), C = corr(the two residuals), ra =
   corr(risk, SSM); checked: |ra - (G Re Rs + C sqrt(1 - Re^2) sqrt(1 - Rs^2))| < 1e-12, at the point estimates and in
   every resample. In every resample the ranks are taken anew and the fits redone.
   Coefficients that are not identified (user, 2026-11-03), one rule for every fit and every resample: the design
   matrix X (intercept and columns) is decomposed by SVD; singular values at most s_max * max(n, p) * machine epsilon
   (numpy's default rank tolerance) count as zero, and their right singular vectors span the null space of X. A
   coefficient with a component of absolute value above NULL_TOL (1e-8; the vectors have length 1) in that null space
   is undefined (NaN): a column of zeros, two equal columns, a constant column (with the intercept), shares adding up
   to 1 with the intercept... The other coefficients are those of the least-squares solution (unique), the fitted
   values the projection of y on the column space, and R2 = 1 - SSR / SST from it. exploratory_rule_undefined.csv:
   every coefficient undefined at the point estimate and why (all zero / constant / the same as <column> / in a linear
   dependence with <columns>; compared within EXACT_TOL 1e-12); in the resamples, nboot of the coefficient table.
 5 exploratory_hazard_camera_modal.csv, exploratory_hazard_camera_tau_b.csv: each frame's most frequent hazard type and
   camera ('tie' when two or more share the top); their distribution; tau-b of the risk score with the seven SSMs on
   the frames whose most frequent hazard type is rear-end and on the others; the same for the camera front.
 6 exploratory_between_segment.csv: the share of the variance of the risk score's rank between segments (variance of
   the segment means weighted by frames / total variance); the seven SSMs alike.
Checks: exploratory_risk_check.csv and exploratory_params.txt; the run ends with exit code 1 when one is not met.

--workers N: the resamples shared out over N processes (fork); every resample is computed on its own from the draws
made beforehand, so the results do not depend on N. BLAS runs single-threaded (unless the environment says otherwise),
so the fits are the same whatever the machine's thread count.

usage: python -I analyze_rules.py <risk analysis subfolder> <quantities analysis subfolder> <frame_ssm.csv>
       <truth.csv> <scene_labels.csv> <nearby_counts.csv> <out_dir> [--n_boot 2000] [--seed 20261021] [--workers 1]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import hashlib  # noqa: E402
import multiprocessing as mp  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_vlm_ssm as av  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
LEAD7 = ['TTC', 'THW', 'DRAC', 'MTTC', 'CI', 'PICUD', 'PSD']
GROUPINGS = ['all', 'segment']
RADII = [10, 20, 30, 50]
QNAMES = ['distance', 'ego_speed', 'lead_speed', 'closing_speed', 'ego_accel', 'lead_accel']
TRUE_CUES = [f'true_{q}' for q in QNAMES]
OWN_CUES = [f'model_{q}' for q in QNAMES]
NEAR_CUES = [f'{k}_{r}m' for r in RADII for k in ('ped_cyc', 'vehicle')]
K_TRUE = ['true_distance', 'true_ego_speed', 'true_closing_speed', 'true_ego_accel', 'true_lead_accel']
K_OWN = ['model_distance', 'model_ego_speed', 'model_closing_speed', 'model_ego_accel', 'model_lead_accel']
E2_VARS = ['true_distance', 'true_ego_speed', 'model_distance', 'model_ego_speed']
WEATHER = ['dry', 'wet', 'rain', 'fog', 'unclear']         # the user's classes (weather labelling page)
HAZ_REF, CAM_REF = 'rear-end', 'front'
LENS = ['Rs', 'Re', 'G', 'C', 'ra']
IDENTITY_TOL = 1e-12
NULL_TOL = 1e-8                    # a coefficient with a null-space component above this is undefined
EXACT_TOL = 1e-12                  # for naming why: all zero, constant, the same as another column

_WORK = {}                         # the one-resample function, inherited by the worker processes (fork)


def _chunk(bs):
    return np.array([_WORK['one'](b) for b in bs])


# ------------------------------------------------------------------ small pieces (tested on their own)
def rank01(x):
    """Rank among the values divided by their number; ties get the mean rank."""
    return rankdata(x) / len(x)


def ols(y, cols):
    """y on an intercept and the columns (a list of arrays), least squares, with the null-space rule of the docstring
    (item 4): coefficients with a component in the null space of the design matrix are NaN; the others are the
    least-squares solution, the fitted values the projection of y. Returns the coefficients (intercept first, then one
    per column), the fitted values and R2 (1 - SSR / SST; all NaN when y has no variation)."""
    n = len(y)
    X = np.column_stack([np.ones(n)] + list(cols))
    p = X.shape[1]
    sst = float(((y - y.mean()) ** 2).sum())
    if sst <= 0:
        return np.full(p, np.nan), np.full(n, np.nan), np.nan
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    r = int((s > s[0] * max(n, p) * np.finfo(float).eps).sum())
    uty = u[:, :r].T @ y
    beta = vt[:r].T @ (uty / s[:r])
    fit = u[:, :r] @ uty
    if r < p:
        beta = np.where((np.abs(vt[r:]) > NULL_TOL).any(0), np.nan, beta)
    return beta, fit, 1.0 - float(((y - fit) ** 2).sum()) / sst


def why_undefined(cols, names, coef):
    """Why each undefined coefficient of a fit is undefined (names: of the columns, not the intercept), compared within
    EXACT_TOL: a column 'all zero', 'constant (with the intercept)' or 'the same as <column>'; the intercept 'with the
    constant column <name>' when there is one; whatever is left 'in a linear dependence with <the others left>'
    (shares adding up to 1 with the intercept, say)."""
    n = len(cols[0]) if cols else 0
    X = [np.ones(n)] + list(cols)
    nm = ['intercept'] + list(names)
    und = [j for j in range(len(nm)) if np.isnan(coef[j])]
    out = {}
    for j in und:
        if j == 0:
            continue
        x = X[j]
        if np.abs(x).max() <= EXACT_TOL:
            out[j] = 'all zero'
        elif np.ptp(x) <= EXACT_TOL:
            out[j] = 'constant (with the intercept)'
        else:
            same = [nm[k] for k in und if k not in (0, j) and np.abs(X[k] - x).max() <= EXACT_TOL]
            if same:
                out[j] = 'the same as ' + ', '.join(same)
    consts = [nm[j] for j in und if out.get(j) == 'constant (with the intercept)']
    if 0 in und and consts:
        out[0] = 'with the constant column ' + ', '.join(consts)
    rest = [j for j in und if j not in out]
    for j in rest:
        out[j] = 'in a linear dependence with ' + ', '.join(nm[k] for k in rest if k != j)
    return {nm[j]: out[j] for j in und}


def demean(x, g):
    """x minus the mean of its group (g: integer group codes)."""
    return x - (np.bincount(g, weights=x) / np.bincount(g))[g]


def between_share(x, g):
    """Share of the variance of x between the groups g: sum over frames of (group mean - mean)^2 / sum (x - mean)^2."""
    m = x.mean()
    tot = float(((x - m) ** 2).sum())
    if tot <= 0:
        return np.nan
    gm = np.bincount(g, weights=x) / np.bincount(g)
    return float(((gm[g] - m) ** 2).sum()) / tot


def corr(x, y):
    sx, sy = x.std(), y.std()
    if not (sx > 0 and sy > 0):
        return np.nan
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def lens(y, fy, e, fe):
    """Brunswik's lens model: y (risk score rank) and e (SSM rank) with their fits fy, fe on the same predictors.
    Returns Rs, Re, G, C, ra and |ra - (G Re Rs + C sqrt(1 - Re^2) sqrt(1 - Rs^2))|."""
    if np.isnan(fy).any() or np.isnan(fe).any():
        return (np.nan,) * 6
    rs, re_, g, ra = corr(y, fy), corr(e, fe), corr(fy, fe), corr(y, e)
    c = corr(y - fy, e - fe)
    tail = np.sqrt(max(1 - re_ ** 2, 0.0)) * np.sqrt(max(1 - rs ** 2, 0.0))
    dev = abs(ra - (g * re_ * rs + (c * tail if tail > 0 else 0.0)))
    return rs, re_, g, c, ra, dev


def modal(counts, cats):
    """Most frequent category per row of a count matrix; 'tie' when two or more share the top, 'no answer' without
    any."""
    tot = counts.sum(1)
    top = counts.max(1)
    n_top = (counts == top[:, None]).sum(1)
    out = np.array(cats, dtype=object)[counts.argmax(1)]
    out = np.where(n_top > 1, 'tie', out)
    return np.where(tot == 0, 'no answer', out)


def tertile(x):
    """1, 2, 3 at the tertiles of x (np.quantile 1/3, 2/3): x <= q1, q1 < x <= q2, x > q2."""
    q1, q2 = np.quantile(x, [1 / 3, 2 / 3])
    return np.where(x <= q1, 1, np.where(x <= q2, 2, 3)), (float(q1), float(q2))


def ci(arr):
    """As in analyze_vlm_ssm.py."""
    ok = ~np.isnan(arr)
    if ok.sum() < 0.9 * len(arr):
        return np.nan, np.nan, int(ok.sum())
    lo, hi = np.percentile(arr[ok], [2.5, 97.5])
    return float(lo), float(hi), int(ok.sum())


# ------------------------------------------------------------------ inputs
def read_params_risk(path):
    """seed, number of resamples, frame count and segment list of the risk analysis's params.txt."""
    txt = open(path, encoding='utf-8').read()
    m = re.search(r'^bootstrap: (\d+) resamples \(segments resampled\), seed (\d+)', txt, re.M)
    f = re.search(r'^target frames: (\d+)', txt, re.M)
    block = txt.split('scene labels per segment:\n', 1)[1] if 'scene labels per segment:\n' in txt else ''
    segs = []
    for line in block.splitlines():
        mm = re.match(r'^  (\S+): ', line)
        if not mm:
            break
        segs.append(mm.group(1))
    return (int(m.group(1)) if m else None, int(m.group(2)) if m else None, int(f.group(1)) if f else None, segs)


def read_hash_quantities(path):
    m = re.search(r'bootstrap: (\d+) resamples of whole segments, seed (\d+) .*drawn segment indices ([0-9a-f]{64})',
                  open(path, encoding='utf-8').read())
    return (int(m.group(1)), int(m.group(2)), m.group(3)) if m else (None, None, None)


def main():
    ap = argparse.ArgumentParser()
    for name in ('risk', 'quantities', 'frame_ssm', 'truth', 'scene_labels', 'nearby', 'out_dir'):
        ap.add_argument(name)
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261021)
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out_dir, 'exploratory_params.txt')):
        sys.exit(f'{a.out_dir} has exploratory_params.txt already: give a new folder')
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    failed = []

    # --- frames, SSMs
    st = pd.read_csv(os.path.join(a.risk, 'ssm_target_frames.csv'))
    frames = st[['segment', 'frame_idx']].sort_values(['segment', 'frame_idx']).reset_index(drop=True)
    n = len(frames)
    mi = pd.MultiIndex.from_frame(frames)
    ssm_df = av.ssm_table(a.frame_ssm, frames)
    for s in LEAD7:
        if not (ssm_df[s + '_conflict'].all() and ssm_df[s + '_use'].all()):
            sys.exit(f'{s}: not every frame has a value (or some frames are left out of it); this program assumes so')
    ssm_r = {s: ssm_df[s + '_risk'].to_numpy(float) for s in LEAD7}
    seg_names = frames['segment'].to_numpy()
    seg_list = np.array(sorted(set(seg_names)))
    seg_frames = {s: np.flatnonzero(seg_names == s) for s in seg_list}
    seg_code = np.searchsorted(seg_list, seg_names)
    ids = np.arange(n)

    # --- risk scores
    fsc = pd.read_csv(os.path.join(a.risk, 'vlm_frame_scores.csv'), float_precision='round_trip')
    cells = sorted(set(zip(fsc.model, fsc.condition)), key=lambda mc: (mc[0], av.CONDITIONS.index(mc[1])))
    risk = {}
    for m, c in cells:
        s = fsc[(fsc.model == m) & (fsc.condition == c)].set_index(['segment', 'frame_idx'])['score_mean']
        if len(s) != n or not s.index.isin(mi).all():
            sys.exit(f'{m} {c}: its frames in vlm_frame_scores.csv are not the {n} frames of ssm_target_frames.csv')
        risk[(m, c)] = s.reindex(mi).to_numpy(float)

    # --- cues: truth, counts, own estimates, scene, hazard type and camera
    tr = frames.merge(pd.read_csv(a.truth, float_precision='round_trip'), on=['segment', 'frame_idx'], how='left',
                      validate='one_to_one', indicator=True)
    if (tr['_merge'] != 'both').any():
        sys.exit(f'{int((tr["_merge"] != "both").sum())} frames are not in {a.truth}')
    cue = {'true_distance': tr['distance'].to_numpy(float), 'true_ego_speed': tr['v_ego'].to_numpy(float),
           'true_lead_speed': tr['v_lead'].to_numpy(float),
           'true_closing_speed': tr['v_ego'].to_numpy(float) - tr['v_lead'].to_numpy(float),
           'true_ego_accel': tr['a_ego'].to_numpy(float), 'true_lead_accel': tr['a_lead'].to_numpy(float)}
    closing_dev = float(np.nanmax(np.abs(cue['true_closing_speed'] - ssm_df['closing'].to_numpy(float))))
    nb = frames.merge(pd.read_csv(a.nearby), on=['segment', 'frame_idx'], how='left', validate='one_to_one',
                      indicator=True)
    if (nb['_merge'] != 'both').any():
        sys.exit(f'{int((nb["_merge"] != "both").sum())} frames are not in {a.nearby}')
    for r in RADII:
        cue[f'ped_cyc_{r}m'] = nb[f'n_ped_cyc_{r}m'].to_numpy(float)
        cue[f'vehicle_{r}m'] = nb[f'n_vehicle_{r}m'].to_numpy(float)
    qf = pd.read_csv(os.path.join(a.quantities, 'quantity_frames.csv'), float_precision='round_trip')
    own = {}
    for (m, c), g in qf.groupby(['model', 'condition']):
        if (m, c) not in risk:
            continue
        e = {q: g[g.quantity == q].set_index(['segment', 'frame_idx'])['estimate'].reindex(mi).to_numpy(float)
             for q in ('distance', 'ego_speed', 'lead_speed', 'ego_accel', 'lead_accel')}
        own[(m, c)] = {'model_distance': e['distance'], 'model_ego_speed': e['ego_speed'],
                       'model_lead_speed': e['lead_speed'], 'model_closing_speed': e['ego_speed'] - e['lead_speed'],
                       'model_ego_accel': e['ego_accel'], 'model_lead_accel': e['lead_accel']}
    own_cells = [mc for mc in cells if mc in own]
    sl = pd.read_csv(a.scene_labels, dtype=str, keep_default_na=False).set_index('segment')
    missing = sorted(set(seg_list) - set(sl.index))
    if missing:
        sys.exit(f'{a.scene_labels}: no row for {len(missing)} segments, e.g. {missing[:3]}')
    s_terms, s_cols, s_info = [], [], {}
    for col in ('time_of_day', 'location', 'weather'):
        v = sl.loc[seg_names, col].str.strip().to_numpy()
        if (v == '').any():
            sys.exit(f'{a.scene_labels}: no {col} for some segments of these frames')
        seg_v = sl.loc[seg_list, col].str.strip()
        if col == 'weather':
            if not seg_v.isin(WEATHER).all():
                sys.exit(f'{a.scene_labels}: weather not one of {WEATHER}: {sorted(set(seg_v) - set(WEATHER))}')
            cats = WEATHER                                   # every class a dummy, none merged (user, 2026-11-03)
            nseg = {k: int((seg_v == k).sum()) for k in cats}
            ref = max(cats, key=lambda k: nseg[k])           # the class with the most segments
        else:
            cats = sorted(set(v))
            nfr = {k: int((v == k).sum()) for k in cats}
            ref = max(cats, key=lambda k: nfr[k])            # the category with the most frames
        s_info[col] = dict(reference=ref, segments={k: int((seg_v == k).sum()) for k in cats},
                           frames={k: int((v == k).sum()) for k in cats})
        for k in cats:
            if k != ref:
                s_terms.append(f'{col}={k}')
                s_cols.append((v == k).astype(float))
    rq = pd.read_csv(os.path.join(a.risk, 'vlm_requests_tidy.csv'), low_memory=False,
                     usecols=['model', 'condition', 'segment', 'frame_idx', 'repeat', 'hazard_type', 'camera'])
    shares, modes, left_out = {}, {}, {}
    for (m, c), g in rq.groupby(['model', 'condition']):
        if (m, c) not in risk:
            continue
        for kind, cats in (('hazard_type', av.HAZARD_TYPES), ('camera', av.CAMERAS)):
            ok = g[kind].isin(cats)
            left_out[(m, c, kind)] = int((~ok).sum())
            cnt = g[ok].groupby(['segment', 'frame_idx', kind]).size().unstack(kind).reindex(index=mi, columns=cats) \
                .fillna(0).to_numpy(float)
            tot = cnt.sum(1)
            shares[(m, c, kind)] = np.where(tot[:, None] > 0, cnt / np.where(tot > 0, tot, 1)[:, None], np.nan)
            modes[(m, c, kind)] = modal(cnt, cats)
    h_terms = [f'hazard_type={k}' for k in av.HAZARD_TYPES if k != HAZ_REF] + \
              [f'camera={k}' for k in av.CAMERAS if k != CAM_REF]

    def h_cols(m, c):
        hz = shares[(m, c, 'hazard_type')]
        cm = shares[(m, c, 'camera')]
        return [hz[:, j] for j, k in enumerate(av.HAZARD_TYPES) if k != HAZ_REF] + \
               [cm[:, j] for j, k in enumerate(av.CAMERAS) if k != CAM_REF]

    hcols = {mc: h_cols(*mc) for mc in cells}
    # frames each model x condition's fits use: every variable known (all of them, in the formal runs)
    fit_ok = {}
    for mc in cells:
        ok = ~np.isnan(risk[mc])
        for v in [cue[k] for k in K_TRUE + NEAR_CUES] + hcols[mc] + ([own[mc][k] for k in K_OWN] if mc in own else []):
            ok &= ~np.isnan(v)
        fit_ok[mc] = ok
    subsets = {}
    for mc in cells:
        hz, cm = modes[(mc[0], mc[1], 'hazard_type')], modes[(mc[0], mc[1], 'camera')]
        subsets[mc] = {('hazard_type', 'rear-end'): hz == 'rear-end',
                       ('hazard_type', 'other than rear-end'): hz != 'rear-end',
                       ('camera', 'front'): cm == 'front', ('camera', 'other than front'): cm != 'front'}

    # --- the resamples, checked against the two analyses before they are used
    n_boot_r, seed_r, frames_r, segs_r = read_params_risk(os.path.join(a.risk, 'params.txt'))
    for what, mine, theirs in (('resamples', a.n_boot, n_boot_r), ('seed', a.seed, seed_r), ('frames', n, frames_r)):
        if mine != theirs:
            failed.append(f'{what}: {mine} here, {theirs} in the risk analysis\'s params.txt')
    if list(seg_list) != segs_r:
        failed.append(f'segment list: {len(seg_list)} here, {len(segs_r)} in the risk analysis\'s params.txt, '
                      'not the same')
    rng = np.random.default_rng(a.seed)
    draws = [rng.choice(seg_list, len(seg_list)) for _ in range(a.n_boot)]
    h = hashlib.sha256()
    for d in draws:
        h.update(np.searchsorted(seg_list, d).astype(np.int32).tobytes())
    nb_q, seed_q, hash_q = read_hash_quantities(os.path.join(a.quantities, 'params.txt'))
    if (nb_q, seed_q, hash_q) != (a.n_boot, a.seed, h.hexdigest()):
        failed.append(f'drawn segments: sha256 {h.hexdigest()} ({a.n_boot}, seed {a.seed}) here, {hash_q} ({nb_q}, '
                      f'seed {seed_q}) in the quantities analysis\'s params.txt')
    if failed:
        sys.exit('stopped before the resamples are used: ' + '; '.join(failed))
    print(f'frames {n} in {len(seg_list)} segments; {len(cells)} model x condition cells, {len(own_cells)} with own '
          f'estimates; resamples as the risk and quantities analyses ({time.time() - t0:.0f} s)')

    # --- everything for one resample (idx: frame positions; gseg: segment group of each position, its own code per
    # drawn copy; gb_seg: the same labels as given to the tau-b functions); diag: at the point estimate, why the
    # undefined coefficients are undefined
    def stats(idx, gseg, gb_seg, diag=None):
        res, cache = {}, {}
        gb = {'all': np.zeros(len(idx), int), 'segment': gb_seg}
        ids_b = ids[idx]

        def tau(x, y, mask):
            """tau-b ('all', 'segment') of x and y over mask, as analyze_quantities.py's taus with every frame a
            'conflict' (all of them have values)."""
            key = mask.tobytes()
            if key not in cache:
                labs = {gn: v[mask] for gn, v in gb.items()}
                cache[key] = labs, {gn: av.within_pairs(v) for gn, v in labs.items()
                                    if gn != 'all' and len(np.unique(v)) > 20}
            labs, pairs = cache[key]
            xs, ys, ids_s = x[mask], y[mask], ids_b[mask]
            cs = np.ones(len(xs), bool)
            out = {}
            for gn in GROUPINGS:
                if len(xs) >= 3:
                    out[gn] = av.stats_by_grouping(xs, cs, ys, ids_s, labs[gn], pairs.get(gn))[3]
                else:
                    out[gn] = np.nan
            return out

        rcache = {}

        def ranked(name, v, sel):
            key = (name, sel.tobytes())
            if key not in rcache:
                rcache[key] = rank01(v[sel])
            return rcache[key]

        def fit(target, y, cols, terms, fname):
            coef, fv, r2 = ols(y, cols)
            res[('r2', *target, fname)] = r2
            for term, b in zip(['intercept'] + terms, coef):
                res[('coef', *target, fname, term)] = b
            if diag is not None and np.isnan(coef).any():
                diag[(*target, fname)] = why_undefined(cols, terms, coef)
            return coef, fv, r2

        ssm_b = {s: ssm_r[s][idx] for s in LEAD7}
        for mc in cells:
            x = risk[mc][idx]
            ok = ~np.isnan(x)
            for s in LEAD7:                                   # the check against agreement.csv
                for gn, v in tau(x, ssm_b[s], ok).items():
                    res[('check', *mc, s, gn)] = v
            for name in TRUE_CUES + NEAR_CUES + (OWN_CUES if mc in own else []):
                v = (own[mc][name] if name.startswith('model_') else cue[name])[idx]
                for gn, t in tau(x, v, ok & ~np.isnan(v)).items():
                    res[('cue', *mc, name, gn)] = t
            for (kind, part), sub in subsets[mc].items():
                sb = sub[idx] & ok
                for s in LEAD7:
                    for gn, t in tau(x, ssm_b[s], sb).items():
                        res[('sub', *mc, kind, part, s, gn)] = t
        # rule. The SSMs' own rule and between share on every frame of the resample
        every = np.ones(len(idx), bool)
        g_all = np.unique(gseg, return_inverse=True)[1]
        K_all = [ranked(k, cue[k][idx], every) for k in K_TRUE]
        for s in LEAD7:
            e = rank01(ssm_b[s])
            fit(('ssm', s, ''), e, K_all, K_TRUE, 'K')
            res[('between', 'ssm', s, '')] = between_share(e, g_all)
        ego = 1 + K_TRUE.index('true_ego_speed')
        efit = {}
        for mc in cells:
            tgt = ('risk', *mc)
            sel = fit_ok[mc][idx]
            y = rank01(risk[mc][idx][sel])
            g = np.unique(gseg[sel], return_inverse=True)[1]
            K = [ranked(k, cue[k][idx], sel) for k in K_TRUE]
            S = [c[idx][sel] for c in s_cols]
            N = {r: [ranked(f'ped_cyc_{r}m', cue[f'ped_cyc_{r}m'][idx], sel),
                     ranked(f'vehicle_{r}m', cue[f'vehicle_{r}m'][idx], sel)] for r in RADII}
            H = [c[idx][sel] for c in hcols[mc]]
            n_terms = {r: [f'ped_cyc_{r}m', f'vehicle_{r}m'] for r in RADII}
            out = {'K': fit(tgt, y, K, K_TRUE, 'K'), 'K+S': fit(tgt, y, K + S, K_TRUE + s_terms, 'K+S')}
            for r in RADII:
                out[f'K+S+N{r}'] = fit(tgt, y, K + S + N[r], K_TRUE + s_terms + n_terms[r], f'K+S+N{r}')
                out[f'K+S+N{r}+H'] = fit(tgt, y, K + S + N[r] + H, K_TRUE + s_terms + n_terms[r] + h_terms,
                                         f'K+S+N{r}+H')
            res[('diff', *mc, 'S added: R2(K+S) - R2(K)')] = out['K+S'][2] - out['K'][2]
            for r in RADII:
                res[('diff', *mc, f'N{r} added: R2(K+S+N{r}) - R2(K+S)')] = out[f'K+S+N{r}'][2] - out['K+S'][2]
                res[('diff', *mc, f'H added: R2(K+S+N{r}+H) - R2(K+S+N{r})')] = \
                    out[f'K+S+N{r}+H'][2] - out[f'K+S+N{r}'][2]
                res[('diff', *mc, f'ego speed coefficient: K+S+N{r} - K')] = \
                    out[f'K+S+N{r}'][0][ego] - out['K'][0][ego]
            if mc in own:
                Ko = [rank01(own[mc][k][idx][sel]) for k in K_OWN]
                r2o = fit(tgt, y, Ko, K_OWN, 'K_own')[2]
                res[('diff', *mc, 'R2(K_own) - R2(K)')] = r2o - out['K'][2]
            yw = demean(y, g)
            Kw = [demean(k, g) for k in K]
            fit(tgt, yw, Kw, K_TRUE, 'within: K')
            for r in RADII:
                fit(tgt, yw, Kw + [demean(v, g) for v in N[r]], K_TRUE + n_terms[r], f'within: K+N{r}')
            res[('between', 'risk', *mc)] = between_share(y, g)
            key = sel.tobytes()                             # lens model: the SSMs on the same frames, on K
            if key not in efit:
                efit[key] = {}
                for s in LEAD7:
                    e = rank01(ssm_b[s][sel])
                    efit[key][s] = (e, ols(e, K)[1])
            for s in LEAD7:
                e, fe = efit[key][s]
                for stat, v in zip(LENS + ['dev'], lens(y, out['K'][1], e, fe)):
                    res[('lens', *mc, s, stat)] = v
        return res

    diag = {}
    point = stats(ids, seg_code, seg_names, diag)
    keys = list(point)
    col = {k: i for i, k in enumerate(keys)}

    def one(b):
        draw = draws[b]
        idx = np.concatenate([seg_frames[s] for s in draw])
        gseg = np.concatenate([np.full(len(seg_frames[s]), k) for k, s in enumerate(draw)])
        r = stats(idx, gseg, gseg)
        return np.array([r.get(k, np.nan) for k in keys], float)

    boot = np.full((a.n_boot, len(keys)), np.nan)
    tb = time.time()
    if a.workers > 1:
        _WORK['one'] = one
        chunks = [c for c in np.array_split(np.arange(a.n_boot), a.workers * 8) if len(c)]
        done, last = 0, 0
        with mp.get_context('fork').Pool(a.workers) as pool:
            for bs, vals in zip(chunks, pool.imap(_chunk, [c.tolist() for c in chunks])):
                boot[bs] = vals
                done += len(bs)
                if done == len(bs) or done // 250 > last or done == a.n_boot:
                    last = done // 250
                    print(f'bootstrap {done}/{a.n_boot} ({(time.time() - tb) / done * (a.n_boot - done):.0f} s left; '
                          f'{a.workers} workers)')
    else:
        for b in range(a.n_boot):
            boot[b] = one(b)
            if b in (0, 9) or (b + 1) % 250 == 0:
                print(f'bootstrap {b + 1}/{a.n_boot} ({(time.time() - tb) / (b + 1) * (a.n_boot - b - 1):.0f} s left)')

    def with_ci(k):
        lo, hi, nv = ci(boot[:, col[k]])
        return point[k], lo, hi, nv

    def write(name, rows, columns=None):
        pd.DataFrame(rows, columns=columns).to_csv(os.path.join(a.out_dir, name), index=False)

    def target_cols(k):            # ('risk', model, condition) or ('ssm', SSM, '')
        return dict(target=k[1], model=k[2] if k[1] == 'risk' else '', condition=k[3] if k[1] == 'risk' else '',
                    ssm=k[2] if k[1] == 'ssm' else '')

    nf = {mc: int(fit_ok[mc].sum()) for mc in cells}
    # --- 4: rule tables
    rows = []
    for k in keys:
        if k[0] == 'coef':
            v, lo, hi, nv = with_ci(k)
            rows.append(dict(**target_cols(k), fit=k[4], term=k[5], coef=v, coef_lo=lo, coef_hi=hi, nboot=nv))
    write('exploratory_rule_coefficients.csv', rows)
    rows = []
    for k in keys:
        if k[0] == 'r2':
            v, lo, hi, nv = with_ci(k)
            rows.append(dict(**target_cols(k), fit=k[4], frames=nf[(k[2], k[3])] if k[1] == 'risk' else n, r2=v,
                             r2_lo=lo, r2_hi=hi, nboot=nv))
    write('exploratory_rule_r2.csv', rows)
    rows = []
    for k in keys:
        if k[0] == 'diff':
            v, lo, hi, nv = with_ci(k)
            rows.append(dict(model=k[1], condition=k[2], comparison=k[3], frames=nf[(k[1], k[2])], diff=v, diff_lo=lo,
                             diff_hi=hi, nboot=nv, clear=bool(lo > 0 or hi < 0) if not np.isnan(lo) else False))
    write('exploratory_rule_differences.csv', rows)
    rows = []
    for (kind, m_or_s, c_, fname), why in diag.items():
        for term, reason in why.items():
            k = ('coef', kind, m_or_s, c_, fname, term)
            rows.append(dict(**target_cols(k), fit=fname, term=term, reason=reason,
                             resamples_defined=int((~np.isnan(boot[:, col[k]])).sum())))
    und = pd.DataFrame(rows, columns=['target', 'model', 'condition', 'ssm', 'fit', 'term', 'reason',
                                      'resamples_defined'])
    und.to_csv(os.path.join(a.out_dir, 'exploratory_rule_undefined.csv'), index=False)

    checks = []
    # --- risk check
    ag = pd.read_csv(os.path.join(a.risk, 'agreement.csv'), dtype=str, keep_default_na=False) \
        .set_index(['model', 'condition', 'ssm', 'grouping'])

    def num(t):
        return float(t) if t != '' else np.nan

    def same(x, y):
        return x == y or (np.isnan(x) and np.isnan(y))

    rows = []
    for mc in cells:
        for s in LEAD7:
            for gn in GROUPINGS:
                v, lo, hi, nv = with_ci(('check', *mc, s, gn))
                r = ag.loc[(*mc, s, gn)]
                old = (num(r['tau_b_conflict']), num(r['tau_b_conflict_lo']), num(r['tau_b_conflict_hi']),
                       int(r['tau_b_conflict_nboot']))
                rows.append(dict(model=mc[0], condition=mc[1], ssm=s, grouping=gn, tau_b=v, tau_b_lo=lo, tau_b_hi=hi,
                                 nboot=nv, risk_tau_b=old[0], risk_tau_b_lo=old[1], risk_tau_b_hi=old[2],
                                 risk_nboot=old[3], identical=all(same(x, y) for x, y in zip((v, lo, hi, nv), old))))
    rc = pd.DataFrame(rows)
    rc.to_csv(os.path.join(a.out_dir, 'exploratory_risk_check.csv'), index=False)
    checks.append(f'risk analysis tau-b and intervals recomputed from these resamples ({len(cells)} model x condition '
                  f'x {len(LEAD7)} SSMs x {GROUPINGS}): {int(rc.identical.sum())} of {len(rc)} identical to its '
                  'agreement.csv')
    if not rc.identical.all():
        failed.append('risk check')

    # --- 1: chain, copied as text
    pt = pd.read_csv(os.path.join(a.quantities, 'perceived_ssm_tau_b.csv'), dtype=str, keep_default_na=False)
    rows = []
    for mc in own_cells:
        for s in LEAD7:
            for gn in GROUPINGS:
                r = ag.loc[(*mc, s, gn)]
                row = dict(model=mc[0], condition=mc[1], ssm=s, grouping=gn, risk_vs_true_ssm_frames=r['n_conflict'],
                           **{f'risk_vs_true_ssm_{x}': r[f'tau_b_conflict{y}'] for x, y in
                              (('tau_b', ''), ('tau_b_lo', '_lo'), ('tau_b_hi', '_hi'), ('nboot', '_nboot'))})
                for comp in ('model_ssm_vs_true_ssm', 'risk_score_vs_model_ssm'):
                    p = pt[(pt.model == mc[0]) & (pt.condition == mc[1]) & (pt.ssm == s) & (pt.comparison == comp)
                           & (pt.grouping == gn)]
                    if len(p) != 1:
                        sys.exit(f'perceived_ssm_tau_b.csv: {len(p)} rows for {mc} {s} {comp} {gn}')
                    p = p.iloc[0]
                    short = 'model_ssm_vs_true_ssm' if comp == 'model_ssm_vs_true_ssm' else 'risk_vs_model_ssm'
                    row.update({f'{short}_frames': p['frames'], f'{short}_tau_b': p['tau_b'],
                                f'{short}_tau_b_lo': p['tau_b_lo'], f'{short}_tau_b_hi': p['tau_b_hi'],
                                f'{short}_nboot': p['nboot']})
                rows.append(row)
    write('exploratory_chain.csv', rows)
    back = pd.read_csv(os.path.join(a.out_dir, 'exploratory_chain.csv'), dtype=str, keep_default_na=False)
    bad = 0
    for r in back.itertuples(index=False):
        src = ag.loc[(r.model, r.condition, r.ssm, r.grouping)]
        bad += (r.risk_vs_true_ssm_tau_b, r.risk_vs_true_ssm_tau_b_lo, r.risk_vs_true_ssm_tau_b_hi,
                r.risk_vs_true_ssm_nboot, r.risk_vs_true_ssm_frames) != \
            (src['tau_b_conflict'], src['tau_b_conflict_lo'], src['tau_b_conflict_hi'], src['tau_b_conflict_nboot'],
             src['n_conflict'])
        for comp, short in (('model_ssm_vs_true_ssm', 'model_ssm_vs_true_ssm'),
                            ('risk_score_vs_model_ssm', 'risk_vs_model_ssm')):
            p = pt[(pt.model == r.model) & (pt.condition == r.condition) & (pt.ssm == r.ssm) &
                   (pt.comparison == comp) & (pt.grouping == r.grouping)].iloc[0]
            bad += tuple(getattr(r, f'{short}_{x}') for x in ('frames', 'tau_b', 'tau_b_lo', 'tau_b_hi', 'nboot')) \
                != tuple(p[x] for x in ('frames', 'tau_b', 'tau_b_lo', 'tau_b_hi', 'nboot'))
    checks.append(f'chain table: {len(back)} rows x 3 tau-b copied from agreement.csv and perceived_ssm_tau_b.csv; '
                  f'rows with a value not the same text as there: {bad}')
    if bad:
        failed.append('chain')

    # --- 2: cues
    rows = []
    for k in keys:
        if k[0] == 'cue':
            mc, name, gn = (k[1], k[2]), k[3], k[4]
            v_full = own[mc][name] if name.startswith('model_') else cue[name]
            v, lo, hi, nv = with_ci(k)
            rows.append(dict(model=mc[0], condition=mc[1], cue=name, grouping=gn,
                             frames=int((~np.isnan(risk[mc]) & ~np.isnan(v_full)).sum()), tau_b=v, tau_b_lo=lo,
                             tau_b_hi=hi, nboot=nv))
    write('exploratory_cue_tau_b.csv', rows)
    mine = pd.read_csv(os.path.join(a.out_dir, 'exploratory_cue_tau_b.csv'), dtype=str, keep_default_na=False)
    e2 = pd.read_csv(os.path.join(a.quantities, 'exploratory_e2_risk_vs_distance_speed.csv'), dtype=str,
                     keep_default_na=False)
    e2 = e2[e2.grouping.isin(GROUPINGS) & e2.variable.isin(E2_VARS)]
    e2 = e2[[(m, c) in risk for m, c in zip(e2.model, e2.condition)]]
    mm = mine[mine.cue.isin(E2_VARS)].rename(columns={'cue': 'variable'})
    cols = ['model', 'condition', 'variable', 'grouping', 'frames', 'tau_b', 'tau_b_lo', 'tau_b_hi', 'nboot']
    j = mm[cols].merge(e2[cols], on=['model', 'condition', 'variable', 'grouping'], how='outer', suffixes=('', '_e2'),
                       indicator=True)
    e2_bad = int((j['_merge'] != 'both').sum()) + int(sum(
        (j[c] != j[c + '_e2']).sum() for c in ['frames', 'tau_b', 'tau_b_lo', 'tau_b_hi', 'nboot']))
    checks.append(f'E2: {len(j)} rows (true distance and ego speed of {len(cells)} cells, own distance and ego speed of '
                  f'{len(own_cells)}, x {GROUPINGS}); rows or values not the same text as '
                  f'exploratory_e2_risk_vs_distance_speed.csv: {e2_bad}')
    if e2_bad:
        failed.append('E2')

    # --- 3: two way, point only
    bins, cuts = {}, {}
    for q in ('true_distance', 'true_ego_speed', 'true_closing_speed'):
        bins[q], cuts[q] = tertile(cue[q])
    targets = [('risk', m, c, '', risk[(m, c)]) for m, c in cells] + [('ssm', '', '', s, ssm_r[s]) for s in LEAD7]
    rows = []
    for tname, other in (('distance x ego speed', 'true_ego_speed'), ('distance x closing speed', 'true_closing_speed')):
        for kind, m, c, s, x in targets:
            ok = ~np.isnan(x)
            rk = np.full(n, np.nan)
            rk[ok] = rank01(x[ok])
            for i in (1, 2, 3):
                for jj in (1, 2, 3):
                    cell_ = ok & (bins['true_distance'] == i) & (bins[other] == jj)
                    rows.append(dict(table=tname, distance_tertile=i, other_tertile=jj, target=kind, model=m,
                                     condition=c, ssm=s, frames=int(cell_.sum()),
                                     mean_rank=float(rk[cell_].mean()) if cell_.any() else np.nan))
    write('exploratory_two_way.csv', rows)

    # --- 4 (lens), 5, 6
    rows = []
    dev_max = 0.0
    for mc in cells:
        for s in LEAD7:
            r = dict(model=mc[0], condition=mc[1], ssm=s, frames=nf[mc])
            for stat in LENS:
                v, lo, hi, nv = with_ci(('lens', *mc, s, stat))
                r.update({stat: v, f'{stat}_lo': lo, f'{stat}_hi': hi, f'{stat}_nboot': nv})
            d = np.concatenate([[point[('lens', *mc, s, 'dev')]], boot[:, col[('lens', *mc, s, 'dev')]]])
            r['identity_max_abs_dev'] = float(np.nanmax(d))
            dev_max = max(dev_max, r['identity_max_abs_dev'])
            rows.append(r)
    write('exploratory_lens.csv', rows)
    checks.append(f'lens model: |ra - (G Re Rs + C sqrt(1 - Re^2) sqrt(1 - Rs^2))| at most {dev_max:.3g} over the point '
                  f'estimates and every resample (must be below {IDENTITY_TOL:g})')
    if not dev_max < IDENTITY_TOL:
        failed.append('lens identity')
    rows = []
    for mc in cells:
        for kind, cats in (('hazard_type', av.HAZARD_TYPES), ('camera', av.CAMERAS)):
            md = modes[(mc[0], mc[1], kind)]
            for k in list(cats) + ['tie', 'no answer']:
                rows.append(dict(model=mc[0], condition=mc[1], kind=kind, most_frequent=k, frames=int((md == k).sum()),
                                 replies_not_in_the_list_or_none=left_out[(mc[0], mc[1], kind)]))
    write('exploratory_hazard_camera_modal.csv', rows)
    rows = []
    for k in keys:
        if k[0] == 'sub':
            mc, kind, part, s, gn = (k[1], k[2]), k[3], k[4], k[5], k[6]
            v, lo, hi, nv = with_ci(k)
            rows.append(dict(model=mc[0], condition=mc[1], kind=kind, frames_part=part, ssm=s, grouping=gn,
                             frames=int((subsets[mc][(kind, part)] & ~np.isnan(risk[mc])).sum()), tau_b=v,
                             tau_b_lo=lo, tau_b_hi=hi, nboot=nv))
    write('exploratory_hazard_camera_tau_b.csv', rows)
    rows = []
    for k in keys:
        if k[0] == 'between':
            v, lo, hi, nv = with_ci(k)
            rows.append(dict(target=k[1], model=k[2] if k[1] == 'risk' else '',
                             condition=k[3] if k[1] == 'risk' else '', ssm=k[2] if k[1] == 'ssm' else '',
                             frames=nf[(k[2], k[3])] if k[1] == 'risk' else n, segments=len(seg_list),
                             between_share=v, between_share_lo=lo, between_share_hi=hi, nboot=nv))
    write('exploratory_between_segment.csv', rows)

    # --- params
    with open(os.path.join(a.out_dir, 'exploratory_params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'EXPLORATORY (user, 2026-11-03, after seeing the risk and quantities results): analyze_rules.py run '
                f'{time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        for k_, v_ in (('risk analysis', a.risk), ('quantities analysis', a.quantities), ('frame_ssm', a.frame_ssm),
                       ('truth', a.truth), ('scene labels', a.scene_labels), ('nearby counts', a.nearby)):
            f.write(f'{k_} = {os.path.abspath(v_)}\n')
        f.write(f'frames {n} in {len(seg_list)} segments; model x condition cells {len(cells)}; with own estimates '
                f'{own_cells}\n')
        f.write(f'frames used by the fits per cell: {sorted(set(nf.values()))}\n')
        f.write(f'resamples: {a.n_boot} of whole segments, seed {a.seed} ({a.workers} processes); the risk analysis\'s '
                f'params.txt: {n_boot_r} resamples, seed {seed_r}, {frames_r} frames, {len(segs_r)} segments, the same '
                f'list as here: {list(seg_list) == segs_r}; sha256 of the drawn segment indices {h.hexdigest()}, the '
                f'quantities analysis\'s {hash_q}\n')
        for c_ in checks:
            f.write(f'check: {c_}\n')
        f.write(f'closing speed = true ego speed - true lead speed; largest difference to frame_ssm\'s closing: '
                f'{closing_dev:.3g} m/s\n')
        for col_, info in s_info.items():
            f.write(f'S: {col_}, reference {info["reference"]!r} (the most '
                    f'{"segments" if col_ == "weather" else "frames"}); segments per category {info["segments"]}, '
                    f'frames {info["frames"]}\n')
        f.write(f'H: shares of the hazard types {av.HAZARD_TYPES} (reference {HAZ_REF}) and cameras {av.CAMERAS} '
                f'(reference {CAM_REF}) among the replies with a listed one; left out per cell (not listed or none): '
                f'hazard type {sorted(set(v_ for (m_, c_, k_), v_ in left_out.items() if k_ == "hazard_type"))}, '
                f'camera {sorted(set(v_ for (m_, c_, k_), v_ in left_out.items() if k_ == "camera"))}\n')
        f.write('ranks: rank among the frames used / their number, ties the mean rank (scipy rankdata); S dummies and '
                'H shares as they are; least squares with an intercept\n')
        f.write(f'undefined coefficients (every fit, every resample): SVD of the design matrix; singular values <= '
                f's_max * max(n, p) * {np.finfo(float).eps:.4g} are zero; a coefficient with a null-space component '
                f'above {NULL_TOL:g} in absolute value is undefined; the others are the least-squares solution and R2 '
                f'comes from the projection. Reasons compared within {EXACT_TOL:g}\n')
        f.write(f'undefined at the point estimate (exploratory_rule_undefined.csv): {len(und)} coefficients in '
                f'{len(diag)} fits; ' + ('; '.join(f'{t}: {c_}' for t, c_ in
                                                   und.groupby('reason').size().sort_values(ascending=False).items())
                                         if len(und) else 'none') + '\n')
        f.write('tertiles (np.quantile 1/3, 2/3): ' + ', '.join(f'{q} {c_[0]!r}, {c_[1]!r}' for q, c_ in cuts.items())
                + '\n')
        f.write(f'checks not met: {failed if failed else "none"}\n')
    print(f'done in {time.time() - t0:.0f} s; checks not met: {failed if failed else "none"}; outputs in '
          f'{os.path.abspath(a.out_dir)}')
    for c_ in checks:
        print('  ' + c_)
    print(f'  undefined at the point estimate: {len(und)} coefficients in {len(diag)} fits')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
