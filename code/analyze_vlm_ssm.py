"""Compare the VLM risk scores with the SSMs on the VLM target frames (pilot analysis).

Method (agreed with the user 2026-10-21, see CLAUDE.md "VLM 打分"):
 0. Frame score = mean of the answered repeats (normally 5) per model x condition x frame. Where a reply has no
    readable answer because the thinking hit max_tokens, the forced answer (Qwen's early-stopping text, made with
    force_answer.py) is used; a request with no answer at all is left out of that frame's mean.
    Every SSM is turned into "larger = riskier": TTC, THW, MTTC, PICUD, PSD, TTC2D and ACT are negated, DRAC and CI
    are kept as they are.
    "Conflict" for an SSM = it has a value: TTC / MTTC / TTC2D / ACT / THW / PICUD / PSD finite (no lead = no value),
    DRAC / CI > 0.
 1. Step 1: AUC of the frame score for conflict vs no-conflict frames
    (= P(score of a conflict frame > score of a no-conflict frame), ties count one half).
 2. Step 2, conflict frames only: Kendall tau-b (main), Spearman rho, and the pairwise agreement
    (share of frame pairs on which both give the same order: riskier, equal or safer; the agreement index of
    Singh et al. 2023, arXiv 2306.14657).
 3. Overall: Kendall tau-b over all target frames with the no-conflict frames tied at the safe end. Every SSM uses the
    same frames, so this is the statistic for "which SSM is the VLM closest to".
 4. Comparison groups (2026-10-21, user's decision), column 'grouping': every statistic is counted over frame pairs
    that share a scene condition, so that the condition cannot drive the result:
      all          every pair of frames (mixed comparison)
      weather      pairs with the same weather label     (labels/scene_labels.csv, our own labels from the images;
      time_of_day  pairs with the same time of day        Waymo's weather field is wrong for at least one segment)
      segment      pairs from the same segment (weather, light, place and road all fixed; the strictest)
    Spearman is only given for 'all'. agreement_by_segment.csv has every segment on its own.
 5. Bootstrap: 95 % intervals of every statistic, and of the differences between SSMs in the overall tau-b and the
    AUC per grouping (paired: one resample serves all SSMs). --resample frame (pilot) resamples frames within each
    segment; --resample segment (202 segments) resamples whole segments.
 6. Checks: frame score without the forced answers; median instead of mean (point estimates).
 Also per model x condition: answers, repeat stability (SD over repeats, test-retest tau-b, ICC), time and tokens.

Inputs: <vlm_scores>/<model>/pilot_<model>.jsonl (+ pilot_<model>_forced_answers.jsonl if present) for every model
folder; smoke_* and *_timeout_retried files are not used. <frame_ssm.csv> from waymo_ssm_frames.py.
<scene_labels.csv> (code/make_scene_labels.py makes the template): one row per segment with the label columns named
in --group_by; an empty label is an error.

usage: python -I analyze_vlm_ssm.py <results/vlm_scores> <frame_ssm.csv> <out_dir>
                                    [--scene_labels labels/scene_labels.csv] [--group_by weather,time_of_day]
                                    [--n_boot 2000] [--seed 20261021] [--resample frame|segment]
"""
import argparse
import glob
import itertools
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore', message='.*constant.*')       # tau / rho of a constant input -> nan, handled

SSMS = ['TTC', 'THW', 'DRAC', 'MTTC', 'CI', 'PICUD', 'PSD', 'TTC2D', 'ACT']
SMALLER_RISKIER = {'TTC', 'THW', 'MTTC', 'PICUD', 'PSD', 'TTC2D', 'ACT'}
POSITIVE_IS_CONFLICT = {'DRAC', 'CI'}                          # 0 = not closing in
CONDITIONS = ['grid', 'sep_time', 'sep_camera', 'ctrl_grid', 'ctrl_sep']
CAMERAS = ['front-left', 'front', 'front-right', 'left-side', 'right-side', 'none']
CAMERA_FIX = {'side left': 'left-side', 'side-left': 'left-side', 'left side': 'left-side',
              'side right': 'right-side', 'side-right': 'right-side', 'right side': 'right-side',
              'front left': 'front-left', 'front right': 'front-right'}
HAZARD_TYPES = ['rear-end', 'lane change or side', 'crossing or intersection', 'pedestrian or cyclist', 'oncoming',
                'other', 'none']
HAZARD_FIX = {'lane change': 'lane change or side', 'intersection': 'crossing or intersection'}
KEY = ['segment', 'frame_idx', 'condition', 'repeat']
STATS = ['auc', 'tau_b_conflict', 'spearman_conflict', 'agreement_conflict', 'tau_b_all']
DIFF_STATS = ['tau_b_all', 'auc']


# ------------------------------------------------------------------ reading the VLM replies
def load_light(raw):
    """One JSON line without its 'logprobs' list (most of the bytes; not needed here). Inside a JSON string a quote
    is escaped, so ', "logprobs": ' can only be the key itself."""
    i = raw.find(b', "logprobs": ')
    if i >= 0:
        j = raw.find(b', "latency_s": ', i)
        raw = raw[:i] + (raw[j:] if j > 0 else b'}')
    return json.loads(raw)


def num(v):
    """None -> NaN, so every model's table has float columns (not set by some servers, e.g. top_k)."""
    return np.nan if v is None else float(v)


def norm_camera(c):
    if c is None:
        return None, False
    c0 = str(c).strip().lower()
    if c0 in CAMERAS:
        return c0, False
    return CAMERA_FIX.get(c0, c0), True


def norm_hazard(h):
    if h is None:
        return None, False
    h0 = str(h).strip().lower()
    if h0 in HAZARD_TYPES:
        return h0, False
    return HAZARD_FIX.get(h0, h0), True


def read_model(model_dir):
    name = os.path.basename(model_dir.rstrip('/\\'))
    main = os.path.join(model_dir, f'pilot_{name}.jsonl')
    forced_path = os.path.join(model_dir, f'pilot_{name}_forced_answers.jsonl')
    rows, seen = [], set()
    with open(main, 'rb') as f:
        for raw in f:
            d = load_light(raw)
            k = tuple(d[c] for c in KEY)
            if k in seen:
                raise ValueError(f'{main}: duplicate request {k}')
            seen.add(k)
            u = d.get('usage') or {}
            det = u.get('completion_tokens_details') or {}
            p = d.get('parsed')
            rows.append(dict(model=name, segment=d['segment'], frame_idx=int(d['frame_idx']),
                             condition=d['condition'], repeat=int(d['repeat']),
                             temperature=num(d.get('temperature')), top_p=num(d.get('top_p')),
                             top_k=num(d.get('top_k')), finish_reason=d.get('finish_reason'), forced=False,
                             risk=float(p['risk']) if p else np.nan,
                             camera_raw=p.get('main_hazard_camera') if p else None,
                             hazard_type_raw=p.get('hazard_type') if p else None,
                             main_hazard=p.get('main_hazard') if p else None,
                             latency_s=num(d.get('latency_s')), prompt_tokens=num(u.get('prompt_tokens')),
                             completion_tokens=num(u.get('completion_tokens')),
                             reasoning_tokens=num(det.get('reasoning_tokens'))))
    req = pd.DataFrame(rows)
    n_forced_used = 0
    if os.path.exists(forced_path):
        idx = req.set_index(KEY).index
        with open(forced_path, 'rb') as f:
            for raw in f:
                d = load_light(raw)
                k = tuple(d[c] for c in KEY)
                i = idx.get_loc(k)
                if not np.isnan(req.at[i, 'risk']):
                    raise ValueError(f'{forced_path}: forced answer for {k}, but the original reply has one')
                p = d.get('parsed')
                if not p:
                    continue                                  # forced, still no readable answer
                u = d.get('usage') or {}
                req.loc[i, ['risk', 'camera_raw', 'hazard_type_raw', 'main_hazard', 'forced']] = \
                    [float(p['risk']), p.get('main_hazard_camera'), p.get('hazard_type'), p.get('main_hazard'), True]
                req.at[i, 'latency_s'] = np.nansum([req.at[i, 'latency_s'], num(d.get('latency_s'))])
                req.at[i, 'completion_tokens'] = np.nansum([req.at[i, 'completion_tokens'],
                                                            num(u.get('completion_tokens'))])
                n_forced_used += 1
    req['forced'] = req['forced'].astype(bool)
    cams = req['camera_raw'].map(norm_camera)
    req['camera'], req['camera_fixed'] = cams.str[0], cams.str[1]
    haz = req['hazard_type_raw'].map(norm_hazard)
    req['hazard_type'], req['hazard_type_fixed'] = haz.str[0], haz.str[1]
    return req, dict(model=name, file=main, lines=len(req), forced_file=os.path.exists(forced_path),
                     forced_used=n_forced_used)


# ------------------------------------------------------------------ SSMs on the target frames
def ssm_table(frame_ssm_csv, frames):
    f = pd.read_csv(frame_ssm_csv)
    t = frames.merge(f, on=['segment', 'frame_idx'], how='left', validate='one_to_one', indicator=True)
    if (t['_merge'] != 'both').any():
        raise ValueError('target frames missing from frame_ssm.csv: '
                         f"{t.loc[t['_merge'] != 'both', ['segment', 'frame_idx']].values.tolist()}")
    if t['edge_frame'].astype(bool).any():
        raise ValueError('some target frames are edge frames; the VLM targets should exclude them')
    out = t[['segment', 'frame_idx']].copy()
    for s in SSMS:
        v = t[s].to_numpy(float)
        conflict = (np.nan_to_num(v, nan=0.0) > 0) if s in POSITIVE_IS_CONFLICT else np.isfinite(v)
        r = np.where(conflict, -v if s in SMALLER_RISKIER else v, np.nan)
        out[s] = v
        out[s + '_conflict'] = conflict
        out[s + '_risk'] = r                                  # larger = riskier; NaN where no conflict
    return out


def scene_groupings(frames, scene_csv, group_by):
    """Frame labels for every grouping: {'all': one group, <label column>: the segment's label, 'segment': segment}."""
    seg = frames['segment'].to_numpy()
    g = {'all': np.zeros(len(frames), int)}
    used = None
    if group_by:
        lab = pd.read_csv(scene_csv, dtype=str).set_index('segment')
        missing = sorted(set(seg) - set(lab.index))
        if missing:
            raise ValueError(f'{scene_csv}: no row for segments {missing}')
        for col in group_by:
            if col not in lab.columns:
                raise ValueError(f'{scene_csv}: no column {col!r}')
            v = lab.loc[seg, col]
            if v.isna().any() or (v.str.strip() == '').any():
                raise ValueError(f'{scene_csv}: empty {col!r} label for {sorted(set(seg[v.isna().to_numpy()]))}')
            g[col] = v.str.strip().to_numpy()
        used = lab.loc[sorted(set(seg)), group_by]
    g['segment'] = seg
    return g, used


# ------------------------------------------------------------------ statistics
def tie_pairs(*cols):
    _, c = np.unique(np.stack(cols, axis=1), axis=0, return_counts=True)
    return float((c * (c - 1) / 2).sum())


def auc(score, conflict):
    """P(score of a conflict frame > score of a no-conflict frame), ties count one half (Mann-Whitney U / n1 n0)."""
    u, d = auc_counts(score, conflict)
    return u / d if d > 0 else np.nan


def auc_counts(score, conflict):
    n1 = int(conflict.sum())
    n0 = len(conflict) - n1
    if n1 == 0 or n0 == 0:
        return 0.0, 0.0
    r = rankdata(score)
    return float(r[conflict].sum() - n1 * (n1 + 1) / 2), float(n1 * n0)


def pair_counts(x, y, ids):
    """Pair counts over all pairs of the given frames, for tau-b and the pairwise agreement:
    S = concordant - discordant, tx / ty = pairs untied in x / in y, agree = pairs with the same order in x and y
    (tied in both counts), n = pairs. Pairs of a frame with its own bootstrap copy are left out of agree and n; they
    are tied in both, so they change neither S nor tx nor ty."""
    n = len(x)
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    n0 = n * (n - 1) / 2
    n1, n2, n12 = tie_pairs(x), tie_pairs(y), tie_pairs(x, y)
    tx, ty = n0 - n1, n0 - n2
    untied = n0 - n1 - n2 + n12                              # pairs untied in x and in y: concordant + discordant
    s = 0.0 if untied == 0 else float(kendalltau(x, y, variant='b').statistic) * np.sqrt(tx * ty)
    n_same = tie_pairs(ids)
    return s, tx, ty, (untied + s) / 2 + n12 - n_same, n0 - n_same


def pair_stats(x, y, ids):
    """Kendall tau-b, Spearman rho and pairwise agreement of x and y (both larger = riskier), over all pairs."""
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    s, tx, ty, ag, npairs = pair_counts(x, y, ids)
    tau = s / np.sqrt(tx * ty) if tx * ty > 0 else np.nan
    return float(tau), float(spearmanr(x, y).statistic), float(ag / npairs) if npairs > 0 else np.nan


def grouped_stats(x, conflict, r_all, ids, labels):
    """AUC, tau-b and agreement among conflict frames, and overall tau-b, counted over frame pairs with the same label
    (stratified: tau-b = sum of S / sqrt(sum of tx * sum of ty); AUC = sum of U / sum of n1 n0). Few, large groups."""
    u = d = sc = txc = tyc = agc = nc = sa = txa = tya = 0.0
    for g in np.unique(labels):
        m = labels == g
        a, b = auc_counts(x[m], conflict[m])
        u, d = u + a, d + b
        mc = m & conflict
        s, tx, ty, ag, npairs = pair_counts(x[mc], r_all[mc], ids[mc])
        sc, txc, tyc, agc, nc = sc + s, txc + tx, tyc + ty, agc + ag, nc + npairs
        s, tx, ty, _, _ = pair_counts(x[m], r_all[m], ids[m])
        sa, txa, tya = sa + s, txa + tx, tya + ty
    return (u / d if d > 0 else np.nan,
            sc / np.sqrt(txc * tyc) if txc * tyc > 0 else np.nan,
            agc / nc if nc > 0 else np.nan,
            sa / np.sqrt(txa * tya) if txa * tya > 0 else np.nan)


def within_pairs(labels):
    """Position pairs (p < q) with the same label."""
    ps, qs = [], []
    for g in np.unique(labels):
        pos = np.flatnonzero(labels == g)
        if len(pos) >= 2:
            i, j = np.triu_indices(len(pos), 1)
            ps.append(pos[i])
            qs.append(pos[j])
    if not ps:
        return np.array([], int), np.array([], int)
    return np.concatenate(ps), np.concatenate(qs)


def paired_stats(x, conflict, r_all, ids, p, q):
    """The same as grouped_stats, counted pair by pair over the given position pairs (fast for many small groups,
    e.g. the 202 segments)."""
    dx = np.sign(x[p] - x[q])
    cp, cq = conflict[p], conflict[q]
    one = cp != cq
    sgn = np.where(cp[one], dx[one], -dx[one])            # +1: the conflict frame scores higher
    auc_ = float(((sgn > 0) + 0.5 * (sgn == 0)).mean()) if one.any() else np.nan

    def tau_agree(mask):
        a_, b_ = dx[mask], np.sign(r_all[p[mask]] - r_all[q[mask]])
        tx, ty = (a_ != 0).sum(), (b_ != 0).sum()
        tau = float((a_ * b_).sum() / np.sqrt(tx * ty)) if tx * ty > 0 else np.nan
        other = ids[p[mask]] != ids[q[mask]]                 # not a frame with its own bootstrap copy
        agree = float(((a_ == b_) & other).sum() / other.sum()) if other.any() else np.nan
        return tau, agree

    tau_c, agree_c = tau_agree(cp & cq)
    tau_all, _ = tau_agree(np.ones(len(p), bool))
    return auc_, tau_c, agree_c, tau_all


def stats_by_grouping(x, conflict, r_all, ids, labels, pairs=None):
    """(auc, tau_c, agree_c, tau_all) over pairs with the same label: pair by pair when there are many small groups,
    otherwise group by group."""
    _, counts = np.unique(labels, return_counts=True)
    if len(counts) > 20 and counts.max() <= 200:
        p, q = pairs if pairs is not None else within_pairs(labels)
        return paired_stats(x, conflict, r_all, ids, p, q)
    return grouped_stats(x, conflict, r_all, ids, labels)


def all_stats(x, ssm, ids, groupings):
    """Statistics of one frame-score vector x against every SSM, for every grouping. Returns
    {ssm: ({grouping: (auc, tau_c, rho_c, agree_c, tau_all)}, n_conflict, n_no_conflict)}; rho_c only for 'all'.
    Frames with x = NaN are left out."""
    ok = ~np.isnan(x)
    xs, ids_s = x[ok], ids[ok]
    labs = {g: v[ok] for g, v in groupings.items()}
    pairs = {g: within_pairs(v) for g, v in labs.items() if g != 'all' and len(np.unique(v)) > 20}
    out = {}
    for s in SSMS:
        conflict = ssm[s + '_conflict'][ok]
        r = ssm[s + '_risk'][ok]
        if conflict.any() and len(xs) >= 3:
            r_all = np.where(conflict, r, np.nanmin(r[conflict]) - 1.0)   # no conflict: one tie at the safe end
        else:
            r_all = None
        per = {}
        for g, lab in labs.items():
            if r_all is None:
                per[g] = (np.nan,) * 5
                continue
            a, tc, ac, ta = stats_by_grouping(xs, conflict, r_all, ids_s, lab, pairs.get(g))
            rho = float(spearmanr(xs[conflict], r[conflict]).statistic) if g == 'all' and conflict.sum() >= 3 \
                else np.nan
            per[g] = (a, tc, rho, ac, ta)
        out[s] = (per, int(conflict.sum()), int((~conflict).sum()))
    return out


def repeat_stability(req_mc):
    """Repeat stability of one model x condition: mean SD over repeats (ddof 0, as reported in docs/handoff.md),
    mean test-retest Kendall tau-b between repeats, and ICC(1) one-way random effects for one answer and for the mean
    of k answers (frames with all k answers)."""
    w = req_mc.pivot_table(index=['segment', 'frame_idx'], columns='repeat', values='risk')
    sd = w.std(axis=1, ddof=0).mean()
    taus = []
    for a, b in itertools.combinations(w.columns, 2):
        m = w[[a, b]].dropna()
        if len(m) >= 3:
            taus.append(kendalltau(m[a], m[b], variant='b').statistic)
    full = w.dropna()
    k, n = full.shape[1], full.shape[0]
    icc1 = icck = np.nan
    if n >= 2 and k >= 2:
        grand = full.values.mean()
        msb = k * ((full.mean(axis=1) - grand) ** 2).sum() / (n - 1)
        msw = ((full.values - full.mean(axis=1).values[:, None]) ** 2).sum() / (n * (k - 1))
        icc1 = (msb - msw) / (msb + (k - 1) * msw)
        icck = (msb - msw) / msb
    return dict(sd_repeats_mean=sd, retest_tau_b_mean=float(np.nanmean(taus)) if taus else np.nan,
                icc_single=icc1, icc_mean_of_k=icck, k_repeats=k, frames_all_k=n)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('vlm_scores')
    ap.add_argument('frame_ssm')
    ap.add_argument('out_dir')
    ap.add_argument('--scene_labels', default=None, help='CSV with one row per segment (labels/scene_labels.csv)')
    ap.add_argument('--group_by', default='weather,time_of_day',
                    help="label columns of --scene_labels to group by ('' = none); 'all' and 'segment' are always on")
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261021)
    ap.add_argument('--resample', choices=['frame', 'segment'], default='frame')
    a = ap.parse_args()
    group_by = [c for c in a.group_by.split(',') if c] if a.scene_labels else []
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()

    # --- replies
    reqs, info = [], []
    for d in sorted(glob.glob(os.path.join(a.vlm_scores, '*', ''))):
        name = os.path.basename(d.rstrip('/\\'))
        if not os.path.exists(os.path.join(d, f'pilot_{name}.jsonl')):
            continue
        r, i = read_model(d)
        reqs.append(r)
        info.append(i)
        print(f"read {name}: {i['lines']} requests, forced answers used {i['forced_used']} ({time.time() - t0:.0f} s)")
    req = pd.concat(reqs, ignore_index=True)
    models = [i['model'] for i in info]
    req.to_csv(os.path.join(a.out_dir, 'vlm_requests_tidy.csv'), index=False)

    # every model must have asked about the same frames
    frames = req[['segment', 'frame_idx']].drop_duplicates().sort_values(['segment', 'frame_idx']).reset_index(drop=True)
    for m in models:
        fm = req.loc[req.model == m, ['segment', 'frame_idx']].drop_duplicates()
        if len(fm) != len(frames):
            raise ValueError(f'{m} asked about {len(fm)} frames, the union is {len(frames)}')

    # --- frame scores
    g = req.groupby(['model', 'condition', 'segment', 'frame_idx'])
    fs = g.agg(n_requests=('repeat', 'size'), n_answers=('risk', 'count'), n_forced=('forced', 'sum'),
               score_mean=('risk', 'mean'), score_median=('risk', 'median'),
               sd_repeats=('risk', lambda s: s.std(ddof=0))).reset_index()
    nf = req[~req.forced].groupby(['model', 'condition', 'segment', 'frame_idx']).risk.mean() \
        .rename('score_mean_no_forced').reset_index()
    fs = fs.merge(nf, on=['model', 'condition', 'segment', 'frame_idx'], how='left')
    fs.to_csv(os.path.join(a.out_dir, 'vlm_frame_scores.csv'), index=False)

    # --- SSMs and scene groupings
    ssm_df = ssm_table(a.frame_ssm, frames)
    groupings, labels_used = scene_groupings(frames, a.scene_labels, group_by)
    for gname in group_by:
        ssm_df[gname] = groupings[gname]
    ssm_df.to_csv(os.path.join(a.out_dir, 'ssm_target_frames.csv'), index=False)
    gnames = list(groupings)
    ssm = {c: ssm_df[c].to_numpy() for c in ssm_df.columns if c.endswith(('_conflict', '_risk'))}
    ssm = {k: (v.astype(bool) if k.endswith('_conflict') else v.astype(float)) for k, v in ssm.items()}
    n_frames = len(frames)
    ids = np.arange(n_frames)
    seg_of = frames['segment'].to_numpy()
    seg_list = np.array(sorted(set(seg_of)))
    seg_frames = {s: np.flatnonzero(seg_of == s) for s in seg_list}
    seg_label = {gname: {s: groupings[gname][seg_frames[s][0]] for s in seg_list} for gname in group_by}

    def score_vec(m, c, col):
        s = fs[(fs.model == m) & (fs.condition == c)].set_index(['segment', 'frame_idx'])[col]
        return s.reindex(pd.MultiIndex.from_frame(frames)).to_numpy(float)

    # --- point estimates (main variant + checks), and each segment on its own
    rows, seg_rows = [], []
    for m, c in itertools.product(models, CONDITIONS):
        for variant, col in [('mean', 'score_mean'), ('mean_no_forced', 'score_mean_no_forced'),
                             ('median', 'score_median')]:
            st = all_stats(score_vec(m, c, col), ssm, ids, groupings)
            for s, (per, n_c, n_n) in st.items():
                for gname, v in per.items():
                    rows.append(dict(model=m, condition=c, ssm=s, grouping=gname,
                                     n_groups=len(np.unique(groupings[gname])), variant=variant,
                                     **dict(zip(STATS, v)), n_conflict=n_c, n_no_conflict=n_n))
        x = score_vec(m, c, 'score_mean')
        for sg, fi in seg_frames.items():
            st = all_stats(x[fi], {k: v[fi] for k, v in ssm.items()}, ids[fi], {'all': np.zeros(len(fi), int)})
            for s, (per, n_c, n_n) in st.items():
                v = per['all']
                seg_rows.append(dict(model=m, condition=c, segment=sg, ssm=s, auc=v[0], tau_b_conflict=v[1],
                                     agreement_conflict=v[3], tau_b_all=v[4], n_conflict=n_c, n_no_conflict=n_n,
                                     score_mean=float(np.nanmean(x[fi])),
                                     **{gname: seg_label[gname][sg] for gname in group_by}))
    est = pd.DataFrame(rows)
    pd.DataFrame(seg_rows).to_csv(os.path.join(a.out_dir, 'agreement_by_segment.csv'), index=False)

    # --- bootstrap (main variant). 'frame': frames resampled within each segment, so every segment keeps its
    # number of target frames; 'segment': whole segments resampled, every drawn copy is its own segment group
    rng = np.random.default_rng(a.seed)
    xs = {(m, c): score_vec(m, c, 'score_mean') for m, c in itertools.product(models, CONDITIONS)}
    boot = np.full((a.n_boot, len(models), len(CONDITIONS), len(SSMS), len(gnames), len(STATS)), np.nan)
    tb = time.time()
    for b in range(a.n_boot):
        if a.resample == 'frame':
            idx = np.concatenate([rng.choice(seg_frames[s], len(seg_frames[s])) for s in seg_list])
            gb = {gname: v[idx] for gname, v in groupings.items()}
        else:
            draw = rng.choice(seg_list, len(seg_list))
            idx = np.concatenate([seg_frames[s] for s in draw])
            gb = {gname: v[idx] for gname, v in groupings.items()}
            gb['segment'] = np.concatenate([np.full(len(seg_frames[s]), k) for k, s in enumerate(draw)])
        sb = {k: v[idx] for k, v in ssm.items()}
        for (mi, m), (ci, c) in itertools.product(enumerate(models), enumerate(CONDITIONS)):
            st = all_stats(xs[(m, c)][idx], sb, ids[idx], gb)
            for si, s in enumerate(SSMS):
                for gi, gname in enumerate(gnames):
                    boot[b, mi, ci, si, gi] = st[s][0][gname]
        if b in (0, 9) or (b + 1) % 250 == 0:
            done = b + 1
            print(f'bootstrap {done}/{a.n_boot} ({(time.time() - tb) / done * (a.n_boot - done):.0f} s left)')

    def ci(arr):
        ok = ~np.isnan(arr)
        if ok.sum() < 0.9 * len(arr):                          # too many undefined resamples
            return np.nan, np.nan, int(ok.sum())
        lo, hi = np.percentile(arr[ok], [2.5, 97.5])
        return float(lo), float(hi), int(ok.sum())

    main_rows = est[est.variant == 'mean'].drop(columns='variant').copy()
    for k, stat in enumerate(STATS):
        lo, hi, nv = [], [], []
        for r in main_rows.itertuples():
            v = ci(boot[:, models.index(r.model), CONDITIONS.index(r.condition), SSMS.index(r.ssm),
                        gnames.index(r.grouping), k])
            lo.append(v[0]), hi.append(v[1]), nv.append(v[2])
        main_rows[stat + '_lo'], main_rows[stat + '_hi'], main_rows[stat + '_nboot'] = lo, hi, nv
    checks = est[est.variant != 'mean'].pivot_table(index=['model', 'condition', 'ssm', 'grouping'],
                                                    columns='variant', values=STATS)
    checks.columns = [f'{s}_{v}' for s, v in checks.columns]
    agreement = main_rows.merge(checks.reset_index(), on=['model', 'condition', 'ssm', 'grouping'], how='left')
    agreement.to_csv(os.path.join(a.out_dir, 'agreement.csv'), index=False)

    # --- differences between SSMs (overall tau-b and AUC, per grouping), paired over the same resamples
    drows = []
    for (mi, m), (ci_, c) in itertools.product(enumerate(models), enumerate(CONDITIONS)):
        for gi, gname in enumerate(gnames):
            pe = main_rows[(main_rows.model == m) & (main_rows.condition == c) &
                           (main_rows.grouping == gname)].set_index('ssm')
            for (i, s1), (j, s2) in itertools.combinations(enumerate(SSMS), 2):
                for stat in DIFF_STATS:
                    k = STATS.index(stat)
                    lo, hi, nv = ci(boot[:, mi, ci_, i, gi, k] - boot[:, mi, ci_, j, gi, k])
                    drows.append(dict(model=m, condition=c, grouping=gname, stat=stat, ssm_1=s1, ssm_2=s2,
                                      diff=pe.at[s1, stat] - pe.at[s2, stat], diff_lo=lo, diff_hi=hi, nboot=nv,
                                      clear=bool(lo > 0 or hi < 0) if not np.isnan(lo) else False))
    pd.DataFrame(drows).to_csv(os.path.join(a.out_dir, 'ssm_differences.csv'), index=False)

    # --- per model x condition: answers, stability, time
    crow = []
    for m, c in itertools.product(models, CONDITIONS):
        q = req[(req.model == m) & (req.condition == c)]
        st = repeat_stability(q)
        crow.append(dict(model=m, condition=c, requests=len(q), answered=int(q.risk.notna().sum()),
                         forced_answers=int(q.forced.sum()), no_answer=int(q.risk.isna().sum()),
                         hit_max_tokens=int((q.finish_reason == 'length').sum()),
                         score_min=q.risk.min(), score_max=q.risk.max(), distinct_scores=q.risk.nunique(),
                         **st, latency_median_s=q.latency_s.median(), latency_max_s=q.latency_s.max(),
                         completion_tokens_median=q.completion_tokens.median(),
                         reasoning_tokens_median=(q.reasoning_tokens.dropna().median()
                                                  if q.reasoning_tokens.notna().any() else np.nan),
                         camera_written_differently=int(q.camera_fixed.sum()),
                         hazard_type_written_differently=int(q.hazard_type_fixed.sum()),
                         temperature=q.temperature.iloc[0], top_p=q.top_p.iloc[0], top_k=q.top_k.iloc[0]))
    cond = pd.DataFrame(crow)
    cond.to_csv(os.path.join(a.out_dir, 'condition_summary.csv'), index=False)

    # --- non-standard field values (normalised in vlm_requests_tidy.csv)
    odd = pd.concat([req.loc[req.camera_fixed, ['model', 'camera_raw', 'camera']].rename(
        columns={'camera_raw': 'written', 'camera': 'used'}).assign(field='main_hazard_camera'),
        req.loc[req.hazard_type_fixed, ['model', 'hazard_type_raw', 'hazard_type']].rename(
        columns={'hazard_type_raw': 'written', 'hazard_type': 'used'}).assign(field='hazard_type')])
    odd = odd.groupby(['field', 'written', 'used']).size().rename('replies').reset_index()

    # --- params.txt
    with open(os.path.join(a.out_dir, 'params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'analyze_vlm_ssm.py run {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'vlm_scores = {os.path.abspath(a.vlm_scores)}\nframe_ssm = {os.path.abspath(a.frame_ssm)}\n')
        f.write(f'scene_labels = {os.path.abspath(a.scene_labels) if a.scene_labels else "(none)"}\n')
        how = 'frames resampled within each segment' if a.resample == 'frame' else 'segments resampled'
        f.write(f'bootstrap: {a.n_boot} resamples ({how}), seed {a.seed}, 95 % percentile intervals\n')
        f.write(f'target frames: {n_frames} ({", ".join(f"{s[:6]} {len(v)}" for s, v in seg_frames.items())})\n')
        f.write('groupings (statistics counted over frame pairs in the same group):\n')
        for gname in gnames:
            vals, cnt = np.unique(groupings[gname], return_counts=True)
            desc = 'one group' if gname == 'all' else ', '.join(
                f'{str(v)[:20]} {k}' for v, k in zip(vals, cnt)) if len(vals) <= 12 else f'{len(vals)} groups'
            f.write(f'  {gname}: {desc} (frames)\n')
        if labels_used is not None:
            f.write('scene labels per segment:\n')
            for s, r in labels_used.iterrows():
                f.write(f'  {s}: ' + ', '.join(f'{k}={v}' for k, v in r.items()) + '\n')
        for i in info:
            f.write(f"model {i['model']}: {i['lines']} requests ({os.path.basename(i['file'])}), "
                    f"forced answers used {i['forced_used']}\n")
        f.write('frames with a value (conflict) / without, per SSM:\n')
        for s in SSMS:
            f.write(f'  {s}: {int(ssm[s + "_conflict"].sum())} / {int((~ssm[s + "_conflict"]).sum())}\n')
        f.write('replies written differently from the prompt (normalised):\n')
        for r in odd.itertuples():
            f.write(f'  {r.field}: "{r.written}" -> "{r.used}": {r.replies}\n')
        f.write('conventions: frame score = mean of answered repeats; SD over repeats with ddof 0; '
                'larger = riskier for every SSM; no-conflict frames tied at the safe end in tau_b_all\n')

    # --- console summary
    pd.set_option('display.width', 200)
    print(f'\ntarget frames {n_frames}; frames with a value per SSM: ' +
          ', '.join(f'{s} {int(ssm[s + "_conflict"].sum())}' for s in SSMS))
    print('\nrepeat stability and cost (main points):')
    print(cond[['model', 'condition', 'answered', 'forced_answers', 'no_answer', 'sd_repeats_mean',
                'retest_tau_b_mean', 'icc_mean_of_k', 'latency_median_s']].round(3).to_string(index=False))
    for gname in gnames:
        print(f'\noverall Kendall tau-b, pairs within the same {gname} group (no conflict tied at the safe end):')
        tab = main_rows[main_rows.grouping == gname].pivot_table(index=['model', 'condition'], columns='ssm',
                                                                 values='tau_b_all')[SSMS]
        print(tab.round(3).to_string())
    print(f'\ndone in {time.time() - t0:.0f} s; outputs in {os.path.abspath(a.out_dir)}')


if __name__ == '__main__':
    main()
