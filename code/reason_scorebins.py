"""EXPLORATORY (user, 2026-11-14; plan in CLAUDE.md "理由分析" -> "分数分段", recorded and committed before this program
was written; not changed after the results): Qwen3.6-27B's risk score cut into fixed score bins, and what the frames of
each bin have in common (what it said, what the scene has, how things move); no clustering. CPU only; no image is opened.
Data: sep_camera, the main result's 1,084 frames (238 segments), 5,420 answers (26 forced). Existing results, not redone:
round 3's frame and answer tables (reason_keywords.py: 20 keyword classes, hazard_type, camera), round 2's 9 SSM risk
percentiles, the 12 scene cues of round 2 (reason_frames.CUES; "red light" is not in the data), quantities/truth_main.csv
(true distance, true closing speed v_ego - v_lead, ego speed). The main_hazard + reason texts are read only for section 4.
Bins (plan section 2 as the user changed it): per frame by score_mean, main 3 bins (< 20, 20 - < 40, >= 40; expected
243 / 532 / 309 frames), sensitivity 5 bins (< 20, 20 - < 30, 30 - < 40, 40 - < 50, >= 50; 243 / 303 / 229 / 154 / 155);
the program stops (exit 2, no tables) when a count differs. Per answer: the answer's own score, the same cuts. The plan's
"三种切法" in sections 3, 6 and 7 is read as these two frame cuts (section 2 lists only two).
Per frame bin (both cuts): the share of frames mentioning each keyword class (3+ of 5 answers "mentioned"), the shares of
each most frequent hazard_type and camera (round 3's columns, 'mixed' on a tie), the share of frames with each cue (over
the frames with a value), the medians of the 9 SSM risk percentiles and of the 3 truth quantities (with quartiles and IQR,
numpy linear, frames with a value); highest bin - lowest bin; Kendall tau-b of each item with the bin number (1..3 or 1..5)
over the frames with a value (scipy kendalltau, variant b); for the highest bin, the segments of its frames with the item
(binary items) or with a value (the others) and the largest segment's frames. Per answer bin: the keyword classes
(mentioned) and hazard_type shares, highest - lowest. Intervals: the main analysis's 2,000 resamples of whole segments
(default_rng(20261021), one choice over the 238 sorted segments per resample, sha256 48bc1f41...), every frame (answer)
of a drawn segment as often as it is drawn; the bins are fixed score cuts, so re-cutting in a resample keeps each frame's
bin; 2.5 / 97.5 % (numpy linear) over the resamples where defined, the others counted as missing.
Distinguishing terms (per answer; main: 3-bin highest vs lowest, sensitivity: 5-bin highest vs lowest): main_hazard and
reason each cleaned as round 1 (reason_clusters.clean: lower case, numbers -> num, punctuation -> space) and counted with
CountVectorizer(stop_words = round 1's custom list + risk words, ngram_range (1, 2), the rest default), the two fields
counted apart and added (no bigram across the two fields); log-odds ratio with an informative Dirichlet prior (Monroe,
Colaresi & Quinn 2008, eq. 22 / 24): alpha_w = the term's count over all 5,420 answers, alpha_0 = their sum, y the term
counts, n the group totals over the whole vocabulary; delta = log((y_i + a_w) / (n_i + a_0 - y_i - a_w)) - log((y_j + a_w)
/ (n_j + a_0 - y_j - a_w)), var = 1 / (y_i + a_w) + 1 / (y_j + a_w), z = delta / sqrt(var); eligible terms: in 20+ answers
and from 5+ segments over all 5,420 answers; 30 highest and 30 lowest z (ties: term order).
Checks (exit 1 when one is not met): 5,420 answers, 1,084 frames, 238 segments, 5 per frame; frame means = score_mean;
round 3's answer table in the same order with the same scores; round 3's keyword counts, hazard_type and camera per frame =
a recomputation from its answer table; the percentiles = round 2's and a recomputation; the cue counts over all frames =
round 2's _cues.csv; every bin cut adds up to 1,084 frames (5,420 answers); the drawn segments' sha256. The run script
compares a second run byte for byte.
usage: python -I reason_scorebins.py <waymo_v143 folder> <round-2 out_dir> <round-3 out_dir> <out_dir> [--workers N]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import collections  # noqa: E402
import hashlib  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau, rankdata  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_rules as ar  # noqa: E402
import analyze_vlm_ssm as av  # noqa: E402
import reason_clusters as rc  # noqa: E402
import reason_frames as rf  # noqa: E402
import reason_keywords as rk  # noqa: E402

MODEL, COND = rk.MODEL, rk.COND
N_BOOT, BOOT_SEED, HASH = rk.N_BOOT, rk.BOOT_SEED, rk.HASH
CUTS = {3: [20.0, 40.0], 5: [20.0, 30.0, 40.0, 50.0]}
EXPECT = {3: [243, 532, 309], 5: [243, 303, 229, 154, 155]}
MIN_ANS, MIN_SEG, N_TOP = 20, 5, 30
PFX = 'exploratory_scorebins_'
LOW_ACC = {19: 'low accuracy (user check 9/20), not interpreted'}
G = {}                                                            # data for the worker processes (fork)


def bin_of(x, cut):
    return np.digitize(x, CUTS[cut]) + 1                          # x < 20 -> 1, 20 <= x < 40 -> 2, ...


def bin_label(cut, k):
    e = CUTS[cut]
    if k == 1:
        return f'< {e[0]:g}'
    if k == len(e) + 1:
        return f'>= {e[-1]:g}'
    return f'{e[k - 2]:g} - < {e[k - 1]:g}'


def stat_cols(V, binary):
    """per column: mean of the non-missing (binary items) or median (the others); nan when none."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        mean, medi = np.nanmean(V, axis=0), np.nanmedian(V, axis=0)
    return np.where(binary, mean, medi)


def tau(b, v):
    ok = ~np.isnan(v)
    if ok.sum() < 2:
        return np.nan
    t = kendalltau(b[ok], v[ok], variant='b').statistic
    return float(t)


def stats(idx):
    g = G
    V, out = g['V'][idx], []
    ia = g['rows5'][idx].ravel()
    Aa = g['A'][ia]
    for cut in (3, 5):
        b = g['bins'][cut][idx]
        per = [stat_cols(V[b == k], g['binary']) if (b == k).any() else np.full(V.shape[1], np.nan)
               for k in range(1, cut + 1)]
        out += [np.concatenate(per), per[-1] - per[0], np.array([tau(b, V[:, j]) for j in range(V.shape[1])])]
        ba = g['abins'][cut][ia]
        pa = [Aa[ba == k].mean(0) if (ba == k).any() else np.full(Aa.shape[1], np.nan) for k in range(1, cut + 1)]
        out += [np.concatenate(pa), pa[-1] - pa[0]]
    return np.concatenate(out)


def one(bi):
    idx = np.concatenate([G['seg_frames'][s] for s in G['draws'][bi]])
    return stats(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('waymo')
    ap.add_argument('round2')
    ap.add_argument('round3')
    ap.add_argument('out')
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    L, R2, R3, O = a.waymo, a.round2, a.round3, a.out
    A, MD, FSSM = rc.paths(L)
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.exists(os.path.join(O, PFX + 'params.txt')):
        sys.exit(f'{O}/{PFX}params.txt exists already: give a new folder')
    os.makedirs(O, exist_ok=True)
    t0 = time.time()
    failed, rows_ck = [], []

    def check(ok, what):
        print(f'  check {"ok" if ok else "NOT MET"}: {what}', flush=True)
        rows_ck.append(dict(check=what, ok=bool(ok)))
        if not ok:
            failed.append(what)

    with open(os.path.abspath(__file__), 'rb') as f:
        prog_sha = hashlib.sha256(f.read()).hexdigest()
    print(f'score bins, {MODEL} {COND} (user, 2026-11-14; EXPLORATORY); run {time.strftime("%Y-%m-%d %H:%M:%S")}; '
          f'program sha256 {prog_sha}\n\n0. inputs', flush=True)

    # ------------------------------------------------------------------------------------------------- answers, texts
    td = pd.read_csv(os.path.join(A, 'vlm_requests_tidy.csv'), low_memory=False, keep_default_na=False, na_values=[''],
                     usecols=['model', 'condition', 'segment', 'frame_idx', 'repeat', 'risk', 'forced'])
    td = td[(td.model == MODEL) & (td.condition == COND)].sort_values(['segment', 'frame_idx', 'repeat']) \
        .reset_index(drop=True)
    td['frame_idx'], td['repeat'] = td.frame_idx.astype(int), td['repeat'].astype(int)
    td['forced'] = td.forced.astype(str).str.lower().eq('true')
    keys = {(s, k, r): i for i, (s, k, r) in enumerate(zip(td.segment, td.frame_idx, td['repeat']))}
    main_f = av.jsonl_file(MD, f'main_{MODEL}')
    forced_f = av.jsonl_file(MD, f'main_{MODEL}_forced_answers')
    parsed = [None] * len(td)
    forced_read = [False] * len(td)
    for fn, is_forced in ((main_f, False), (forced_f, True)):
        with av.open_jsonl(fn) as f:
            for raw in f:
                d = av.load_light(raw)
                if d['condition'] != COND:
                    continue
                i = keys.get((d['segment'], int(d['frame_idx']), int(d['repeat'])))
                if i is None:
                    continue
                if not is_forced:
                    if parsed[i] is not None:
                        sys.exit(f'duplicate request {d["segment"]} {d["frame_idx"]} {d["repeat"]}')
                    parsed[i] = d.get('parsed') or {}
                elif d.get('parsed'):
                    if parsed[i]:
                        sys.exit(f'forced answer for a reply that has one: {d["segment"]} {d["frame_idx"]}')
                    parsed[i], forced_read[i] = d['parsed'], True
    ok_all = all(p for p in parsed)
    check(ok_all and np.array_equal(np.array([float(p['risk']) for p in parsed]) if ok_all else [],
                                    td.risk.to_numpy(float)) and forced_read == td.forced.tolist(),
          f'{len(td)} answers read from {main_f} and {forced_f}; the risk = the tidy table\'s; forced as the tidy '
          f'table ({int(td.forced.sum())})')
    if not ok_all:
        sys.exit('stopped: not every answer was read')

    def fld(p, k):
        v = p.get(k)
        return v if isinstance(v, str) else ''

    t_mh = [fld(p, 'main_hazard') for p in parsed]
    t_rs = [fld(p, 'reason') for p in parsed]
    n_empty = sum(not (x.strip() or y.strip()) for x, y in zip(t_mh, t_rs))

    frames = pd.read_csv(os.path.join(A, 'ssm_target_frames.csv'))[['segment', 'frame_idx']] \
        .sort_values(['segment', 'frame_idx']).reset_index(drop=True)
    n = len(frames)
    mi = pd.MultiIndex.from_frame(frames)
    fpos = pd.Series(np.arange(n), index=mi).reindex(pd.MultiIndex.from_arrays([td.segment, td.frame_idx])).to_numpy()
    reps = collections.defaultdict(dict)
    for i, (fp, rp) in enumerate(zip(fpos, td['repeat'])):
        if not np.isnan(fp):
            reps[int(fp)][int(rp)] = i
    check(len(td) == 5420 and n == 1084 and frames.segment.nunique() == 238 and not np.isnan(fpos.astype(float)).any()
          and all(sorted(reps[f_]) == [0, 1, 2, 3, 4] for f_ in range(n)),
          f'{len(td)} answers, {n} frames, {frames.segment.nunique()} segments, 5 answers per frame (repeats 0-4)')
    rows5 = np.array([[reps[f_][r] for r in range(5)] for f_ in range(n)])
    p_fs = os.path.join(A, 'vlm_frame_scores.csv')
    fs = pd.read_csv(p_fs, float_precision='round_trip')
    score = fs[(fs.model == MODEL) & (fs.condition == COND)].set_index(['segment', 'frame_idx'])['score_mean'] \
        .reindex(mi).to_numpy(float)
    fm = td.groupby(['segment', 'frame_idx'], sort=True)['risk'].mean().reindex(mi).to_numpy(float)
    check(not np.isnan(score).any() and np.array_equal(fm, score),
          f'frame means of the 5 answers = {p_fs} score_mean frame by frame')
    risk_a = td.risk.to_numpy(float)

    # ------------------------------------------------------------------------------------------------- round 3
    p_a3 = os.path.join(R3, 'exploratory_keywords_answers.csv')
    p_f3 = os.path.join(R3, 'exploratory_keywords_frames.csv')
    a3 = pd.read_csv(p_a3, keep_default_na=False, float_precision='round_trip')
    f3 = pd.read_csv(p_f3, keep_default_na=False, float_precision='round_trip').set_index(['segment', 'frame_idx']) \
        .reindex(mi)
    check(len(a3) == len(td) and (a3.segment.to_numpy() == td.segment.to_numpy()).all()
          and (a3.frame_idx.to_numpy() == td.frame_idx.to_numpy()).all()
          and (a3['repeat'].to_numpy() == td['repeat'].to_numpy()).all()
          and np.array_equal(a3.risk.to_numpy(float), risk_a),
          f'{p_a3}: the same answers in the same order, the same scores')
    check(np.array_equal(f3.score_mean.to_numpy(float), score), f'{p_f3} score_mean = score_mean frame by frame')
    kw_a = np.stack([(a3[f'k{c:02d}'] == 'mentioned').to_numpy() for c in range(1, 21)], axis=1)
    kw_cnt = np.stack([f3[f'k{c:02d}_answers'].to_numpy(int) for c in range(1, 21)], axis=1)
    check(np.array_equal(kw_a[rows5].sum(1), kw_cnt), f'the 20 keyword counts of {p_f3} = a recomputation from {p_a3}')

    def mode5(vals):
        c = collections.Counter(vals)
        top = max(c.values())
        w = sorted(k for k, v in c.items() if v == top)
        return w[0] if len(w) == 1 else 'mixed'

    hz_a, cam_a = a3.hazard_type.to_numpy(object), a3.main_hazard_camera.to_numpy(object)
    hz_f, cam_f = f3.hazard_type_most.to_numpy(object), f3.camera_most.to_numpy(object)
    check([mode5(list(hz_a[r])) for r in rows5] == list(hz_f) and [mode5(list(cam_a[r])) for r in rows5] == list(cam_f),
          f'hazard_type_most and camera_most of {p_f3} = a recomputation from {p_a3}')
    p_lab = os.path.join(R3, 'exploratory_keywords_sample_labeled.csv')
    lab = pd.read_csv(p_lab, dtype=str, keep_default_na=False)
    acc = {}
    for c_, sub in lab.groupby('category'):
        if c_.isdigit():
            v = sub.correct_1_or_0.str.strip()
            acc[int(c_)] = f'{int((v == "1").sum())}/{len(v)}'

    # ------------------------------------------------------------------------------------------------- SSMs, truth, cues
    ssm_df = av.ssm_table(FSSM, frames)
    risk = {s: ssm_df[s + '_risk'].to_numpy(float) for s in av.SSMS}
    p_r2 = os.path.join(R2, 'exploratory_reason_frames.csv')
    r2 = pd.read_csv(p_r2, float_precision='round_trip').set_index(['segment', 'frame_idx']).reindex(mi)
    pct = {s: r2[f'{s}_risk_percentile'].to_numpy(float) for s in av.SSMS}
    check(all(np.array_equal(pct[s], (rankdata(risk[s], method='average') - 1) / (n - 1) * 100)
              and np.array_equal(pct[s], f3[f'{s}_risk_percentile'].to_numpy(float)) for s in av.SSMS),
          f'the 9 SSM risk percentiles of {p_r2} (used) = a recomputation = {p_f3}, frame by frame')
    p_tr = os.path.join(L, 'quantities', 'truth_main.csv')
    tr = frames.merge(pd.read_csv(p_tr, float_precision='round_trip'), on=['segment', 'frame_idx'], how='left',
                      validate='one_to_one')
    truth = {'true_distance': tr['distance'].to_numpy(float),
             'true_closing_speed': (tr['v_ego'] - tr['v_lead']).to_numpy(float),
             'ego_speed': tr['v_ego'].to_numpy(float)}
    labels_csv = os.path.join(REPO, 'labels', 'scene_labels.csv')
    sl = pd.read_csv(labels_csv, dtype=str, keep_default_na=False).set_index('segment')
    cue_val, cue_src = {}, {}
    for name, file_, col, rule, _ in rf.CUES:
        if file_ is None:
            continue
        if file_ == 'labels':
            v = sl.loc[frames.segment, col].str.strip().to_numpy()
            cue_val[name] = np.where(v == rule.split('== ')[1], 1.0, 0.0)
            cue_src[name] = labels_csv
        else:
            p_ = os.path.join(L, file_)
            t_ = pd.read_csv(p_, float_precision='round_trip')
            t_ = frames.merge(t_[['segment', 'frame_idx', col]], on=['segment', 'frame_idx'], how='left',
                              validate='one_to_one')[col].to_numpy(float)
            cue_val[name] = np.where(np.isnan(t_), np.nan, (t_ >= float(rule.split('>= ')[1])).astype(float))
            cue_src[name] = p_
    p_q2 = os.path.join(R2, 'exploratory_reason_frames_cues.csv')
    q2 = pd.read_csv(p_q2, keep_default_na=False)
    q2 = q2[q2.cluster.astype(str) == 'all']
    same_q = all((int(np.sum(~np.isnan(cue_val[r.cue]))), int(np.nansum(cue_val[r.cue])))
                 == (int(r.frames_with_value), int(float(r.frames_with_cue))) for r in q2.itertuples()
                 if r.cue in cue_val) and len(q2) == len(rf.CUES)
    check(same_q, f'the cue counts over all frames = {p_q2} (cluster "all")')

    # ------------------------------------------------------------------------------------------------- bins
    bins = {c: bin_of(score, c) for c in CUTS}
    abins = {c: bin_of(risk_a, c) for c in CUTS}
    seg_names = frames.segment.to_numpy()
    seg_a = td.segment.to_numpy()
    rows_b = []
    for c in CUTS:
        cnts = [int((bins[c] == k).sum()) for k in range(1, c + 1)]
        acnt = [int((abins[c] == k).sum()) for k in range(1, c + 1)]
        check(sum(cnts) == n and sum(acnt) == len(td), f'{c} bins: frames add up to {sum(cnts)}, answers {sum(acnt)}')
        for k in range(1, c + 1):
            m, ma = bins[c] == k, abins[c] == k
            rows_b.append(dict(bins=c, level='frame', bin=k, scores=bin_label(c, k), count=int(m.sum()),
                               frames=int(m.sum()), segments=int(len(set(seg_names[m]))),
                               min_score=float(score[m].min()) if m.any() else np.nan,
                               max_score=float(score[m].max()) if m.any() else np.nan,
                               expected_frames=EXPECT[c][k - 1]))
            rows_b.append(dict(bins=c, level='answer', bin=k, scores=bin_label(c, k), count=int(ma.sum()),
                               frames=int(len(set(fpos[ma].astype(int)))), segments=int(len(set(seg_a[ma]))),
                               min_score=float(risk_a[ma].min()) if ma.any() else np.nan,
                               max_score=float(risk_a[ma].max()) if ma.any() else np.nan, expected_frames=np.nan))
        print(f'  {c} bins: frames {cnts} (expected {EXPECT[c]}), answers {acnt}', flush=True)
    stop_counts = [c for c in CUTS if [int((bins[c] == k).sum()) for k in range(1, c + 1)] != EXPECT[c]]

    def W(name, rows):
        pd.DataFrame(rows).to_csv(os.path.join(O, PFX + name), index=False)

    if stop_counts:
        W('bins.csv', rows_b)
        W('check.csv', rows_ck)
        sys.exit(f'stopped: the frame counts of the {stop_counts} bin cut(s) differ from the plan; only bins.csv and '
                 'check.csv written')

    # ------------------------------------------------------------------------------------------------- items
    items = []                                                    # (group, name, values, binary, keyword class)
    for j in range(20):
        c = j + 1
        assert rk.CATS[j][0] == c
        items.append(('keyword', f'k{c:02d} {rk.CATS[j][1]}',
                      (kw_cnt[:, j] >= 3).astype(float), True, c))
    HZ = sorted(set(hz_f), key=lambda v: (v == 'mixed', v))
    CAM = sorted(set(cam_f), key=lambda v: (v == 'mixed', v))
    items += [('hazard_type_most', v, (hz_f == v).astype(float), True, None) for v in HZ]
    items += [('camera_most', v, (cam_f == v).astype(float), True, None) for v in CAM]
    items += [('cue', nm, cue_val[nm], True, None) for nm, file_, _, _, _ in rf.CUES if file_ is not None]
    items += [('ssm_risk_percentile', s, pct[s], False, None) for s in av.SSMS]
    items += [('truth', k, v, False, None) for k, v in truth.items()]
    V = np.stack([it[2] for it in items], axis=1)
    binary = np.array([it[3] for it in items])
    HZA = sorted(set(hz_a))
    aitems = [('keyword', items[j][1], kw_a[:, j].astype(float), j + 1) for j in range(20)] + \
             [('hazard_type', v, (hz_a == v).astype(float), None) for v in HZA]
    Am = np.stack([it[2] for it in aitems], axis=1)

    # ------------------------------------------------------------------------------------------------- resamples
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
    G.update(V=V, binary=binary, A=Am, rows5=rows5, bins=bins, abins=abins, draws=draws,
             seg_frames={s: np.flatnonzero(seg_names == s) for s in seg_list})
    point = stats(np.arange(n))
    I, J = V.shape[1], Am.shape[1]
    lay, pos = {}, 0
    for c in (3, 5):
        for part, size in (('f', c * I), ('fd', I), ('ft', I), ('a', c * J), ('ad', J)):
            lay[(c, part)] = pos
            pos += size
    assert pos == len(point)
    print(f'\n1. resamples: {N_BOOT}, {a.workers} worker(s), {len(point)} numbers each', flush=True)
    tb = time.time()
    with mp.get_context('fork').Pool(max(a.workers, 1)) as pool:
        boot = np.stack(list(pool.imap(one, range(N_BOOT), chunksize=8)))
    print(f'  done in {time.time() - tb:.0f} s', flush=True)

    comps = []

    def iv(c, part, j, kind, what):
        col = lay[(c, part)] + j
        lo, hi, nv, miss = rk.ci(boot[:, col])
        sd = rk.side(lo, hi, 0.0)
        if not np.isnan(lo):
            comps.append((kind, c, what, sd))
        return dict(value=float(point[col]), lo=lo, hi=hi, nboot=nv, missing=miss, side=sd)

    def note(it):
        if it[0] != 'keyword':
            return '', ''
        return acc.get(it[-1], ''), LOW_ACC.get(it[-1], '')

    # ------------------------------------------------------------------------------------------------- tables
    rows_f, rows_d, rows_t, rows_a, rows_ad = [], [], [], [], []
    for c in (3, 5):
        top = bins[c] == c
        for j, it in enumerate(items):
            grp, name, v, isb = it[0], it[1], it[2], it[3]
            ac, lw = note(it)
            for k in range(1, c + 1):
                m = bins[c] == k
                vv = v[m]
                ok = ~np.isnan(vv)
                r = dict(bins=c, bin=k, scores=bin_label(c, k), group=grp, item=name, statistic='share' if isb else
                         'median', frames=int(m.sum()), frames_with_value=int(ok.sum()),
                         frames_with_item=int(np.nansum(vv)) if isb else np.nan,
                         value=float(point[lay[(c, 'f')] + (k - 1) * I + j]))
                if not isb:
                    q1, q3 = (np.percentile(vv[ok], [25, 75]) if ok.any() else (np.nan, np.nan))
                    r.update(q25=float(q1), q75=float(q3), iqr=float(q3 - q1))
                else:
                    r.update(q25=np.nan, q75=np.nan, iqr=np.nan)
                r.update(keyword_accuracy=ac, note=lw)
                rows_f.append(r)
            sel = top & ((v == 1) if isb else ~np.isnan(v))
            segc = collections.Counter(seg_names[sel])
            d = iv(c, 'fd', j, 'frame: highest - lowest', f'{c} bins {grp} {name}')
            rows_d.append(dict(bins=c, group=grp, item=name, statistic='share difference' if isb else
                               'median difference', highest=float(point[lay[(c, 'f')] + (c - 1) * I + j]),
                               lowest=float(point[lay[(c, 'f')] + j]), difference=d['value'], lo=d['lo'], hi=d['hi'],
                               nboot=d['nboot'], missing=d['missing'], side=d['side'],
                               highest_bin_frames_with_item=int(sel.sum()), highest_bin_segments=len(segc),
                               highest_bin_largest_segment_frames=max(segc.values()) if segc else 0,
                               highest_bin_counted='frames with the item' if isb else 'frames with a value',
                               keyword_accuracy=ac, note=lw))
            t = iv(c, 'ft', j, 'frame: trend tau-b', f'{c} bins {grp} {name}')
            rows_t.append(dict(bins=c, group=grp, item=name, frames_with_value=int((~np.isnan(v)).sum()),
                               tau_b=t['value'], lo=t['lo'], hi=t['hi'], nboot=t['nboot'], missing=t['missing'],
                               side=t['side'], keyword_accuracy=ac, note=lw))
        for j, it in enumerate(aitems):
            ac, lw = note(it)
            for k in range(1, c + 1):
                ma = abins[c] == k
                rows_a.append(dict(bins=c, bin=k, scores=bin_label(c, k), group=it[0], item=it[1],
                                   answers=int(ma.sum()), answers_with_item=int(it[2][ma].sum()),
                                   share=float(point[lay[(c, 'a')] + (k - 1) * J + j]), keyword_accuracy=ac, note=lw))
            d = iv(c, 'ad', j, 'answer: highest - lowest', f'{c} bins {it[0]} {it[1]}')
            rows_ad.append(dict(bins=c, group=it[0], item=it[1], highest=float(point[lay[(c, 'a')] + (c - 1) * J + j]),
                                lowest=float(point[lay[(c, 'a')] + j]), difference=d['value'], lo=d['lo'], hi=d['hi'],
                                nboot=d['nboot'], missing=d['missing'], side=d['side'], keyword_accuracy=ac, note=lw))
    # 3 vs 5 bins, item by item
    rows_x = []
    for tab, kind in ((rows_d, 'frame: highest - lowest'), (rows_t, 'frame: trend tau-b'),
                      (rows_ad, 'answer: highest - lowest')):
        s3 = {(r['group'], r['item']): r['side'] for r in tab if r['bins'] == 3}
        s5 = {(r['group'], r['item']): r['side'] for r in tab if r['bins'] == 5}
        for k_ in s3:
            rows_x.append(dict(kind=kind, group=k_[0], item=k_[1], side_3_bins=s3[k_], side_5_bins=s5.get(k_, ''),
                               different=s3[k_] != s5.get(k_, '')))
    rows_c = []
    for c in (None, 3, 5):
        for kind in [None] + sorted({x[0] for x in comps}):
            sub = [x for x in comps if (c is None or x[1] == c) and (kind is None or x[0] == kind)]
            rows_c.append(dict(bins='all' if c is None else c, kind=kind or 'all', intervals=len(sub),
                               leaving_out_0=sum(x[3].startswith(('above', 'below')) for x in sub),
                               by_chance_about=0.05 * len(sub)))

    # ------------------------------------------------------------------------------------------------- terms
    print('\n2. distinguishing terms', flush=True)
    stop = sorted(set(rc.CUSTOM_STOP) | set(rc.RISK_WORDS))
    stop_sha = hashlib.sha256('\n'.join(stop).encode()).hexdigest()
    cv = CountVectorizer(stop_words=stop, ngram_range=(1, 2))
    c_mh = [rc.clean(x) for x in t_mh]
    c_rs = [rc.clean(x) for x in t_rs]
    cv.fit(c_mh + c_rs)
    C = (cv.transform(c_mh) + cv.transform(c_rs)).tocsc()
    vocab = cv.get_feature_names_out()
    alpha = np.asarray(C.sum(0)).ravel().astype(float)
    a0 = alpha.sum()
    D = (C > 0).astype(np.int8).tocsc()
    dfreq = np.asarray(D.sum(0)).ravel()
    seg_codes = np.searchsorted(seg_list, seg_a)
    nseg = np.array([len(set(seg_codes[D.indices[D.indptr[w]:D.indptr[w + 1]]])) for w in range(len(vocab))])
    elig = (dfreq >= MIN_ANS) & (nseg >= MIN_SEG)
    print(f'  vocabulary {len(vocab)} terms, eligible {int(elig.sum())}; answers with no text {n_empty}', flush=True)
    Cr = C.tocsr()
    Dr = D.tocsr()
    rows_w = []
    for c, lab_ in ((3, 'main'), (5, 'sensitivity')):
        hi_m, lo_m = abins[c] == c, abins[c] == 1
        yi = np.asarray(Cr[hi_m].sum(0)).ravel().astype(float)
        yj = np.asarray(Cr[lo_m].sum(0)).ravel().astype(float)
        ni, nj = yi.sum(), yj.sum()
        delta = np.log((yi + alpha) / (ni + a0 - yi - alpha)) - np.log((yj + alpha) / (nj + a0 - yj - alpha))
        z = delta / np.sqrt(1.0 / (yi + alpha) + 1.0 / (yj + alpha))
        ai = np.asarray(Dr[hi_m].sum(0)).ravel()
        aj = np.asarray(Dr[lo_m].sum(0)).ravel()
        both = np.flatnonzero(hi_m | lo_m)
        Db = D[both].tocsc()
        fb, sb = fpos[both].astype(int), seg_codes[both]
        cand = np.flatnonzero(elig)
        order_hi = cand[np.lexsort((cand, -z[cand]))][:N_TOP]
        order_lo = cand[np.lexsort((cand, z[cand]))][:N_TOP]
        for side_, order in ((f'highest bin ({bin_label(c, c)})', order_hi), (f'lowest bin ({bin_label(c, 1)})',
                                                                                order_lo)):
            for rank, w in enumerate(order, 1):
                rws = Db.indices[Db.indptr[w]:Db.indptr[w + 1]]
                rows_w.append(dict(comparison=lab_, bins=c, side=side_, rank=rank, term=vocab[w], z=float(z[w]),
                                   delta=float(delta[w]), answers_highest_bin=int(ai[w]), answers_lowest_bin=int(aj[w]),
                                   frames_both_bins=int(len(set(fb[rws]))), segments_both_bins=int(len(set(sb[rws]))),
                                   answers_all=int(dfreq[w]), segments_all=int(nseg[w]), count_all=int(alpha[w])))
        print(f'  {lab_}: {int(hi_m.sum())} vs {int(lo_m.sum())} answers, tokens {int(ni)} vs {int(nj)}', flush=True)

    # ------------------------------------------------------------------------------------------------- write
    fr = frames.copy()
    fr['score_mean'] = score
    fr['bin_3'] = bins[3]
    fr['bin_5'] = bins[5]
    W('bins.csv', rows_b)
    W('features.csv', rows_f)
    W('differences.csv', rows_d)
    W('trend_tau_b.csv', rows_t)
    W('answers.csv', rows_a)
    W('answers_differences.csv', rows_ad)
    W('terms.csv', rows_w)
    W('three_vs_five.csv', rows_x)
    W('comparisons.csv', rows_c)
    W('comparisons_list.csv', [dict(kind=k_, bins=c_, what=w_, side=s_) for k_, c_, w_, s_ in comps])
    W('check.csv', rows_ck)
    fr.to_csv(os.path.join(O, PFX + 'frames.csv'), index=False)
    with open(os.path.join(O, PFX + 'params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'EXPLORATORY: score bins (user, 2026-11-14; plan in CLAUDE.md "理由分析" -> "分数分段")\n'
                f'run {time.strftime("%Y-%m-%d %H:%M:%S")}; out_dir {os.path.abspath(O)}\n'
                f'program {os.path.abspath(__file__)} sha256 {prog_sha}\n')
        f.write(f'bins: frames by score_mean, answers by their own score; cuts {CUTS} (a cut belongs to the bin above); '
                f'expected frames {EXPECT}; the plan\'s "三种切法" read as these two frame cuts\n')
        f.write(f'inputs: {main_f}; {forced_f}; {p_fs}; {p_a3}; {p_f3}; {p_r2}; {p_q2}; {p_tr}; {p_lab}; cues: '
                + '; '.join(f'{nm}: {cue_src.get(nm, "none")} {c_ or ""} {r_}' for nm, _, c_, r_, _ in rf.CUES) + '\n')
        f.write('items: keyword class mentioned in 3+ of 5 answers (round 3 k*_answers); hazard_type_most, camera_most '
                '(round 3, mixed on a tie); cues (share over the frames with a value); 9 SSM risk percentiles (round 2) '
                'and true distance, true closing speed v_ego - v_lead, ego speed (quantities/truth_main.csv): medians, '
                'quartiles numpy linear, over the frames with a value; trend: scipy kendalltau variant b of the bin number '
                'and the item over the frames with a value\n')
        f.write(f'intervals: {N_BOOT} resamples of whole segments, default_rng({BOOT_SEED}), sha256 {hh.hexdigest()}; '
                '2.5 / 97.5 % numpy linear over the resamples where defined\n')
        f.write(f'terms: main_hazard and reason each reason_clusters.clean() then CountVectorizer(stop_words, '
                f'ngram_range=(1, 2)), the two fields added; stop words {len(stop)} (round 1 custom list + risk words) '
                f'sha256 {stop_sha}: {" ".join(stop)}; prior alpha = counts over all {len(td)} answers (alpha_0 '
                f'{int(a0)}); eligible: {MIN_ANS}+ answers and {MIN_SEG}+ segments over all answers; {N_TOP} each side; '
                f'vocabulary {len(vocab)}, eligible {int(elig.sum())}; answers with no text {n_empty}\n')
        f.write(f'keyword accuracy: {p_lab} column correct_1_or_0 per class; class 19: {LOW_ACC[19]}\n')
    print('\n' + '=' * 110)
    for r_ in rows_c:
        if r_['kind'] == 'all':
            print(f"  {r_['bins']} bins: {r_['intervals']} intervals, {r_['leaving_out_0']} leave out 0, by chance "
                  f"about {r_['by_chance_about']:.2f}")
    print(f'\ndone in {time.time() - t0:.0f} s; checks not met: {len(failed)}')
    for w_ in failed:
        print('  ', w_)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
