"""EXPLORATORY (user, 2026-11-13; the plan is in CLAUDE.md "理由分析" -> "第二轮", recorded and committed before this program
was written; it is not changed after the results): Qwen3.6-27B's reasons at the frame level. sep_camera, the main
result's 1,084 frames (238 segments); each frame's 5 reasons are 5 descriptions of the same frame and are combined into
one output per frame. CPU only; no image is opened; only the text fields of the replies are read.
Inputs: round 1's work folder (reason_clusters.py extract / reason_embed.py: answers_base.csv, reasons.jsonl,
embeddings.npy, whose sha256 must be round 1's) and round 1's answer table (its k = 7 clusters); the risk analysis
analysis/after_gemma-4-31b-it/main_1084 (vlm_frame_scores.csv score_mean, ssm_target_frames.csv, agreement.csv,
params.txt); quantities/truth_main.csv; the scene cues of the D / E analyses (analysis/rule_explore/nearby_counts.csv,
new_cues/new_cues.csv, new_cues/new_cues_e.csv, new_cues/new_cues_e2.csv) and labels/scene_labels.csv.
Frame representations: the mean of the frame's 5 sentence vectors, normalised to length 1 (main); the most frequent of
round 1's k = 7 clusters among its 5 answers, 'mixed' on a tie (comparison); the mean of the 10 pairwise cosine
similarities of its 5 unit vectors (how alike the 5 reasons are); TF-IDF of the 5 reasons joined in repeat order with a
space, cleaned and vectorised as round 1 (lower case, numbers -> num, punctuation -> space; the user's custom stop list
plus the risk-level words; ngram (1, 2), min_df 5, sublinear tf; the rest default).
Clusters: frame vectors -> KMeans(k, n_init=20, random_state 20261113), k = 4..12; silhouette (cosine); the main k =
the highest (a tie: the smaller k); stability with random_state 20261113..20261122 (ARI); TF-IDF -> TruncatedSVD(100,
random_state 20261113) -> normalised -> KMeans for the same k, ARI at the main k; the cross table and ARI of the main
clusters against the most frequent round-1 cluster ('mixed' its own label); t-SNE (2, perplexity 30, cosine,
random_state 20261113) on the frame vectors.
Per main cluster: frames, segments, the largest segment's frames; the 10 terms of highest mean TF-IDF weight (ties
alphabetical); the 3 frames nearest the KMeans centre (Euclidean, frame-vector space; ties by frame order) with their 5
reasons as written and score_mean; score_mean mean, median, SD (ddof 0), mean - the mean of all 1,084 frames;
hazard_type shares over the 5 answers of its frames (the risk analysis's normalised hazard_type); the mean within-frame
similarity; 9 SSMs as risk percentiles (each *_risk column ranked over all 1,084 frames, ties the average rank,
(rank - 1) / (1,084 - 1) x 100, 100 = the most dangerous): the median per cluster; raw values (the 9 SSMs in their own
units, the true distance, the true closing speed v_ego - v_lead, the ego speed): median, quartiles, IQR (numpy linear);
scene cues: the share of the cluster's frames with the cue (frames with a value) and of all frames (definitions in
CUES); within-cluster tau-b of score_mean with each SSM over the pairs inside the cluster, tau-b = S / sqrt(nx ny) with
S, nx, ny summed over the pairs (as the 4.3 groups), and pooled over all clusters (pairs in the same cluster), beside
all pairs; eta^2 = between-cluster / total sum of squares of score_mean, and the same by segment, and a reference from
1,000 shuffles of the cluster labels among the frames (default_rng(20261111)): mean and 2.5 / 97.5 % percentiles.
Intervals: the main analysis's 2,000 resamples of whole segments (default_rng(20261021), one choice over the 238 sorted
segments per resample, drawn segments' sha256 48bc1f41...; every frame of a drawn segment, as often as the segment is
drawn; clusters, percentiles and cues not redone): the cluster mean, the difference, the percentile medians and the
tau-b recomputed on the drawn frames; 2.5 / 97.5 % percentiles (numpy linear) over the resamples where the statistic is
defined, the missing ones counted (no 90 % rule, as the plan has none).
Checks (exit code 1 when one is not met): 1,084 frames, 238 segments, exactly 5 reasons per frame; the frame means of
the 5 answers = score_mean frame by frame; the 9 tau-b over all pairs = agreement.csv lines 1001-1025 within 1e-12; the
sentence vectors' sha256 = round 1's. The run script compares a second run byte for byte.
Outputs in <out_dir>: exploratory_reason_frames.csv (one row per frame, no reason text) and exploratory_reason_frames_
k, _stability, _crosstab, _clusters, _terms, _examples (the only file with reason text), _hazard, _ssm_percentile, _raw,
_cues, _tau_b, _eta2, _check (.csv) and _params.txt.
usage: python -I reason_frames.py <waymo_v143 folder> <round-1 work folder> <round-1 out_dir> <out_dir> [--workers N]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_rules as ar  # noqa: E402
import analyze_vlm_ssm as av  # noqa: E402
import reason_clusters as rc  # noqa: E402

MODEL, COND = 'qwen3.6-27b', 'sep_camera'
SEED, N_BOOT, BOOT_SEED = 20261113, 2000, 20261021
HASH = '48bc1f413f9d5b99da57ff4edde7bbb4eb378c5a9343c44ccad4e0165466604c'
EMB_SHA = '30ba000292417d5047d2279dba1fa1a084883a49ebc9cd379a9aeee6a46296be'      # round 1's embeddings.npy
REASONS_SHA = 'c718132acf962585cb1463f0fc1736b77d984b598808498f2bd8e5a3c7345373'  # round 1's reasons
KS = list(range(4, 13))
N_STAB, N_TERMS, N_EXAMPLES, MIN_FRAMES = 10, 10, 3, 50
SHUFFLE_SEED, N_SHUFFLE = 20261111, 1000
AG_FIRST, AG_STEP, TOL = 1001, 3, 1e-12
NE = 'analysis/rule_explore/new_cues'
# name, file (relative to waymo_v143, or 'labels' for the repository's labels/scene_labels.csv), column, rule, in the
# user's list (True) or added by Claude (False)
CUES = [('pedestrian within 20 m (in a camera view, occluded ones too)', 'analysis/rule_explore/nearby_counts.csv',
         'n_pedestrian_20m', '>= 1', True),
        ('rain (the user\'s weather label)', 'labels', 'weather', '== rain', True),
        ('night (Waymo time of day)', 'labels', 'time_of_day', '== Night', True),
        ('red light', None, None, 'not in the data (no traffic-signal state in the v1.4.3 perception frames)', True),
        ('roadside parking: a stationary vehicle not in a lane within 20 m', f'{NE}/new_cues_e.csv',
         'n_vehicle_stationary_other_20m', '>= 1', True),
        ('large vehicle: a vehicle box of 7.0 m or longer within 30 m (lead included)', f'{NE}/new_cues_e2.csv',
         'max_vehicle_length_30m', '>= 7.0', True),
        ('pedestrian or cyclist on the road ahead (0-30 m, |y| <= 2.5 m)', f'{NE}/new_cues.csv', 'n_ped_cyc_on_road',
         '>= 1', False),
        ('pedestrian or cyclist in a crosswalk within 50 m', f'{NE}/new_cues_e.csv', 'n_ped_cyc_in_crosswalk_50m', '>= 1',
         False),
        ('wet road (the user\'s weather label)', 'labels', 'weather', '== wet', False),
        ('crossing vehicle ahead', f'{NE}/new_cues.csv', 'n_crossing_vehicle', '>= 1', False),
        ('oncoming vehicle', f'{NE}/new_cues.csv', 'n_oncoming_vehicle', '>= 1', False),
        ('vehicle in the next lane nearby (-5..15 m, 1.5 < |y| <= 5 m)', f'{NE}/new_cues_e.csv', 'n_side_vehicle',
         '>= 1', False)]
G = {}                                                            # data for the worker processes (fork)


def txt(x):
    return '' if x is None or (isinstance(x, float) and np.isnan(x)) else repr(float(x))


def pct(v):
    ok = ~np.isnan(v)
    if not ok.any():
        return np.nan, np.nan, 0, int((~ok).sum())
    lo, hi = np.percentile(v[ok], [2.5, 97.5])
    return float(lo), float(hi), int(ok.sum()), int((~ok).sum())


def tau_of(S, nx, ny):
    return float(S / np.sqrt(nx * ny)) if nx * ny > 0 else np.nan


def pair_counts(idx):
    """S, nx, ny per pair code (cluster of both frames, K = different clusters) for each SSM, frames idx."""
    K = G['K']
    lab = G['lab'][idx]
    p, q = np.triu_indices(len(idx), 1)
    code = np.where(lab[p] == lab[q], lab[p], K).astype(np.int64)
    x = G['score'][idx]
    sx = np.sign(x[p] - x[q])
    nx = np.bincount(code, weights=np.abs(sx), minlength=K + 1)
    out = {}
    for s in av.SSMS:
        y = G['risk'][s][idx]
        sy = np.sign(y[p] - y[q])
        out[s] = (np.bincount(code, weights=sx * sy, minlength=K + 1), nx, np.bincount(code, weights=np.abs(sy),
                                                                                         minlength=K + 1))
    return out


def stats(idx):
    K = G['K']
    lab, score = G['lab'][idx], G['score'][idx]
    res = {}
    ov = float(score.mean())
    for c in range(K):
        m = lab == c
        res[('mean', c)] = float(score[m].mean()) if m.any() else np.nan
        res[('diff', c)] = res[('mean', c)] - ov
        for s in av.SSMS:
            res[('pct', c, s)] = float(np.median(G['pct'][s][idx][m])) if m.any() else np.nan
    pc = pair_counts(idx)
    for s in av.SSMS:
        S, nx, ny = pc[s]
        for c in range(K):
            res[('tau', c, s)] = tau_of(S[c], nx[c], ny[c])
        res[('tau', 'within', s)] = tau_of(S[:K].sum(), nx[:K].sum(), ny[:K].sum())
        res[('tau', 'all', s)] = tau_of(S.sum(), nx.sum(), ny.sum())
    return res


def one(b):
    idx = np.concatenate([G['seg_frames'][s] for s in G['draws'][b]])
    rr = stats(idx)
    return np.array([rr.get(k, np.nan) for k in G['keys']], float)


def eta2(y, groups):
    m = y.mean()
    tot = ((y - m) ** 2).sum()
    between = sum(len(y[groups == g]) * (y[groups == g].mean() - m) ** 2 for g in np.unique(groups))
    return float(between / tot)


def main():
    import sklearn
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.manifold import TSNE
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.preprocessing import normalize
    ap = argparse.ArgumentParser()
    ap.add_argument('waymo')
    ap.add_argument('work1')
    ap.add_argument('out1')
    ap.add_argument('out')
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    L, W1, R1, O = a.waymo, a.work1, a.out1, a.out
    A, _, FSSM = rc.paths(L)
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.exists(os.path.join(O, 'exploratory_reason_frames_params.txt')):
        sys.exit(f'{O}/exploratory_reason_frames_params.txt exists already: give a new folder')
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
    print(f'reason clusters at the frame level, {MODEL} {COND} (user, 2026-11-13; EXPLORATORY); run '
          f'{time.strftime("%Y-%m-%d %H:%M:%S")}; program sha256 {prog_sha}; scikit-learn {sklearn.__version__}\n\n0. inputs')

    # ------------------------------------------------------------------------------------------------- inputs
    td = pd.read_csv(os.path.join(W1, 'answers_base.csv'), keep_default_na=False, na_values=[''],
                     dtype={'hazard_type': str, 'camera': str, 'missing_kind': str})
    td['has_reason'] = td['has_reason'].astype(str).str.lower().eq('true')
    recs = [json.loads(x) for x in open(os.path.join(W1, 'reasons.jsonl'), encoding='utf-8')]
    reasons = [r['reason'] for r in recs]
    h = hashlib.sha256()
    for r in reasons:
        h.update(r.encode('utf-8') + b'\n')
    p_emb = os.path.join(W1, 'embeddings.npy')
    with open(p_emb, 'rb') as f:
        emb_sha = hashlib.sha256(f.read()).hexdigest()
    meta = json.load(open(os.path.join(W1, 'embeddings.json')))
    E = normalize(np.load(p_emb).astype(np.float64))
    check(emb_sha == EMB_SHA and meta['reasons_sha256'] == REASONS_SHA == h.hexdigest()
          and [r['row'] for r in recs] == list(range(len(td))) and td.has_reason.all() and E.shape[0] == len(td),
          f'sentence vectors {p_emb}: sha256 {emb_sha} = round 1\'s; reasons\' sha256 = round 1\'s; every one of the '
          f'{len(td)} answers has a reason and a vector, in row order')
    r1 = pd.read_csv(os.path.join(R1, 'exploratory_reason_answers.csv'))
    check(len(r1) == len(td) and (r1.segment.to_numpy() == td.segment.to_numpy()).all()
          and (r1.frame_idx.to_numpy() == td.frame_idx.to_numpy()).all()
          and (r1['repeat'].to_numpy() == td['repeat'].to_numpy()).all() and set(r1.cluster.dropna().unique()) <= set(range(7)),
          f'{R1}/exploratory_reason_answers.csv: the same {len(r1)} answers in the same order, round-1 clusters 0..6')
    frames = pd.read_csv(os.path.join(A, 'ssm_target_frames.csv'))[['segment', 'frame_idx']] \
        .sort_values(['segment', 'frame_idx']).reset_index(drop=True)
    n = len(frames)
    mi = pd.MultiIndex.from_frame(frames)
    key = pd.MultiIndex.from_arrays([td.segment, td.frame_idx])
    fpos = pd.Series(np.arange(n), index=mi).reindex(key).to_numpy()
    reps = {}
    for i, (fp, rp) in enumerate(zip(fpos, td['repeat'])):
        reps.setdefault(int(fp), {})[int(rp)] = i
    rows5 = np.array([[reps[f][r] for r in range(5)] for f in range(n)])
    check(n == 1084 and frames.segment.nunique() == 238 and not np.isnan(fpos.astype(float)).any()
          and all(sorted(reps[f]) == [0, 1, 2, 3, 4] for f in range(n)) and len(td) == 5 * n,
          f'{n} frames, {frames.segment.nunique()} segments, each with exactly 5 reasons (repeats 0-4)')
    p_fs = os.path.join(A, 'vlm_frame_scores.csv')
    fs = pd.read_csv(p_fs, float_precision='round_trip')
    fs = fs[(fs.model == MODEL) & (fs.condition == COND)].set_index(['segment', 'frame_idx'])['score_mean'].reindex(mi)
    score = fs.to_numpy(float)
    fm = td.groupby(['segment', 'frame_idx'], sort=True)['risk'].mean().reindex(mi).to_numpy(float)
    check(not np.isnan(score).any() and np.array_equal(fm, score),
          f'the frame means of the 5 answers = {p_fs} score_mean frame by frame (largest |difference| '
          f'{float(np.max(np.abs(fm - score)))!r})')

    # ------------------------------------------------------------------------------------------------- frame representations
    F = normalize(E[rows5].mean(axis=1))
    sim = np.array([np.mean([E[rows5[f, i]] @ E[rows5[f, j]] for i in range(5) for j in range(i + 1, 5)])
                    for f in range(n)])
    c1 = r1.cluster.to_numpy()
    mode = []
    for f in range(n):
        cnt = np.bincount(c1[rows5[f]].astype(int), minlength=7)
        top = np.flatnonzero(cnt == cnt.max())
        mode.append(str(int(top[0])) if len(top) == 1 else 'mixed')
    mode = np.array(mode)
    n_mixed = int((mode == 'mixed').sum())
    docs = [' '.join(reasons[rows5[f, r]] for r in range(5)) for f in range(n)]
    stop = sorted(set(rc.CUSTOM_STOP) | set(rc.RISK_WORDS))
    vec = TfidfVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=5, sublinear_tf=True)
    X = vec.fit_transform([rc.clean(d) for d in docs])
    terms = np.array(vec.get_feature_names_out())
    print(f'  frame vectors {F.shape}; most frequent round-1 cluster: mixed {n_mixed} frames; within-frame similarity mean '
          f'{float(sim.mean())!r}; TF-IDF {X.shape[0]} x {X.shape[1]} terms')

    # ------------------------------------------------------------------------------------------------- clusters
    print('\n1. frame vectors -> KMeans')
    rows_kk, fits = [], {}
    for k in KS:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(F)
        sil = float(silhouette_score(F, km.labels_, metric='cosine'))
        fits[k] = km
        sizes = np.bincount(km.labels_, minlength=k)
        rows_kk.append(dict(vectors='frame_sentence', k=k, silhouette=sil, sizes=';'.join(str(int(x)) for x in sizes)))
        print(f'  k {k:2d}: silhouette {sil!r}; sizes {sizes.tolist()}', flush=True)
    best = max(rows_kk, key=lambda d: (d['silhouette'], -d['k']))
    K = best['k']
    km = fits[K]
    lab = km.labels_
    print(f'  main k = {K}')
    rows_s = []
    for s in range(SEED, SEED + N_STAB):
        rows_s.append(dict(what=f'frame vectors, KMeans random_state {s}', ari=float(adjusted_rand_score(
            lab, KMeans(n_clusters=K, n_init=20, random_state=s).fit(F).labels_))))
    aris = np.array([d['ari'] for d in rows_s])
    svd = TruncatedSVD(n_components=100, random_state=SEED)
    Zn = normalize(svd.fit_transform(X))
    tf_lab = {}
    for k in KS:
        km2 = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Zn)
        tf_lab[k] = km2.labels_
        sizes = np.bincount(km2.labels_, minlength=k)
        rows_kk.append(dict(vectors='frame_tfidf_svd100', k=k, silhouette=float(silhouette_score(Zn, km2.labels_,
                                                                                                 metric='cosine')),
                            sizes=';'.join(str(int(x)) for x in sizes)))
    ari_tf = float(adjusted_rand_score(lab, tf_lab[K]))
    ari_mode = float(adjusted_rand_score(lab, mode))
    rows_s.append(dict(what=f'TF-IDF -> SVD 100 -> KMeans, k {K}, against the main clusters', ari=ari_tf))
    rows_s.append(dict(what='the most frequent round-1 cluster (mixed its own label) against the main clusters',
                       ari=ari_mode))
    ct = pd.crosstab(pd.Series(lab, name='cluster'), pd.Series(mode, name='round1_most_frequent'))
    print(f'  stability ARI mean {float(aris.mean())!r} ({float(aris.min())!r}..{float(aris.max())!r}); TF-IDF '
          f'{ari_tf!r}; round-1 most frequent {ari_mode!r}')
    tsne = TSNE(n_components=2, perplexity=30, metric='cosine', random_state=SEED)
    Y = tsne.fit_transform(F)

    # ------------------------------------------------------------------------------------------------- SSMs, truth, cues
    ssm_df = av.ssm_table(FSSM, frames)
    risk = {s: ssm_df[s + '_risk'].to_numpy(float) for s in av.SSMS}
    check(all(ssm_df[s + '_conflict'].all() and ssm_df[s + '_use'].all() for s in av.SSMS),
          'every frame has a value of every SSM')
    pctl = {s: (rankdata(risk[s], method='average') - 1) / (n - 1) * 100 for s in av.SSMS}
    truth = pd.read_csv(os.path.join(L, 'quantities', 'truth_main.csv'), float_precision='round_trip')
    tr = frames.merge(truth, on=['segment', 'frame_idx'], how='left', validate='one_to_one')
    raw = {**{s: ssm_df[s].to_numpy(float) for s in av.SSMS}, 'true_distance': tr['distance'].to_numpy(float),
           'true_closing_speed': (tr['v_ego'] - tr['v_lead']).to_numpy(float), 'ego_speed': tr['v_ego'].to_numpy(float)}
    labels_csv = os.path.join(REPO, 'labels', 'scene_labels.csv')
    sl = pd.read_csv(labels_csv, dtype=str, keep_default_na=False).set_index('segment')
    cue_val, cue_src = {}, {}
    for name, file_, col, rule, _ in CUES:
        if file_ is None:
            continue
        if file_ == 'labels':
            v = sl.loc[frames.segment, col].str.strip().to_numpy()
            want = rule.split('== ')[1]
            cue_val[name] = np.where(v == want, 1.0, 0.0)
            cue_src[name] = labels_csv
        else:
            p_ = os.path.join(L, file_)
            t_ = pd.read_csv(p_, float_precision='round_trip')
            t_ = frames.merge(t_[['segment', 'frame_idx', col]], on=['segment', 'frame_idx'], how='left',
                              validate='one_to_one')[col].to_numpy(float)
            thr = float(rule.split('>= ')[1])
            cue_val[name] = np.where(np.isnan(t_), np.nan, (t_ >= thr).astype(float))
            cue_src[name] = p_

    # ------------------------------------------------------------------------------------------------- point, resamples
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
    G.update(K=K, lab=lab, score=score, risk=risk, pct=pctl, draws=draws,
             seg_frames={s: np.flatnonzero(seg_names == s) for s in seg_list})
    point = stats(np.arange(n))
    p_ag = os.path.join(A, 'agreement.csv')
    with open(p_ag, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        ag = {d['ssm']: (rd.line_num, d) for d in rd
              if d['model'] == MODEL and d['condition'] == COND and d['grouping'] == 'all'}
    lines = [ag[s][0] for s in av.SSMS]
    dmax = max(abs(point[('tau', 'all', s)] - float(ag[s][1]['tau_b_conflict'])) for s in av.SSMS)
    same_text = sum(txt(point[('tau', 'all', s)]) == ag[s][1]['tau_b_conflict'] for s in av.SSMS)
    check(lines == list(range(AG_FIRST, AG_FIRST + 9 * AG_STEP, AG_STEP)) and dmax <= TOL,
          f'the 9 tau-b over all frame pairs (S / sqrt(nx ny)) = {p_ag} lines {lines}: largest |difference| {dmax!r}; the '
          f'same text {same_text} of 9')
    keys = list(point)
    G['keys'] = keys
    col = {k: i for i, k in enumerate(keys)}
    print(f'\n2. resamples: {N_BOOT}, {a.workers} worker(s)', flush=True)
    tb = time.time()
    with mp.get_context('fork').Pool(max(a.workers, 1)) as pool:
        boot = np.stack(list(pool.imap(one, range(N_BOOT), chunksize=8)))
    print(f'  done in {time.time() - tb:.0f} s')

    def B(k):
        return boot[:, col[k]]

    # ------------------------------------------------------------------------------------------------- tables
    overall = float(score.mean())
    rows_c, rows_t, rows_e, rows_h, rows_p, rows_r, rows_q, rows_tau = [], [], [], [], [], [], [], []
    hazard_vals = sorted(td.hazard_type.fillna('(missing)').unique())
    for c in range(K):
        m = lab == c
        segc = pd.Series(seg_names[m]).value_counts()
        sc = score[m]
        mlo, mhi, mn, mmiss = pct(B(('mean', c)))
        dlo, dhi, dn, dmiss = pct(B(('diff', c)))
        centre = np.asarray(X[m].mean(axis=0)).ravel()
        order = np.lexsort((terms, -centre))[:N_TERMS]
        rows_t += [dict(cluster=c, rank=i + 1, term=terms[j], mean_weight=float(centre[j])) for i, j in enumerate(order)]
        idx_c = np.flatnonzero(m)
        dist = np.linalg.norm(F[idx_c] - km.cluster_centers_[c], axis=1)
        for i, o in enumerate(np.argsort(dist, kind='stable')[:N_EXAMPLES]):
            f_ = idx_c[o]
            for r in range(5):
                rows_e.append(dict(cluster=c, rank=i + 1, distance=float(dist[o]), segment=seg_names[f_],
                                   frame_idx=int(frames.frame_idx[f_]), score_mean=float(score[f_]), repeat=r,
                                   risk=float(td.risk[rows5[f_, r]]), reason=reasons[rows5[f_, r]]))
        hz = td.hazard_type.iloc[rows5[idx_c].ravel()].fillna('(missing)').value_counts()
        for v in hazard_vals:
            rows_h.append(dict(cluster=c, hazard_type=v, answers=int(hz.get(v, 0)), share=float(hz.get(v, 0) / (5 * m.sum()))))
        for s in av.SSMS:
            lo, hi, nv, miss = pct(B(('pct', c, s)))
            rows_p.append(dict(cluster=c, ssm=s, frames=int(m.sum()), median_percentile=point[('pct', c, s)], lo=lo, hi=hi,
                               nboot=nv, missing=miss,
                               side=('no interval' if np.isnan(lo) else 'above 50' if lo > 50 else 'below 50' if hi < 50
                                     else 'contains 50')))
        rows_c.append(dict(cluster=c, frames=int(m.sum()), segments=int(len(segc)), largest_segment=segc.index[0],
                           largest_segment_frames=int(segc.iloc[0]), mean=float(sc.mean()), median=float(np.median(sc)),
                           sd=float(sc.std(ddof=0)), mean_minus_all=float(sc.mean() - overall), mean_lo=mlo, mean_hi=mhi,
                           mean_nboot=mn, mean_missing=mmiss, diff_lo=dlo, diff_hi=dhi, diff_nboot=dn,
                           diff_missing=dmiss,
                           diff_side=('no interval' if np.isnan(dlo) else 'above 0' if dlo > 0 else 'below 0' if dhi < 0
                                      else 'contains 0'),
                           within_similarity_mean=float(sim[m].mean()), under_50_frames=bool(m.sum() < MIN_FRAMES),
                           top_terms='; '.join(terms[j] for j in order)))
    for c in list(range(K)) + ['all']:
        m = np.ones(n, bool) if c == 'all' else lab == c
        for q_, v in raw.items():
            q1, med, q3 = np.percentile(v[m], [25, 50, 75])
            rows_r.append(dict(cluster=c, quantity='TTC2D' if q_ == 'TTC2D' else q_, frames=int(m.sum()), median=float(med),
                               q1=float(q1), q3=float(q3), iqr=float(q3 - q1)))
        for name, file_, col_, rule, users in CUES:
            if file_ is None:
                rows_q.append(dict(cue=name, users_list=users, source='none', column='', rule=rule, cluster=c,
                                   frames_with_value=0, frames_with_cue=np.nan, share=np.nan))
                continue
            v = cue_val[name][m]
            ok = ~np.isnan(v)
            rows_q.append(dict(cue=name, users_list=users, source=cue_src[name], column=col_, rule=rule, cluster=c,
                               frames_with_value=int(ok.sum()), frames_with_cue=int(np.nansum(v)),
                               share=float(np.nansum(v) / ok.sum()) if ok.any() else np.nan))
    pc0 = pair_counts(np.arange(n))
    for g in list(range(K)) + ['within', 'all']:
        nf = int((lab == g).sum()) if g not in ('within', 'all') else n
        for s in av.SSMS:
            S, nx, ny = pc0[s]
            if g == 'within':
                Sg, xg, yg = S[:K].sum(), nx[:K].sum(), ny[:K].sum()
            elif g == 'all':
                Sg, xg, yg = S.sum(), nx.sum(), ny.sum()
            else:
                Sg, xg, yg = S[g], nx[g], ny[g]
            lo, hi, nv, miss = pct(B(('tau', g, s)))
            rows_tau.append(dict(group=g if g not in ('within', 'all') else
                                 ('within clusters (pooled)' if g == 'within' else 'all pairs'), ssm=s, frames=nf,
                                 S=int(Sg), nx=int(xg), ny=int(yg), tau_b=point[('tau', g, s)], lo=lo, hi=hi, nboot=nv,
                                 missing=miss,
                                 side=('no interval' if np.isnan(lo) else 'above 0' if lo > 0 else 'below 0' if hi < 0
                                       else 'contains 0'),
                                 under_50_frames=bool(g not in ('within', 'all') and nf < MIN_FRAMES)))
    e_cl, e_sg = eta2(score, lab), eta2(score, seg_names)
    rng2 = np.random.default_rng(SHUFFLE_SEED)
    sh = np.array([eta2(score, rng2.permutation(lab)) for _ in range(N_SHUFFLE)])
    slo, shi = np.percentile(sh, [2.5, 97.5])
    rows_eta = [dict(grouping=f'main clusters (k {K})', groups=K, eta2=e_cl, shuffle_mean=float(sh.mean()),
                     shuffle_lo=float(slo), shuffle_hi=float(shi), shuffles=N_SHUFFLE, shuffle_seed=SHUFFLE_SEED),
                dict(grouping='segment', groups=len(seg_list), eta2=e_sg, shuffle_mean=np.nan, shuffle_lo=np.nan,
                     shuffle_hi=np.nan, shuffles=0, shuffle_seed=np.nan)]

    # ------------------------------------------------------------------------------------------------- write
    out = frames.copy()
    out['score_mean'], out['cluster'], out['round1_most_frequent'] = score, lab, mode
    out['within_similarity'], out['tsne_x'], out['tsne_y'] = sim, Y[:, 0], Y[:, 1]
    for s in av.SSMS:
        out[f'{s}_risk_percentile'] = pctl[s]
    out.to_csv(os.path.join(O, 'exploratory_reason_frames.csv'), index=False)
    for name, rows in (('k', rows_kk), ('stability', rows_s), ('clusters', rows_c), ('terms', rows_t), ('examples', rows_e),
                       ('hazard', rows_h), ('ssm_percentile', rows_p), ('raw', rows_r), ('cues', rows_q),
                       ('tau_b', rows_tau), ('eta2', rows_eta), ('check', rows_ck)):
        pd.DataFrame(rows).to_csv(os.path.join(O, f'exploratory_reason_frames_{name}.csv'), index=False)
    ct.to_csv(os.path.join(O, 'exploratory_reason_frames_crosstab.csv'))
    with open(os.path.join(O, 'exploratory_reason_frames_params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'EXPLORATORY: reason clusters at the frame level, {MODEL} {COND}, main result 1,084 frames (user, '
                f'2026-11-13; plan in CLAUDE.md "理由分析" -> "第二轮")\nrun {time.strftime("%Y-%m-%d %H:%M:%S")}; out_dir '
                f'{os.path.abspath(O)}\n')
        f.write(f'program {os.path.abspath(__file__)} sha256 {prog_sha}; scikit-learn {sklearn.__version__}, numpy '
                f'{np.__version__}, pandas {pd.__version__}\n')
        f.write(f'sentence vectors {os.path.abspath(p_emb)} sha256 {emb_sha} (round 1); {meta["repo"]} revision '
                f'{meta["revision"]}; reasons sha256 {h.hexdigest()}; round-1 clusters from {os.path.abspath(R1)}\n')
        f.write(f'inputs: {p_fs}; {p_ag}; {A}/ssm_target_frames.csv; {FSSM}; {L}/quantities/truth_main.csv; cues: '
                + '; '.join(f'{nm}: {cue_src.get(nm, "none")} {c_ or ""} {r_}' for nm, _, c_, r_, _ in CUES) + '\n')
        f.write(f'frame vector: mean of the 5 unit vectors, normalised; most frequent round-1 cluster ({n_mixed} mixed); '
                f'within-frame similarity: mean of the 10 pairwise dot products of the unit vectors\n')
        f.write(f'TF-IDF: the 5 reasons in repeat order joined with a space, round 1\'s cleaning and stop list '
                f'({len(stop)} words), ngram (1, 2), min_df 5, sublinear tf; vocabulary {X.shape[1]}\n')
        f.write(f'KMeans(n_clusters=k, n_init=20, random_state={SEED}), k {KS[0]}..{KS[-1]}, silhouette cosine; main k {K}; '
                f'stability random_state {SEED}..{SEED + N_STAB - 1}; TF-IDF comparison TruncatedSVD(100, '
                f'random_state={SEED}) -> normalised -> KMeans; TSNE(2, perplexity 30, cosine, random_state {SEED})\n')
        f.write(f'risk percentile: rankdata(*_risk, average) over the {n} frames, (rank - 1) / ({n} - 1) x 100\n')
        f.write(f'intervals: {N_BOOT} resamples of whole segments, default_rng({BOOT_SEED}), sha256 {hh.hexdigest()}; '
                f'percentiles 2.5 / 97.5 (numpy linear) over the defined resamples; no 90 % rule\n')
        f.write(f'eta2 shuffle reference: {N_SHUFFLE} permutations of the cluster labels, default_rng({SHUFFLE_SEED})\n')

    # ------------------------------------------------------------------------------------------------- report
    print('\n' + '=' * 110)
    print(f'frames {n}; mixed {n_mixed}; within-frame similarity mean {float(sim.mean())!r}; all frames\' mean {overall!r}')
    for i, d in enumerate(rows_kk):
        print(f"  k line {i + 2}: {d['vectors']:18s} k {d['k']:2d} silhouette {d['silhouette']!r} sizes {d['sizes']}")
    print(f'main k {K}; stability ARI mean {float(aris.mean())!r} range {float(aris.min())!r}..{float(aris.max())!r}; '
          f'TF-IDF {ari_tf!r}; round-1 most frequent {ari_mode!r}')
    for i, d in enumerate(rows_c):
        print(f"  cluster line {i + 2}: {d['cluster']}: {d['frames']} frames, {d['segments']} segments (largest "
              f"{d['largest_segment_frames']}); mean {d['mean']!r} [{txt(d['mean_lo'])}, {txt(d['mean_hi'])}]; diff "
              f"{d['mean_minus_all']!r} [{txt(d['diff_lo'])}, {txt(d['diff_hi'])}] ({d['diff_side']}); similarity "
              f"{d['within_similarity_mean']!r}; terms {d['top_terms']}")
    for d in rows_eta:
        print(f"  eta2 {d['grouping']}: {d['eta2']!r} (shuffle mean {txt(d['shuffle_mean'])} [{txt(d['shuffle_lo'])}, "
              f"{txt(d['shuffle_hi'])}])")
    print(f'\ndone in {time.time() - t0:.0f} s; checks not met: {len(failed)}')
    for w in failed:
        print('  ', w)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
