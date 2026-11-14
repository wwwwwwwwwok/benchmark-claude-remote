"""EXPLORATORY (user, 2026-11-13; the plan is in CLAUDE.md "理由分析" with sections 2 and 3 as the user revised them, all
recorded and committed before this program ran; it is not changed after the results): Qwen3.6-27B's reasons (the
`reason` field of its final JSON) and its risk scores. sep_camera, the main result's 1,084 frames (238 segments), 5
answers each. CPU only; no image is opened; only the text fields of the replies are read.
Two steps around code/reason_embed.py (which computes the sentence vectors in the sentence environment):
  extract <waymo_v143> <work folder>: the 5,420 requests from the risk analysis's vlm_requests_tidy.csv
    (analysis/after_gemma-4-31b-it/main_1084: risk, forced, and the normalised hazard_type and camera); the reason from
    the same parse result the score comes from: `parsed` of vlm_scores/qwen3.6-27b/main_qwen3.6-27b.jsonl.gz, and the
    forced answer's `parsed` from main_qwen3.6-27b_forced_answers.jsonl.gz where the reply has none
    (reason_words.read_replies, which follows analyze_vlm_ssm.read_model's rule); checked: every request once, the risk
    the same as the tidy table's. Writes <work>/answers_base.csv (the requests, no text) and <work>/reasons.jsonl (the
    answers with a reason, as written, in row order; the only file with reason text besides the examples). Missing or
    empty reasons (None, not a string, or only white space) are counted and left out.
  analyze <waymo_v143> <work folder> <out_dir>: reads those two files and <work>/embeddings.npy / .json (the sha256 of
    the reasons must match), then:
    main clusters: the sentence vectors (unit length) -> KMeans(k, n_init=20, random_state 20261113), k = 4..12;
      silhouette (cosine distance); the main k = the highest silhouette (a tie: the smaller k). Stability: the main k
      with random_state 20261113..20261122 (the first is the main run itself), ARI against the main labels.
    comparison: TF-IDF of the cleaned text (lower case; every number, a run of digits with . or , between digits, -> the
      word num; punctuation, anything not a letter, digit, underscore or white space -> space) with TfidfVectorizer(
      stop_words = the user's custom list + the risk-level words, ngram_range=(1, 2), min_df=5, sublinear_tf=True; the
      rest default: scikit-learn drops the stop words before it forms the bigrams; negations, quantities and directions
      are not in the list) -> TruncatedSVD(100, random_state 20261113) -> rows normalised -> KMeans for the same k;
      silhouettes and sizes too; the ARI of the two clusterings at the main k.
    terms: per main cluster, the 10 terms with the highest mean TF-IDF weight over its answers (ties alphabetical).
    t-SNE (2 components, perplexity 30, cosine, random_state 20261113, the rest default) on the sentence vectors.
    per main cluster: answers, frames, segments, the largest segment's answers; the 5 reasons nearest the KMeans centre
      (Euclidean distance in the sentence-vector space, where KMeans works; ties by row order), as written; score mean,
      median, SD (ddof 0), and mean - the mean of all 5,420 answers; hazard_type shares; intervals of the mean and of
      the difference from the main analysis's 2,000 resamples of whole segments (default_rng(20261021), one choice over
      the 238 sorted segments per resample, drawn segments' sha256 48bc1f41...; every answer of a drawn segment, as often
      as the segment is drawn; the clusters are not redone): 2.5 / 97.5 % percentiles (numpy, linear) over the
      resamples in which the cluster has an answer, the others counted as missing (the plan has no 90 % rule, so none
      is applied); among the frames with an answer in the cluster, the share whose 5 answers are all in it; overall, the
      share of the 1,084 frames whose 5 answers are all in one cluster.
    checks (exit code 1 when one is not met): answers clustered + missing reasons = 5,420; the frame means of the
      answers' scores = vlm_frame_scores.csv score_mean (qwen3.6-27b, sep_camera) frame by frame, exactly; the 9 tau-b of
      sep_camera, grouping all, from those means (analyze_vlm_ssm.all_stats, tau_b_conflict) = agreement.csv lines
      1001-1025 within 1e-12. A second run with the same parameters is compared by the run script.
    outputs in <out_dir>: exploratory_reason_k.csv, _clusters.csv, _terms.csv, _examples.csv (the 5 nearest reasons),
      _hazard.csv, _stability.csv, _answers.csv (one row per answer, no reason text), _check.csv, _params.txt.
usage: python -I reason_clusters.py extract <waymo_v143 folder> <work folder>
       python -I reason_clusters.py analyze <waymo_v143 folder> <work folder> <out_dir>
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_rules as ar  # noqa: E402
import analyze_vlm_ssm as av  # noqa: E402

MODEL, COND = 'qwen3.6-27b', 'sep_camera'
SEED, N_BOOT, BOOT_SEED = 20261113, 2000, 20261021
HASH = '48bc1f413f9d5b99da57ff4edde7bbb4eb378c5a9343c44ccad4e0165466604c'
KS = list(range(4, 13))
N_STAB, N_TERMS, N_EXAMPLES, MIN_ANSWERS = 10, 10, 5, 50
ST_REPO, ST_REV = 'sentence-transformers/all-mpnet-base-v2', 'e8c3b32edf5434bc2275fc9bab85f82640a19130'
CUSTOM_STOP = ['a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am', 'it', 'its', 'this', 'that',
               'these', 'those', 'there', 'here', 'of', 'to', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'as', 'and',
               'or', 'but', 'so', 'than', 'then', 'which', 'who', 'whom', 'whose', 'what', 'while', 'also', 'has', 'have',
               'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'i', 'you', 'he',
               'she', 'they', 'we', 'my', 'your', 'their', 'our', 'his', 'her', 'them', 'us']
RISK_WORDS = ['risk', 'risks', 'risky', 'low', 'lower', 'moderate', 'moderately', 'high', 'higher', 'minimal', 'minor',
              'slight', 'slightly', 'significant', 'safe', 'safely', 'safety', 'danger', 'dangerous', 'hazard', 'hazards',
              'hazardous', 'potential', 'potentially', 'situation', 'immediate', 'imminent', 'concern', 'concerns']
NUM = re.compile(r'\d+(?:[.,]\d+)*')
PUNCT = re.compile(r'[^\w\s]')
AG_FIRST, AG_STEP = 1001, 3
TOL = 1e-12


def clean(text):
    return PUNCT.sub(' ', NUM.sub(' num ', text.lower()))


def txt(x):
    return '' if x is None or (isinstance(x, float) and np.isnan(x)) else repr(float(x))


def paths(L):
    A = os.path.join(L, 'analysis', 'after_gemma-4-31b-it', 'main_1084')
    return A, os.path.join(L, 'vlm_scores', MODEL), os.path.join(L, 'results_lead_final_excl',
                                                                  'frame_ssm_validation_training.csv')


def extract(L, W):
    import reason_words as rw
    A, MD, _ = paths(L)
    if os.path.exists(os.path.join(W, 'reasons.jsonl')):
        sys.exit(f'{W}/reasons.jsonl exists already: give a new folder')
    os.makedirs(W, exist_ok=True)
    bad = []
    p_td = os.path.join(A, 'vlm_requests_tidy.csv')
    td = pd.read_csv(p_td, low_memory=False, keep_default_na=False, na_values=[''],
                     usecols=['model', 'condition', 'segment', 'frame_idx', 'repeat', 'risk', 'forced', 'hazard_type',
                              'camera'], dtype={'hazard_type': str, 'camera': str})
    td = td[(td.model == MODEL) & (td.condition == COND)].reset_index(drop=True)
    td['frame_idx'], td['repeat'] = td['frame_idx'].astype(int), td['repeat'].astype(int)
    td['forced'] = td['forced'].astype(str).str.lower().eq('true')
    n_frames = td[['segment', 'frame_idx']].drop_duplicates().shape[0]
    ok = len(td) == 5420 and n_frames == 1084 and td.segment.nunique() == 238 and \
        td.groupby(['segment', 'frame_idx']).size().eq(5).all()
    print(f'  check {"ok" if ok else "NOT MET"}: {p_td}: {len(td)} {MODEL} {COND} requests, {n_frames} frames x 5, '
          f'{td.segment.nunique()} segments')
    if not ok:
        bad.append('requests')
    keys = set(zip(td.segment, td.frame_idx, td.condition, td['repeat']))
    rep, info = rw.read_replies(MD, keys)
    risk_r, reason = [], []
    for s, k, c, r_ in zip(td.segment, td.frame_idx, td.condition, td['repeat']):
        rr, _, rs = rep[(s, k, c, r_)]
        risk_r.append(rr)
        reason.append(rs)
    same = np.array_equal(np.nan_to_num(np.array(risk_r, float), nan=-1), np.nan_to_num(td.risk.to_numpy(float), nan=-1))
    ok = len(rep) == len(td) and same
    print(f'  check {"ok" if ok else "NOT MET"}: reasons (parsed.reason) read for {len(rep)} requests from {info["main"]} '
          f'and {info["forced"]}; the risk of the same parse result = the tidy table\'s on every request')
    if not ok:
        bad.append('replies')
    kind = []
    for rs in reason:
        kind.append('None' if rs is None else 'not a string' if not isinstance(rs, str) else
                    'empty or white space' if not rs.strip() else '')
    td['has_reason'] = [k_ == '' for k_ in kind]
    td['missing_kind'] = kind
    td.to_csv(os.path.join(W, 'answers_base.csv'), index=False)
    with open(os.path.join(W, 'reasons.jsonl'), 'w', encoding='utf-8') as f:
        for i in np.flatnonzero(td.has_reason):
            f.write(json.dumps(dict(row=int(i), segment=td.segment[i], frame_idx=int(td.frame_idx[i]),
                                    repeat=int(td['repeat'][i]), reason=reason[i]), ensure_ascii=False) + '\n')
    with open(os.path.join(W, 'extract_info.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(tidy=p_td, main=info['main'], forced=info['forced'], requests=len(td),
                       forced_answers=int(td.forced.sum()), no_answer=int(td.risk.isna().sum()),
                       missing={k_: kind.count(k_) for k_ in ('None', 'not a string', 'empty or white space')},
                       with_reason=int(td.has_reason.sum())), f, indent=1)
    print(f'  answers {len(td)} (forced {int(td.forced.sum())}, no readable answer {int(td.risk.isna().sum())}); with a '
          f'reason {int(td.has_reason.sum())}; missing or empty {int((~td.has_reason).sum())}; written {W}/answers_base.csv, '
          f'reasons.jsonl, extract_info.json')
    sys.exit(1 if bad else 0)


def analyze(L, W, O):
    import sklearn
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.manifold import TSNE
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.preprocessing import normalize
    A, MD, FSSM = paths(L)
    if os.path.exists(os.path.join(O, 'exploratory_reason_params.txt')):
        sys.exit(f'{O}/exploratory_reason_params.txt exists already: give a new folder')
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
    print(f'reason clusters of {MODEL} {COND} (user, 2026-11-13; EXPLORATORY); run {time.strftime("%Y-%m-%d %H:%M:%S")}; '
          f'program sha256 {prog_sha}; scikit-learn {sklearn.__version__}, numpy {np.__version__}\n\n0. inputs')
    td = pd.read_csv(os.path.join(W, 'answers_base.csv'), keep_default_na=False, na_values=[''],
                     dtype={'hazard_type': str, 'camera': str, 'missing_kind': str})
    td['has_reason'] = td['has_reason'].astype(str).str.lower().eq('true')
    td['forced'] = td['forced'].astype(str).str.lower().eq('true')
    info = json.load(open(os.path.join(W, 'extract_info.json')))
    recs = [json.loads(x) for x in open(os.path.join(W, 'reasons.jsonl'), encoding='utf-8')]
    reasons = [r['reason'] for r in recs]
    has = td.has_reason.to_numpy()
    rows_r = np.flatnonzero(has)
    h = hashlib.sha256()
    for r in reasons:
        h.update(r.encode('utf-8') + b'\n')
    meta = json.load(open(os.path.join(W, 'embeddings.json')))
    E = np.load(os.path.join(W, 'embeddings.npy')).astype(np.float64)
    check([r['row'] for r in recs] == rows_r.tolist() and meta['reasons_sha256'] == h.hexdigest()
          and E.shape == (len(reasons), meta['dim']) and meta['repo'] == ST_REPO and meta['revision'] == ST_REV,
          f'{len(reasons)} reasons in row order; their sha256 = the embeddings\' ({meta["reasons_sha256"][:16]}...); '
          f'vectors {E.shape}, {meta["repo"]} rev {meta["revision"][:8]}, norms {meta["norms_min"]!r}..{meta["norms_max"]!r}')
    n_missing = int((~has).sum())
    check(int(has.sum()) + n_missing == 5420, f'answers clustered {int(has.sum())} + missing or empty reasons {n_missing} '
          f'= 5,420 ({info["missing"]})')
    E = normalize(E)                                       # already unit length (normalize_embeddings=True); exact again

    # ------------------------------------------------------------------------------------------------- main clusters
    rows_kk, fits = [], {}
    print('\n1. sentence vectors -> KMeans')
    for k in KS:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(E)
        sil = float(silhouette_score(E, km.labels_, metric='cosine'))
        sizes = np.bincount(km.labels_, minlength=k)
        fits[k] = km
        rows_kk.append(dict(vectors='sentence', k=k, silhouette=sil, sizes=';'.join(str(int(x)) for x in sizes)))
        print(f'  k {k:2d}: silhouette {sil!r}; sizes {sizes.tolist()}', flush=True)
    best = max([d for d in rows_kk if d['vectors'] == 'sentence'], key=lambda d: (d['silhouette'], -d['k']))
    K = best['k']
    km = fits[K]
    lab = km.labels_
    print(f'  main k = {K} (silhouette {best["silhouette"]!r})')
    rows_s = []
    for s in range(SEED, SEED + N_STAB):
        l2 = KMeans(n_clusters=K, n_init=20, random_state=s).fit(E).labels_
        rows_s.append(dict(what=f'sentence vectors, KMeans random_state {s}', ari=float(adjusted_rand_score(lab, l2))))
    aris = np.array([d['ari'] for d in rows_s])
    print(f'  stability: ARI mean {float(aris.mean())!r}, min {float(aris.min())!r}, max {float(aris.max())!r}')

    # ------------------------------------------------------------------------------------------------- TF-IDF
    stop = sorted(set(CUSTOM_STOP) | set(RISK_WORDS))
    vec = TfidfVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=5, sublinear_tf=True)
    X = vec.fit_transform([clean(r) for r in reasons])
    terms = np.array(vec.get_feature_names_out())
    print(f'\n2. TF-IDF: {X.shape[0]} x {X.shape[1]} terms (unigrams {sum(" " not in t for t in terms)}, bigrams '
          f'{sum(" " in t for t in terms)}); answers without any term {int((X.getnnz(axis=1) == 0).sum())}')
    svd = TruncatedSVD(n_components=100, random_state=SEED)
    Zn = normalize(svd.fit_transform(X))
    tf_lab = {}
    for k in KS:
        km2 = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Zn)
        tf_lab[k] = km2.labels_
        sil = float(silhouette_score(Zn, km2.labels_, metric='cosine'))
        sizes = np.bincount(km2.labels_, minlength=k)
        rows_kk.append(dict(vectors='tfidf_svd100', k=k, silhouette=sil, sizes=';'.join(str(int(x)) for x in sizes)))
        print(f'  k {k:2d}: silhouette {sil!r}; sizes {sizes.tolist()}', flush=True)
    ari_tf = float(adjusted_rand_score(lab, tf_lab[K]))
    rows_s.append(dict(what=f'TF-IDF -> SVD 100 -> KMeans, k {K}, against the sentence-vector clusters', ari=ari_tf))
    print(f'  ARI sentence vs TF-IDF clusters at k {K}: {ari_tf!r}')

    # ------------------------------------------------------------------------------------------------- t-SNE
    tsne = TSNE(n_components=2, perplexity=30, metric='cosine', random_state=SEED)
    Y = tsne.fit_transform(E)
    print(f'\n3. t-SNE on the sentence vectors done (KL divergence {float(tsne.kl_divergence_)!r})')

    # ------------------------------------------------------------------------------------------------- answers table
    cl = np.full(len(td), np.nan)
    cl[rows_r] = lab
    tx, ty = np.full(len(td), np.nan), np.full(len(td), np.nan)
    tx[rows_r], ty[rows_r] = Y[:, 0], Y[:, 1]
    ans = td[['segment', 'frame_idx', 'repeat', 'risk', 'forced']].copy()
    ans['cluster'] = pd.array(cl, dtype='Float64').astype('Int64')
    ans['tsne_x'], ans['tsne_y'] = tx, ty
    ans['hazard_type'], ans['main_hazard_camera'] = td['hazard_type'], td['camera']
    risk = td.risk.to_numpy(float)
    overall = float(np.nanmean(risk))

    # ------------------------------------------------------------------------------------------------- resamples
    seg_names = td.segment.to_numpy()
    seg_list = np.array(sorted(set(seg_names)))
    nb_r, seed_r, frames_r, segs_r = ar.read_params_risk(os.path.join(A, 'params.txt'))
    check((nb_r, seed_r, frames_r) == (N_BOOT, BOOT_SEED, 1084) and list(seg_list) == segs_r,
          f'{A}/params.txt: {nb_r} resamples, seed {seed_r}, {frames_r} frames, the same {len(seg_list)} segments')
    rng = np.random.default_rng(BOOT_SEED)
    draws = [rng.choice(seg_list, len(seg_list)) for _ in range(N_BOOT)]
    hh = hashlib.sha256()
    for dr in draws:
        hh.update(np.searchsorted(seg_list, dr).astype(np.int32).tobytes())
    check(hh.hexdigest() == HASH, f"drawn segments' sha256 {hh.hexdigest()}")
    seg_rows = {s: np.flatnonzero(seg_names == s) for s in seg_list}
    bmean = np.full((N_BOOT, K), np.nan)
    bdiff = np.full((N_BOOT, K), np.nan)
    for b, dr in enumerate(draws):
        idx = np.concatenate([seg_rows[s] for s in dr])
        r_, c_ = risk[idx], cl[idx]
        ov = np.nanmean(r_)
        for c in range(K):
            sel = c_ == c
            if sel.any():
                bmean[b, c] = r_[sel].mean()
                bdiff[b, c] = bmean[b, c] - ov

    def pct(v):
        ok = ~np.isnan(v)
        if not ok.any():
            return np.nan, np.nan, 0, int((~ok).sum())
        lo, hi = np.percentile(v[ok], [2.5, 97.5])
        return float(lo), float(hi), int(ok.sum()), int((~ok).sum())

    # ------------------------------------------------------------------------------------------------- per cluster
    rows_c, rows_t, rows_e, rows_h = [], [], [], []
    fr = pd.DataFrame({'segment': td.segment, 'frame_idx': td.frame_idx, 'cluster': cl})
    per_frame = fr.groupby(['segment', 'frame_idx'])['cluster'].agg(lambda s: tuple(s.tolist()))
    all_one = per_frame.map(lambda t: len(t) == 5 and not any(np.isnan(t)) and len(set(t)) == 1)
    hazard_vals = sorted(td.hazard_type.fillna('(missing)').unique())
    for c in range(K):
        m = cl == c
        sub = td[m]
        segc = sub.segment.value_counts()
        rc = risk[m]
        mlo, mhi, mn, mmiss = pct(bmean[:, c])
        dlo, dhi, dn, dmiss = pct(bdiff[:, c])
        touching = per_frame.map(lambda t: c in t)
        frames_all_c = int(per_frame.map(lambda t: len(t) == 5 and all(x == c for x in t)).sum())
        centre = np.asarray(X[lab == c].mean(axis=0)).ravel()
        order = np.lexsort((terms, -centre))[:N_TERMS]
        rows_t += [dict(cluster=c, rank=i + 1, term=terms[j], mean_weight=float(centre[j])) for i, j in enumerate(order)]
        rows_in = np.flatnonzero(lab == c)
        dist = np.linalg.norm(E[rows_in] - km.cluster_centers_[c], axis=1)
        for i, o in enumerate(np.argsort(dist, kind='stable')[:N_EXAMPLES]):
            j = rows_in[o]
            t = rows_r[j]
            rows_e.append(dict(cluster=c, rank=i + 1, distance=float(dist[o]), segment=td.segment[t],
                               frame_idx=int(td.frame_idx[t]), repeat=int(td['repeat'][t]), risk=float(risk[t]),
                               reason=reasons[j]))
        hz = sub.hazard_type.fillna('(missing)').value_counts()
        for v in hazard_vals:
            rows_h.append(dict(cluster=c, hazard_type=v, answers=int(hz.get(v, 0)), share=float(hz.get(v, 0) / len(sub))))
        rows_c.append(dict(cluster=c, answers=int(m.sum()),
                           frames=int(sub[['segment', 'frame_idx']].drop_duplicates().shape[0]),
                           segments=int(sub.segment.nunique()), largest_segment=segc.index[0],
                           largest_segment_answers=int(segc.iloc[0]), forced=int(sub.forced.sum()),
                           mean=float(rc.mean()), median=float(np.median(rc)), sd=float(rc.std(ddof=0)),
                           mean_minus_all=float(rc.mean() - overall), mean_lo=mlo, mean_hi=mhi, mean_nboot=mn,
                           mean_missing=mmiss, diff_lo=dlo, diff_hi=dhi, diff_nboot=dn, diff_missing=dmiss,
                           diff_side=('no interval' if np.isnan(dlo) else 'above 0' if dlo > 0 else 'below 0' if dhi < 0
                                      else 'contains 0'),
                           frames_touching=int(touching.sum()), frames_all_5_here=frames_all_c,
                           share_all_5_here=frames_all_c / int(touching.sum()) if touching.any() else np.nan,
                           under_50_answers=bool(m.sum() < MIN_ANSWERS), top_terms='; '.join(terms[j] for j in order)))

    # ------------------------------------------------------------------------------------------------- checks
    print('\n4. checks of the scores')
    fsm = ans.groupby(['segment', 'frame_idx'], sort=True)['risk'].mean()
    p_fs = os.path.join(A, 'vlm_frame_scores.csv')
    fs = pd.read_csv(p_fs, float_precision='round_trip')
    fs = fs[(fs.model == MODEL) & (fs.condition == COND)].set_index(['segment', 'frame_idx'])['score_mean'].reindex(fsm.index)
    dmax = float(np.nanmax(np.abs(fsm.to_numpy() - fs.to_numpy())))
    check(len(fsm) == 1084 and not fs.isna().any() and np.array_equal(fsm.to_numpy(), fs.to_numpy()),
          f'frame means of the answers\' scores = {p_fs} score_mean on {len(fsm)} frames, exactly (largest |difference| '
          f'{dmax!r})')
    frames = pd.read_csv(os.path.join(A, 'ssm_target_frames.csv'))[['segment', 'frame_idx']] \
        .sort_values(['segment', 'frame_idx']).reset_index(drop=True)
    ssm_df = av.ssm_table(FSSM, frames)
    ssm = {c: ssm_df[c].to_numpy() for c in ssm_df.columns if c.endswith(('_conflict', '_risk', '_use'))}
    ssm = {k_: (v.astype(float) if k_.endswith('_risk') else v.astype(bool)) for k_, v in ssm.items()}
    x = fsm.reindex(pd.MultiIndex.from_frame(frames)).to_numpy(float)
    st = av.all_stats(x, ssm, np.arange(len(frames)), {'all': np.zeros(len(frames), int)})
    p_ag = os.path.join(A, 'agreement.csv')
    with open(p_ag, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        ag = {d['ssm']: (rd.line_num, d) for d in rd
              if d['model'] == MODEL and d['condition'] == COND and d['grouping'] == 'all'}
    lines = [ag[s][0] for s in av.SSMS]
    diffs = [abs(st[s][0]['all'][1] - float(ag[s][1]['tau_b_conflict'])) for s in av.SSMS]
    same_text = sum(txt(st[s][0]['all'][1]) == ag[s][1]['tau_b_conflict'] for s in av.SSMS)
    check(lines == list(range(AG_FIRST, AG_FIRST + 9 * AG_STEP, AG_STEP)) and max(diffs) <= TOL,
          f'the 9 tau-b of {COND}, all, from these frame means = {p_ag} lines {lines} (tau_b_conflict): largest '
          f'|difference| {max(diffs)!r}; the same text {same_text} of 9')

    # ------------------------------------------------------------------------------------------------- write
    pd.DataFrame(rows_kk).to_csv(os.path.join(O, 'exploratory_reason_k.csv'), index=False)
    pd.DataFrame(rows_c).to_csv(os.path.join(O, 'exploratory_reason_clusters.csv'), index=False)
    pd.DataFrame(rows_t).to_csv(os.path.join(O, 'exploratory_reason_terms.csv'), index=False)
    pd.DataFrame(rows_e).to_csv(os.path.join(O, 'exploratory_reason_examples.csv'), index=False)
    pd.DataFrame(rows_h).to_csv(os.path.join(O, 'exploratory_reason_hazard.csv'), index=False)
    pd.DataFrame(rows_s).to_csv(os.path.join(O, 'exploratory_reason_stability.csv'), index=False)
    ans.to_csv(os.path.join(O, 'exploratory_reason_answers.csv'), index=False)
    pd.DataFrame(rows_ck).to_csv(os.path.join(O, 'exploratory_reason_check.csv'), index=False)
    with open(os.path.join(O, 'exploratory_reason_params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'EXPLORATORY: reason clusters of {MODEL} {COND}, main result 1,084 frames (user, 2026-11-13; plan in '
                f'CLAUDE.md "理由分析", sections 2 and 3 as revised)\nrun {time.strftime("%Y-%m-%d %H:%M:%S")}; out_dir '
                f'{os.path.abspath(O)}; work folder {os.path.abspath(W)}\n')
        f.write(f'program {os.path.abspath(__file__)} sha256 {prog_sha}; scikit-learn {sklearn.__version__}, numpy '
                f'{np.__version__}, pandas {pd.__version__}\n')
        f.write(f'inputs: {info["tidy"]} (risk, forced, hazard_type, camera); reasons: parsed.reason of {info["main"]}, '
                f'forced answers of {info["forced"]} (reason_words.read_replies); {p_fs}; {p_ag}; {FSSM}\n')
        f.write(f'answers {info["requests"]}; forced answers among them {info["forced_answers"]}; no readable answer '
                f'{info["no_answer"]}; reasons missing or empty {n_missing} {info["missing"]}; clustered {int(has.sum())}\n')
        f.write(f'sentence vectors: {meta["repo"]} revision {meta["revision"]} ({meta["model_dir"]}), sentence-transformers '
                f'{meta["sentence_transformers"]}, transformers {meta["transformers"]}, torch {meta["torch"]}, '
                f'normalize_embeddings=True, max_seq_length {meta["max_seq_length"]} (longest reason '
                f'{meta["longest_reason_tokens"]} tokens), {meta["threads"]} threads, modules {meta["modules"]}; reasons as '
                f'written; sha256 of the reasons {meta["reasons_sha256"]}\n')
        f.write(f'main clusters: KMeans(n_clusters=k, n_init=20, random_state={SEED}), the rest default, on the sentence '
                f'vectors, k {KS[0]}..{KS[-1]}; silhouette metric cosine; main k {K}\n')
        f.write(f'stability: KMeans(n_clusters={K}, n_init=20, random_state={SEED}..{SEED + N_STAB - 1}); ARI\n')
        f.write('TF-IDF text: lower case; numbers (\\d+(?:[.,]\\d+)*) -> " num "; [^\\w\\s] -> space\n')
        f.write(f'TF-IDF: TfidfVectorizer(stop_words=<list below>, ngram_range=(1, 2), min_df=5, sublinear_tf=True), the '
                f'rest default; vocabulary {X.shape[1]} terms\n')
        f.write(f'stop words ({len(stop)}: the custom list {len(CUSTOM_STOP)} + the risk words {len(RISK_WORDS)}): '
                + ', '.join(stop) + '\n')
        f.write(f'comparison: TruncatedSVD(n_components=100, random_state={SEED}) -> rows normalised -> KMeans as above; '
                f'ARI at k {K} against the sentence-vector clusters {ari_tf!r}\n')
        f.write(f'terms: per cluster the mean TF-IDF vector of its answers, top {N_TERMS}, ties alphabetical\n')
        f.write(f'TSNE(n_components=2, perplexity=30, metric="cosine", random_state={SEED}), the rest default, on the '
                f'sentence vectors\n')
        f.write(f'examples: the {N_EXAMPLES} answers nearest the KMeans centre (Euclidean, sentence-vector space), ties by '
                f'row order\n')
        f.write(f'intervals: {N_BOOT} resamples of whole segments, default_rng({BOOT_SEED}), drawn segments\' sha256 '
                f'{hh.hexdigest()}; percentiles 2.5 / 97.5 (numpy linear) over the resamples where the cluster has an '
                f'answer; no 90 % rule (the plan has none)\n')
        f.write(f'frames with all 5 answers in one cluster: {int(all_one.sum())} of {len(all_one)}\n')

    # ------------------------------------------------------------------------------------------------- report
    print('\n' + '=' * 110)
    print(f'answers {info["requests"]} (forced {info["forced_answers"]}); clustered {int(has.sum())}, missing or empty '
          f'{n_missing}; TF-IDF vocabulary {X.shape[1]}; all answers\' mean score {overall!r}')
    print('\nk, silhouette, sizes (exploratory_reason_k.csv)')
    for i, d in enumerate(rows_kk):
        print(f"  line {i + 2}: {d['vectors']:12s} k {d['k']:2d} silhouette {d['silhouette']!r} sizes {d['sizes']}")
    print(f'main k {K}; stability (exploratory_reason_stability.csv lines 2-{N_STAB + 1}): ARI mean {float(aris.mean())!r}, '
          f'range {float(aris.min())!r} to {float(aris.max())!r}; TF-IDF clusters vs sentence (line {N_STAB + 2}) {ari_tf!r}')
    print(f'frames with all 5 answers in one cluster: {int(all_one.sum())} of {len(all_one)}')
    print('\nclusters (exploratory_reason_clusters.csv)')
    for i, d in enumerate(rows_c):
        print(f"  line {i + 2}: cluster {d['cluster']}: {d['answers']} answers, {d['frames']} frames, {d['segments']} "
              f"segments, largest segment {d['largest_segment_answers']}; mean {d['mean']!r} [{txt(d['mean_lo'])}, "
              f"{txt(d['mean_hi'])}] (missing {d['mean_missing']}), median {d['median']!r}, SD {d['sd']!r}; mean - all "
              f"{d['mean_minus_all']!r} [{txt(d['diff_lo'])}, {txt(d['diff_hi'])}] ({d['diff_side']}); all 5 here "
              f"{d['frames_all_5_here']} of {d['frames_touching']}; terms {d['top_terms']}"
              + ('; FEWER THAN 50 ANSWERS' if d['under_50_answers'] else ''))
    print(f'\ndone in {time.time() - t0:.0f} s; checks not met: {len(failed)}')
    for w in failed:
        print('  ', w)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    if len(sys.argv) >= 4 and sys.argv[1] == 'extract':
        extract(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 5 and sys.argv[1] == 'analyze':
        analyze(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        sys.exit(__doc__)
