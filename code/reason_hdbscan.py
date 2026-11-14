"""EXPLORATORY (user, 2026-11-14; the plan is in CLAUDE.md "理由分析" -> "HDBSCAN 稳健性检查", recorded as given and committed
before this program was written; it is not changed after the results): HDBSCAN on UMAP as a robustness check of the
KMeans reason clusters. Qwen3.6-27B, sep_camera, the main result's 1,084 frames (238 segments). No embedding is redone,
no KMeans is run. CPU only; no image is opened.
Methods (inputs, all checked):
  M1, answers: round 1's 5,420 sentence vectors (tmp_checks/reason_27b/run1/embeddings.npy, sha256 30ba0002..., as
    stored, float32), rows = answers_base.csv; score = the answer's own score; KMeans k 7 labels =
    main_1084/exploratory_reason_answers.csv (same rows, same order); text = the reason.
  M2, frames, joined texts: step 3's 1,084 vectors (tmp_checks/reason_27b/concat/run1/embeddings.npy, job 31795100,
    sha256 3c778ec2..., as stored, float32), frames sorted by segment, frame_idx; KMeans k 5 labels =
    main_1084_answers/exploratory_answer_concat_frames.csv column cluster; text = the joined text.
  M3, frames, mean vectors: round 2's frame vectors (df630fb). They are not stored as a file: round 2 computed them in
    memory as normalize(E[rows5].mean(axis=1)) (sklearn.preprocessing.normalize, float64) from round 1's vectors; they
    are recomputed here by that same line from the same (sha256-checked) vectors; KMeans k 4 labels =
    main_1084_frames/exploratory_reason_frames.csv column cluster; text = the joined text of M2 (examples only).
  M2 / M3 score = score_mean (= the mean of the 5 answers' scores, checked).
Fixed settings: umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=20261021), the
  rest default; sklearn.cluster.HDBSCAN(min_cluster_size=m, min_samples=None, metric='euclidean',
  cluster_selection_method='eom'), the rest default, on the UMAP output; m: M1 25, 50, 100 (main 50); M2, M3 10, 20,
  40 (main 20). Seed stability: the main m with UMAP random_state 20261022 and 20261023 too (HDBSCAN unchanged).
Per method x m: clusters, noise (label -1) count and share, cluster sizes; ARI (adjusted_rand_score) against the KMeans
  labels on the non-noise points; the cross table with the noise row. Seeds: pairwise ARI on the points non-noise in
  both runs, noise share of each run. A setting with 0 or 1 cluster is reported as such; its per-cluster tables (the
  main-setting tables below) are skipped.
Main setting of each method: per cluster and for noise: answers (M1), frames, segments, mean score, mean minus the
  overall mean (M1: all 5,420 answers; M2 / M3: all 1,084 frames), 95% interval; 10 examples per cluster (the highest
  probabilities_, ties by segment, frame_idx, repeat; text cut to its first 300 characters). M1: the within-frame
  comparison as step 2 of the answer analysis (frames with at least 1 answer in c and 1 not in c, noise counted as not
  in c; per frame mean in c minus mean not in c; frames weighted equally; frames, segments, frames with the same score
  in all 5 answers).
95% intervals: the main analysis's 2,000 resamples of whole segments (default_rng(20261021), drawn segments' sha256
  48bc1f41...; every answer / frame of a drawn segment, as often as drawn; labels not redone); per resample the group's
  mean minus the mean of all drawn answers / frames (as before); 2.5 / 97.5 % percentiles (numpy, linear) over the
  resamples where the group is present, the others counted as missing.
Checks (exit 1 when one is not met): the vectors' sha256; 5,420 answers, 1,084 frames, 238 segments, 5 per frame; label
  tables in the same order with the expected label sets; frame means = score_mean exactly; the joined texts = M2's.
  The run script compares a second run byte for byte and the sha256 of all earlier reason outputs before and after.
Outputs (out_dir, prefix exploratory_hdbscan_): settings, crosstab, stability, groups, examples (the only file with
  reason text), within, labels_m1 / _m2 / _m3 (one row per point, no text), check (.csv) and params.txt.
usage: python -I reason_hdbscan.py <waymo_v143 folder> <reason work folder (tmp_checks/reason_27b)> <out_dir>
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_rules as ar  # noqa: E402

MODEL, COND = 'qwen3.6-27b', 'sep_camera'
N_BOOT, BOOT_SEED = 2000, 20261021
HASH = '48bc1f413f9d5b99da57ff4edde7bbb4eb378c5a9343c44ccad4e0165466604c'
EMB1_SHA = '30ba000292417d5047d2279dba1fa1a084883a49ebc9cd379a9aeee6a46296be'
EMB2_SHA = '3c778ec2f95faaf6cd21b5ac83f3a0839600da71baf41896cd47737e37bb292a'
REASONS_SHA = 'c718132acf962585cb1463f0fc1736b77d984b598808498f2bd8e5a3c7345373'
TEXTS2_SHA = '70ca6cd42d52b55a9a70483bb7fd7a6d9b50a7824c135ea8a53eafd029d0343a'
UMAP_SEEDS = [20261021, 20261022, 20261023]
MCS = {'M1': [25, 50, 100], 'M2': [10, 20, 40], 'M3': [10, 20, 40]}
MAIN = {'M1': 50, 'M2': 20, 'M3': 20}
KM_K = {'M1': 7, 'M2': 5, 'M3': 4}
N_EX, N_CHARS = 10, 300
failed, rows_ck = [], []


def check(ok, what):
    rows_ck.append(dict(check=what, ok=bool(ok)))
    print(('  check ok: ' if ok else '  CHECK NOT MET: ') + what, flush=True)
    if not ok:
        failed.append(what)


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def pct(v):
    ok = ~np.isnan(v)
    if not ok.any():
        return np.nan, np.nan, 0, int((~ok).sum())
    lo, hi = np.percentile(v[ok], [2.5, 97.5])
    return float(lo), float(hi), int(ok.sum()), int((~ok).sum())


def side(lo, hi):
    return 'no interval' if np.isnan(lo) else 'above 0' if lo > 0 else 'below 0' if hi < 0 else 'contains 0'


def main():
    import sklearn
    import umap
    import numba
    from sklearn.cluster import HDBSCAN
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import normalize
    t0 = time.time()
    L, WR, O = sys.argv[1:4]
    os.makedirs(O, exist_ok=True)
    A = os.path.join(L, 'analysis', 'after_gemma-4-31b-it', 'main_1084')
    RA = os.path.join(L, 'analysis', 'reason_27b')
    W1, W2 = os.path.join(WR, 'run1'), os.path.join(WR, 'concat', 'run1')
    prog_sha = sha(os.path.abspath(__file__))
    print(f'EXPLORATORY: HDBSCAN robustness check, {MODEL} {COND}; program sha256 {prog_sha}; umap-learn '
          f'{umap.__version__}, scikit-learn {sklearn.__version__}, numba {numba.__version__}\n\n1. inputs')

    # ---- answers (M1)
    td = pd.read_csv(os.path.join(W1, 'answers_base.csv'), keep_default_na=False, na_values=[''],
                     float_precision='round_trip')
    recs = [json.loads(x) for x in open(os.path.join(W1, 'reasons.jsonl'), encoding='utf-8')]
    reasons = [r['reason'] for r in recs]
    h = hashlib.sha256()
    for r in reasons:
        h.update(r.encode('utf-8') + b'\n')
    p_e1 = os.path.join(W1, 'embeddings.npy')
    E1 = np.load(p_e1)
    check(sha(p_e1) == EMB1_SHA and h.hexdigest() == REASONS_SHA and [r['row'] for r in recs] == list(range(len(td)))
          and td.has_reason.all() and E1.shape == (5420, 768),
          f'M1 vectors {p_e1}: sha256 = round 1\'s, {E1.shape} {E1.dtype}; reasons as round 1, in row order')
    r1 = pd.read_csv(os.path.join(RA, 'main_1084', 'exploratory_reason_answers.csv'))
    check(len(r1) == len(td) and (r1.segment.to_numpy() == td.segment.to_numpy()).all()
          and (r1.frame_idx.to_numpy() == td.frame_idx.to_numpy()).all()
          and (r1['repeat'].to_numpy() == td['repeat'].to_numpy()).all()
          and np.array_equal(r1.risk.to_numpy(float), td.risk.to_numpy(float)) and set(r1.cluster) == set(range(7)),
          'M1 KMeans labels: main_1084/exploratory_reason_answers.csv, same answers in the same order, same risk, k 7')
    risk = td.risk.to_numpy(float)
    aseg = td.segment.to_numpy()

    # ---- frames
    frames = pd.read_csv(os.path.join(A, 'ssm_target_frames.csv'))[['segment', 'frame_idx']] \
        .sort_values(['segment', 'frame_idx']).reset_index(drop=True)
    n = len(frames)
    mi = pd.MultiIndex.from_frame(frames)
    fpos = pd.Series(np.arange(n), index=mi).reindex(pd.MultiIndex.from_arrays([td.segment, td.frame_idx])).to_numpy()
    reps = {}
    for i, (fp, rp) in enumerate(zip(fpos, td['repeat'])):
        reps.setdefault(int(fp), {})[int(rp)] = i
    check(n == 1084 and frames.segment.nunique() == 238 and not np.isnan(fpos.astype(float)).any()
          and all(sorted(reps[f]) == [0, 1, 2, 3, 4] for f in range(n)) and len(td) == 5 * n,
          f'{n} frames, {frames.segment.nunique()} segments, 5 answers each (repeats 0-4)')
    rows5 = np.array([[reps[f][r] for r in range(5)] for f in range(n)])
    fpos = fpos.astype(int)
    fseg = frames.segment.to_numpy()
    p_fs = os.path.join(A, 'vlm_frame_scores.csv')
    fs = pd.read_csv(p_fs, float_precision='round_trip')
    fs = fs[(fs.model == MODEL) & (fs.condition == COND)].set_index(['segment', 'frame_idx'])['score_mean'].reindex(mi)
    score = fs.to_numpy(float)
    fm = td.groupby(['segment', 'frame_idx'], sort=True)['risk'].mean().reindex(mi).to_numpy(float)
    check(not np.isnan(score).any() and np.array_equal(fm, score), f'frame score = mean of the 5 answers = {p_fs} '
          f'score_mean, exactly')

    # ---- M2
    p_e2 = os.path.join(W2, 'embeddings.npy')
    E2 = np.load(p_e2)
    trec = [json.loads(x) for x in open(os.path.join(W2, 'texts.jsonl'), encoding='utf-8')]
    texts = [r['reason'] for r in trec]
    h2 = hashlib.sha256()
    for t in texts:
        h2.update(t.encode('utf-8') + b'\n')
    rebuilt = ['\n'.join(reasons[rows5[f, r]] for r in range(5)) for f in range(n)]
    check(sha(p_e2) == EMB2_SHA and E2.shape == (n, 768) and h2.hexdigest() == TEXTS2_SHA and texts == rebuilt
          and [(r['segment'], r['frame_idx']) for r in trec] == list(zip(fseg, frames.frame_idx.astype(int))),
          f'M2 vectors {p_e2}: sha256 = job 31795100\'s, {E2.shape} {E2.dtype}; texts = the joined reasons, frame order')
    c2 = pd.read_csv(os.path.join(RA, 'main_1084_answers', 'exploratory_answer_concat_frames.csv'))
    check(len(c2) == n and (c2.segment.to_numpy() == fseg).all()
          and (c2.frame_idx.to_numpy() == frames.frame_idx.to_numpy()).all() and set(c2.cluster) == set(range(5)),
          'M2 KMeans labels: exploratory_answer_concat_frames.csv column cluster, same frames, k 5')

    # ---- M3
    E3 = normalize(E1.astype(np.float64)[rows5].mean(axis=1))
    c3 = pd.read_csv(os.path.join(RA, 'main_1084_frames', 'exploratory_reason_frames.csv'), float_precision='round_trip')
    check(len(c3) == n and (c3.segment.to_numpy() == fseg).all()
          and (c3.frame_idx.to_numpy() == frames.frame_idx.to_numpy()).all() and set(c3.cluster) == set(range(4))
          and np.array_equal(c3.score_mean.to_numpy(float), score),
          'M3 KMeans labels: main_1084_frames/exploratory_reason_frames.csv column cluster, same frames, k 4, same '
          'score_mean')
    print(f'  M3 vectors recomputed as round 2: {E3.shape} {E3.dtype}')

    meth = {
        'M1': dict(X=E1, km=r1.cluster.to_numpy(int), score=risk, seg=aseg, fr=fpos,
                   key=list(zip(td.segment, td.frame_idx.astype(int), td['repeat'].astype(int))), text=reasons),
        'M2': dict(X=E2, km=c2.cluster.to_numpy(int), score=score, seg=fseg, fr=np.arange(n),
                   key=list(zip(fseg, frames.frame_idx.astype(int), [-1] * n)), text=texts),
        'M3': dict(X=E3, km=c3.cluster.to_numpy(int), score=score, seg=fseg, fr=np.arange(n),
                   key=list(zip(fseg, frames.frame_idx.astype(int), [-1] * n)), text=texts),
    }
    if failed:
        sys.exit('input checks not met; nothing written')

    # ---- resamples
    seg_list = np.array(sorted(set(fseg)))
    nb_r, seed_r, frames_r, segs_r = ar.read_params_risk(os.path.join(A, 'params.txt'))
    check((nb_r, seed_r, frames_r) == (N_BOOT, BOOT_SEED, 1084) and list(seg_list) == segs_r,
          f'{A}/params.txt: {nb_r} resamples, seed {seed_r}, {frames_r} frames, the same {len(seg_list)} segments')
    rng = np.random.default_rng(BOOT_SEED)
    draws = [rng.choice(seg_list, len(seg_list)) for _ in range(N_BOOT)]
    hh = hashlib.sha256()
    for dr in draws:
        hh.update(np.searchsorted(seg_list, dr).astype(np.int32).tobytes())
    check(hh.hexdigest() == HASH, f"drawn segments' sha256 {hh.hexdigest()}")

    # ---- UMAP + HDBSCAN
    print('\n2. UMAP + HDBSCAN', flush=True)
    res = {}
    for m, d in meth.items():
        for s in UMAP_SEEDS:
            mcs_list = MCS[m] if s == UMAP_SEEDS[0] else [MAIN[m]]
            t1 = time.time()
            U = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=s).fit_transform(d['X'])
            for mc in mcs_list:
                hd = HDBSCAN(min_cluster_size=mc, min_samples=None, metric='euclidean',
                             cluster_selection_method='eom').fit(U)
                res[(m, mc, s)] = (hd.labels_.astype(int), hd.probabilities_.astype(float))
            print(f'  {m} UMAP seed {s}: {time.time() - t1:.0f} s; ' + '; '.join(
                f'm {mc}: {len(set(res[(m, mc, s)][0]) - {-1})} clusters, noise {int((res[(m, mc, s)][0] == -1).sum())}'
                for mc in mcs_list), flush=True)

    rows_s, rows_x, rows_st = [], [], []
    for (m, mc, s), (lab, prob) in res.items():
        d = meth[m]
        cl = sorted(set(lab) - {-1})
        nn = lab != -1
        sizes = [int((lab == c).sum()) for c in cl]
        ari = float(adjusted_rand_score(d['km'][nn], lab[nn])) if nn.any() else np.nan
        rows_s.append(dict(method=m, min_cluster_size=mc, umap_seed=s, main=(mc == MAIN[m] and s == UMAP_SEEDS[0]),
                           points=len(lab), clusters=len(cl), noise=int((~nn).sum()), noise_share=float((~nn).mean()),
                           sizes=';'.join(map(str, sizes)), non_noise=int(nn.sum()),
                           ari_vs_kmeans_non_noise=ari, kmeans_k=KM_K[m],
                           note='0 or 1 cluster: per-cluster tables skipped' if len(cl) <= 1 else ''))
        if s == UMAP_SEEDS[0]:
            for c in [-1] + cl:
                for k in range(KM_K[m]):
                    rows_x.append(dict(method=m, min_cluster_size=mc, hdbscan_cluster=('noise' if c == -1 else c),
                                       kmeans_cluster=k, count=int(((lab == c) & (d['km'] == k)).sum())))
    for m in meth:
        for i in range(3):
            for j in range(i + 1, 3):
                la, lb = res[(m, MAIN[m], UMAP_SEEDS[i])][0], res[(m, MAIN[m], UMAP_SEEDS[j])][0]
                both = (la != -1) & (lb != -1)
                rows_st.append(dict(method=m, min_cluster_size=MAIN[m], seed_a=UMAP_SEEDS[i], seed_b=UMAP_SEEDS[j],
                                    both_non_noise=int(both.sum()),
                                    ari=float(adjusted_rand_score(la[both], lb[both])) if both.any() else np.nan,
                                    noise_share_a=float((la == -1).mean()), noise_share_b=float((lb == -1).mean())))

    # ---- main-setting tables
    rows_g, rows_e, rows_w = [], [], []
    for m, d in meth.items():
        lab, prob = res[(m, MAIN[m], UMAP_SEEDS[0])]
        cl = sorted(set(lab) - {-1})
        if len(cl) <= 1:
            print(f'  {m} main setting: {len(cl)} cluster(s), per-cluster tables skipped')
            continue
        sc, sg = d['score'], d['seg']
        seg_rows = {s: np.flatnonzero(sg == s) for s in seg_list}
        groups = cl + [-1]
        bd = np.full((N_BOOT, len(groups)), np.nan)
        for b, dr in enumerate(draws):
            idx = np.concatenate([seg_rows[s] for s in dr])
            x, lb = sc[idx], lab[idx]
            ov = float(x.mean())
            for gi, g in enumerate(groups):
                sel = lb == g
                if sel.any():
                    bd[b, gi] = float(x[sel].mean()) - ov
        overall = float(sc.mean())
        for gi, g in enumerate(groups):
            sel = lab == g
            lo, hi, nb, miss = pct(bd[:, gi])
            rows_g.append(dict(method=m, min_cluster_size=MAIN[m], group=('noise' if g == -1 else g),
                               answers=(int(sel.sum()) if m == 'M1' else np.nan),
                               frames=int(np.unique(d['fr'][sel]).size), segments=int(np.unique(sg[sel]).size),
                               mean=float(sc[sel].mean()), overall_mean=overall,
                               mean_minus_overall=float(sc[sel].mean() - overall), lo95=lo, hi95=hi, nboot=nb,
                               missing=miss, side=side(lo, hi)))
        for c in cl:
            ii = np.flatnonzero(lab == c)
            order = sorted(ii, key=lambda i: (-prob[i], d['key'][i]))[:N_EX]
            for r, i in enumerate(order):
                sgm, fi, rp = d['key'][i]
                rows_e.append(dict(method=m, cluster=c, rank=r + 1, probability=float(prob[i]), segment=sgm,
                                   frame_idx=fi, repeat=(rp if m == 'M1' else ''), score=float(sc[i]),
                                   text_first_300_characters=d['text'][i][:N_CHARS]))
        if m == 'M1':
            fseg5 = aseg[rows5[:, 0]]
            same5 = (risk[rows5] == risk[rows5][:, :1]).all(1)
            fr_rows = {s: np.flatnonzero(fseg5 == s) for s in seg_list}
            for c in cl:
                mf = lab[rows5] == c
                nin = mf.sum(1)
                sp = (nin >= 1) & (nin <= 4)
                rf = risk[rows5]
                dd = np.full(n, np.nan)
                dd[sp] = (np.where(mf, rf, 0).sum(1)[sp] / nin[sp]) - (np.where(~mf, rf, 0).sum(1)[sp] / (5 - nin[sp]))
                bw = np.full(N_BOOT, np.nan)
                for b, dr in enumerate(draws):
                    fidx = np.concatenate([fr_rows[s] for s in dr])
                    fs_ = fidx[sp[fidx]]
                    if len(fs_):
                        bw[b] = dd[fs_].mean()
                lo, hi, nb, miss = pct(bw)
                rows_w.append(dict(method=m, min_cluster_size=MAIN[m], cluster=c, split_frames=int(sp.sum()),
                                   segments=int(np.unique(fseg5[sp]).size),
                                   mean_diff=float(dd[sp].mean()) if sp.any() else np.nan, lo95=lo, hi95=hi, nboot=nb,
                                   missing=miss, side=side(lo, hi),
                                   split_frames_same_score_all_5=int((sp & same5).sum()),
                                   frames_all_5_in=int((nin == 5).sum())))

    # ---- write
    pfx = os.path.join(O, 'exploratory_hdbscan_')
    pd.DataFrame(rows_s).to_csv(pfx + 'settings.csv', index=False)
    pd.DataFrame(rows_x).to_csv(pfx + 'crosstab.csv', index=False)
    pd.DataFrame(rows_st).to_csv(pfx + 'stability.csv', index=False)
    pd.DataFrame(rows_g).to_csv(pfx + 'groups.csv', index=False)
    pd.DataFrame(rows_e).to_csv(pfx + 'examples.csv', index=False)
    pd.DataFrame(rows_w).to_csv(pfx + 'within.csv', index=False)
    for m, d in meth.items():
        out = pd.DataFrame(d['key'], columns=['segment', 'frame_idx', 'repeat'])
        if m != 'M1':
            out = out.drop(columns='repeat')
        out['score'] = d['score']
        out[f'kmeans_k{KM_K[m]}'] = d['km']
        for (mm, mc, s), (lab, prob) in res.items():
            if mm == m:
                out[f'hdbscan_m{mc}_seed{s}'] = lab
                if mc == MAIN[m] and s == UMAP_SEEDS[0]:
                    out['probability_main'] = prob
        out.to_csv(pfx + f'labels_{m.lower()}.csv', index=False)
    pd.DataFrame(rows_ck).to_csv(pfx + 'check.csv', index=False)
    with open(pfx + 'params.txt', 'w', encoding='utf-8') as f:
        f.write(f'EXPLORATORY: HDBSCAN robustness check of the reason clusters, {MODEL} {COND}, 1,084 frames (user, '
                f'2026-11-14; plan in CLAUDE.md "理由分析" -> "HDBSCAN 稳健性检查")\nrun {time.strftime("%Y-%m-%d %H:%M:%S")}; '
                f'out_dir {os.path.abspath(O)}\n')
        f.write(f'program {os.path.abspath(__file__)} sha256 {prog_sha}; umap-learn {umap.__version__}, scikit-learn '
                f'{sklearn.__version__}, numba {numba.__version__}, numpy {np.__version__}, pandas {pd.__version__}\n')
        f.write(f'M1 {p_e1} (sha256 {EMB1_SHA}); M2 {p_e2} (sha256 {EMB2_SHA}); M3 normalize(mean of the 5 M1 vectors), '
                f'float64, as round 2 (not stored by round 2)\n')
        f.write(f'UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=<seed>), the rest default; '
                f'seeds {UMAP_SEEDS} (the last two for the main min_cluster_size only)\n')
        f.write(f'HDBSCAN(min_cluster_size=<m>, min_samples=None, metric="euclidean", cluster_selection_method="eom"), the '
                f'rest default; m {MCS}; main {MAIN}\n')
        f.write(f'95% intervals: {N_BOOT} resamples of whole segments, default_rng({BOOT_SEED}), drawn segments\' sha256 '
                f'{hh.hexdigest()}; per resample group mean minus the mean of all drawn points; percentiles 2.5 / 97.5 '
                f'(numpy linear) over the resamples where the group is present\n')
        f.write(f'examples: the {N_EX} highest probabilities_ per cluster, ties by segment, frame_idx, repeat; text cut to '
                f'{N_CHARS} characters (M1 the reason, M2 / M3 the joined text)\n')

    # ---- report
    print('\n' + '=' * 110)
    print('settings (exploratory_hdbscan_settings.csv)')
    for i, d in enumerate(rows_s):
        print(f"  line {i + 2}: {d['method']} m {d['min_cluster_size']} seed {d['umap_seed']}{' MAIN' if d['main'] else ''}: "
              f"{d['clusters']} clusters, noise {d['noise']} ({d['noise_share']!r}), sizes {d['sizes']}; ARI vs KMeans "
              f"k {d['kmeans_k']} (non-noise {d['non_noise']}) {d['ari_vs_kmeans_non_noise']!r} {d['note']}")
    print('\nseed stability (exploratory_hdbscan_stability.csv)')
    for i, d in enumerate(rows_st):
        print(f"  line {i + 2}: {d['method']} seeds {d['seed_a']}-{d['seed_b']}: both non-noise {d['both_non_noise']}, "
              f"ARI {d['ari']!r}; noise shares {d['noise_share_a']!r} / {d['noise_share_b']!r}")
    print('\ngroups, main settings (exploratory_hdbscan_groups.csv)')
    for i, d in enumerate(rows_g):
        print(f"  line {i + 2}: {d['method']} {d['group']}: answers {d['answers']}, frames {d['frames']}, segments "
              f"{d['segments']}; mean {d['mean']!r}; minus overall {d['mean_minus_overall']!r} 95% interval "
              f"[{d['lo95']!r}, {d['hi95']!r}] ({d['side']}; missing {d['missing']})")
    print('\nwithin frame, M1 main (exploratory_hdbscan_within.csv)')
    for i, d in enumerate(rows_w):
        print(f"  line {i + 2}: cluster {d['cluster']}: {d['split_frames']} split frames, {d['segments']} segments; mean "
              f"difference {d['mean_diff']!r} 95% interval [{d['lo95']!r}, {d['hi95']!r}] ({d['side']}; missing "
              f"{d['missing']}); same score in all 5: {d['split_frames_same_score_all_5']}")
    print(f'\ndone in {time.time() - t0:.0f} s; checks not met: {len(failed)}')
    for w in failed:
        print('  ', w)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main()
