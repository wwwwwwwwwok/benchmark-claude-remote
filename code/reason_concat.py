"""EXPLORATORY (user, 2026-11-14; the plan is in CLAUDE.md "理由分析" -> "逐条答案" -> "第 3 步", recorded as given and
committed before this program was written; it is not changed after the results): frame-level clusters on the 5 reasons
joined into one text, a robustness check of round 2's frame themes (df630fb: the mean of the 5 sentence vectors, k 4).
Qwen3.6-27B, sep_camera, the main result's 1,084 frames (238 segments). CPU only; no image is opened.
Three steps (the middle one is code/reason_embed.py, unchanged, in the sentence environment: all-mpnet-base-v2 rev
e8c3b32, normalize_embeddings=True, max_seq_length 384, the same settings as round 1; it counts the texts over 384
tokens, which it truncates, and that count is reported before the clusters are made):
  texts <waymo_v143> <round-1 work folder> <work folder>: round 1's reasons (reasons.jsonl, sha256 = round 1's) per frame
    joined in run order (repeat 0, 1, 2, 3, 4) with a newline ("\n") between them, no other change; frames sorted by
    segment, frame_idx (round 2's order). Writes <work>/texts.jsonl (row, segment, frame_idx, reason = the joined text).
  analyze <waymo_v143> <round-1 work folder> <work folder> <round-2 out_dir> <out_dir>: the vectors of those texts
    (<work>/embeddings.npy, the texts' sha256 must match) -> normalised (as round 2) -> KMeans(k, n_init=20,
    random_state 20261113), k = 4..12, silhouette (cosine); the chosen k = the highest silhouette (a tie: the smaller k),
    as df630fb. Frame score = the mean of the 5 answers' scores (= vlm_frame_scores.csv score_mean, checked).
    Per cluster at the chosen k: frames, segments, mean score, mean minus the mean of all 1,084 frames, its 95% interval
    from the main analysis's 2,000 resamples of whole segments (default_rng(20261021), drawn segments' sha256
    48bc1f41...; every frame of a drawn segment, as often as drawn; clusters not redone; per resample the cluster's mean
    minus the mean of all drawn frames, as round 2; 2.5 / 97.5 % percentiles, numpy linear, over the resamples where the
    cluster has a frame, the others counted as missing); 10 example frames per cluster: the 10 nearest the KMeans centre
    (Euclidean, as round 2's examples; ties by frame order), the joined text cut to its first 300 characters.
    Agreement: ARI (scikit-learn adjusted_rand_score) of the chosen-k clusters and of the k = 4 clusters against round
    2's clusters (exploratory_reason_frames.csv column cluster, k 4), and their cross tables.
Checks (exit 1 when one is not met): 1,084 frames, 238 segments, 5 reasons per frame (repeats 0-4); round 1's reasons
and vectors as recorded; the texts rebuilt = texts.jsonl, their sha256 = the embeddings' record; frame means =
score_mean exactly; round 2's frames table has the same 1,084 frames, clusters 0..3. The run script compares a second
run byte for byte.
Outputs in <out_dir>: exploratory_answer_concat_k.csv, _clusters.csv, _examples.csv (the only file with reason text),
_agreement.csv, _crosstab.csv, _frames.csv (one row per frame, no text), _check.csv, _params.txt.
usage: python -I reason_concat.py texts <waymo_v143> <round-1 work> <work>
       python -I reason_concat.py analyze <waymo_v143> <round-1 work> <work> <round-2 out_dir> <out_dir>
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
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
SEED, N_BOOT, BOOT_SEED = 20261113, 2000, 20261021
HASH = '48bc1f413f9d5b99da57ff4edde7bbb4eb378c5a9343c44ccad4e0165466604c'
EMB_SHA = '30ba000292417d5047d2279dba1fa1a084883a49ebc9cd379a9aeee6a46296be'      # round 1's embeddings.npy
REASONS_SHA = 'c718132acf962585cb1463f0fc1736b77d984b598808498f2bd8e5a3c7345373'  # round 1's reasons
KS = list(range(4, 13))
N_EXAMPLES, N_CHARS, K_REF = 10, 300, 4
failed, rows_ck = [], []


def check(ok, what):
    rows_ck.append(dict(check=what, ok=bool(ok)))
    print(('  check ok: ' if ok else '  CHECK NOT MET: ') + what, flush=True)
    if not ok:
        failed.append(what)


def build(L, W1):
    """Round 1's reasons joined per frame; returns frames (sorted), rows5, the answers table, the texts."""
    A = os.path.join(L, 'analysis', 'after_gemma-4-31b-it', 'main_1084')
    td = pd.read_csv(os.path.join(W1, 'answers_base.csv'), keep_default_na=False, na_values=[''],
                     float_precision='round_trip')
    recs = [json.loads(x) for x in open(os.path.join(W1, 'reasons.jsonl'), encoding='utf-8')]
    reasons = [r['reason'] for r in recs]
    h = hashlib.sha256()
    for r in reasons:
        h.update(r.encode('utf-8') + b'\n')
    check(h.hexdigest() == REASONS_SHA and [r['row'] for r in recs] == list(range(len(td))) and td.has_reason.all()
          and all((r['segment'], int(r['frame_idx']), int(r['repeat'])) == (s, int(f), int(p)) for r, s, f, p in
                  zip(recs, td.segment, td.frame_idx, td['repeat'])),
          f'round 1\'s reasons: sha256 {h.hexdigest()} = round 1\'s; every one of the {len(td)} answers has one, in row order')
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
          f'{n} frames, {frames.segment.nunique()} segments, each with exactly 5 reasons (repeats 0-4)')
    rows5 = np.array([[reps[f][r] for r in range(5)] for f in range(n)])
    texts = ['\n'.join(reasons[rows5[f, r]] for r in range(5)) for f in range(n)]
    return A, frames, rows5, td, texts


def texts_step(L, W1, W):
    os.makedirs(W, exist_ok=True)
    _, frames, _, _, texts = build(L, W1)
    if failed:
        sys.exit('checks not met; nothing written')
    with open(os.path.join(W, 'texts.jsonl'), 'w', encoding='utf-8') as f:
        for i, (s, fi, t) in enumerate(zip(frames.segment, frames.frame_idx, texts)):
            f.write(json.dumps(dict(row=i, segment=s, frame_idx=int(fi), reason=t), ensure_ascii=False) + '\n')
    print(f'{len(texts)} joined texts written to {os.path.join(W, "texts.jsonl")}; longest {max(map(len, texts))} '
          f'characters')


def pct(v):
    ok = ~np.isnan(v)
    if not ok.any():
        return np.nan, np.nan, 0, int((~ok).sum())
    lo, hi = np.percentile(v[ok], [2.5, 97.5])
    return float(lo), float(hi), int(ok.sum()), int((~ok).sum())


def side(lo, hi):
    return 'no interval' if np.isnan(lo) else 'above 0' if lo > 0 else 'below 0' if hi < 0 else 'contains 0'


def analyze(L, W1, W, R2, O):
    import sklearn
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.preprocessing import normalize
    t0 = time.time()
    os.makedirs(O, exist_ok=True)
    prog_sha = hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()
    print(f'EXPLORATORY: frame clusters on the joined reasons, {MODEL} {COND}; program sha256 {prog_sha}\n\n1. inputs')
    A, frames, rows5, td, texts = build(L, W1)
    with open(os.path.join(W1, 'embeddings.npy'), 'rb') as f:
        check(hashlib.sha256(f.read()).hexdigest() == EMB_SHA, 'round 1\'s sentence vectors as recorded (sha256)')
    recs = [json.loads(x) for x in open(os.path.join(W, 'texts.jsonl'), encoding='utf-8')]
    h = hashlib.sha256()
    for r in recs:
        h.update(r['reason'].encode('utf-8') + b'\n')
    meta = json.load(open(os.path.join(W, 'embeddings.json')))
    p_emb = os.path.join(W, 'embeddings.npy')
    emb_sha = hashlib.sha256(open(p_emb, 'rb').read()).hexdigest()
    E = normalize(np.load(p_emb).astype(np.float64))
    check([r['reason'] for r in recs] == texts and [(r['segment'], r['frame_idx']) for r in recs] ==
          list(zip(frames.segment, frames.frame_idx.astype(int))) and meta['reasons_sha256'] == h.hexdigest()
          and E.shape == (1084, 768),
          f'texts.jsonl = the joined texts rebuilt, in frame order; their sha256 {h.hexdigest()} = the vectors\' record; '
          f'vectors {E.shape}')
    n = len(frames)
    mi = pd.MultiIndex.from_frame(frames)
    p_fs = os.path.join(A, 'vlm_frame_scores.csv')
    fs = pd.read_csv(p_fs, float_precision='round_trip')
    fs = fs[(fs.model == MODEL) & (fs.condition == COND)].set_index(['segment', 'frame_idx'])['score_mean'].reindex(mi)
    score = fs.to_numpy(float)
    fm = td.groupby(['segment', 'frame_idx'], sort=True)['risk'].mean().reindex(mi).to_numpy(float)
    check(not np.isnan(score).any() and np.array_equal(fm, score),
          f'frame score = the mean of the 5 answers\' scores = {p_fs} score_mean, exactly')
    p_r2 = os.path.join(R2, 'exploratory_reason_frames.csv')
    r2 = pd.read_csv(p_r2, float_precision='round_trip')
    check(len(r2) == n and (r2.segment.to_numpy() == frames.segment.to_numpy()).all()
          and (r2.frame_idx.to_numpy() == frames.frame_idx.to_numpy()).all() and set(r2.cluster) == set(range(K_REF)),
          f'{p_r2}: the same {n} frames in the same order, clusters 0..{K_REF - 1}')
    lab2 = r2.cluster.to_numpy(int)
    print(f'  texts over {meta["max_seq_length"]} tokens (truncated by the model): {meta["reasons_over_max_seq_length"]}; '
          f'longest {meta["longest_reason_tokens"]} tokens')
    if failed:
        sys.exit('input checks not met; nothing written')

    print('\n2. KMeans on the joined-text vectors')
    rows_k, fits = [], {}
    for k in KS:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(E)
        sil = float(silhouette_score(E, km.labels_, metric='cosine'))
        fits[k] = km
        sizes = np.bincount(km.labels_, minlength=k)
        rows_k.append(dict(k=k, silhouette=sil, sizes=';'.join(str(int(x)) for x in sizes),
                           ari_vs_round2_k4=float(adjusted_rand_score(lab2, km.labels_))))
        print(f'  k {k:2d}: silhouette {sil!r}; sizes {sizes.tolist()}', flush=True)
    K = max(rows_k, key=lambda d: (d['silhouette'], -d['k']))['k']
    km = fits[K]
    lab = km.labels_
    lab4 = fits[K_REF].labels_
    print(f'  chosen k = {K}')

    seg = frames.segment.to_numpy()
    seg_list = np.array(sorted(set(seg)))
    nb_r, seed_r, frames_r, segs_r = ar.read_params_risk(os.path.join(A, 'params.txt'))
    check((nb_r, seed_r, frames_r) == (N_BOOT, BOOT_SEED, 1084) and list(seg_list) == segs_r,
          f'{A}/params.txt: {nb_r} resamples, seed {seed_r}, {frames_r} frames, the same {len(seg_list)} segments')
    rng = np.random.default_rng(BOOT_SEED)
    draws = [rng.choice(seg_list, len(seg_list)) for _ in range(N_BOOT)]
    hh = hashlib.sha256()
    for dr in draws:
        hh.update(np.searchsorted(seg_list, dr).astype(np.int32).tobytes())
    check(hh.hexdigest() == HASH, f"drawn segments' sha256 {hh.hexdigest()}")
    seg_rows = {s: np.flatnonzero(seg == s) for s in seg_list}
    overall = float(score.mean())
    bd = np.full((N_BOOT, K), np.nan)
    for b, dr in enumerate(draws):
        idx = np.concatenate([seg_rows[s] for s in dr])
        sc, lb = score[idx], lab[idx]
        ov = float(sc.mean())
        for c in range(K):
            m = lb == c
            if m.any():
                bd[b, c] = float(sc[m].mean()) - ov

    rows_c, rows_e = [], []
    for c in range(K):
        m = lab == c
        lo, hi, nb, miss = pct(bd[:, c])
        rows_c.append(dict(cluster=c, frames=int(m.sum()), segments=int(np.unique(seg[m]).size),
                           mean=float(score[m].mean()), mean_all_frames=overall, mean_minus_all=float(score[m].mean() - overall),
                           lo95=lo, hi95=hi, nboot=nb, missing=miss, side=side(lo, hi)))
        rows_in = np.flatnonzero(m)
        dist = np.linalg.norm(E[rows_in] - km.cluster_centers_[c], axis=1)
        for i, o in enumerate(np.argsort(dist, kind='stable')[:N_EXAMPLES]):
            f = rows_in[o]
            rows_e.append(dict(cluster=c, rank=i + 1, distance=float(dist[o]), segment=seg[f],
                               frame_idx=int(frames.frame_idx[f]), score_mean=float(score[f]),
                               text_first_300_characters=texts[f][:N_CHARS]))

    rows_a = [dict(what=f'joined texts, chosen k {K}, against round 2 (df630fb, mean vectors, k 4)',
                   ari=float(adjusted_rand_score(lab2, lab))),
              dict(what=f'joined texts, k {K_REF}, against round 2 (df630fb, mean vectors, k 4)',
                   ari=float(adjusted_rand_score(lab2, lab4)))]
    rows_x = []
    for name, l in ((f'chosen k {K}', lab), (f'k {K_REF}', lab4)):
        ct = pd.crosstab(pd.Series(l, name='joined'), pd.Series(lab2, name='round2'))
        for a_ in ct.index:
            rows_x.append(dict(table=name, joined_cluster=int(a_),
                               **{f'round2_cluster_{b_}': int(ct.loc[a_, b_]) for b_ in ct.columns}))

    out = frames.copy()
    out['score_mean'] = score
    out['cluster'] = lab
    out[f'cluster_k{K_REF}'] = lab4
    out['round2_cluster'] = lab2
    pfx = os.path.join(O, 'exploratory_answer_concat_')
    pd.DataFrame(rows_k).to_csv(pfx + 'k.csv', index=False)
    pd.DataFrame(rows_c).to_csv(pfx + 'clusters.csv', index=False)
    pd.DataFrame(rows_e).to_csv(pfx + 'examples.csv', index=False)
    pd.DataFrame(rows_a).to_csv(pfx + 'agreement.csv', index=False)
    pd.DataFrame(rows_x).to_csv(pfx + 'crosstab.csv', index=False)
    out.to_csv(pfx + 'frames.csv', index=False)
    pd.DataFrame(rows_ck).to_csv(pfx + 'check.csv', index=False)
    with open(pfx + 'params.txt', 'w', encoding='utf-8') as f:
        f.write(f'EXPLORATORY: frame clusters on the joined reasons, {MODEL} {COND}, 1,084 frames (user, 2026-11-14; plan '
                f'in CLAUDE.md "理由分析" -> "逐条答案" -> "第 3 步")\nrun {time.strftime("%Y-%m-%d %H:%M:%S")}; out_dir '
                f'{os.path.abspath(O)}; work folder {os.path.abspath(W)}\n')
        f.write(f'program {os.path.abspath(__file__)} sha256 {prog_sha}; scikit-learn {sklearn.__version__}, numpy '
                f'{np.__version__}, pandas {pd.__version__}\n')
        f.write('texts: the 5 reasons of a frame (round 1\'s reasons.jsonl) joined in run order, repeat 0..4, with "\\n"\n')
        f.write(f'vectors: {p_emb} sha256 {emb_sha}; {meta["repo"]} revision {meta["revision"]}, sentence-transformers '
                f'{meta["sentence_transformers"]}, transformers {meta["transformers"]}, torch {meta["torch"]}, '
                f'normalize_embeddings=True, max_seq_length {meta["max_seq_length"]}; texts over it (truncated) '
                f'{meta["reasons_over_max_seq_length"]}, longest {meta["longest_reason_tokens"]} tokens; texts\' sha256 '
                f'{meta["reasons_sha256"]}\n')
        f.write(f'KMeans(n_clusters=k, n_init=20, random_state={SEED}), the rest default, k {KS[0]}..{KS[-1]}; silhouette '
                f'cosine; chosen k {K} (highest silhouette, a tie: the smaller k)\n')
        f.write(f'all frames\' mean score {overall!r}\n')
        f.write(f'examples: the {N_EXAMPLES} frames nearest the KMeans centre (Euclidean), ties by frame order; text cut to '
                f'its first {N_CHARS} characters\n')
        f.write(f'95% intervals: {N_BOOT} resamples of whole segments, default_rng({BOOT_SEED}), drawn segments\' sha256 '
                f'{hh.hexdigest()}; per resample the cluster mean minus the mean of all drawn frames; percentiles 2.5 / '
                f'97.5 (numpy linear) over the resamples where the cluster has a frame\n')
        f.write(f'ARI: adjusted_rand_score against {p_r2} column cluster\n')

    print('\n' + '=' * 110)
    print(f'all frames\' mean score {overall!r}; texts over {meta["max_seq_length"]} tokens {meta["reasons_over_max_seq_length"]}')
    print('\nk (exploratory_answer_concat_k.csv)')
    for i, d in enumerate(rows_k):
        print(f"  line {i + 2}: k {d['k']:2d} silhouette {d['silhouette']!r} sizes {d['sizes']} ARI vs round 2 "
              f"{d['ari_vs_round2_k4']!r}")
    print(f'chosen k {K}\n\nclusters (exploratory_answer_concat_clusters.csv)')
    for i, d in enumerate(rows_c):
        print(f"  line {i + 2}: cluster {d['cluster']}: {d['frames']} frames, {d['segments']} segments; mean {d['mean']!r}; "
              f"minus all {d['mean_minus_all']!r} 95% interval [{d['lo95']!r}, {d['hi95']!r}] ({d['side']}; missing "
              f"{d['missing']})")
    print('\nagreement (exploratory_answer_concat_agreement.csv)')
    for i, d in enumerate(rows_a):
        print(f"  line {i + 2}: {d['what']}: ARI {d['ari']!r}")
    print('\ncross tables (exploratory_answer_concat_crosstab.csv)')
    for i, d in enumerate(rows_x):
        print(f'  line {i + 2}: {d}')
    print(f'\ndone in {time.time() - t0:.0f} s; checks not met: {len(failed)}')
    for w in failed:
        print('  ', w)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == 'texts':
        texts_step(*sys.argv[2:5])
    elif len(sys.argv) == 7 and sys.argv[1] == 'analyze':
        analyze(*sys.argv[2:7])
    else:
        sys.exit(__doc__)
