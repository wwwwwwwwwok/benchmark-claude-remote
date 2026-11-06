"""EXPLORATORY, decided after D's results were seen (user, 2026-11-05, item 2; CLAUDE.md "我定（2026-11-05，看过 0beb2bb…
以后…）"): what Qwen3.6-27B says in the part of its risk score that D's additive model does not get. Only counts are
written; no reply's text is written or printed.

Residual: per condition, the rank score (score_mean ranked among the fit frames / their number, ties the mean rank:
new_cues_fit.py's y; checked against vlm_frame_scores.csv) minus D's additive boosting leave-one-segment-out prediction
(new_cues_fit.py --predictions; checked: every prediction column's out-of-sample R2 recomputed from the file equals
exploratory_d_oos.csv within R2_TOL). Its between-segment share (analyze_rules.between_share), with the risk analysis's
2,000 resamples restricted to the fit frames (the residuals not recomputed; every drawn copy its own segment); the rank
score's share beside (Claude's addition) and the reference for random scores, (segments - 1) / (frames - 1).
Halves: the fit frames' segments sorted, numpy default_rng(SPLIT_SEED).permutation; the first half "find", the rest
"validate".
Words as reason_words.py: of every reply with a readable answer the words of main_hazard and reason (lower case, runs of
a-z, ENGLISH_STOP_WORDS out, no stemming); a frame has a word when one of its replies has it. Hazard types: per frame the
share of each of the prompt's hazard types among the replies that give one of them (analyze_rules.py's H; NaN without).
Find (per condition, the find half): reason_words.word_table and top_lists with the residual in place of the score:
tiers at the tertiles of the residual's rank (high = scored above the prediction), words in at least MIN_FRAMES frames of
the half, the 25 with the largest and the 25 with the smallest difference (share of the high tier's frames with the word
minus the low tier's); hazard types: the mean share in the high tier minus in the low tier (frames with a share).
Candidates: the words in the high 25 of at least MIN_COND conditions (side 'high'); the same for the low 25 ('low').
Validate (per condition, the validate half): tiers at the tertiles of the residual's rank in the half; per candidate word
the difference (high - low share of frames with it), per hazard type (every one) the difference of the mean shares;
intervals: N_BOOT resamples of the half's segments with replacement (numpy default_rng(BOOT_SEED), one rng.choice over the
sorted segments per resample, the same resamples for every condition), the tiers recomputed on each resample's frames
(copies counted); validated = the interval excludes 0 on the side of the find half's difference of the same condition.
Clusters (descriptive; per condition, all the fit frames): a frame's text = the reasons of its replies with a readable
answer joined with spaces; scikit-learn TfidfVectorizer on the reason_words words (repeats counted; the rest default),
KMeans(N_CLUSTERS, random_state=0, n_init=N_INIT, the rest default; Claude's choice: with the default single start,
three made-up, fully separate word lists came out wrong for 1 of 4 seeds, adjusted Rand 0.44); the adjusted Rand index
against random_state 1, 2, 3. Per cluster
(numbered by size): frames, the N_TOP_WORDS words in the most of its frames (ties alphabetical), the N_TOP_WORDS terms
with the largest centre weight (Claude's addition), the mean residual with an interval from the 2,000 resamples
restricted to the fit frames (clusters fixed).
Outputs (out_dir): exploratory_residual_*.csv, exploratory_residual_params.txt; exit code 1 when a check is not met.
usage: python -I residual_words.py <new_cues_fit.py out_dir run with --predictions> <risk subfolder>
       <quantities subfolder> <model dir (vlm_scores/qwen3.6-27b)> <out_dir>
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import hashlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_rules as ar  # noqa: E402
import analyze_vlm_ssm as av  # noqa: E402
import reason_words as rw  # noqa: E402

MODEL = rw.MODEL
PRED_COL = 'pred_boost_add_D'
SPLIT_SEED = 20261105
BOOT_SEED = 20261021
N_BOOT = 2000
MIN_COND = 4
N_CLUSTERS = 12
N_INIT = 10                        # k-means starts (the best kept); scikit-learn's default 'auto' is one start
CLUSTER_SEEDS = (0, 1, 2, 3)
N_TOP_WORDS = 10
R2_TOL = 1e-12


# ------------------------------------------------------------------ small pieces (tested on their own)
def tokens(text):
    """reason_words.words_of's words with their repeats (for TF-IDF)."""
    if text is None:
        return []
    if not isinstance(text, str):
        text = str(text)
    return [w for w in rw.WORD.findall(text.lower()) if w not in ENGLISH_STOP_WORDS]


def tiers_of(x):
    """reason_words.word_table's tiers: 1, 2, 3 at the tertiles of the rank of x."""
    return rw.tertile(rankdata(x) / len(x))[0]


def word_diff(has, tier):
    """has: frames x words (0/1); per word the share of the high tier's frames with it minus the low tier's."""
    return has[tier == 3].mean(0) - has[tier == 1].mean(0)


def hazard_diff(sh, tier):
    """sh: frames x hazard types (NaN: the frame has no listed type); per type the mean share in the high tier minus in
    the low tier, over the frames with a share."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(sh[tier == 3], 0) - np.nanmean(sh[tier == 1], 0)


def candidates(tops, min_cond=MIN_COND):
    """tops: {condition: (high list, low list)} -> {word: (side, number of conditions)} for the words in a side's list
    of at least min_cond conditions."""
    out = {}
    for side, i in (('high', 0), ('low', 1)):
        cnt = {}
        for hi_lo in tops.values():
            for w in hi_lo[i]:
                cnt[w] = cnt.get(w, 0) + 1
        for w, k in cnt.items():
            if k >= min_cond:
                if w in out:
                    raise ValueError(f'{w} is a candidate on both sides')
                out[w] = (side, k)
    return out


def validate(resid, has, sh, seg_frames, draws):
    """The validate half: point differences and their resampled values (tiers anew on every resample's frames).
    resid: the half's residuals; has, sh: its frames x words, frames x types; seg_frames: {segment: positions in the
    half}; draws: list of arrays of drawn segments. Returns (word diff, type diff, n_boot x words, n_boot x types)."""
    t = tiers_of(resid)
    wd, hd = word_diff(has, t), hazard_diff(sh, t)
    W = np.full((len(draws), has.shape[1]), np.nan)
    H = np.full((len(draws), sh.shape[1]), np.nan)
    for b, d in enumerate(draws):
        idx = np.concatenate([seg_frames[s] for s in d])
        tb = tiers_of(resid[idx])
        W[b] = word_diff(has[idx], tb)
        H[b] = hazard_diff(sh[idx], tb)
    return wd, hd, W, H


def judged(find_d, lo, hi):
    """validated: the interval excludes 0 on the side of the find half's difference."""
    if np.isnan(lo) or np.isnan(hi) or np.isnan(find_d) or find_d == 0:
        return False
    return bool(lo > 0) if find_d > 0 else bool(hi < 0)


def clusters(docs, seeds=CLUSTER_SEEDS, k=N_CLUSTERS):
    """TF-IDF of the docs (tokens) and k-means per seed. Returns the labels per seed, the first seed's centres and the
    feature names."""
    vec = TfidfVectorizer(analyzer=tokens)
    X = vec.fit_transform(docs)
    labels, centres = {}, None
    for s in seeds:
        km = KMeans(n_clusters=k, random_state=s, n_init=N_INIT).fit(X)
        labels[s] = km.labels_.copy()
        if centres is None:
            centres = km.cluster_centers_
    return labels, centres, np.array(vec.get_feature_names_out())


def by_size(lab):
    """Cluster numbers 1.. by size (largest first, ties by the k-means label): {k-means label: number}."""
    sz = pd.Series(lab).value_counts()
    order = sorted(sz.index, key=lambda k: (-sz[k], k))
    return {k: i + 1 for i, k in enumerate(order)}


def top_words(word_sets, n=N_TOP_WORDS):
    """The n words in the most of the frames (ties alphabetical): [(word, frames), ...]."""
    cnt = {}
    for ws in word_sets:
        for w in ws:
            cnt[w] = cnt.get(w, 0) + 1
    return sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def main():
    fitdir, risk, quant, model_dir, out = sys.argv[1:6]
    if os.path.exists(os.path.join(out, 'exploratory_residual_params.txt')):
        sys.exit(f'{out} has exploratory_residual_params.txt already: give a new folder')
    t0 = time.time()
    failed, checks = [], []
    conds = av.CONDITIONS

    # --- residuals: predictions of new_cues_fit.py --predictions, checked
    pr = pd.read_csv(os.path.join(fitdir, 'exploratory_d_predictions.csv'), float_precision='round_trip')
    pr = pr[pr.model == MODEL]
    oos = pd.read_csv(os.path.join(fitdir, 'exploratory_d_oos.csv'), float_precision='round_trip')
    oos = oos[oos.model == MODEL].set_index(['condition', 'fit', 'cues'])
    fr = pr[pr.condition == conds[0]][['segment', 'frame_idx']].reset_index(drop=True)
    nf = len(fr)
    y, resid = {}, {}
    dev, pcols = 0.0, [c_ for c_ in pr.columns if c_.startswith('pred_')]
    for c in conds:
        q = pr[pr.condition == c].reset_index(drop=True)
        if not q[['segment', 'frame_idx']].equals(fr):
            sys.exit(f'{c}: the predictions are not on the same frames in the same order')
        y[c] = q.y_rank.to_numpy(float)
        resid[c] = y[c] - q[PRED_COL].to_numpy(float)
        for col in pcols:
            m_, s_ = col[len('pred_'):].rsplit('_', 1)
            p = q[col].to_numpy(float)
            r2 = 1.0 - float(((y[c] - p) ** 2).sum()) / float(((y[c] - y[c].mean()) ** 2).sum())
            dev = max(dev, abs(r2 - oos.loc[(c, m_, s_), 'r2_oos']))
    checks.append(f'out-of-sample R2 of the {len(pcols)} prediction columns x {len(conds)} conditions recomputed from '
                  f'exploratory_d_predictions.csv against exploratory_d_oos.csv: largest difference {dev:.3g} '
                  f'(tolerance {R2_TOL:g})')
    if not dev <= R2_TOL:
        failed.append('predictions against the out-of-sample table')
    fsc = pd.read_csv(os.path.join(risk, 'vlm_frame_scores.csv'), float_precision='round_trip')
    bad_y = 0
    for c in conds:
        s = fsc[(fsc.model == MODEL) & (fsc.condition == c)].set_index(['segment', 'frame_idx'])['score_mean']
        bad_y += int((ar.rank01(s.reindex(pd.MultiIndex.from_frame(fr)).to_numpy(float)) != y[c]).sum())
    checks.append(f'y_rank = rank of score_mean among the {nf} fit frames / {nf}: frames not the same {bad_y}')
    if bad_y:
        failed.append('y_rank')

    # --- resamples (as new_cues_fit.py), restricted to the fit frames
    st = pd.read_csv(os.path.join(risk, 'ssm_target_frames.csv'))[['segment', 'frame_idx']]
    seg_list = np.array(sorted(set(st.segment)))
    n_boot_r, seed_r, frames_r, segs_r = ar.read_params_risk(os.path.join(risk, 'params.txt'))
    if (n_boot_r, seed_r, frames_r) != (N_BOOT, BOOT_SEED, len(st)) or list(seg_list) != segs_r:
        sys.exit('stopped before the resamples are used: resamples, seed, frames or segments not the risk analysis\'s')
    rng = np.random.default_rng(BOOT_SEED)
    draws = [rng.choice(seg_list, len(seg_list)) for _ in range(N_BOOT)]
    h = hashlib.sha256()
    for d in draws:
        h.update(np.searchsorted(seg_list, d).astype(np.int32).tobytes())
    nb_q, seed_q, hash_q = ar.read_hash_quantities(os.path.join(quant, 'params.txt'))
    if (nb_q, seed_q, hash_q) != (N_BOOT, BOOT_SEED, h.hexdigest()):
        sys.exit('stopped before the resamples are used: the drawn segments are not the quantities analysis\'s')
    seg_f = fr.segment.to_numpy()
    segs_fit = np.array(sorted(set(seg_f)))
    seg_frames = {s_: np.flatnonzero(seg_f == s_) for s_ in segs_fit}
    g_fit = np.searchsorted(segs_fit, seg_f)

    def drawn(d):
        idx = [seg_frames[x] for x in d if x in seg_frames]
        gg = np.concatenate([np.full(len(v), k) for k, v in enumerate(idx)])
        return np.concatenate(idx), gg

    dd = [drawn(d) for d in draws]

    # --- 1. between-segment share of the residual (and of the rank score)
    brow = []
    ref = (len(segs_fit) - 1) / (nf - 1)
    for c in conds:
        for what, v in (('residual', resid[c]), ('rank score', y[c])):
            pt = ar.between_share(v, g_fit)
            bb = np.array([ar.between_share(v[i], gg) for i, gg in dd])
            lo, hi, nv = ar.ci(bb)
            brow.append(dict(model=MODEL, condition=c, of=what, frames=nf, segments=len(segs_fit), between_share=pt,
                             lo=lo, hi=hi, nboot=nv, random_reference=ref))
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(brow).to_csv(os.path.join(out, 'exploratory_residual_between.csv'), index=False)
    print(f'residuals ready ({time.time() - t0:.0f} s)', flush=True)

    # --- 2. halves
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(segs_fit))
    half = len(segs_fit) // 2
    find_s, val_s = np.array(sorted(segs_fit[perm[:half]])), np.array(sorted(segs_fit[perm[half:]]))
    fidx = np.flatnonzero(np.isin(seg_f, find_s))
    vidx = np.flatnonzero(np.isin(seg_f, val_s))
    pd.DataFrame({'segment': np.concatenate([find_s, val_s]),
                  'half': ['find'] * len(find_s) + ['validate'] * len(val_s)}) \
        .to_csv(os.path.join(out, 'exploratory_residual_halves.csv'), index=False)
    hs = hashlib.sha256('\n'.join(find_s).encode()).hexdigest()
    vpos = {s_: np.flatnonzero(seg_f[vidx] == s_) for s_ in val_s}
    rngv = np.random.default_rng(BOOT_SEED)
    draws_v = [rngv.choice(val_s, len(val_s)) for _ in range(N_BOOT)]
    hv = hashlib.sha256()
    for d in draws_v:
        hv.update(np.searchsorted(val_s, d).astype(np.int32).tobytes())

    # --- replies (reason_words.py), checked against the tidy table
    rq = pd.read_csv(os.path.join(risk, 'vlm_requests_tidy.csv'), low_memory=False,
                     usecols=['model', 'condition', 'segment', 'frame_idx', 'repeat', 'risk', 'main_hazard',
                              'hazard_type'], keep_default_na=False, na_values=[''],
                     dtype={'main_hazard': str, 'hazard_type': str})
    rq = rq[rq.model == MODEL].merge(fr, on=['segment', 'frame_idx'], how='inner').reset_index(drop=True)
    ks = list(zip(rq.segment, rq.frame_idx.astype(int), rq.condition, rq['repeat'].astype(int)))
    rep, info = rw.read_replies(model_dir, set(ks))
    missing = sum(1 for k in ks if k not in rep)
    r_here = np.array([rep[k][0] if k in rep else np.nan for k in ks])
    r_tidy = rq.risk.to_numpy(float)
    bad_risk = int((~((r_here == r_tidy) | (np.isnan(r_here) & np.isnan(r_tidy)))).sum())
    bad_mh = int(sum(1 for k, y_ in zip(ks, rq.main_hazard) if not rw.same_text(rep[k][1] if k in rep else None, y_)))
    checks.append(f'requests of {MODEL} on the fit frames in vlm_requests_tidy.csv: {len(ks)}; not among the replies '
                  f'read {missing}; risk score not the same {bad_risk}; main_hazard not the same {bad_mh}')
    if missing or bad_risk or bad_mh:
        failed.append('replies against vlm_requests_tidy.csv')
    print(f'read {info["main_lines"]} lines of {info["main"]}; forced answers used {info["forced_used"]} '
          f'({time.time() - t0:.0f} s)', flush=True)
    pos = {k: i for i, k in enumerate(zip(fr.segment, fr.frame_idx.astype(int)))}

    # --- per condition: words, hazard shares, documents
    fw, rwds, docs, sh = {}, {}, {}, {}
    nlist = len(av.HAZARD_TYPES)
    for c in conds:
        q = rq[rq.condition == c]
        reps = [[] for _ in range(nf)]
        cnt = np.zeros((nf, nlist))
        for sg, fi, rp, hz in zip(q.segment, q.frame_idx.astype(int), q['repeat'].astype(int), q.hazard_type):
            i = pos[(sg, fi)]
            reps[i].append(rep.get((sg, fi, c, rp), (np.nan, None, None)))
            if isinstance(hz, str) and hz in av.HAZARD_TYPES:
                cnt[i, av.HAZARD_TYPES.index(hz)] += 1
        fw[c] = [rw.frame_words_of(v) for v in reps]
        docs[c] = [' '.join(str(rs) for r_, _, rs in v if not np.isnan(r_) and rs is not None) for v in reps]
        rwds[c] = [set(tokens(d_)) for d_ in docs[c]]
        tot = cnt.sum(1)
        sh[c] = np.where(tot[:, None] > 0, cnt / np.where(tot > 0, tot, 1)[:, None], np.nan)

    # --- 3. find
    words_rows, haz_rows, tier_rows, tops, find_t = [], [], [], {}, {}
    for c in conds:
        tier, cuts, n_t, t = rw.word_table([fw[c][i] for i in fidx], resid[c][fidx])
        find_t[c] = tier
        hi, lo = rw.top_lists(t)
        tops[c] = (list(hi.word), list(lo.word))
        words_rows.append(t.assign(condition=c, rank_high=t.word.map({w: i + 1 for i, w in enumerate(hi.word)}),
                                   rank_low=t.word.map({w: i + 1 for i, w in enumerate(lo.word)})))
        hd = hazard_diff(sh[c][fidx], tier)
        for j, k in enumerate(av.HAZARD_TYPES):
            v = sh[c][fidx, j]
            haz_rows.append(dict(condition=c, hazard_type=k, mean_share_high=float(np.nanmean(v[tier == 3])),
                                 mean_share_low=float(np.nanmean(v[tier == 1])), difference=float(hd[j]),
                                 frames_high_with_share=int((~np.isnan(v[tier == 3])).sum()),
                                 frames_low_with_share=int((~np.isnan(v[tier == 1])).sum())))
        for half_name, ix, tr_ in (('find', fidx, tier), ('validate', vidx, tiers_of(resid[c][vidx]))):
            r_ = resid[c][ix]
            tier_rows.append(dict(condition=c, half=half_name, frames=len(ix), segments=len(set(seg_f[ix])),
                                  frames_low=int((tr_ == 1).sum()), frames_middle=int((tr_ == 2).sum()),
                                  frames_high=int((tr_ == 3).sum()), residual_low_max=float(r_[tr_ == 1].max()),
                                  residual_high_min=float(r_[tr_ == 3].min()),
                                  words_in_min_frames=len(t) if half_name == 'find' else np.nan))
    pd.concat(words_rows, ignore_index=True).to_csv(os.path.join(out, 'exploratory_residual_words_find.csv'),
                                                    index=False)
    pd.DataFrame(haz_rows).to_csv(os.path.join(out, 'exploratory_residual_hazard_find.csv'), index=False)
    pd.DataFrame(tier_rows).to_csv(os.path.join(out, 'exploratory_residual_tiers.csv'), index=False)
    cand = candidates(tops)
    cw = sorted(cand, key=lambda w: (cand[w][0], -cand[w][1], w))
    crow = []
    for w in cw:
        row = dict(word=w, side=cand[w][0], conditions=cand[w][1])
        for c in conds:
            row[f'rank_{c}'] = (tops[c][0].index(w) + 1 if w in tops[c][0] else np.nan) if cand[w][0] == 'high' else \
                (tops[c][1].index(w) + 1 if w in tops[c][1] else np.nan)
        crow.append(row)
    pd.DataFrame(crow, columns=['word', 'side', 'conditions'] + [f'rank_{c}' for c in conds]) \
        .to_csv(os.path.join(out, 'exploratory_residual_candidates.csv'), index=False)
    print(f'find done: {len(cw)} candidate words ({time.time() - t0:.0f} s)', flush=True)

    # --- 4. validate
    vrow = []
    for c in conds:
        has_all = np.array([[w in fw[c][i] for w in cw] for i in range(nf)], float).reshape(nf, len(cw))
        fd_w = word_diff(has_all[fidx], find_t[c]) if cw else np.array([])
        fd_h = hazard_diff(sh[c][fidx], find_t[c])
        wd, hd, W, H = validate(resid[c][vidx], has_all[vidx], sh[c][vidx], vpos, draws_v)
        tv = tiers_of(resid[c][vidx])
        for j, w in enumerate(cw):
            lo, hi, nv = ar.ci(W[:, j])
            vrow.append(dict(kind='word', item=w, side=cand[w][0], condition=c, find_difference=float(fd_w[j]),
                             validate_high=float(has_all[vidx][tv == 3, j].mean()),
                             validate_low=float(has_all[vidx][tv == 1, j].mean()), validate_difference=float(wd[j]),
                             lo=lo, hi=hi, nboot=nv, validated=judged(fd_w[j], lo, hi)))
        for j, k in enumerate(av.HAZARD_TYPES):
            lo, hi, nv = ar.ci(H[:, j])
            v = sh[c][vidx, j]
            vrow.append(dict(kind='hazard_type', item=k, side='', condition=c, find_difference=float(fd_h[j]),
                             validate_high=float(np.nanmean(v[tv == 3])), validate_low=float(np.nanmean(v[tv == 1])),
                             validate_difference=float(hd[j]), lo=lo, hi=hi, nboot=nv,
                             validated=judged(fd_h[j], lo, hi)))
    pd.DataFrame(vrow).to_csv(os.path.join(out, 'exploratory_residual_validate.csv'), index=False)
    print(f'validate done ({time.time() - t0:.0f} s)', flush=True)

    # --- 5. clusters
    clrow, arow = [], []
    for c in conds:
        labels, centres, feats = clusters(docs[c])
        lab = labels[CLUSTER_SEEDS[0]]
        for s_ in CLUSTER_SEEDS[1:]:
            arow.append(dict(condition=c, seed=CLUSTER_SEEDS[0], other_seed=s_,
                             adjusted_rand_index=adjusted_rand_score(lab, labels[s_])))
        num = by_size(lab)
        mb = np.full((N_BOOT, N_CLUSTERS), np.nan)
        for b, (i, _) in enumerate(dd):
            li, ri = lab[i], resid[c][i]
            for k in range(N_CLUSTERS):
                m = li == k
                if m.any():
                    mb[b, k] = ri[m].mean()
        for k in sorted(num, key=num.get):
            m = lab == k
            lo, hi, nv = ar.ci(mb[:, k])
            order = sorted(range(len(feats)), key=lambda j: (-centres[k, j], feats[j]))[:N_TOP_WORDS]
            clrow.append(dict(condition=c, cluster=num[k], kmeans_label=int(k), frames=int(m.sum()),
                              top_words='; '.join(f'{w} {n_}' for w, n_ in top_words([rwds[c][i] for i in
                                                                                      np.flatnonzero(m)])),
                              centre_terms='; '.join(f'{feats[j]} {centres[k, j]:.3f}' for j in order),
                              mean_residual=float(resid[c][m].mean()), lo=lo, hi=hi, nboot=nv))
        print(f'clusters {c} done ({time.time() - t0:.0f} s)', flush=True)
    pd.DataFrame(clrow).to_csv(os.path.join(out, 'exploratory_residual_clusters.csv'), index=False)
    pd.DataFrame(arow).to_csv(os.path.join(out, 'exploratory_residual_cluster_ari.csv'), index=False)

    with open(os.path.join(out, 'exploratory_residual_params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'EXPLORATORY, decided after D\'s results were seen (user, 2026-11-05, item 2): residual_words.py run '
                f'{time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'predictions = {os.path.abspath(fitdir)} ({PRED_COL}); risk = {os.path.abspath(risk)}; quantities = '
                f'{os.path.abspath(quant)}\n')
        f.write(f'replies: {info["main"]} ({info["main_lines"]} lines) and {info["forced"]} (forced answers used for '
                f'these frames {info["forced_used"]}); fields main_hazard, reason of the final JSON, not the thinking\n')
        f.write(f'versions: python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, scipy '
                f'{scipy.__version__}, scikit-learn {sklearn.__version__}\n')
        f.write(f'fit frames {nf} in {len(segs_fit)} segments; random reference for the between share '
                f'({len(segs_fit)} - 1) / ({nf} - 1) = {ref:.6f}\n')
        f.write(f'halves: numpy default_rng({SPLIT_SEED}).permutation of the {len(segs_fit)} sorted segments; find '
                f'{len(find_s)} segments, {len(fidx)} frames (sha256 of the sorted names joined by newlines {hs}); '
                f'validate {len(val_s)} segments, {len(vidx)} frames (exploratory_residual_halves.csv)\n')
        f.write(f'words: reason_words.py (lower case, {rw.WORD.pattern}, ENGLISH_STOP_WORDS out, no stemming), words in '
                f'at least {rw.MIN_FRAMES} frames of the find half, top {rw.TOP} per side; candidates: in a side\'s top '
                f'{rw.TOP} of at least {MIN_COND} of the {len(conds)} conditions\n')
        f.write(f'validate resamples: {N_BOOT}, numpy default_rng({BOOT_SEED}).choice of the {len(val_s)} validate '
                f'segments, sha256 of the drawn indices {hv.hexdigest()}; tiers recomputed per resample\n')
        f.write(f'between share and cluster intervals: the risk analysis\'s {N_BOOT} resamples (seed {BOOT_SEED}, '
                f'sha256 {h.hexdigest()}) restricted to the fit frames, residuals and clusters fixed\n')
        f.write(f'clusters: TfidfVectorizer(analyzer = the words with repeats), KMeans({N_CLUSTERS}, random_state '
                f'{CLUSTER_SEEDS[0]}, n_init {N_INIT}, the rest scikit-learn\'s defaults); adjusted Rand index against '
                f'random_state {list(CLUSTER_SEEDS[1:])}\n')
        for c_ in checks:
            f.write(f'check: {c_}\n')
        f.write(f'checks not met: {failed if failed else "none"}\n')
    print(f'done in {time.time() - t0:.0f} s; checks not met: {failed if failed else "none"}')
    for c_ in checks:
        print('  ' + c_)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
