"""EXPLORATORY (user, 2026-11-06, the plan of round E; E's cues were chosen after these residuals were seen): last
time's 12 clusters of Qwen3.6-27B's reasons (code/residual_words.py: the same documents and settings, the clustering
redone; checked: with D's residuals every cluster's frames, mean residual and interval equal its
exploratory_residual_clusters.csv), now with the residual of E's additive model (rank score - E's leave-one-segment-out
prediction, new_cues_fit_e.py): per condition and cluster the mean residual with its interval (the risk analysis's
2,000 resamples restricted to the fit frames, clusters fixed), D's beside, and E - D with its interval from the same
resamples. Also both residuals' between-segment shares. Only counts and words; no reply's text.
usage: python -I residual_clusters_e.py <D predictions folder (new_cues_fit.py --predictions)> <new_cues_fit_e.py
       out_dir> <risk subfolder> <quantities subfolder> <model dir (vlm_scores/qwen3.6-27b)> <residual_words.py out_dir>
       <out_dir>
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import hashlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_rules as ar  # noqa: E402
import analyze_vlm_ssm as av  # noqa: E402
import reason_words as rw  # noqa: E402
import residual_words as res  # noqa: E402

MODEL = rw.MODEL
TOL = 1e-12


def main():
    dpred, efit, risk, quant, model_dir, rwdir, out = sys.argv[1:8]
    if os.path.exists(os.path.join(out, 'exploratory_e_residual_params.txt')):
        sys.exit(f'{out} has exploratory_e_residual_params.txt already: give a new folder')
    t0 = time.time()
    failed, checks = [], []
    conds = av.CONDITIONS
    pd_ = pd.read_csv(os.path.join(dpred, 'exploratory_d_predictions.csv'), float_precision='round_trip')
    pe = pd.read_csv(os.path.join(efit, 'exploratory_e_predictions.csv'), float_precision='round_trip')
    pd_, pe = pd_[pd_.model == MODEL].reset_index(drop=True), pe[pe.model == MODEL].reset_index(drop=True)
    same = pd_[['condition', 'segment', 'frame_idx', 'y_rank']].equals(pe[['condition', 'segment', 'frame_idx', 'y_rank']])
    checks.append(f'D and E predictions on the same frames with the same rank scores: {same}')
    if not same:
        sys.exit('the D and E predictions are not on the same frames')
    fr = pd_[pd_.condition == conds[0]][['segment', 'frame_idx']].reset_index(drop=True)
    nf = len(fr)
    rd, re_ = {}, {}
    for c in conds:
        q, qe = pd_[pd_.condition == c], pe[pe.condition == c]
        rd[c] = q.y_rank.to_numpy(float) - q.pred_boost_add_D.to_numpy(float)
        re_[c] = qe.y_rank.to_numpy(float) - qe.pred_boost_add_E.to_numpy(float)

    # resamples, restricted to the fit frames (as residual_words.py)
    st = pd.read_csv(os.path.join(risk, 'ssm_target_frames.csv'))[['segment', 'frame_idx']]
    seg_list = np.array(sorted(set(st.segment)))
    n_boot_r, seed_r, frames_r, segs_r = ar.read_params_risk(os.path.join(risk, 'params.txt'))
    if (n_boot_r, seed_r, frames_r) != (res.N_BOOT, res.BOOT_SEED, len(st)) or list(seg_list) != segs_r:
        sys.exit('stopped before the resamples are used: resamples, seed, frames or segments not the risk analysis\'s')
    rng = np.random.default_rng(res.BOOT_SEED)
    draws = [rng.choice(seg_list, len(seg_list)) for _ in range(res.N_BOOT)]
    h = hashlib.sha256()
    for d in draws:
        h.update(np.searchsorted(seg_list, d).astype(np.int32).tobytes())
    nb_q, seed_q, hash_q = ar.read_hash_quantities(os.path.join(quant, 'params.txt'))
    if (nb_q, seed_q, hash_q) != (res.N_BOOT, res.BOOT_SEED, h.hexdigest()):
        sys.exit('stopped before the resamples are used: the drawn segments are not the quantities analysis\'s')
    seg_f = fr.segment.to_numpy()
    segs_fit = np.array(sorted(set(seg_f)))
    seg_frames = {s_: np.flatnonzero(seg_f == s_) for s_ in segs_fit}
    g_fit = np.searchsorted(segs_fit, seg_f)
    dd = []
    for d in draws:
        idx = [seg_frames[x] for x in d if x in seg_frames]
        dd.append((np.concatenate(idx), np.concatenate([np.full(len(v), k) for k, v in enumerate(idx)])))

    # between-segment shares
    brow = []
    for c in conds:
        for what, v in (('D residual', rd[c]), ('E residual', re_[c])):
            lo, hi, nv = ar.ci(np.array([ar.between_share(v[i], gg) for i, gg in dd]))
            brow.append(dict(model=MODEL, condition=c, of=what, frames=nf, between_share=ar.between_share(v, g_fit),
                             lo=lo, hi=hi, nboot=nv, random_reference=(len(segs_fit) - 1) / (nf - 1)))
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(brow).to_csv(os.path.join(out, 'exploratory_e_residual_between.csv'), index=False)

    # the documents and the clusters, as residual_words.py
    rq = pd.read_csv(os.path.join(risk, 'vlm_requests_tidy.csv'), low_memory=False,
                     usecols=['model', 'condition', 'segment', 'frame_idx', 'repeat', 'risk'])
    rq = rq[rq.model == MODEL].merge(fr, on=['segment', 'frame_idx'], how='inner').reset_index(drop=True)
    rep, info = rw.read_replies(model_dir, set(zip(rq.segment, rq.frame_idx.astype(int), rq.condition,
                                                   rq['repeat'].astype(int))))
    print(f'read {info["main_lines"]} lines; forced answers used {info["forced_used"]} ({time.time() - t0:.0f} s)',
          flush=True)
    pos = {k: i for i, k in enumerate(zip(fr.segment, fr.frame_idx.astype(int)))}
    old = pd.read_csv(os.path.join(rwdir, 'exploratory_residual_clusters.csv'), float_precision='round_trip')
    rows = []
    dev = 0.0
    for c in conds:
        q = rq[rq.condition == c]
        reps = [[] for _ in range(nf)]
        for sg, fi, rp in zip(q.segment, q.frame_idx.astype(int), q['repeat'].astype(int)):
            reps[pos[(sg, fi)]].append(rep.get((sg, fi, c, rp), (np.nan, None, None)))
        docs = [' '.join(str(rs) for r_, _, rs in v if not np.isnan(r_) and rs is not None) for v in reps]
        labels, _, _ = res.clusters(docs, seeds=(res.CLUSTER_SEEDS[0],))
        lab = labels[res.CLUSTER_SEEDS[0]]
        num = res.by_size(lab)
        mb = np.full((res.N_BOOT, res.N_CLUSTERS, 2), np.nan)
        for b, (i, _) in enumerate(dd):
            li = lab[i]
            for k in range(res.N_CLUSTERS):
                m = li == k
                if m.any():
                    mb[b, k] = rd[c][i][m].mean(), re_[c][i][m].mean()
        o = old[old.condition == c].set_index('cluster')
        for k in sorted(num, key=num.get):
            m = lab == k
            lo_d, hi_d, _ = ar.ci(mb[:, k, 0])
            lo_e, hi_e, _ = ar.ci(mb[:, k, 1])
            lo_x, hi_x, nv = ar.ci(mb[:, k, 1] - mb[:, k, 0])
            md, me = float(rd[c][m].mean()), float(re_[c][m].mean())
            ref = o.loc[num[k]]
            dev = max(dev, abs(int(ref.frames) - int(m.sum())), abs(ref.mean_residual - md), abs(ref.lo - lo_d),
                      abs(ref.hi - hi_d))
            rows.append(dict(condition=c, cluster=num[k], frames=int(m.sum()), top_words=ref.top_words,
                             mean_residual_D=md, lo_D=lo_d, hi_D=hi_d, mean_residual_E=me, lo_E=lo_e, hi_E=hi_e,
                             E_minus_D=me - md, lo_diff=lo_x, hi_diff=hi_x, nboot=nv))
        print(f'clusters {c} done ({time.time() - t0:.0f} s)', flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, 'exploratory_e_residual_clusters.csv'), index=False)
    checks.append(f'the clusters redone with D\'s residuals against {rwdir}/exploratory_residual_clusters.csv '
                  f'(frames, mean residual, interval): largest difference {dev:.3g} (tolerance {TOL:g})')
    if not dev <= TOL:
        failed.append('clusters not reproduced')
    with open(os.path.join(out, 'exploratory_e_residual_params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'EXPLORATORY (user, 2026-11-06; E\'s cues were chosen after these residuals were seen): '
                f'residual_clusters_e.py run {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'D predictions = {os.path.abspath(dpred)}; E fit = {os.path.abspath(efit)}; residual_words = '
                f'{os.path.abspath(rwdir)}; risk = {os.path.abspath(risk)}\n')
        f.write(f'replies: {info["main"]} ({info["main_lines"]} lines), forced answers used {info["forced_used"]}; '
                f'clusters as residual_words.py: TF-IDF of the reasons, KMeans({res.N_CLUSTERS}, random_state '
                f'{res.CLUSTER_SEEDS[0]}, n_init {res.N_INIT}); intervals: {res.N_BOOT} resamples, seed '
                f'{res.BOOT_SEED}, restricted to the {nf} fit frames, clusters fixed\n')
        for c_ in checks:
            f.write(f'check: {c_}\n')
        f.write(f'checks not met: {failed if failed else "none"}\n')
    print(f'done in {time.time() - t0:.0f} s; checks not met: {failed if failed else "none"}')
    for c_ in checks:
        print('  ' + c_)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
