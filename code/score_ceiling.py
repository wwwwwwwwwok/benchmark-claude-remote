"""EXPLORATORY, added after the trees were seen (user, 2026-11-04; CLAUDE.md "找模型自己的规则", the entry of 2026-11-04
after the trees were seen): the most of Qwen3.6-27B's risk-score ranks that any function of the frame could reproduce
("the ceiling"), from how well its repeated answers agree.

Per condition of qwen3.6-27b: the scores of the 5 repeats behind score_mean (<risk>/vlm_requests_tidy.csv risk, forced
answers included, as analyze_vlm_ssm.py; checked: their mean per frame is vlm_frame_scores.csv score_mean, and every
frame has 5 requests), on the frames whose 5 repeats all have a score (the others counted and left out). Frames as the
rule analysis (ssm_target_frames.csv, sorted by segment and frame). Each repeat's scores are ranked among these frames
(ties: the mean rank; scipy rankdata); r = the mean of the 10 correlations between the repeats' ranks (Spearman), also
their minimum and maximum; ceiling = 5 r / (1 + 4 r) (Spearman-Brown: the reliability of the mean of 5). Interval: the
risk analysis's resamples of whole segments (default_rng(--seed), checked before they are used as analyze_rules.py), in
each the ranks and r taken anew on the drawn frames that have 5 scores (a segment drawn twice counted twice); 2.5 / 97.5 %
percentiles, none when fewer than 90 % are defined.
Beside it, copied from the existing tables (nothing rerun): the out-of-sample R2 with interval of the depth-3 A and B
trees (<trees>/exploratory_tree_oos.csv), of K, K+S+N10, K+S+N20, K+S+N30, K+S+N50 and K+S+N20+H
(<rules>/exploratory_oos_r2.csv); and the ICC of the risk analysis (<risk>/condition_summary.csv, analyze_vlm_ssm.py
repeat_stability: ICC(1), one-way random effects, on the raw scores of the frames with all k answers: icc_single for one
answer, icc_mean_of_k for the mean of the k answers).
Outputs (out_dir): exploratory_ceiling.csv, exploratory_ceiling_pairs.csv (the 10 correlations), exploratory_ceiling_vs_r2.csv,
exploratory_ceiling_params.txt; exit code 1 when a check is not met.

usage: python -I score_ceiling.py <risk analysis subfolder> <quantities analysis subfolder> <rule analysis subfolder>
       <tree subfolder (rule_tree.py)> <out_dir> [--n_boot 2000] [--seed 20261021]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import hashlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_rules as ar  # noqa: E402
import analyze_vlm_ssm as av  # noqa: E402

MODEL = 'qwen3.6-27b'
K_REP = 5
MEAN_TOL = 1e-12                   # mean of the repeats against score_mean
BESIDE_TREES = ['A', 'B']
BESIDE_LINEAR = ['K', 'K+S+N10', 'K+S+N20', 'K+S+N30', 'K+S+N50', 'K+S+N20+H']


def pair_corr(R):
    """R: frames x repeats (ranks). The correlations between every two columns (10 for 5), their mean, min, max."""
    v = np.corrcoef(R, rowvar=False)[np.triu_indices(R.shape[1], 1)]
    return v, float(v.mean()), float(v.min()), float(v.max())


def spearman_brown(r, k=K_REP):
    return k * r / (1 + (k - 1) * r)


def main():
    ap = argparse.ArgumentParser()
    for name in ('risk', 'quantities', 'rules', 'trees', 'out_dir'):
        ap.add_argument(name)
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261021)
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out_dir, 'exploratory_ceiling_params.txt')):
        sys.exit(f'{a.out_dir} has exploratory_ceiling_params.txt already: give a new folder')
    t0 = time.time()
    failed, checks = [], []

    # --- frames and resamples, as analyze_rules.py
    st = pd.read_csv(os.path.join(a.risk, 'ssm_target_frames.csv'))
    frames = st[['segment', 'frame_idx']].sort_values(['segment', 'frame_idx']).reset_index(drop=True)
    n = len(frames)
    mi = pd.MultiIndex.from_frame(frames)
    seg_names = frames['segment'].to_numpy()
    seg_list = np.array(sorted(set(seg_names)))
    nseg = len(seg_list)
    seg_frames = {s: np.flatnonzero(seg_names == s) for s in seg_list}
    n_boot_r, seed_r, frames_r, segs_r = ar.read_params_risk(os.path.join(a.risk, 'params.txt'))
    for what, mine, theirs in (('resamples', a.n_boot, n_boot_r), ('seed', a.seed, seed_r), ('frames', n, frames_r)):
        if mine != theirs:
            failed.append(f'{what}: {mine} here, {theirs} in the risk analysis\'s params.txt')
    if list(seg_list) != segs_r:
        failed.append(f'segment list: {nseg} here, {len(segs_r)} in the risk analysis\'s params.txt, not the same')
    rng = np.random.default_rng(a.seed)
    draws = [rng.choice(seg_list, nseg) for _ in range(a.n_boot)]
    h = hashlib.sha256()
    for d in draws:
        h.update(np.searchsorted(seg_list, d).astype(np.int32).tobytes())
    nb_q, seed_q, hash_q = ar.read_hash_quantities(os.path.join(a.quantities, 'params.txt'))
    if (nb_q, seed_q, hash_q) != (a.n_boot, a.seed, h.hexdigest()):
        failed.append(f'drawn segments: sha256 {h.hexdigest()} ({a.n_boot}, seed {a.seed}) here, {hash_q} ({nb_q}, '
                      f'seed {seed_q}) in the quantities analysis\'s params.txt')
    if failed:
        sys.exit('stopped before the resamples are used: ' + '; '.join(failed))
    os.makedirs(a.out_dir, exist_ok=True)

    # --- the repeats behind score_mean
    rq = pd.read_csv(os.path.join(a.risk, 'vlm_requests_tidy.csv'), low_memory=False,
                     usecols=['model', 'condition', 'segment', 'frame_idx', 'repeat', 'forced', 'risk'])
    rq = rq[rq.model == MODEL]
    fsc = pd.read_csv(os.path.join(a.risk, 'vlm_frame_scores.csv'), float_precision='round_trip')
    conds = [c for c in av.CONDITIONS if (rq.condition == c).any()]
    if conds != av.CONDITIONS:
        sys.exit(f'{MODEL}: conditions {conds} in vlm_requests_tidy.csv, not {av.CONDITIONS}')
    rows, pair_rows, mean_dev = [], [], 0.0
    for c in conds:
        g = rq[rq.condition == c]
        if g.duplicated(['segment', 'frame_idx', 'repeat']).any():
            sys.exit(f'{MODEL} {c}: a repeat twice for some frame')
        reps = sorted(g['repeat'].unique())
        per_frame = g.groupby(['segment', 'frame_idx']).size().reindex(mi)
        if len(reps) != K_REP or per_frame.isna().any() or (per_frame != K_REP).any() or len(g) != n * K_REP:
            sys.exit(f'{MODEL} {c}: not {K_REP} requests (repeats {reps}) for each of the {n} frames')
        w = g.pivot(index=['segment', 'frame_idx'], columns='repeat', values='risk').reindex(index=mi, columns=reps)
        sm = fsc[(fsc.model == MODEL) & (fsc.condition == c)].set_index(['segment', 'frame_idx'])['score_mean'] \
            .reindex(mi).to_numpy(float)
        mh = w.mean(axis=1).to_numpy(float)
        same_nan = bool((np.isnan(mh) == np.isnan(sm)).all())
        dev = float(np.nanmax(np.abs(mh - sm))) if (~np.isnan(mh)).any() else 0.0
        mean_dev = max(mean_dev, dev)
        if not (same_nan and dev <= MEAN_TOL):
            failed.append(f'{c}: the mean of the repeats is not score_mean (largest difference {dev:.3g}, frames '
                          f'without a score the same: {same_nan})')
        W = w.to_numpy(float)
        full = ~np.isnan(W).any(axis=1)
        nk = (~np.isnan(W)).sum(axis=1)
        R = np.column_stack([ar.rank01(W[full, k]) for k in range(K_REP)])
        v, r, rmin, rmax = pair_corr(R)
        for (i, j), x in zip(zip(*np.triu_indices(K_REP, 1)), v):
            pair_rows.append(dict(model=MODEL, condition=c, repeat_a=reps[i], repeat_b=reps[j], frames=int(full.sum()),
                                  spearman=float(x)))
        cb = np.full(a.n_boot, np.nan)
        for b, draw in enumerate(draws):
            idx = np.concatenate([seg_frames[s] for s in draw])
            idx = idx[full[idx]]
            cb[b] = spearman_brown(pair_corr(np.column_stack([ar.rank01(W[idx, k]) for k in range(K_REP)]))[1])
        lo, hi, nv = ar.ci(cb)
        rows.append(dict(model=MODEL, condition=c, frames=n, frames_5_scores=int(full.sum()),
                         frames_fewer_than_5=int((~full).sum()),
                         frames_with_4_3_2_1_0_scores='/'.join(str(int((nk == k).sum())) for k in (4, 3, 2, 1, 0)),
                         forced_answers_used=int(g.forced.astype(bool).sum()), r_mean=r, r_min=rmin, r_max=rmax,
                         ceiling=spearman_brown(r), ceiling_lo=lo, ceiling_hi=hi, nboot=nv))
        print(f'{c}: {int(full.sum())} frames with 5 scores, r {r:.4f}, ceiling {spearman_brown(r):.4f} [{lo:.4f}, '
              f'{hi:.4f}] ({time.time() - t0:.0f} s)', flush=True)
    checks.append(f'the mean of the 5 repeats per frame (pandas, NaN skipped) against vlm_frame_scores.csv score_mean: '
                  f'largest difference {mean_dev:.3g} (tolerance {MEAN_TOL:g}); every frame {K_REP} requests')

    # --- beside: existing tables, copied
    to = pd.read_csv(os.path.join(a.trees, 'exploratory_tree_oos.csv'), float_precision='round_trip')
    to = to[(to.target == 'risk') & (to.model == MODEL) & (to.max_depth == 3)]
    oo = pd.read_csv(os.path.join(a.rules, 'exploratory_oos_r2.csv'), float_precision='round_trip')
    oo = oo[(oo.target == 'risk') & (oo.model == MODEL)]
    cs_ = pd.read_csv(os.path.join(a.risk, 'condition_summary.csv'), float_precision='round_trip')
    cs_ = cs_[cs_.model == MODEL].set_index('condition')
    vs = []
    for r_ in rows:
        c = r_['condition']
        vs.append(dict(condition=c, measure='ceiling 5r/(1+4r) (this program)', value=r_['ceiling'],
                       lo=r_['ceiling_lo'], hi=r_['ceiling_hi'], frames=r_['frames_5_scores'], source='this program'))
        for cs in BESIDE_TREES:
            t = to[(to.condition == c) & (to.cues == cs)]
            if len(t) != 1:
                failed.append(f'{c} tree {cs}: {len(t)} rows in exploratory_tree_oos.csv')
                continue
            t = t.iloc[0]
            vs.append(dict(condition=c, measure=f'tree {cs}, depth 3, out-of-sample R2', value=t.r2_oos,
                           lo=t.r2_oos_lo, hi=t.r2_oos_hi, frames=int(t.frames), source='exploratory_tree_oos.csv'))
        for f_ in BESIDE_LINEAR:
            t = oo[(oo.condition == c) & (oo.fit == f_)]
            if len(t) != 1:
                failed.append(f'{c} {f_}: {len(t)} rows in exploratory_oos_r2.csv')
                continue
            t = t.iloc[0]
            vs.append(dict(condition=c, measure=f'{f_} (linear), out-of-sample R2', value=t.r2_oos, lo=t.r2_oos_lo,
                           hi=t.r2_oos_hi, frames=int(t.frames), source='exploratory_oos_r2.csv'))
        if c in cs_.index:
            q = cs_.loc[c]
            for col, what in (('icc_single', 'ICC(1) one answer'), ('icc_mean_of_k', 'ICC(1,k) mean of k answers')):
                vs.append(dict(condition=c, measure=f'{what} (one-way random effects, raw scores, k = '
                                                    f'{int(q.k_repeats)})', value=q[col], lo=np.nan, hi=np.nan,
                               frames=int(q.frames_all_k), source='condition_summary.csv (risk analysis)'))
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, 'exploratory_ceiling.csv'), index=False)
    pd.DataFrame(pair_rows).to_csv(os.path.join(a.out_dir, 'exploratory_ceiling_pairs.csv'), index=False)
    pd.DataFrame(vs).to_csv(os.path.join(a.out_dir, 'exploratory_ceiling_vs_r2.csv'), index=False)
    with open(os.path.join(a.out_dir, 'exploratory_ceiling_params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'EXPLORATORY, added after the trees were seen (user, 2026-11-04): score_ceiling.py run '
                f'{time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        for k_, v_ in (('risk analysis', a.risk), ('quantities analysis', a.quantities), ('rule analysis', a.rules),
                       ('trees (rule_tree.py)', a.trees)):
            f.write(f'{k_} = {os.path.abspath(v_)}\n')
        f.write(f'versions: python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, scipy '
                f'{scipy.__version__}\n')
        f.write(f'frames {n} in {nseg} segments; model {MODEL}, conditions {conds}\n')
        f.write(f'scores: vlm_requests_tidy.csv risk of the {K_REP} repeats (forced answers included, as score_mean); '
                f'only the frames with all {K_REP} scores; each repeat ranked among them (scipy rankdata, ties the mean '
                f'rank); r = mean of the {K_REP * (K_REP - 1) // 2} correlations of the ranks (Spearman); ceiling = '
                f'{K_REP} r / (1 + {K_REP - 1} r) (Spearman-Brown)\n')
        f.write(f'interval: the {a.n_boot} resamples (seed {a.seed}, sha256 of the drawn segment indices '
                f'{h.hexdigest()}), the ranks and r taken anew on the drawn frames with {K_REP} scores (a segment drawn '
                'twice counted twice); 2.5 / 97.5 % percentiles, none when fewer than 90 % are defined\n')
        f.write(f'beside (copied, not rerun): trees {BESIDE_TREES} at depth 3 (exploratory_tree_oos.csv), linear '
                f'{BESIDE_LINEAR} (exploratory_oos_r2.csv), out of sample; the ICC of condition_summary.csv (ICC(1), '
                'one-way random effects, raw scores, frames with all k answers; icc_single one answer, icc_mean_of_k '
                'the mean of k)\n')
        for c_ in checks:
            f.write(f'check: {c_}\n')
        f.write(f'checks not met: {failed if failed else "none"}\n')
    print(f'done in {time.time() - t0:.0f} s; checks not met: {failed if failed else "none"}; outputs in '
          f'{os.path.abspath(a.out_dir)}')
    for c_ in checks:
        print('  ' + c_)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
