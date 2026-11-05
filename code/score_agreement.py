"""EXPLORATORY, added after the trees, the ceiling and the resampled trees were seen (user, 2026-11-05; CLAUDE.md
"找模型自己的规则", the entry of 2026-11-05): do the 5 conditions (ways of giving the pictures) rank the frames alike?

For qwen3.6-27b: each condition's score_mean (<risk>/vlm_frame_scores.csv) ranked among the frames (ssm_target_frames.csv,
sorted by segment and frame; ties: the mean rank, scipy rankdata); between every two conditions the correlation of the
ranks (Spearman; 10 pairs); divided by the square root of the two conditions' ceilings (<ceiling> = exploratory_ceiling.csv
of score_ceiling.py), "the correlation with each condition's own random part taken out": near 1, the frames are ranked
the same way whatever the condition. Interval: the risk analysis's resamples of whole segments (checked as
analyze_rules.py), in each the ranks, the correlations and the two ceilings taken anew on the drawn frames (a segment
drawn twice counted twice; the ceilings as score_ceiling.py: the frames with all 5 scores, each repeat ranked, the mean
of the 10 correlations r, 5 r / (1 + 4 r)); 2.5 / 97.5 % percentiles, none when fewer than 90 % are defined; values
above 1 are kept as they are. Checked: the ceilings recomputed here equal <ceiling> within 1e-12; every frame has a
score_mean.
Outputs (out_dir): exploratory_agreement.csv, exploratory_agreement_params.txt; exit code 1 when a check is not met.
usage: python -I score_agreement.py <risk analysis subfolder> <quantities analysis subfolder> <exploratory_ceiling.csv>
       <out_dir> [--n_boot 2000] [--seed 20261021]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_rules as ar  # noqa: E402
import analyze_vlm_ssm as av  # noqa: E402
from score_ceiling import MODEL, K_REP, pair_corr, spearman_brown  # noqa: E402

CEIL_TOL = 1e-12


def main():
    ap = argparse.ArgumentParser()
    for name in ('risk', 'quantities', 'ceiling', 'out_dir'):
        ap.add_argument(name)
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261021)
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out_dir, 'exploratory_agreement_params.txt')):
        sys.exit(f'{a.out_dir} has exploratory_agreement_params.txt already: give a new folder')
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

    # --- score_mean and the repeats of every condition
    fsc = pd.read_csv(os.path.join(a.risk, 'vlm_frame_scores.csv'), float_precision='round_trip')
    rq = pd.read_csv(os.path.join(a.risk, 'vlm_requests_tidy.csv'), low_memory=False,
                     usecols=['model', 'condition', 'segment', 'frame_idx', 'repeat', 'risk'])
    rq = rq[rq.model == MODEL]
    conds = [c for c in av.CONDITIONS if ((fsc.model == MODEL) & (fsc.condition == c)).any()]
    if conds != av.CONDITIONS:
        sys.exit(f'{MODEL}: conditions {conds} in vlm_frame_scores.csv, not {av.CONDITIONS}')
    sm, W, full = {}, {}, {}
    for c in conds:
        v = fsc[(fsc.model == MODEL) & (fsc.condition == c)].set_index(['segment', 'frame_idx'])['score_mean'] \
            .reindex(mi).to_numpy(float)
        if np.isnan(v).any():
            sys.exit(f'{MODEL} {c}: {int(np.isnan(v).sum())} frames without score_mean')
        sm[c] = v
        g = rq[rq.condition == c]
        reps = sorted(g['repeat'].unique())
        if len(reps) != K_REP or len(g) != n * K_REP or g.duplicated(['segment', 'frame_idx', 'repeat']).any():
            sys.exit(f'{MODEL} {c}: not {K_REP} requests for each of the {n} frames')
        W[c] = g.pivot(index=['segment', 'frame_idx'], columns='repeat', values='risk').reindex(index=mi, columns=reps) \
            .to_numpy(float)
        full[c] = ~np.isnan(W[c]).any(axis=1)

    def ceil_of(c, idx):
        i = idx[full[c][idx]]
        return spearman_brown(pair_corr(np.column_stack([ar.rank01(W[c][i, k]) for k in range(K_REP)]))[1])

    ce = pd.read_csv(a.ceiling, float_precision='round_trip')
    ce = ce[ce.model == MODEL].set_index('condition')
    every = np.arange(n)
    ceil = {c: ceil_of(c, every) for c in conds}
    dev = max(abs(ceil[c] - ce.loc[c, 'ceiling']) for c in conds)
    checks.append(f'ceilings recomputed here against {a.ceiling}: largest difference {dev:.3g} (tolerance {CEIL_TOL:g})')
    if not dev <= CEIL_TOL:
        failed.append('ceilings')
    pairs = list(itertools.combinations(conds, 2))

    def stats(idx):
        R = {c: ar.rank01(sm[c][idx]) for c in conds}
        cl = {c: ceil_of(c, idx) for c in conds}
        out = []
        for c1, c2 in pairs:
            r = float(np.corrcoef(R[c1], R[c2])[0, 1])
            out.append((r, r / np.sqrt(cl[c1] * cl[c2])))
        return out

    point = stats(every)
    boot = np.array([[x for pr in stats(np.concatenate([seg_frames[s] for s in d])) for x in pr] for d in draws])
    rows = []
    for j, (c1, c2) in enumerate(pairs):
        lo, hi, nv = ar.ci(boot[:, 2 * j])
        lo2, hi2, nv2 = ar.ci(boot[:, 2 * j + 1])
        rows.append(dict(model=MODEL, condition_1=c1, condition_2=c2, frames=n, spearman=point[j][0],
                         spearman_lo=lo, spearman_hi=hi, ceiling_1=ce.loc[c1, 'ceiling'], ceiling_2=ce.loc[c2, 'ceiling'],
                         disattenuated=point[j][1], disattenuated_lo=lo2, disattenuated_hi=hi2, nboot=min(nv, nv2)))
        print(f'{c1} x {c2}: Spearman {point[j][0]:.3f} [{lo:.3f}, {hi:.3f}], / sqrt(ceilings) {point[j][1]:.3f} '
              f'[{lo2:.3f}, {hi2:.3f}]', flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, 'exploratory_agreement.csv'), index=False)
    with open(os.path.join(a.out_dir, 'exploratory_agreement_params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'EXPLORATORY, added after the trees, the ceiling and the resampled trees were seen (user, 2026-11-05): '
                f'score_agreement.py run {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        for k_, v_ in (('risk analysis', a.risk), ('quantities analysis', a.quantities), ('ceiling', a.ceiling)):
            f.write(f'{k_} = {os.path.abspath(v_)}\n')
        f.write(f'versions: python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, scipy '
                f'{scipy.__version__}\n')
        f.write(f'frames {n} in {nseg} segments; model {MODEL}, conditions {conds}; 10 pairs\n')
        f.write('Spearman: score_mean of each condition ranked among the frames (scipy rankdata, ties the mean rank), '
                'Pearson correlation of the ranks; disattenuated = Spearman / sqrt(ceiling_1 x ceiling_2), ceilings of '
                'score_ceiling.py (frames with all 5 scores); values above 1 kept\n')
        f.write(f'interval: the {a.n_boot} resamples (seed {a.seed}, sha256 of the drawn segment indices '
                f'{h.hexdigest()}), ranks, correlations and ceilings taken anew on the drawn frames; 2.5 / 97.5 % '
                'percentiles, none when fewer than 90 % are defined\n')
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
