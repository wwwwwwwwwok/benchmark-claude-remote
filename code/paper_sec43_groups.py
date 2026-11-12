"""Paper section 4.3, tau-b within each group (user, 2026-11-12; EXPLORATORY): Qwen3.6-27B with thinking on, sep_camera,
the main result's 1,084 frames. CPU only; no image is opened. Tables <out_dir>/exploratory_sec43_groups_*.csv, the
report on stdout.

Groups: time of day (Day, Dawn/Dusk, Night), weather (dry, wet, rain), city (location_sf, location_phx, location_other);
labels/scene_labels.csv. In each group only the pairs of frames inside the group: tau-b of score_mean with each of the
nine SSMs (as the main analysis: larger = riskier, every frame has a value; analyze_vlm_ssm.all_stats with one group,
tau_b_all), with its segments, frames and frame pairs. Intervals: the main analysis's 2,000 resamples of whole segments
(seed 20261021, drawn as the risk analysis draws them, every drawn copy its own segment), in each resample the drawn
frames that fall in the group; 2.5 / 97.5 % percentiles (numpy, linear), none when fewer than 90 % of the resamples are
defined (a resample with fewer than 3 frames of the group is undefined). Differences between groups, per SSM, over the
same resamples: Dawn/Dusk - Day, Night - Day, rain - dry, wet - dry, location_phx - location_sf,
location_other - location_sf; with each interval's width (hi - lo).
Checks: the drawn segments' sha256 = the quantities analysis's; every group's tau-b = scipy kendalltau(variant='b') on
the group's frames and = a pair-by-pair count without scipy (S = concordant - discordant pairs, tx, ty = pairs untied in
the score, in the SSM), within 1e-12; per grouping the counts summed over its groups (sum S / sqrt(sum tx * sum ty))
= the grouped tau-b already computed: "same time of day" and "same weather" in after_qwen3.5-4b_weather/main_1084/
agreement.csv, "same location" in <out_dir>/exploratory_sec43_more_tau_b.csv (within 1e-12, and whether the text is the
same).
Reading (fixed before the run): a difference whose interval contains 0 = no difference from the reference group seen;
an interval leaving out 0 = its direction and size reported as they are. The widths are reported because wet has 7
segments and Dawn/Dusk and Night 20 each.
usage: python -I paper_sec43_groups.py <waymo_v143 folder> <scene_labels.csv> <out_dir>
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import csv  # noqa: E402
import hashlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_rules as ar  # noqa: E402
import analyze_vlm_ssm as av  # noqa: E402

MODEL, COND = 'qwen3.6-27b', 'sep_camera'
N_BOOT, SEED = 2000, 20261021
HASH = '48bc1f413f9d5b99da57ff4edde7bbb4eb378c5a9343c44ccad4e0165466604c'
GROUPS = {'time_of_day': ['Day', 'Dawn/Dusk', 'Night'], 'weather': ['dry', 'wet', 'rain'],
          'location': ['location_sf', 'location_phx', 'location_other']}
DIFFS = [('time_of_day', 'Dawn/Dusk', 'Day'), ('time_of_day', 'Night', 'Day'), ('weather', 'rain', 'dry'),
         ('weather', 'wet', 'dry'), ('location', 'location_phx', 'location_sf'),
         ('location', 'location_other', 'location_sf')]
TOL = 1e-12


def txt(x):
    return '' if x is None or (isinstance(x, float) and np.isnan(x)) else repr(float(x))


def r(x):
    return 'nan' if x is None or (isinstance(x, float) and np.isnan(x)) else repr(float(x))


def side(lo, hi):
    if lo is None or np.isnan(lo):
        return 'no interval'
    return 'above 0' if lo > 0 else ('below 0' if hi < 0 else 'contains 0')


def csv_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        for d in rd:
            yield rd.line_num, d


def counts(x, y):
    """S = sum of sign(dx) sign(dy) over all pairs, and the pairs untied in x and in y (no scipy)."""
    iu = np.triu_indices(len(x), 1)
    dx = np.sign(x[:, None] - x[None, :])[iu]
    dy = np.sign(y[:, None] - y[None, :])[iu]
    return float((dx * dy).sum()), float((dx != 0).sum()), float((dy != 0).sum())


def main():
    L, SL, O = sys.argv[1:4]
    A = os.path.join(L, 'analysis')
    RISK = os.path.join(A, 'after_qwen3.5-4b', 'main_1084')
    WEA = os.path.join(A, 'after_qwen3.5-4b_weather', 'main_1084')
    QUANT = os.path.join(A, 'quantities_after_27b_think', 'main_1084')
    FSSM = os.path.join(L, 'results_lead_final_excl', 'frame_ssm_validation_training.csv')
    if os.path.exists(os.path.join(O, 'exploratory_sec43_groups_tau_b.csv')):
        sys.exit(f'{O}/exploratory_sec43_groups_tau_b.csv exists already: give a new folder')
    t0 = time.time()
    failed = []

    def check(ok, what):
        print(f'  check {"ok" if ok else "NOT MET"}: {what}', flush=True)
        if not ok:
            failed.append(what)

    print(f'paper section 4.3, tau-b within each group (user, 2026-11-12); EXPLORATORY; {MODEL} x {COND}, main_1084; run '
          f'{time.strftime("%Y-%m-%d %H:%M:%S")}')
    frames = pd.read_csv(os.path.join(RISK, 'ssm_target_frames.csv'))[['segment', 'frame_idx']] \
        .sort_values(['segment', 'frame_idx']).reset_index(drop=True)
    n = len(frames)
    mi = pd.MultiIndex.from_frame(frames)
    seg_names = frames['segment'].to_numpy()
    seg_list = np.array(sorted(set(seg_names)))
    seg_frames = {s: np.flatnonzero(seg_names == s) for s in seg_list}
    ids = np.arange(n)
    nb_r, seed_r, frames_r, segs_r = ar.read_params_risk(os.path.join(RISK, 'params.txt'))
    check((nb_r, seed_r, frames_r) == (N_BOOT, SEED, n) and list(seg_list) == segs_r,
          f'{n} frames in {len(seg_list)} segments; resamples, seed, frames, segment list as the risk analysis\'s '
          f'params.txt')
    rng = np.random.default_rng(SEED)
    draws = [rng.choice(seg_list, len(seg_list)) for _ in range(N_BOOT)]
    h = hashlib.sha256()
    for d in draws:
        h.update(np.searchsorted(seg_list, d).astype(np.int32).tobytes())
    nb_q, seed_q, hash_q = ar.read_hash_quantities(os.path.join(QUANT, 'params.txt'))
    check(h.hexdigest() == HASH and (nb_q, seed_q, hash_q) == (N_BOOT, SEED, HASH),
          f'drawn segments\' sha256 {h.hexdigest()} = the quantities analysis\'s')
    fs = pd.read_csv(os.path.join(RISK, 'vlm_frame_scores.csv'), float_precision='round_trip')
    s_ = fs[(fs.model == MODEL) & (fs.condition == COND)].set_index(['segment', 'frame_idx'])['score_mean']
    if len(s_) != n or not s_.index.isin(mi).all():
        sys.exit(f'the frames of {MODEL} {COND} are not the {n} frames')
    risk = s_.reindex(mi).to_numpy(float)
    ssm_df = av.ssm_table(FSSM, frames)
    ssm = {c: ssm_df[c].to_numpy() for c in ssm_df.columns if c.endswith(('_conflict', '_risk', '_use'))}
    ssm = {k: (v.astype(float) if k.endswith('_risk') else v.astype(bool)) for k, v in ssm.items()}
    check(not np.isnan(risk).any() and all(ssm[s + '_conflict'].all() and ssm[s + '_use'].all() for s in av.SSMS),
          'every frame has a score and a value of every SSM')
    lab = pd.read_csv(SL, dtype=str, keep_default_na=False).set_index('segment')
    labels = {g: lab.loc[seg_names, g].str.strip().to_numpy() for g in GROUPS}
    for g, classes in GROUPS.items():
        check(set(labels[g]) == set(classes), f'{g}: classes {sorted(set(labels[g]))}')

    def taus(idx):
        """tau_b_all of every SSM on the frames idx, one group (the main analysis's function)."""
        if len(idx) < 3:
            return {s: np.nan for s in av.SSMS}
        st = av.all_stats(risk[idx], {k: v[idx] for k, v in ssm.items()}, ids[idx], {'all': np.zeros(len(idx), int)})
        return {s: st[s][0]['all'][4] for s in av.SSMS}

    keys = [(g, c) for g, cl in GROUPS.items() for c in cl]
    point = {k: taus(np.flatnonzero(labels[k[0]] == k[1])) for k in keys}
    boot = {k: np.full((N_BOOT, len(av.SSMS)), np.nan) for k in keys}
    tb = time.time()
    for b, draw in enumerate(draws):
        idx = np.concatenate([seg_frames[s] for s in draw])
        for k in keys:
            t_ = taus(idx[labels[k[0]][idx] == k[1]])
            boot[k][b] = [t_[s] for s in av.SSMS]
        if (b + 1) % 500 == 0:
            print(f'  [resample {b + 1}/{N_BOOT}, {(time.time() - tb) / (b + 1) * (N_BOOT - b - 1):.0f} s left]', flush=True)

    # checks: scipy, pair by pair, the sums over the groups
    rows_t, dmax_sp, dmax_bf = [], 0.0, 0.0
    sums = {g: {s: [0.0, 0.0, 0.0] for s in av.SSMS} for g in GROUPS}
    for g, c in keys:
        m = labels[g] == c
        nf = int(m.sum())
        for si, s in enumerate(av.SSMS):
            x, y = risk[m], ssm[s + '_risk'][m]
            sp = float(kendalltau(x, y, variant='b').statistic)
            S, tx, ty = counts(x, y)
            for j, v in enumerate((S, tx, ty)):
                sums[g][s][j] += v
            bf = S / np.sqrt(tx * ty)
            dmax_sp = max(dmax_sp, abs(point[(g, c)][s] - sp))
            dmax_bf = max(dmax_bf, abs(point[(g, c)][s] - bf))
            lo, hi, nv = ar.ci(boot[(g, c)][:, si])
            rows_t.append(dict(grouping=g, group=c, segments=len(set(seg_names[m])), frames=nf,
                               frame_pairs=nf * (nf - 1) // 2, ssm=s, tau_b=point[(g, c)][s], tau_b_lo=lo, tau_b_hi=hi,
                               nboot=nv, side=side(lo, hi)))
    check(dmax_sp <= TOL, f'every group\'s tau-b = scipy kendalltau(variant=\'b\') on its frames: largest |difference| '
          f'{float(dmax_sp)!r} (9 groups x 9 SSMs)')
    check(dmax_bf <= TOL, f'every group\'s tau-b = the pair-by-pair count (no scipy): largest |difference| {float(dmax_bf)!r}')
    ag = {(d['ssm'], d['grouping']): (ln, d) for ln, d in csv_rows(os.path.join(WEA, 'agreement.csv'))
          if d['model'] == MODEL and d['condition'] == COND}
    loc = {d['ssm']: (ln, d) for ln, d in csv_rows(os.path.join(O, 'exploratory_sec43_more_tau_b.csv'))
           if d['frames'] == 'main_1084' and d['condition'] == COND and d['grouping'] == 'location'}
    rows_p = []
    for g in GROUPS:
        for s in av.SSMS:
            S, tx, ty = sums[g][s]
            pooled = S / np.sqrt(tx * ty)
            if g == 'location':
                ln, d = loc[s]
                ref, src = d['tau_b'], f'exploratory_sec43_more_tau_b.csv line {ln}'
            else:
                ln, d = ag[(s, g)]
                ref, src = d['tau_b_all'], f'{WEA}/agreement.csv line {ln}'
            rows_p.append(dict(grouping=g, ssm=s, sum_S=S, sum_tx=tx, sum_ty=ty, pooled_tau_b=pooled, reference=ref,
                               difference=pooled - float(ref), same_text=txt(pooled) == ref, source=src))
    dmax_p = max(abs(d['difference']) for d in rows_p)
    check(dmax_p <= TOL, f'per grouping, the counts summed over its groups = the grouped tau-b already computed (27 '
          f'values): largest |difference| {float(dmax_p)!r}; the same text {sum(d["same_text"] for d in rows_p)} of 27')
    rows_d = []
    for g, c1, c0 in DIFFS:
        for si, s in enumerate(av.SSMS):
            lo, hi, nv = ar.ci(boot[(g, c1)][:, si] - boot[(g, c0)][:, si])
            rows_d.append(dict(comparison=f'{c1} - {c0}', grouping=g, ssm=s, tau_b_1=point[(g, c1)][s],
                               tau_b_0=point[(g, c0)][s], diff=point[(g, c1)][s] - point[(g, c0)][s], diff_lo=lo,
                               diff_hi=hi, width=hi - lo if not np.isnan(lo) else np.nan, nboot=nv, side=side(lo, hi)))
    pd.DataFrame(rows_t).to_csv(os.path.join(O, 'exploratory_sec43_groups_tau_b.csv'), index=False)
    pd.DataFrame(rows_d).to_csv(os.path.join(O, 'exploratory_sec43_groups_differences.csv'), index=False)
    pd.DataFrame(rows_p).to_csv(os.path.join(O, 'exploratory_sec43_groups_pooled_check.csv'), index=False)
    print('\n' + '=' * 110 + '\ntau-b within each group (exploratory_sec43_groups_tau_b.csv, line = row + 1)')
    for i, d in enumerate(rows_t):
        print(f"  line {i + 2}: {d['grouping']:11s} {d['group']:14s} ({d['segments']} segments, {d['frames']} frames, "
              f"{d['frame_pairs']} pairs) {d['ssm']:6s} {r(d['tau_b'])} [{r(d['tau_b_lo'])}, {r(d['tau_b_hi'])}] "
              f"(nboot {d['nboot']}; {d['side']})")
    print('\ndifferences between groups (exploratory_sec43_groups_differences.csv, line = row + 1)')
    for i, d in enumerate(rows_d):
        print(f"  line {i + 2}: {d['comparison']:28s} {d['ssm']:6s} {r(d['diff'])} [{r(d['diff_lo'])}, {r(d['diff_hi'])}] "
              f"width {r(d['width'])} (nboot {d['nboot']}; {d['side']})")
    print('\nsums over the groups (exploratory_sec43_groups_pooled_check.csv, line = row + 1)')
    for i, d in enumerate(rows_p):
        print(f"  line {i + 2}: {d['grouping']:11s} {d['ssm']:6s} sum S {d['sum_S']:g}, tx {d['sum_tx']:g}, ty "
              f"{d['sum_ty']:g}: {r(d['pooled_tau_b'])}; {d['source']}: {d['reference']} (the same text: {d['same_text']})")
    print(f'\ndone in {time.time() - t0:.0f} s; checks not met: {len(failed)}' + (f': {failed}' if failed else ''))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
