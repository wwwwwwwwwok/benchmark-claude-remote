"""Only the first k repeats (user, 2026-10-29; for choosing how many repeats the Qwen3.6-27B quantities main run gets):
for each model of the quantities main runs, k = 1 .. 4 against k = 5, on the frames of the main result (closing in at
least --closing_min). The first k repeats are repeats 0 .. k-1; the frame estimate is, as in analyze_quantities.py, the
median of their readable answers, missing when there is none.
 - tau-b of the frame estimate against the truth for the groupings 'all' and 'segment'; the difference k - 5, with a
   95 % interval for 'all' (the risk analysis's resamples: whole segments, numpy default_rng(--seed), as
   analyze_quantities.py; each cell on its own frames, so the difference is exactly the difference of the two values);
 - tau-b of the model's SSMs (analyze_quantities.perceived_ssms on the first-k estimates) against the true SSMs,
   'all', minus k = 5, with intervals;
 - the median absolute error; the frames without an estimate, per quantity and for the model's SSMs;
 - the rule fixed by the user before seeing the results: the smallest k of at least 3 with |tau-b(k) - tau-b(5)| below
   0.02 in every model x condition x quantity cell ('all'); 5 when neither 3 nor 4 qualifies.
Check: the k = 5 values must equal those of the analysis folder given (--check: quantity_tau_b.csv, quantity_errors.csv,
perceived_ssm_tau_b.csv; the 'all' intervals too, which shows the resamples are the same), and the sha256 of the drawn
segment indices its params.txt.
usage: python -I quantities_first_k.py <vlm_scores> <truth.csv> <frame_ssm.csv> <out_dir> --check <analysis subfolder>
                                       [--run main] [--closing_min 0.1] [--ks 1,2,3,4] [--n_boot 2000] [--seed 20261021]
"""
import argparse
import glob
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd

CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'code')
sys.path.insert(0, CODE)
import analyze_quantities as aq  # noqa: E402
import analyze_vlm_ssm as av  # noqa: E402
from quantity_truth import TRUTH  # noqa: E402

RULE_MIN_K, RULE_LIMIT = 3, 0.02          # user, 2026-10-29

ap = argparse.ArgumentParser()
ap.add_argument('vlm_scores')
ap.add_argument('truth')
ap.add_argument('frame_ssm')
ap.add_argument('out_dir')
ap.add_argument('--check', required=True)
ap.add_argument('--run', default='main')
ap.add_argument('--closing_min', type=float, default=av.CLOSING_MIN)
ap.add_argument('--ks', default='1,2,3,4')
ap.add_argument('--n_boot', type=int, default=2000)
ap.add_argument('--seed', type=int, default=20261021)
a = ap.parse_args()
ks = [int(k) for k in a.ks.split(',')] + [5]
os.makedirs(a.out_dir, exist_ok=True)
t0 = time.time()

parts, files = [], {}
for d in sorted(glob.glob(os.path.join(a.vlm_scores, '*', ''))):
    name = os.path.basename(d.rstrip('/\\'))
    f = av.jsonl_file(d, f'{a.run}_quantities_{name}')
    if f:
        parts.append(aq.read_quantities(f, name))
        files[name] = f
        print(f'read {name}: {len(parts[-1])} requests ({time.time() - t0:.0f} s)', flush=True)
req = pd.concat(parts, ignore_index=True)
models = sorted(files)
if sorted(req.repeat.unique()) != [0, 1, 2, 3, 4]:
    sys.exit(f'repeats {sorted(req.repeat.unique())}, expected 0-4')

# frames, truth, true SSMs, groupings: as analyze_quantities.py
frames = req[['segment', 'frame_idx']].drop_duplicates().sort_values(['segment', 'frame_idx']).reset_index(drop=True)
ssm_df = av.ssm_table(a.frame_ssm, frames)
closing = ssm_df['closing'].to_numpy(float)
slow = (closing > 0) & (closing < a.closing_min)
frames, ssm_df = frames[~slow].reset_index(drop=True), ssm_df[~slow].reset_index(drop=True)
mi = pd.MultiIndex.from_frame(frames)
req = req[pd.MultiIndex.from_frame(req[['segment', 'frame_idx']]).isin(mi)].reset_index(drop=True)
tr = frames.merge(pd.read_csv(a.truth, float_precision='round_trip'), on=['segment', 'frame_idx'], how='left',
                  validate='one_to_one')
if tr['camera_offset'].isna().any():
    sys.exit(f'{a.truth}: no camera offset for some frames')
offset = tr['camera_offset'].to_numpy(float)
truth = {q: tr[TRUTH[q]].to_numpy(float) for q in aq.QUANTS}
groupings, _ = av.scene_groupings(frames, None, [])            # 'all' and 'segment'
ssm = {c: ssm_df[c].to_numpy() for c in ssm_df.columns if c.endswith(('_conflict', '_risk', '_use'))}
ssm = {k: (v.astype(float) if k.endswith('_risk') else v.astype(bool)) for k, v in ssm.items()}
lead_use = ssm['TTC_use']
n_frames = len(frames)
ids = np.arange(n_frames)
seg_of = frames['segment'].to_numpy()
seg_list = np.array(sorted(set(seg_of)))
seg_frames = {s: np.flatnonzero(seg_of == s) for s in seg_list}
print(f'frames {n_frames} in {len(seg_list)} segments ({int(slow.sum())} closing in more slowly than '
      f'{a.closing_min} m/s left out)', flush=True)

# frame estimates from the first k repeats; the model's SSMs from them
est, perc, pmiss = {}, {}, []
for k in ks:
    sub = req[req.repeat < k]
    for (m, c, q), s in sub.groupby(['model', 'condition', 'quantity']):
        est[(m, c, q, k)] = s.groupby(['segment', 'frame_idx']).value.median().reindex(mi).to_numpy(float)
    for m in models:
        for c in av.CONDITIONS:
            p, counts, _ = aq.perceived_ssms(*(est[(m, c, q, k)] for q in ('distance', 'ego_speed', 'lead_speed',
                                                                             'ego_accel', 'lead_accel')), offset)
            perc[(m, c, k)] = p
            pmiss.append(dict(model=m, condition=c, k=k, frames=n_frames, **counts))
cells = [(m, c, q) for m in models for c in av.CONDITIONS for q in aq.QUANTS]


def stats(idx, gb, groups):
    cache, res = {}, {}
    ones = np.ones(len(idx), bool)

    def taus(x, y, conflict, mask):
        key = mask.tobytes()
        if key not in cache:
            labs = {gn: gb[gn][mask] for gn in groups}
            cache[key] = labs, {gn: av.within_pairs(v) for gn, v in labs.items()
                                if gn != 'all' and len(np.unique(v)) > 20}
        labs, pairs = cache[key]
        xs, ys, cs, ids_s = x[mask], y[mask], conflict[mask], ids[idx][mask]
        o = {}
        for gn, lab in labs.items():
            if len(xs) >= 3 and cs.any():
                _, tc, _, ta = av.stats_by_grouping(xs, cs, ys, ids_s, lab, pairs.get(gn))
            else:
                tc = ta = np.nan
            o[gn] = ta
        return o

    for m, c, q in cells:
        y = truth[q][idx]
        for k in ks:
            x = est[(m, c, q, k)][idx]
            mask = ~np.isnan(x) & ~np.isnan(y)
            for gn, ta in taus(x, y, ones, mask).items():
                res[('tau', m, c, q, k, gn)] = ta
            if len(groups) > 1:                      # the point estimate only
                e = x[mask] - y[mask]
                res[('mae', m, c, q, k)] = float(np.median(np.abs(e))) if len(e) else np.nan
    for (m, c, k), p in perc.items():
        use = lead_use[idx] & p['valid'][idx]
        for s in aq.LEAD7:
            for gn, ta in taus(p[s + '_all'][idx], np.nan_to_num(ssm[s + '_risk'][idx]), ones,
                               use & ssm[s + '_conflict'][idx]).items():
                res[('p_true', m, c, s, k, gn)] = ta
    return res


point = stats(ids, groupings, ['all', 'segment'])
keys = [k for k in point if k[-1] == 'all' and k[0] in ('tau', 'p_true')]
boot = np.full((a.n_boot, len(keys)), np.nan)
rng = np.random.default_rng(a.seed)                      # the risk analysis's resamples, as analyze_quantities.py
draws_hash = hashlib.sha256()
tb = time.time()
for b in range(a.n_boot):
    draw = rng.choice(seg_list, len(seg_list))
    draws_hash.update(np.searchsorted(seg_list, draw).astype(np.int32).tobytes())
    idx = np.concatenate([seg_frames[s] for s in draw])
    gb = {'all': groupings['all'][idx]}
    r = stats(idx, gb, ['all'])
    boot[b] = [r.get(k, np.nan) for k in keys]
    if b in (0, 9) or (b + 1) % 250 == 0:
        print(f'bootstrap {b + 1}/{a.n_boot} ({(time.time() - tb) / (b + 1) * (a.n_boot - b - 1):.0f} s left)',
              flush=True)
col = {k: i for i, k in enumerate(keys)}


def diff(k1, k2):
    lo, hi, nv = aq.ci(boot[:, col[k1]] - boot[:, col[k2]])
    return point[k1] - point[k2], lo, hi, nv


rows = []
for m, c, q in cells:
    for k in ks:
        n_ok = int((~np.isnan(est[(m, c, q, k)]) & ~np.isnan(truth[q])).sum())
        for gn in ('all', 'segment'):
            r = dict(model=m, condition=c, quantity=q, k=k, grouping=gn, frames=n_ok,
                     tau_b=point[('tau', m, c, q, k, gn)], tau_b_5=point[('tau', m, c, q, 5, gn)])
            r['diff'] = r['tau_b'] - r['tau_b_5']
            if gn == 'all':
                lo, hi, nv = aq.ci(boot[:, col[('tau', m, c, q, k, gn)]]) if k == 5 else (np.nan,) * 3
                _, dlo, dhi, dnv = diff(('tau', m, c, q, k, gn), ('tau', m, c, q, 5, gn))
                r.update(tau_b_lo=lo, tau_b_hi=hi, diff_lo=dlo, diff_hi=dhi, nboot=dnv)
            rows.append(r)
tau = pd.DataFrame(rows)
tau.to_csv(os.path.join(a.out_dir, 'first_k_tau_b.csv'), index=False)
rows = []
for m, c, k in perc:
    for s in aq.LEAD7:
        p = perc[(m, c, k)]
        r = dict(model=m, condition=c, ssm=s, k=k, grouping='all',
                 frames=int((lead_use & p['valid'] & ssm[s + '_conflict']).sum()),
                 tau_b=point[('p_true', m, c, s, k, 'all')], tau_b_5=point[('p_true', m, c, s, 5, 'all')])
        lo, hi, nv = aq.ci(boot[:, col[('p_true', m, c, s, k, 'all')]]) if k == 5 else (np.nan,) * 3
        d, dlo, dhi, dnv = diff(('p_true', m, c, s, k, 'all'), ('p_true', m, c, s, 5, 'all'))
        r.update(tau_b_lo=lo, tau_b_hi=hi, diff=d, diff_lo=dlo, diff_hi=dhi, nboot=dnv)
        rows.append(r)
pss = pd.DataFrame(rows)
pss.to_csv(os.path.join(a.out_dir, 'first_k_perceived_ssm_tau_b.csv'), index=False)
rows = []
for m, c, q in cells:
    for k in ks:
        rows.append(dict(model=m, condition=c, quantity=q, k=k, frames=n_frames,
                         frames_missing=int(np.isnan(est[(m, c, q, k)]).sum()),
                         median_abs_error=point[('mae', m, c, q, k)], median_abs_error_5=point[('mae', m, c, q, 5)],
                         mae_diff=point[('mae', m, c, q, k)] - point[('mae', m, c, q, 5)]))
err = pd.DataFrame(rows)
err.to_csv(os.path.join(a.out_dir, 'first_k_errors_missing.csv'), index=False)
pd.DataFrame(pmiss).to_csv(os.path.join(a.out_dir, 'first_k_perceived_ssm_missing.csv'), index=False)

# --- the rule
lines = [f'first k repeats: models {models}; frames {n_frames} in {len(seg_list)} segments; bootstrap {a.n_boot} '
         f'resamples of whole segments, seed {a.seed}; sha256 of the drawn segment indices {draws_hash.hexdigest()}',
         f"rule (user, 2026-10-29): the smallest k >= {RULE_MIN_K} with |tau-b(k) - tau-b(5)| < {RULE_LIMIT} in every "
         f"model x condition x quantity cell, grouping 'all'"]
ta = tau[tau.grouping == 'all']
chosen = 5
for k in ks[:-1]:
    t = ta[ta.k == k]
    worst = t.loc[t['diff'].abs().idxmax()]
    over = t[~(t['diff'].abs() < RULE_LIMIT)]                 # a cell without a value does not qualify either
    lines.append(f'k = {k}: largest |difference| {abs(worst["diff"]):.4f} ({worst.model} {worst.condition} '
                 f'{worst.quantity}); cells at {RULE_LIMIT} or more: {len(over)} of {len(t)}'
                 + (': ' + ', '.join(f'{r.model} {r.condition} {r.quantity} {r.diff:+.4f}' for r in over.itertuples())
                    if 0 < len(over) <= 12 else ''))
    if k >= RULE_MIN_K and chosen == 5 and len(over) == 0:
        chosen = k
lines.append(f'the rule gives k = {chosen}')

# --- check against the analysis folder (k = 5)
qt = pd.read_csv(os.path.join(a.check, 'quantity_tau_b.csv'), float_precision='round_trip') \
    .set_index(['model', 'condition', 'quantity', 'grouping'])
qe = pd.read_csv(os.path.join(a.check, 'quantity_errors.csv'), float_precision='round_trip') \
    .set_index(['model', 'condition', 'quantity'])
pt = pd.read_csv(os.path.join(a.check, 'perceived_ssm_tau_b.csv'), float_precision='round_trip')
pt = pt[pt.comparison == 'model_ssm_vs_true_ssm'].set_index(['model', 'condition', 'ssm', 'grouping'])
n_same = n_all = 0


def eq(*pairs):                                          # both missing counts as the same
    return all(x == y or (np.isnan(x) and np.isnan(y)) for x, y in pairs)


for r in tau[tau.k == 5].itertuples():
    o = qt.loc[(r.model, r.condition, r.quantity, r.grouping)]
    same = eq((r.tau_b, o.tau_b)) and (r.grouping != 'all' or eq((r.tau_b_lo, o.tau_b_lo), (r.tau_b_hi, o.tau_b_hi)))
    n_same, n_all = n_same + same, n_all + 1
for r in err[err.k == 5].itertuples():
    n_same += eq((r.median_abs_error, qe.loc[(r.model, r.condition, r.quantity)].median_abs_error))
    n_all += 1
for r in pss[pss.k == 5].itertuples():
    o = pt.loc[(r.model, r.condition, r.ssm, 'all')]
    n_same, n_all = n_same + eq((r.tau_b, o.tau_b), (r.tau_b_lo, o.tau_b_lo), (r.tau_b_hi, o.tau_b_hi)), n_all + 1
params = open(os.path.join(a.check, 'params.txt'), encoding='utf-8').read()
lines.append(f'check against {a.check} (k = 5): {n_same} of {n_all} values identical (tau-b all / segment, the all '
             f'intervals, median absolute error, model SSM tau-b and intervals); drawn segments the same as its '
             f'params.txt: {draws_hash.hexdigest() in params}')
lines.append(f'done in {time.time() - t0:.0f} s')
open(os.path.join(a.out_dir, 'first_k_rule.txt'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
