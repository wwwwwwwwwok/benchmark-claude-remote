"""Analysis of the quantities task (user's design, 2026-10-28, decided before the main runs were seen; CLAUDE.md
"物理量的正式分析"): how well each model reads the distance to the lead, the ego and lead speeds and accelerations off
the pictures, and what its reading implies for the lead-vehicle SSMs.

 - Frames: the frames asked; those closing in more slowly than --closing_min (closing speed above 0) are left out, as
   in analyze_vlm_ssm.py: 1,084 frames for the main result (0.1 m/s), 1,123 for the sensitivity run (0).
 - Frame estimate per model x condition x quantity: the median of the repeats with a readable number (no answer is
   dropped; a frame without any is missing in that cell). Implausible answers (distance above 200 m, a speed above
   60 m/s, an acceleration beyond +-10 m/s2) are only counted.
 - Against the truth (code/quantity_truth.py: distance = gap + the segment's front-bumper-to-front-camera distance;
   v_ego, v_lead, a_ego, a_lead): Kendall tau-b for the groupings of analyze_vlm_ssm.py (pairs within the same group);
   median absolute error and median error (estimate - truth), grouping 'all' only; for the accelerations the share
   right in each truth class (+-0.5 m/s2: speeding up, steady, slowing down) and that of answering "steady" always.
 - Comparisons fixed in advance, on the same resamples, each cell on its own frames (so the difference is exactly
   the difference of the two point estimates): grid - ctrl_grid, sep_time - ctrl_sep, sep_camera - ctrl_sep per model
   and quantity; MODEL_PAIRS per condition and quantity; tau-b (every grouping) and median absolute error ('all').
 - The model's SSMs: the five frame estimates, distance minus the camera offset as the gap, through lead_ssms of
   waymo_ssm_frames.py (its PARAMS) -> TTC, THW, DRAC, MTTC, CI, PICUD, PSD; a frame is missing when a quantity is
   missing, the gap would be negative or a speed is negative. "Larger = riskier" and "has a value" as in
   analyze_vlm_ssm.py; frames where the model's SSM has no value are tied at the safe end (its overall tau-b).
   tau-b (1) of the model's SSM against the true SSM (frames where the true one has a value), (2) of the model's risk
   score (score_mean of the risk analysis's vlm_frame_scores.csv, same condition) against the model's SSM; for (2)
   also over the frames where the model's SSM has a value. Frames where the ego or the lead reverses are left out of
   both, as in the risk analysis.
 - Bootstrap: the risk analysis's resamples (whole segments, numpy default_rng(--seed), one rng.choice over the sorted
   segments per resample, every drawn copy its own segment group), shared by all models; 95 % percentile intervals,
   none when fewer than 90 % of the resamples are defined. Check: with these resamples the risk analysis's tau-b and
   intervals of every model's grid condition (9 SSMs x every grouping) are recomputed with its own functions and must
   equal its agreement.csv (risk_check.csv).

Inputs: <vlm_scores>/<model>/<run>_quantities_<model>.jsonl(.gz) for every model folder that has one; the truth csv of
code/quantity_truth.py for (at least) these frames; frame_ssm.csv; the risk analysis subfolder of the same frames
(vlm_frame_scores.csv, agreement.csv; e.g. analysis/after_qwen3.5-4b/main_1084).
usage: python -I analyze_quantities.py <vlm_scores> <truth.csv> <frame_ssm.csv> <risk analysis subfolder> <out_dir>
                                       [--run main] [--closing_min 0.1] [--scene_labels <csv>] [--group_by time_of_day]
                                       [--n_boot 2000] [--seed 20261021]
"""
import argparse
import glob
import hashlib
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_vlm_ssm as av  # noqa: E402
from quantity_truth import TRUTH  # noqa: E402
from run_vlm import QUANTITIES  # noqa: E402
from waymo_ssm_frames import PARAMS, lead_ssms  # noqa: E402

QUANTS = list(QUANTITIES)
LEAD7 = ['TTC', 'THW', 'DRAC', 'MTTC', 'CI', 'PICUD', 'PSD']
IMPLAUSIBLE = {'distance': lambda v: v > 200, 'ego_speed': lambda v: v > 60, 'lead_speed': lambda v: v > 60,
               'ego_accel': lambda v: np.abs(v) > 10, 'lead_accel': lambda v: np.abs(v) > 10}
STEADY = 0.5                                   # m/s2: speeding up above, slowing down below -STEADY
TIME_PAIRS = [('grid', 'ctrl_grid'), ('sep_time', 'ctrl_sep'), ('sep_camera', 'ctrl_sep')]
MODEL_PAIRS = [('qwen-drive-1.0-4b', 'qwen3.5-4b')]   # model_1 - model_2 (user, 2026-10-28)


def read_quantities(path, name):
    rows, seen = [], set()
    with av.open_jsonl(path) as f:
        for raw in f:
            d = av.load_light(raw)
            k = (d['segment'], int(d['frame_idx']), d['condition'], d['quantity'], int(d['repeat']))
            if k in seen:
                raise ValueError(f'{path}: duplicate request {k}')
            seen.add(k)
            p = d.get('parsed')
            rows.append(dict(model=name, segment=k[0], frame_idx=k[1], condition=k[2], quantity=k[3], repeat=k[4],
                             value=float(p[QUANTITIES[k[3]][1]]) if p else np.nan,
                             finish_reason=d.get('finish_reason')))
    return pd.DataFrame(rows)


def accel_class(a):
    return np.where(a > STEADY, 1, np.where(a < -STEADY, -1, 0))


def ci(arr):
    """As in analyze_vlm_ssm.py."""
    ok = ~np.isnan(arr)
    if ok.sum() < 0.9 * len(arr):
        return np.nan, np.nan, int(ok.sum())
    lo, hi = np.percentile(arr[ok], [2.5, 97.5])
    return float(lo), float(hi), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('vlm_scores')
    ap.add_argument('truth')
    ap.add_argument('frame_ssm')
    ap.add_argument('risk_analysis')
    ap.add_argument('out_dir')
    ap.add_argument('--run', default='main')
    ap.add_argument('--closing_min', type=float, default=av.CLOSING_MIN)
    ap.add_argument('--scene_labels', default=None)
    ap.add_argument('--group_by', default='time_of_day')
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261021)
    a = ap.parse_args()
    group_by = [c for c in a.group_by.split(',') if c] if a.scene_labels else []
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()

    # --- replies
    parts, files = [], {}
    for d in sorted(glob.glob(os.path.join(a.vlm_scores, '*', ''))):
        name = os.path.basename(d.rstrip('/\\'))
        f = av.jsonl_file(d, f'{a.run}_quantities_{name}')
        if f:
            parts.append(read_quantities(f, name))
            files[name] = f
            print(f'read {name}: {len(parts[-1])} requests ({time.time() - t0:.0f} s)')
    if not parts:
        sys.exit(f'no {a.run}_quantities_<model>.jsonl(.gz) in the model folders of {a.vlm_scores}')
    req = pd.concat(parts, ignore_index=True)
    models = sorted(files)
    frames = req[['segment', 'frame_idx']].drop_duplicates().sort_values(['segment', 'frame_idx']).reset_index(drop=True)
    for m in models:
        n = len(req.loc[req.model == m, ['segment', 'frame_idx']].drop_duplicates())
        if n != len(frames):
            raise ValueError(f'{m} asked about {n} frames, the union is {len(frames)}')

    # --- frames, true SSMs, truth, groupings (as analyze_vlm_ssm.py)
    n_asked = len(frames)
    ssm_df = av.ssm_table(a.frame_ssm, frames)
    closing = ssm_df['closing'].to_numpy(float)
    slow = (closing > 0) & (closing < a.closing_min)
    frames, ssm_df = frames[~slow].reset_index(drop=True), ssm_df[~slow].reset_index(drop=True)
    keep = pd.MultiIndex.from_frame(frames)
    req = req[pd.MultiIndex.from_frame(req[['segment', 'frame_idx']]).isin(keep)].reset_index(drop=True)
    tr = frames.merge(pd.read_csv(a.truth, float_precision='round_trip'), on=['segment', 'frame_idx'], how='left',
                      validate='one_to_one', indicator=True)
    if (tr['_merge'] != 'both').any():
        sys.exit(f'{int((tr["_merge"] != "both").sum())} frames are not in {a.truth}')
    # frame_ssm read as quantity_truth.py and analyze_vlm_ssm.ssm_table read it (pandas' default parser); the truth
    # file written from it is read back exactly (round_trip)
    fs = frames.merge(pd.read_csv(a.frame_ssm, usecols=['segment', 'frame_idx', 'gap', 'v_ego', 'v_lead', 'a_ego',
                                                           'a_lead']),
                      on=['segment', 'frame_idx'], how='left', validate='one_to_one')
    for c in ['gap', 'v_ego', 'v_lead', 'a_ego', 'a_lead']:
        x, y = tr[c].to_numpy(float), fs[c].to_numpy(float)
        if not np.array_equal(x, y, equal_nan=True):
            sys.exit(f'{a.truth}: column {c} differs from {a.frame_ssm}')
    if tr['camera_offset'].isna().any():
        sys.exit(f'{a.truth}: no camera offset for some frames')
    offset = tr['camera_offset'].to_numpy(float)
    truth = {q: tr[TRUTH[q]].to_numpy(float) for q in QUANTS}
    groupings, labels_used = av.scene_groupings(frames, a.scene_labels, group_by)
    gnames = list(groupings)
    ssm = {c: ssm_df[c].to_numpy() for c in ssm_df.columns if c.endswith(('_conflict', '_risk', '_use'))}
    ssm = {k: (v.astype(float) if k.endswith('_risk') else v.astype(bool)) for k, v in ssm.items()}
    n_frames = len(frames)
    ids = np.arange(n_frames)
    seg_of = frames['segment'].to_numpy()
    seg_list = np.array(sorted(set(seg_of)))
    seg_frames = {s: np.flatnonzero(seg_of == s) for s in seg_list}
    mi = pd.MultiIndex.from_frame(frames)

    # --- frame estimates
    g = req.groupby(['model', 'condition', 'quantity', 'segment', 'frame_idx'])
    fr = g.agg(n_requests=('repeat', 'size'), n_read=('value', 'count'), estimate=('value', 'median')).reset_index()
    est = {}
    for (m, c, q), sub in fr.groupby(['model', 'condition', 'quantity']):
        est[(m, c, q)] = sub.set_index(['segment', 'frame_idx'])['estimate'].reindex(mi).to_numpy(float)
    cells = [(m, c, q) for m in models for c in av.CONDITIONS for q in QUANTS]
    missing_cells = [k for k in cells if k not in est]
    if missing_cells:
        sys.exit(f'cells without any request: {missing_cells[:5]}')
    out = fr.copy()
    out['truth'] = [truth[q][i] for q, i in zip(out.quantity, mi.get_indexer(pd.MultiIndex.from_frame(
        out[['segment', 'frame_idx']])))]
    out.to_csv(os.path.join(a.out_dir, 'quantity_frames.csv'), index=False)

    arow = []
    for m, c, q in cells:
        s = req[(req.model == m) & (req.condition == c) & (req.quantity == q)]
        v = s.value.dropna().to_numpy()
        arow.append(dict(model=m, condition=c, quantity=q, requests=len(s), read=len(v), read_rate=len(v) / len(s),
                         at_max_tokens=int((s.finish_reason == 'length').sum()),
                         implausible=int(IMPLAUSIBLE[q](v).sum()), negative=int((v < 0).sum()),
                         frames=n_frames, frames_without_answer=int(np.isnan(est[(m, c, q)]).sum()),
                         frames_with_truth=int((~np.isnan(truth[q])).sum())))
    pd.DataFrame(arow).to_csv(os.path.join(a.out_dir, 'quantity_answers.csv'), index=False)

    # --- the model's SSMs
    perc, pmiss = {}, []
    for m, c in itertools.product(models, av.CONDITIONS):
        d, vf, vl, af, al = (est[(m, c, q)] for q in ('distance', 'ego_speed', 'lead_speed', 'ego_accel', 'lead_accel'))
        gap = d - offset
        lack = np.isnan(d) | np.isnan(vf) | np.isnan(vl) | np.isnan(af) | np.isnan(al)
        neg_gap = ~lack & (gap < 0)
        neg_v = ~lack & ((vf < 0) | (vl < 0))
        valid = ~lack & ~neg_gap & ~neg_v
        vals = {s: np.full(n_frames, np.nan) for s in LEAD7}
        for i in np.flatnonzero(valid):
            o = lead_ssms(gap[i], vf[i], vl[i], af[i], al[i])
            for s in LEAD7:
                vals[s][i] = o[s]
        p = {'valid': valid, 'gap': gap}
        for s in LEAD7:
            v = vals[s]
            conflict = valid & ((np.nan_to_num(v, nan=0.0) > 0) if s in av.POSITIVE_IS_CONFLICT else np.isfinite(v))
            r = np.where(conflict, -v if s in av.SMALLER_RISKIER else v, np.nan)
            p[s] = v
            p[s + '_conflict'] = conflict
            p[s + '_risk'] = r
            # frames without a value tied at the safe end, as in analyze_vlm_ssm.all_stats
            p[s + '_all'] = np.where(conflict, r, np.nanmin(r[conflict]) - 1.0) if conflict.any() else \
                np.where(valid, 0.0, np.nan)
        perc[(m, c)] = p
        pmiss.append(dict(model=m, condition=c, frames=n_frames, missing=int((~valid).sum()),
                          quantity_missing=int(lack.sum()), gap_negative=int(neg_gap.sum()),
                          speed_negative=int(neg_v.sum()), gap_zero=int((valid & (gap == 0)).sum()),
                          **{f'with_value_{s}': int(p[s + '_conflict'].sum()) for s in LEAD7}))
    pd.DataFrame(pmiss).to_csv(os.path.join(a.out_dir, 'perceived_ssm_missing.csv'), index=False)
    pf = []
    for (m, c), p in perc.items():
        pf.append(pd.DataFrame({'model': m, 'condition': c, 'segment': frames.segment, 'frame_idx': frames.frame_idx,
                                'valid': p['valid'], 'gap': p['gap'], **{s: p[s] for s in LEAD7}}))
    pd.concat(pf, ignore_index=True).to_csv(os.path.join(a.out_dir, 'perceived_ssm_frames.csv'), index=False)

    # --- risk scores of the risk analysis (same frames)
    rs = pd.read_csv(os.path.join(a.risk_analysis, 'vlm_frame_scores.csv'), float_precision='round_trip')
    risk = {}
    for m, c in itertools.product(models, av.CONDITIONS):
        s = rs[(rs.model == m) & (rs.condition == c)].set_index(['segment', 'frame_idx'])['score_mean']
        if len(s):
            v = s.reindex(mi).to_numpy(float)
            if np.isnan(v).all():
                sys.exit(f'{a.risk_analysis}: no risk scores for {m} {c} on these frames')
            risk[(m, c)] = v
    risk_models = sorted({m for m, _ in risk})
    lead_use = ssm['TTC_use']                          # the seven lead-vehicle SSMs share it (no reversing)

    # --- all statistics for one resample (idx: frame positions; gb: group labels of those positions)
    def stats(idx, gb):
        cache, res = {}, {}

        def taus(x, y, conflict, mask):
            key = mask.tobytes()
            if key not in cache:
                labs = {gn: v[mask] for gn, v in gb.items()}
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
                o[gn] = (tc, ta)
            return o

        ones = np.ones(len(idx), bool)
        for m, c, q in cells:
            x, y = est[(m, c, q)][idx], truth[q][idx]
            mask = ~np.isnan(x) & ~np.isnan(y)
            for gn, (_, ta) in taus(x, y, ones, mask).items():
                res[('tau', m, c, q, gn)] = ta
            e = x[mask] - y[mask]
            res[('mae', m, c, q)] = float(np.median(np.abs(e))) if len(e) else np.nan
            res[('bias', m, c, q)] = float(np.median(e)) if len(e) else np.nan
        for (m, c), p in perc.items():
            for s in LEAD7:
                use = lead_use[idx] & p['valid'][idx]
                x_all, true_r, true_c = p[s + '_all'][idx], ssm[s + '_risk'][idx], ssm[s + '_conflict'][idx]
                for gn, (_, ta) in taus(x_all, np.nan_to_num(true_r), ones, use & true_c).items():
                    res[('p_true', m, c, s, gn)] = ta
                if (m, c) in risk:
                    xr = risk[(m, c)][idx]
                    for gn, (tc, ta) in taus(xr, x_all, p[s + '_conflict'][idx], use & ~np.isnan(xr)).items():
                        res[('p_risk', m, c, s, gn)] = ta
                        res[('p_risk_c', m, c, s, gn)] = tc
        for m in risk_models:
            st = av.all_stats(risk[(m, 'grid')][idx], {k: v[idx] for k, v in ssm.items()}, ids[idx], gb)
            for s in av.SSMS:
                for gn in gnames:
                    res[('risk_check', m, s, gn)] = st[s][0][gn][1]
        return res

    point = stats(ids, groupings)
    keys = list(point)
    boot = np.full((a.n_boot, len(keys)), np.nan)
    rng = np.random.default_rng(a.seed)                 # the risk analysis's resamples (analyze_vlm_ssm.main)
    draws_hash = hashlib.sha256()
    tb = time.time()
    for b in range(a.n_boot):
        draw = rng.choice(seg_list, len(seg_list))
        draws_hash.update(np.searchsorted(seg_list, draw).astype(np.int32).tobytes())
        idx = np.concatenate([seg_frames[s] for s in draw])
        gb = {gn: v[idx] for gn, v in groupings.items()}
        gb['segment'] = np.concatenate([np.full(len(seg_frames[s]), k) for k, s in enumerate(draw)])
        r = stats(idx, gb)
        boot[b] = [r.get(k, np.nan) for k in keys]
        if b in (0, 9) or (b + 1) % 250 == 0:
            done = b + 1
            print(f'bootstrap {done}/{a.n_boot} ({(time.time() - tb) / done * (a.n_boot - done):.0f} s left)')
    col = {k: i for i, k in enumerate(keys)}

    def with_ci(k):
        lo, hi, nv = ci(boot[:, col[k]])
        return point[k], lo, hi, nv

    # --- tables
    rows = []
    for m, c, q in cells:
        n_ok = int((~np.isnan(est[(m, c, q)]) & ~np.isnan(truth[q])).sum())
        for gn in gnames:
            v, lo, hi, nv = with_ci(('tau', m, c, q, gn))
            rows.append(dict(model=m, condition=c, quantity=q, grouping=gn, frames=n_ok, tau_b=v, tau_b_lo=lo,
                             tau_b_hi=hi, nboot=nv))
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, 'quantity_tau_b.csv'), index=False)
    rows = []
    for m, c, q in cells:
        r = dict(model=m, condition=c, quantity=q, grouping='all',
                 frames=int((~np.isnan(est[(m, c, q)]) & ~np.isnan(truth[q])).sum()))
        for st in ('mae', 'bias'):
            v, lo, hi, nv = with_ci((st, m, c, q))
            name = 'median_abs_error' if st == 'mae' else 'median_error'
            r.update({name: v, name + '_lo': lo, name + '_hi': hi, name + '_nboot': nv})
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, 'quantity_errors.csv'), index=False)
    rows = []
    for m, c, q in cells:
        if not q.endswith('accel'):
            continue
        x, y = est[(m, c, q)], truth[q]
        ok = ~np.isnan(x) & ~np.isnan(y)
        ct, cp = accel_class(y[ok]), accel_class(x[ok])
        r = dict(model=m, condition=c, quantity=q, frames=int(ok.sum()))
        for k, nm in ((1, 'up'), (0, 'steady'), (-1, 'down')):
            n_k = int((ct == k).sum())
            r[f'truth_{nm}'] = n_k
            r[f'right_{nm}'] = int(((ct == k) & (cp == k)).sum())
            r[f'share_right_{nm}'] = r[f'right_{nm}'] / n_k if n_k else np.nan
        r['share_right_all'] = float((ct == cp).mean()) if ok.any() else np.nan
        r['share_always_steady'] = float((ct == 0).mean()) if ok.any() else np.nan
        for kt, nt in ((1, 'up'), (0, 'steady'), (-1, 'down')):
            for ke, ne in ((1, 'up'), (0, 'steady'), (-1, 'down')):
                r[f'n_truth_{nt}_answer_{ne}'] = int(((ct == kt) & (cp == ke)).sum())
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, 'accel_classes.csv'), index=False)

    rows = []

    def diff_row(kind, m1, c1, m2, c2, q, st, gn, k1, k2):
        b = boot[:, col[k1]] - boot[:, col[k2]]
        lo, hi, nv = ci(b)
        v1, v2 = point[k1], point[k2]
        rows.append(dict(comparison=kind, model_1=m1, condition_1=c1, model_2=m2, condition_2=c2, quantity=q, stat=st,
                         grouping=gn, value_1=v1, value_2=v2, diff=v1 - v2, diff_lo=lo, diff_hi=hi, nboot=nv,
                         clear=bool(lo > 0 or hi < 0) if not np.isnan(lo) else False))

    for m in models:
        for c1, c2 in TIME_PAIRS:
            for q in QUANTS:
                for gn in gnames:
                    diff_row('time', m, c1, m, c2, q, 'tau_b', gn, ('tau', m, c1, q, gn), ('tau', m, c2, q, gn))
                diff_row('time', m, c1, m, c2, q, 'median_abs_error', 'all', ('mae', m, c1, q), ('mae', m, c2, q))
    for m1, m2 in MODEL_PAIRS:
        if m1 not in models or m2 not in models:
            print(f'model comparison {m1} - {m2}: skipped, not in this run')
            continue
        for c in av.CONDITIONS:
            for q in QUANTS:
                for gn in gnames:
                    diff_row('model', m1, c, m2, c, q, 'tau_b', gn, ('tau', m1, c, q, gn), ('tau', m2, c, q, gn))
                diff_row('model', m1, c, m2, c, q, 'median_abs_error', 'all', ('mae', m1, c, q), ('mae', m2, c, q))
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, 'quantity_differences.csv'), index=False)

    rows = []
    for (m, c), p in perc.items():
        for s in LEAD7:
            use = lead_use & p['valid']
            for gn in gnames:
                v, lo, hi, nv = with_ci(('p_true', m, c, s, gn))
                rows.append(dict(model=m, condition=c, ssm=s, comparison='model_ssm_vs_true_ssm', grouping=gn,
                                 frames=int((use & ssm[s + '_conflict']).sum()),
                                 frames_model_ssm_with_value=int((use & ssm[s + '_conflict'] & p[s + '_conflict']).sum()),
                                 tau_b=v, tau_b_lo=lo, tau_b_hi=hi, nboot=nv))
            if (m, c) not in risk:
                continue
            for gn in gnames:
                v, lo, hi, nv = with_ci(('p_risk', m, c, s, gn))
                vc, loc, hic, nvc = with_ci(('p_risk_c', m, c, s, gn))
                ok = use & ~np.isnan(risk[(m, c)])
                rows.append(dict(model=m, condition=c, ssm=s, comparison='risk_score_vs_model_ssm', grouping=gn,
                                 frames=int(ok.sum()), frames_model_ssm_with_value=int((ok & p[s + '_conflict']).sum()),
                                 tau_b=v, tau_b_lo=lo, tau_b_hi=hi, nboot=nv, tau_b_with_value=vc,
                                 tau_b_with_value_lo=loc, tau_b_with_value_hi=hic, nboot_with_value=nvc))
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, 'perceived_ssm_tau_b.csv'), index=False)

    # --- check: the risk analysis's intervals from these resamples
    ag = pd.read_csv(os.path.join(a.risk_analysis, 'agreement.csv'), dtype=str, keep_default_na=False) \
        .set_index(['model', 'condition', 'ssm', 'grouping'])

    def num(t):
        return float(t) if t != '' else np.nan

    def same(x, y):
        return x == y or (np.isnan(x) and np.isnan(y))

    rows = []
    for m in risk_models:
        for s in av.SSMS:
            for gn in gnames:
                v, lo, hi, nv = with_ci(('risk_check', m, s, gn))
                r = ag.loc[(m, 'grid', s, gn)]
                old = (num(r['tau_b_conflict']), num(r['tau_b_conflict_lo']), num(r['tau_b_conflict_hi']),
                       int(r['tau_b_conflict_nboot']))
                rows.append(dict(model=m, condition='grid', ssm=s, grouping=gn, tau_b=v, tau_b_lo=lo, tau_b_hi=hi,
                                 nboot=nv, risk_tau_b=old[0], risk_tau_b_lo=old[1], risk_tau_b_hi=old[2],
                                 risk_nboot=old[3], identical=all(same(x, y) for x, y in zip((v, lo, hi, nv), old))))
    rc = pd.DataFrame(rows)
    rc.to_csv(os.path.join(a.out_dir, 'risk_check.csv'), index=False)

    # --- params.txt
    with open(os.path.join(a.out_dir, 'params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'analyze_quantities.py run {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'vlm_scores = {os.path.abspath(a.vlm_scores)}\ntruth = {os.path.abspath(a.truth)}\n'
                f'frame_ssm = {os.path.abspath(a.frame_ssm)}\nrisk analysis = {os.path.abspath(a.risk_analysis)}\n'
                f'scene_labels = {os.path.abspath(a.scene_labels) if a.scene_labels else "(none)"}\n')
        for m in models:
            f.write(f'model {m}: {int((req.model == m).sum())} requests on the frames kept ({files[m]})\n')
        f.write(f'frames asked {n_asked}; closing in more slowly than {a.closing_min} m/s (above 0): {int(slow.sum())}, '
                f'left out; frames {n_frames} in {len(seg_list)} segments\n')
        f.write(f'bootstrap: {a.n_boot} resamples of whole segments, seed {a.seed} (the risk analysis\'s); sha256 of the '
                f'drawn segment indices {draws_hash.hexdigest()}\n')
        f.write(f'risk analysis intervals recomputed from these resamples (grid, {len(av.SSMS)} SSMs x {len(gnames)} '
                f'groupings per model, models {risk_models}): {int(rc.identical.sum()) if len(rc) else 0} of {len(rc)} '
                f'identical to its agreement.csv\n')
        f.write(f'lead_ssms parameters: {PARAMS}\n')
        f.write('implausible (counted only): distance > 200 m, speed > 60 m/s, |acceleration| > 10 m/s2; '
                f'acceleration classes at +-{STEADY} m/s2\n')
        for gn in gnames:
            vals, cnt = np.unique(groupings[gn], return_counts=True)
            f.write(f'grouping {gn}: {len(vals)} groups\n')
    print(f'\nframes {n_frames}; risk check {int(rc.identical.sum()) if len(rc) else 0}/{len(rc)} identical; '
          f'done in {time.time() - t0:.0f} s; outputs in {os.path.abspath(a.out_dir)}')


if __name__ == '__main__':
    main()
