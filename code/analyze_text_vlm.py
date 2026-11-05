"""DRAFT (user, 2026-11-05, item 5; the analysis plan in CLAUDE.md, marked "起草中，用户还没定"): analysis of the
described-situation replies of code/run_text_vlm.py. Not run on real replies yet (none exist); tested on fake replies.

Scores: per request the parsed risk of the reply, or of its forced answer when the reply stopped at max_tokens without
one (as analyze_vlm_ssm.py); per combination the mean of its scores (combinations with fewer than 5 counted).
 1 factors: per level of X (ego speed), Z (lead speed), Y (distance), traffic (many / few), weather (rain / dry), the mean
   score over the replies of the 500 combinations with a scene; X, Z, Y also over the 125 without one. Interval: 2,000
   resamples of each combination's replies with replacement (seed BOOT_SEED), the means taken anew.
 2 ceiling: as score_ceiling.py on the 500 combinations with a scene and 5 scores (each repeat ranked among them, the mean
   r of the 10 correlations, 5 r / (1 + 4 r)); interval: 2,000 resamples of the combinations with replacement.
 3 rank regression as the rule analysis (analyze_rules.ols, ranks / n, ties the mean rank) on the 500: the rank of the
   combination's mean score on the ranks of X, Z, Y and the 0/1 many and rain; a second version with the closing speed
   X - Z instead of Z. Coefficients and R2 in sample; coefficient and R2 intervals from the combination resamples; R2 by
   10-fold cross-validation over the combinations (KFold shuffled, seed 0) beside it. The 125 without a scene: per
   kinematic combination its mean score minus the mean of its 4 scene versions (mean and interval over the
   combinations' resamples), and the same regression without many and rain on the 125 beside that on the 500.
 4 SSMs from the design (waymo_ssm_frames.lead_ssms, the project's parameters; accelerations 0; gap = Y - the median
   camera offset of the 1,084 frames when Y is measured from the front camera): TTC, THW, DRAC, PSD, PICUD "larger =
   riskier"; not closing in: TTC infinite and DRAC 0, the safe end (tied there); tau-b of the combination mean score with
   each over the 500 (and the 125), interval from the combination resamples.
 5 tree: DecisionTreeRegressor(max_depth 3, min_samples_leaf 50, random_state 0) on X, Z, Y, many, rain (their values),
   the rank of the mean score, 10-fold cross-validated R2 over the 500 combinations (KFold shuffled, seed 0); the tree on
   all of them written out.
 6 directions beside the pictures (copied): X's coefficient here against the true ego speed's in K (exploratory_rule_
   coefficients.csv, qwen3.6-27b, each condition); many against tau-b of the risk score with the vehicle and pedestrian
   + cyclist counts within 20 m (exploratory_cue_tau_b.csv, grouping all); rain against weather=rain in K+S+N20.
Outputs (out_dir): exploratory_text_*.csv, exploratory_text_tree.txt, exploratory_text_params.txt.
usage: python -I analyze_text_vlm.py <replies.jsonl> <design.csv of run_text_vlm.py> <camera offset (m) or 0> <out_dir>
       [--rule_dir <analysis/rule_explore/main_1084>] [--n_boot 2000]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_rules as ar  # noqa: E402
import analyze_vlm_ssm as av  # noqa: E402
from score_ceiling import pair_corr, spearman_brown  # noqa: E402
from waymo_ssm_frames import lead_ssms  # noqa: E402

BOOT_SEED = 20261105
SSM5 = ['TTC', 'THW', 'DRAC', 'PSD', 'PICUD']
K_REP = 5


def load(path):
    """Final score per (combo, repeat): the reply's parsed risk, else its forced answer's (<stem>_forced_answers.jsonl)."""
    s, n_forced = {}, 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            o = json.loads(line)
            p = o.get('parsed')
            s[(o['combo'], o['repeat'])] = float(p['risk']) if p else np.nan
    fp = path[:-len('.jsonl')] + '_forced_answers.jsonl'
    if os.path.exists(fp):
        with open(fp, encoding='utf-8') as f:
            for line in f:
                o = json.loads(line)
                k = (o['combo'], o['repeat'])
                if not np.isnan(s[k]):
                    raise ValueError(f'{fp}: forced answer for {k}, but the reply has one')
                p = o.get('parsed')
                if p:
                    s[k] = float(p['risk'])
                    n_forced += 1
    return s, n_forced


def tau(x, y):
    return float(kendalltau(x, y, variant='b').statistic)


def ssm_risk(d, offset):
    out = {k: np.empty(len(d)) for k in SSM5}
    for i, r in enumerate(d.itertuples()):
        o = lead_ssms(r.y - offset, r.x, r.z, 0.0, 0.0)
        for k in SSM5:
            out[k][i] = o[k]
    risk = {}
    for k in SSM5:
        v = out[k]
        conflict = (v > 0) if k in av.POSITIVE_IS_CONFLICT else np.isfinite(v)
        r = np.where(conflict, -v if k in av.SMALLER_RISKIER else v, np.nan)
        risk[k] = np.where(conflict, r, (np.nanmin(r) - 1.0) if conflict.any() else 0.0)   # not closing: the safe end
        out[k + '_tied_safe'] = int((~conflict).sum())
    return out, risk


def cv_r2(X, y, fit_predict):
    pred = np.empty(len(y))
    for tr, te in KFold(10, shuffle=True, random_state=0).split(X):
        pred[te] = fit_predict(X[tr], y[tr], X[te])
    return 1.0 - float(((y - pred) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    for name in ('replies', 'design', 'offset', 'out_dir'):
        ap.add_argument(name)
    ap.add_argument('--rule_dir', default=None)
    ap.add_argument('--n_boot', type=int, default=2000)
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out_dir, 'exploratory_text_params.txt')):
        sys.exit(f'{a.out_dir} has exploratory_text_params.txt already')
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    d = pd.read_csv(a.design)
    d['x'], d['z'], d['y'] = d.x_text.astype(float), d.z_text.astype(float), d.y_text.astype(float)   # as written
    s, n_forced = load(a.replies)
    S = np.array([[s.get((c, k), np.nan) for k in range(K_REP)] for c in d.combo])
    n_have = (~np.isnan(S)).sum(1)
    d['score'] = np.nanmean(np.where(np.isnan(S), np.nan, S), axis=1)
    d['many'] = (d.traffic == 'many').astype(float)
    d['rain'] = (d.weather == 'rain').astype(float)
    d['closing_speed'] = d.x - d.z
    rng = np.random.default_rng(BOOT_SEED)
    sc = d[d.scene_said].reset_index(drop=True)
    ns = d[~d.scene_said].reset_index(drop=True)
    Ssc, Sns = S[d.scene_said.to_numpy()], S[~d.scene_said.to_numpy()]
    info = [f'requests {S.size}; with a score {int((~np.isnan(S)).sum())} (forced answers {n_forced}); combinations with '
            f'fewer than {K_REP} scores {int((n_have < K_REP).sum())}']
    # --- 1 factors: replies resampled within each combination
    pick = [rng.integers(0, K_REP, (a.n_boot, len(d), K_REP))]
    Sb = np.take_along_axis(np.broadcast_to(S, (a.n_boot,) + S.shape), pick[0], axis=2)   # n_boot x combos x 5
    rows = []
    for part, mask in (('with a scene (500)', d.scene_said.to_numpy()), ('without a scene (125)', ~d.scene_said.to_numpy())):
        for f, col in (('X ego speed', 'x_level'), ('Z lead speed', 'z_level'), ('Y distance', 'y_level'),
                       ('traffic', 'traffic'), ('weather', 'weather')):
            if part.startswith('without') and f in ('traffic', 'weather'):
                continue
            for lv in sorted(d.loc[mask, col].unique(), key=str):
                m = mask & (d[col] == lv).to_numpy()
                point = float(np.nanmean(S[m]))
                b = np.nanmean(Sb[:, m, :].reshape(a.n_boot, -1), axis=1)
                lo, hi, nv = ar.ci(b)
                rows.append(dict(part=part, factor=f, level=lv, value=d.loc[m, col.replace('_level', '') + '_text'].iloc[0]
                                 if col.endswith('_level') else lv, combinations=int(m.sum()), mean_score=point,
                                 lo=lo, hi=hi, nboot=nv))
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, 'exploratory_text_factors.csv'), index=False)
    # --- combination resamples for 2-4
    cb = [rng.integers(0, len(sc), len(sc)) for _ in range(a.n_boot)]
    full = ~np.isnan(Ssc).any(1)

    def ceil_of(idx):
        i = idx[full[idx]]
        return spearman_brown(pair_corr(np.column_stack([ar.rank01(Ssc[i, k]) for k in range(K_REP)]))[1])

    ce = ceil_of(np.arange(len(sc)))
    ceb = np.array([ceil_of(i) for i in cb])
    lo, hi, nv = ar.ci(ceb)
    out = [dict(what='ceiling 5r/(1+4r), 500 combinations with a scene', value=ce, lo=lo, hi=hi, nboot=nv)]
    designs = {'X, Z, Y, many, rain': ['x', 'z', 'y', 'many', 'rain'],
               'X, closing speed, Y, many, rain': ['x', 'closing_speed', 'y', 'many', 'rain']}

    def reg(df, cols, idx=None):
        q = df if idx is None else df.iloc[idx]
        y = ar.rank01(q.score.to_numpy(float))
        X = [ar.rank01(q[c].to_numpy(float)) if c not in ('many', 'rain') else q[c].to_numpy(float) for c in cols]
        return ar.ols(y, X)

    coef_rows = []
    for name, cols in designs.items():
        b0, _, r2 = reg(sc, cols)
        bb = np.array([np.concatenate([reg(sc, cols, i)[0], [reg(sc, cols, i)[2]]]) for i in cb])
        y = ar.rank01(sc.score.to_numpy(float))
        Xr = np.column_stack([ar.rank01(sc[c].to_numpy(float)) if c not in ('many', 'rain') else sc[c].to_numpy(float)
                              for c in cols])
        r2cv = cv_r2(Xr, y, lambda Xa, ya, Xb: np.column_stack([np.ones(len(Xb)), Xb]) @
                     np.linalg.lstsq(np.column_stack([np.ones(len(Xa)), Xa]), ya, rcond=None)[0])
        for j, term in enumerate(['intercept'] + cols):
            lo, hi, nv = ar.ci(bb[:, j])
            coef_rows.append(dict(fit=name, term=term, coef=b0[j], lo=lo, hi=hi, nboot=nv))
        lo, hi, nv = ar.ci(bb[:, -1])
        out.append(dict(what=f'rank regression {name}: R2 in sample', value=r2, lo=lo, hi=hi, nboot=nv))
        out.append(dict(what=f'rank regression {name}: R2 10-fold cross-validated', value=r2cv, lo=np.nan, hi=np.nan,
                        nboot=0))
    pd.DataFrame(coef_rows).to_csv(os.path.join(a.out_dir, 'exploratory_text_regression.csv'), index=False)
    # the 125 without a scene against their 4 scene versions
    key = ['x_level', 'z_level', 'y_level']
    m4 = sc.groupby(key).score.mean().rename('scene_mean').reset_index()
    j = ns.merge(m4, on=key, validate='one_to_one')
    diff = (j.score - j.scene_mean).to_numpy(float)
    nb_ = [rng.integers(0, len(j), len(j)) for _ in range(a.n_boot)]
    lo, hi, nv = ar.ci(np.array([diff[i].mean() for i in nb_]))
    out.append(dict(what='without a scene minus the mean of the 4 scene versions (125)', value=float(diff.mean()), lo=lo,
                    hi=hi, nboot=nv))
    for nm, df_ in (('500 with a scene', sc), ('125 without', ns)):
        b_, _, r2_ = reg(df_, ['x', 'z', 'y'])
        out.append(dict(what=f'rank regression X, Z, Y on the {nm}: R2 in sample; coefficients '
                             + ', '.join(f'{t} {v:+.3f}' for t, v in zip(['intercept', 'x', 'z', 'y'], b_)),
                        value=r2_, lo=np.nan, hi=np.nan, nboot=0))
    # --- 4 SSMs
    off = float(a.offset)
    raw_sc, risk_sc = ssm_risk(sc, off)
    raw_ns, risk_ns = ssm_risk(ns, off)
    ssm_rows = []
    for k in SSM5:
        t_ = tau(sc.score, risk_sc[k])
        tb = np.array([tau(sc.score.to_numpy()[i], risk_sc[k][i]) for i in cb])
        lo, hi, nv = ar.ci(tb)
        ssm_rows.append(dict(ssm=k, part='500 with a scene', tau_b=t_, lo=lo, hi=hi, nboot=nv,
                             tied_at_the_safe_end=raw_sc[k + '_tied_safe']))
        ssm_rows.append(dict(ssm=k, part='125 without', tau_b=tau(ns.score, risk_ns[k]), lo=np.nan, hi=np.nan, nboot=0,
                             tied_at_the_safe_end=raw_ns[k + '_tied_safe']))
    pd.DataFrame(ssm_rows).to_csv(os.path.join(a.out_dir, 'exploratory_text_ssm.csv'), index=False)
    # --- 5 tree
    cols = ['x', 'z', 'y', 'many', 'rain']
    Xt = sc[cols].to_numpy(float)
    yt = ar.rank01(sc.score.to_numpy(float))
    grow = lambda Xa, ya: DecisionTreeRegressor(max_depth=3, min_samples_leaf=50, random_state=0).fit(Xa, ya)  # noqa
    r2t = cv_r2(Xt, yt, lambda Xa, ya, Xb: grow(Xa, ya).predict(Xb))
    tr = grow(Xt, yt)
    out.append(dict(what='tree (max_depth 3, 50 per leaf) on X, Z, Y, many, rain: R2 10-fold cross-validated', value=r2t,
                    lo=np.nan, hi=np.nan, nboot=0))
    t = tr.tree_
    lines = []

    def walk(jn, ind, path):
        if t.children_left[jn] == -1:
            lines.append('    ' * ind + f'leaf: {path or "all"} -> {int(t.n_node_samples[jn])} combinations, mean rank '
                                       f'{float(t.value[jn].ravel()[0]):.3f}')
            return
        c_, th = cols[t.feature[jn]], float(t.threshold[jn])
        lines.append('    ' * ind + f'cut {c_} <= {th:.3f} ({int(t.n_node_samples[jn])} combinations)')
        walk(t.children_left[jn], ind + 1, f'{path} & {c_} <= {th:.3f}'.strip(' &'))
        walk(t.children_right[jn], ind + 1, f'{path} & {c_} > {th:.3f}'.strip(' &'))
    walk(0, 0, '')
    open(os.path.join(a.out_dir, 'exploratory_text_tree.txt'), 'w').write('\n'.join(lines) + '\n')
    pd.DataFrame(out).to_csv(os.path.join(a.out_dir, 'exploratory_text_summary.csv'), index=False)
    # --- 6 directions beside the pictures
    side = []
    xb = [r for r in coef_rows if r['fit'] == 'X, Z, Y, many, rain']
    get = {r['term']: r for r in xb}
    for term, what in (('x', 'ego speed'), ('many', 'many vehicles and pedestrians'), ('rain', 'rain')):
        side.append(dict(cue=what, source='described situations (this run)', condition='', value=get[term]['coef'],
                         lo=get[term]['lo'], hi=get[term]['hi']))
    if a.rule_dir:
        co = pd.read_csv(os.path.join(a.rule_dir, 'exploratory_rule_coefficients.csv'))
        co = co[(co.target == 'risk') & (co.model == 'qwen3.6-27b')]
        cu = pd.read_csv(os.path.join(a.rule_dir, 'exploratory_cue_tau_b.csv'))
        cu = cu[(cu.model == 'qwen3.6-27b') & (cu.grouping == 'all')]
        for c in av.CONDITIONS:
            q = co[(co.condition == c) & (co.fit == 'K') & (co.term == 'true_ego_speed')].iloc[0]
            side.append(dict(cue='ego speed', source='pictures: K coefficient of the true ego speed', condition=c,
                             value=q.coef, lo=q.coef_lo, hi=q.coef_hi))
            for nm in ('vehicle_20m', 'ped_cyc_20m'):
                q = cu[(cu.condition == c) & (cu.cue == nm)].iloc[0]
                side.append(dict(cue='many vehicles and pedestrians', source=f'pictures: tau-b with {nm}', condition=c,
                                 value=q.tau_b, lo=q.tau_b_lo, hi=q.tau_b_hi))
            q = co[(co.condition == c) & (co.fit == 'K+S+N20') & (co.term == 'weather=rain')].iloc[0]
            side.append(dict(cue='rain', source='pictures: K+S+N20 coefficient of weather=rain', condition=c,
                             value=q.coef, lo=q.coef_lo, hi=q.coef_hi))
    pd.DataFrame(side).to_csv(os.path.join(a.out_dir, 'exploratory_text_directions.csv'), index=False)
    with open(os.path.join(a.out_dir, 'exploratory_text_params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'DRAFT analysis of the described situations (user, 2026-11-05, item 5): analyze_text_vlm.py run '
                f'{time.strftime("%Y-%m-%d %H:%M:%S")}\nreplies = {os.path.abspath(a.replies)}\ndesign = '
                f'{os.path.abspath(a.design)}\ncamera offset used for the gap: {off} m\nresamples {a.n_boot}, seed '
                f'{BOOT_SEED}\n')
        for x in info:
            f.write(x + '\n')
    print(f'done in {time.time() - t0:.0f} s; ' + '; '.join(info))


if __name__ == '__main__':
    main()
