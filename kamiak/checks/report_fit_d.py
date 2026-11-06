"""Read-only (user, 2026-11-05, item 1, exploratory): the numbers of code/new_cues_fit.py's real run (fit_d/main_1084,
sensitivity_1123) put together for the report, with the reading rules written before the results:
  new cues useful (linear, 20 m main): D - B wholly above 0 in at least 4 of the 5 inputs;
  interactions useful (D): boost_full - boost_add wholly above 0 in at least 4 of 5; otherwise "not seen";
  proxy (20 m main): the D - B difference of the ego-speed coefficient: containing 0 = no change seen; not containing 0
    and smaller in size: D's interval containing 0 = mostly a proxy, not = partly; larger = reported as it is; a
    conclusion only when at least 4 of 5 inputs agree, otherwise "inconsistent";
  share of the ceiling: out-of-sample R2 / ceiling, the highest of the three models (linear R20, tree, boost_full).
No answer text is read. usage: python -I report_fit_d.py <fit_d folder>"""
import os
import sys

import numpy as np
import pandas as pd

O = sys.argv[1]
CONDS = ['grid', 'sep_time', 'sep_camera', 'ctrl_grid', 'ctrl_sep']
MODELS = ['linear_R10', 'linear_R20', 'linear_R30', 'linear_R50', 'tree', 'boost_full', 'boost_add']
f3 = lambda v: 'nan' if pd.isna(v) else f'{v:+.3f}'  # noqa: E731
iv = lambda p, lo, hi: f'{f3(p)} [{f3(lo)}, {f3(hi)}]'  # noqa: E731


def load(sub):
    d = os.path.join(O, sub)
    rd = lambda n: pd.read_csv(os.path.join(d, f'exploratory_d_{n}.csv'), float_precision='round_trip')  # noqa: E731
    return {n: rd(n) for n in ('oos', 'differences', 'first_cuts', 'ego_coefficient', 'within', 'tau_b',
                               'partial_dependence', 'beside')}, open(os.path.join(d, 'exploratory_d_params.txt'),
                                                                      encoding='utf-8').read()


def classify(r):
    if r.diff_lo <= 0 <= r.diff_hi:
        return 'no change seen'
    if abs(r.ego_D) < abs(r.ego_B):
        return 'mostly a proxy' if r.ego_D_lo <= 0 <= r.ego_D_hi else 'partly a proxy'
    return 'larger in D'


def verdict(labels):
    vc = pd.Series(labels).value_counts()
    return vc.index[0] if vc.iloc[0] >= 4 else f'inconsistent ({dict(vc)})'


summary = {}
for sub in ('main_1084', 'sensitivity_1123'):
    T, par = load(sub)
    print(f'\n################ {sub}')
    for line in par.splitlines():
        if line.startswith(('frames ', 'check', 'resamples', 'control_crossing', 'new_cues =')):
            print('  ' + line)
    oos = T['oos'].set_index(['condition', 'fit', 'cues'])
    nf = int(T['oos'].frames.iloc[0])
    print(f'\n== out-of-sample R2 on the {nf} fit frames [95 % interval]')
    for c in CONDS:
        print(f'  {c}')
        for m in MODELS:
            print(f'    {m:11s} ' + '  '.join(f'{s}: {iv(*oos.loc[(c, m, s), ["r2_oos", "r2_oos_lo", "r2_oos_hi"]])}'
                                            for s in ('B', 'D', 'Dn')) +
                  f'   (in sample B {oos.loc[(c, m, "B"), "r2_in"]:.3f}, D {oos.loc[(c, m, "D"), "r2_in"]:.3f})')
    df = T['differences'].set_index(['condition', 'comparison'])
    comps = [f'{k} - B, {m}' for k in ('D', 'Dn') for m in MODELS] + [f'D - Dn, {m}' for m in MODELS] + \
        [f'boost_full - {x}, {s}' for s in ('B', 'D') for x in ('tree', 'linear_R20', 'boost_add')]
    print('\n== differences, same resamples (* = interval wholly above 0, - = wholly below 0)')
    cnt = {}
    for comp in comps:
        rs = [df.loc[(c, comp)] for c in CONDS]
        cnt[comp] = sum(bool(r.above_0) for r in rs)
        print(f'  {comp:28s} above 0 in {cnt[comp]}/5: ' + '; '.join(
            f'{c} {iv(r["diff"], r.lo, r.hi)}{"*" if r.above_0 else ("-" if r.below_0 else "")}'
            for c, r in zip(CONDS, rs)))
    useful = cnt['D - B, linear_R20'] >= 4
    inter = cnt['boost_full - boost_add, D'] >= 4
    print(f'\n  READING new cues (linear R20): D - B above 0 in {cnt["D - B, linear_R20"]}/5 -> '
          f'{"useful" if useful else "not shown useful"}; sensitivity radii: ' +
          ', '.join(f'R{r} {cnt[f"D - B, linear_R{r}"]}/5' for r in (10, 30, 50)) +
          f'; tree {cnt["D - B, tree"]}/5, boost_full {cnt["D - B, boost_full"]}/5, boost_add {cnt["D - B, boost_add"]}/5')
    print(f'  READING interactions (D, full - additive): above 0 in {cnt["boost_full - boost_add, D"]}/5 -> '
          f'{"useful" if inter else "not seen (which is not the same as none)"}; on B '
          f'{cnt["boost_full - boost_add, B"]}/5')
    bes = T['beside']
    print('\n== the ceiling on the fit frames and shares (out-of-sample R2 / ceiling)')
    shares = {}
    for c in CONDS:
        q = bes[bes.condition == c]
        ce = q[q.measure.str.startswith('ceiling')].iloc[0]
        r2 = {m: oos.loc[(c, m, 'D'), 'r2_oos'] for m in ('linear_R20', 'tree', 'boost_full', 'boost_add')}
        r2b = {m: oos.loc[(c, m, 'B'), 'r2_oos'] for m in ('linear_R20', 'tree', 'boost_full', 'boost_add')}
        best = max(('linear_R20', 'tree', 'boost_full'), key=lambda m: r2[m])
        bestb = max(('linear_R20', 'tree', 'boost_full'), key=lambda m: r2b[m])
        shares[c] = (ce.value, best, r2[best] / ce.value, r2['boost_add'] / ce.value)
        print(f'  {c}: {ce.measure} {ce.value:.3f} [{ce.lo:.3f}, {ce.hi:.3f}]; D: ' +
              ', '.join(f'{m} {r2[m] / ce.value:.3f}' for m in r2) +
              f'; highest of the three: {best} {r2[best]:.3f} = {r2[best] / ce.value:.3f} of the ceiling; '
              f'B highest {bestb} {r2b[bestb]:.3f} = {r2b[bestb] / ce.value:.3f}')
        for _, r in q[~q.measure.str.startswith('ceiling')].iterrows():
            print(f'      beside: {r.measure}: {r.value:.3f} [{r.lo:.3f}, {r.hi:.3f}]')
    print('\n== D additive boosting: share of the ceiling and how much lower than the full model')
    for c in CONDS:
        r = df.loc[(c, 'boost_full - boost_add, D')]
        print(f'  {c}: additive {oos.loc[(c, "boost_add", "D"), "r2_oos"]:.3f} = {shares[c][3]:.3f} of the ceiling; '
              f'full {oos.loc[(c, "boost_full", "D"), "r2_oos"]:.3f}; full - additive {iv(r["diff"], r.lo, r.hi)}')
    eg = T['ego_coefficient']
    print('\n== proxy test: ego-speed coefficient (linear, ranks), B and D, D - B, refitted per resample')
    verdicts = {}
    for R in (20, 10, 30, 50):
        labs = []
        for c in CONDS:
            r = eg[(eg.condition == c) & (eg.R == R)].iloc[0]
            labs.append(classify(r))
            if R == 20:
                print(f'  R20 {c}: B {iv(r.ego_B, r.ego_B_lo, r.ego_B_hi)}, D {iv(r.ego_D, r.ego_D_lo, r.ego_D_hi)}, '
                      f'D - B {iv(r.diff_D_minus_B, r.diff_lo, r.diff_hi)} -> {labs[-1]}')
        verdicts[R] = verdict(labs)
        print(f'  R{R} verdict: {verdicts[R]}' + ('' if R == 20 else f'  ({", ".join(labs)})'))
    fc = T['first_cuts']
    print('\n== resampled trees, share of first cuts on true_ego_speed (fake noise reference 0.043-0.092)')
    for c in CONDS:
        out = []
        for s in ('B', 'D'):
            q = fc[(fc.condition == c) & (fc.cues == s)]
            out.append(f'{s} {q[q.first_cue == "true_ego_speed"].share.sum():.3f} (most frequent {q.iloc[0].first_cue} '
                       f'{q.iloc[0].share:.3f})')
        print(f'  {c}: ' + '; '.join(out))
    tb = T['tau_b']
    print('\n== single-cue tau-b with the score (* = interval excludes 0); all / within segment')
    for cue in tb.cue.unique():
        q = tb[tb.cue == cue]
        parts = []
        for c in CONDS:
            a_ = q[(q.condition == c) & (q.grouping == 'all')].iloc[0]
            s_ = q[(q.condition == c) & (q.grouping == 'segment')].iloc[0]
            m = lambda r: '*' if (r.tau_b_lo > 0 or r.tau_b_hi < 0) else ''  # noqa: E731
            parts.append(f'{c} {a_.tau_b:+.3f}{m(a_)}/{s_.tau_b:+.3f}{m(s_)}')
        print(f'  {cue:28s} ({int(q.frames.iloc[0])} fr) ' + '  '.join(parts))
    wi = T['within']
    print('\n== within segments (descriptive): in-sample R2; coefficients whose interval excludes 0')
    for c in CONDS:
        q = wi[wi.condition == c]
        r2 = q[q.term == 'R2 in sample'].set_index('fit')
        print(f'  {c}: ' + '; '.join(f'{f_} R2 {iv(r.value, r.lo, r.hi)}' for f_, r in r2.iterrows()))
        for f_ in r2.index:
            z = q[(q.fit == f_) & (q.term != 'R2 in sample') & (q.term != 'intercept') & ((q.lo > 0) | (q.hi < 0))]
            print(f'      {f_}: ' + (', '.join(f'{r.term} {iv(r.value, r.lo, r.hi)}' for _, r in z.iterrows()) or 'none'))
    pdp = T['partial_dependence']
    print('\n== D additive boosting, partial dependence (mean predicted rank score) at the 10/30/50/70/90 % quantiles; '
          'cues ordered by range, range >= 0.01 shown')
    for c in CONDS:
        q = pdp[pdp.condition == c]
        rng = q.groupby('cue').partial_dependence.agg(lambda v: v.max() - v.min()).sort_values(ascending=False)
        print(f'  {c}:')
        for cue, rg in rng.items():
            if rg < 0.01:
                continue
            z = q[q.cue == cue].sort_values('quantile')
            print(f'    {cue:26s} range {rg:.3f}: ' + ', '.join(f'{v:.4g}->{p:.3f}' for v, p in
                                                             zip(z.value, z.partial_dependence)))
        print(f'    ({int((rng < 0.01).sum())} cues with range < 0.01, {int((rng == 0).sum())} of them flat)')
    summary[sub] = (oos, df, verdicts, cnt, shares)

print('\n################ sensitivity against main')
(om, dm, vm, cm, sm), (os_, ds, vs, cs, ss) = summary['main_1084'], summary['sensitivity_1123']
d1 = (os_.r2_oos - om.r2_oos).abs()
print(f'  out-of-sample R2: largest |sensitivity - main| {d1.max():.3f} at {d1.idxmax()}; median {d1.median():.3f}')
d2 = (ds['diff'] - dm['diff']).abs()
print(f'  differences: largest |sensitivity - main| {d2.max():.3f} at {d2.idxmax()}')
chg = [k for k in cm if (cm[k] >= 4) != (cs[k] >= 4)]
print(f'  comparisons whose 4-of-5 reading differs: {chg if chg else "none"}; counts that differ: ' +
      ', '.join(f'{k} {cm[k]}->{cs[k]}' for k in cm if cm[k] != cs[k]))
print(f'  proxy verdicts main {vm} / sensitivity {vs}')
print('  shares of the ceiling (highest of three) main / sensitivity: ' +
      ', '.join(f'{c} {sm[c][2]:.3f} ({sm[c][1]}) / {ss[c][2]:.3f} ({ss[c][1]})' for c in CONDS))
