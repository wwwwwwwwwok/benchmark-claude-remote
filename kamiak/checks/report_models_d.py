"""Read-only (user, 2026-11-05, item 3, exploratory): D for every model (new_cues_fit.py --model, main result, 1,065 fit
frames) with the reading rules of 27B's run (report_fit_d.py): new cues useful = linear R20 D - B above 0 in at least
4 of 5; interactions useful = D's boost_full - boost_add above 0 in at least 4 of 5; proxy (R20): the D - B difference of
the ego-speed coefficient, a conclusion only when 4 of 5 agree; share of the ceiling: the highest of linear R20, the
tree and boost_full, with the model's own ceiling on the fit frames. The earlier K+S+N20(+H) rows (1,084 frames) are
out-of-sample R2, compared with D's out-of-sample R2 only (no 1,084-frame ceiling for these models).
usage: python -I report_models_d.py <new_cues folder> (fit_d_pred/, fit_d_models/<model>/, fit_d_models/compare/)"""
import os
import sys

import pandas as pd

O = sys.argv[1]
CONDS = ['grid', 'sep_time', 'sep_camera', 'ctrl_grid', 'ctrl_sep']
MODELS = ['qwen3.6-27b', 'qwen3-vl-32b', 'qwen3-vl-32b-thinking', 'internvl3.5-38b', 'internvl3.5-38b-thinking',
          'qwen-drive-1.0-4b', 'qwen3.5-4b']
f3 = lambda v: 'nan' if pd.isna(v) else f'{v:+.3f}'  # noqa: E731
iv = lambda p, lo, hi: f'{f3(p)} [{f3(lo)}, {f3(hi)}]'  # noqa: E731


def folder(m):
    return os.path.join(O, 'fit_d_pred', 'main_1084') if m == 'qwen3.6-27b' else \
        os.path.join(O, 'fit_d_models', m, 'main_1084')


def classify(r):
    if r.diff_lo <= 0 <= r.diff_hi:
        return 'no change seen'
    if abs(r.ego_D) < abs(r.ego_B):
        return 'mostly a proxy' if r.ego_D_lo <= 0 <= r.ego_D_hi else 'partly a proxy'
    return 'larger in D'


def verdict(labels):
    vc = pd.Series(labels).value_counts()
    return vc.index[0] if vc.iloc[0] >= 4 else f'inconsistent ({", ".join(f"{k} {v}" for k, v in vc.items())})'


summary = []
for m in MODELS:
    d = folder(m)
    rd = lambda n: pd.read_csv(os.path.join(d, f'exploratory_d_{n}.csv'), float_precision='round_trip')  # noqa: E731
    par = open(os.path.join(d, 'exploratory_d_params.txt'), encoding='utf-8').read()
    oos = rd('oos').set_index(['condition', 'fit', 'cues'])
    df = rd('differences').set_index(['condition', 'comparison'])
    bes, eg, fc, tb, wi = rd('beside'), rd('ego_coefficient'), rd('first_cuts'), rd('tau_b'), rd('within')
    print(f'\n################ {m}  ({d})')
    print('  ' + [x for x in par.splitlines() if x.startswith('checks not met')][0] + '; ' +
          '; '.join(x[len('check: '):] for x in par.splitlines() if x.startswith('check: linear')))
    cnt = {k[1]: 0 for k in df.index}
    for (c, comp), r in df.iterrows():
        cnt[comp] += bool(r.above_0)
    print('  out-of-sample R2, B -> D [D interval]:')
    for c in CONDS:
        print(f'    {c:10s} ' + '  '.join(
            f'{f_} {oos.loc[(c, f_, "B"), "r2_oos"]:.3f} -> {oos.loc[(c, f_, "D"), "r2_oos"]:.3f} '
            f'[{oos.loc[(c, f_, "D"), "r2_oos_lo"]:.3f}, {oos.loc[(c, f_, "D"), "r2_oos_hi"]:.3f}]'
            for f_ in ('linear_R20', 'tree', 'boost_full', 'boost_add')))
    print('  D - B linear R20: ' + '; '.join(f'{c} {iv(*df.loc[(c, "D - B, linear_R20"), ["diff", "lo", "hi"]])}'
                                             f'{"*" if df.loc[(c, "D - B, linear_R20"), "above_0"] else ""}'
                                             for c in CONDS))
    print('  above 0 in x/5: D - B ' + ', '.join(f'{f_} {cnt[f"D - B, {f_}"]}' for f_ in
                                                ('linear_R10', 'linear_R20', 'linear_R30', 'linear_R50', 'tree',
                                                 'boost_full', 'boost_add')) +
          f' | Dn - B R20 {cnt["Dn - B, linear_R20"]}, D - Dn R20 {cnt["D - Dn, linear_R20"]}, Dn - B boost_full '
          f'{cnt["Dn - B, boost_full"]}, D - Dn boost_full {cnt["D - Dn, boost_full"]}')
    print('  boosting - other, above 0 in x/5: ' + ', '.join(
        f'{x} on {s} {cnt[f"boost_full - {x}, {s}"]}' for s in ('B', 'D') for x in ('tree', 'linear_R20', 'boost_add')))
    print('  D full - additive: ' + '; '.join(f'{c} {iv(*df.loc[(c, "boost_full - boost_add, D"), ["diff", "lo", "hi"]])}'
                                              for c in CONDS))
    useful, inter = cnt['D - B, linear_R20'] >= 4, cnt['boost_full - boost_add, D'] >= 4
    print(f'  READING new cues: {"useful" if useful else "not shown useful"} ({cnt["D - B, linear_R20"]}/5); '
          f'interactions: {"useful" if inter else "not seen (not the same as none)"} '
          f'({cnt["boost_full - boost_add, D"]}/5)')
    shares = []
    for c in CONDS:
        q = bes[bes.condition == c]
        ce = q[q.measure.str.startswith('ceiling')].iloc[0]
        best = max(('linear_R20', 'tree', 'boost_full'), key=lambda f_: oos.loc[(c, f_, 'D'), 'r2_oos'])
        bestb = max(('linear_R20', 'tree', 'boost_full'), key=lambda f_: oos.loc[(c, f_, 'B'), 'r2_oos'])
        h = q[q.measure.str.contains(r'K\+S\+N20\+H')].iloc[0]
        k = q[q.measure.str.contains('K\\+S\\+N20 linear')].iloc[0]
        shares.append(dict(c=c, ceil=ce.value, ceil_m=ce.measure, best=best, d=oos.loc[(c, best, 'D'), 'r2_oos'],
                           b=oos.loc[(c, bestb, 'B'), 'r2_oos'], bestb=bestb, add=oos.loc[(c, 'boost_add', 'D'), 'r2_oos'],
                           h=h.value, k=k.value))
    print('  ceiling on the fit frames: ' + '; '.join(f'{s["c"]} {s["ceil"]:.3f} ({s["ceil_m"].split("(")[1][:-1]})'
                                                       for s in shares))
    print('  share of the ceiling (highest of three, D | B | D additive): ' + '; '.join(
        f'{s["c"]} {s["d"] / s["ceil"]:.3f} ({s["best"]}) | {s["b"] / s["ceil"]:.3f} | {s["add"] / s["ceil"]:.3f}'
        for s in shares))
    print('  out-of-sample R2 (D best | D additive | earlier K+S+N20 | earlier K+S+N20+H, 1,084 frames): ' + '; '.join(
        f'{s["c"]} {s["d"]:.3f} | {s["add"]:.3f} | {s["k"]:.3f} | {s["h"]:.3f}' for s in shares))
    print(f'  D best below K+S+N20+H (out-of-sample R2): {sum(s["d"] < s["h"] for s in shares)}/5; D additive: '
          f'{sum(s["add"] < s["h"] for s in shares)}/5')
    labs20 = []
    for R in (20, 10, 30, 50):
        labs = [classify(eg[(eg.condition == c) & (eg.R == R)].iloc[0]) for c in CONDS]
        if R == 20:
            labs20 = labs
            print('  proxy R20: ' + '; '.join(
                f'{c} B {r.ego_B:+.3f}, D {r.ego_D:+.3f}, D - B {iv(r.diff_D_minus_B, r.diff_lo, r.diff_hi)}'
                for c, r in ((c, eg[(eg.condition == c) & (eg.R == 20)].iloc[0]) for c in CONDS)))
        print(f'    R{R} verdict: {verdict(labs)}')
    print('  first cut on true_ego_speed (B / D; noise 0.043-0.092): ' + '; '.join(
        f'{c} ' + ' / '.join(f'{fc[(fc.condition == c) & (fc.cues == s) & (fc.first_cue == "true_ego_speed")].share.sum():.3f}'
                             for s in ('B', 'D')) for c in CONDS))
    print('  single-cue tau-b, conditions with the interval excluding 0 (all / segment), range of the all ones:')
    for cue in tb.cue.unique():
        q = tb[tb.cue == cue]
        a_, s_ = q[q.grouping == 'all'], q[q.grouping == 'segment']
        na = int(((a_.tau_b_lo > 0) | (a_.tau_b_hi < 0)).sum())
        ns = int(((s_.tau_b_lo > 0) | (s_.tau_b_hi < 0)).sum())
        print(f'    {cue:28s} {na}/{ns}  {a_.tau_b.min():+.3f} to {a_.tau_b.max():+.3f}')
    r2w = wi[wi.term == 'R2 in sample']
    print('  within segments, in-sample R2 (K+N20 | + new): ' + '; '.join(
        f'{c} {r2w[(r2w.condition == c) & (r2w.fit == "within: K+N20")].value.iloc[0]:.3f} | '
        f'{r2w[(r2w.condition == c) & (r2w.fit == "within: K+N20 split + new")].value.iloc[0]:.3f}' for c in CONDS))
    z = wi[(wi.fit == 'within: K+N20 split + new') & ~wi.term.isin(['R2 in sample', 'intercept']) &
           ((wi.lo > 0) | (wi.hi < 0))]
    print('    coefficients with the interval excluding 0 (+ new): ' + ', '.join(
        f'{t} {n}/5' for t, n in z.term.value_counts().items()))
    summary.append(dict(model=m, new_cues=cnt['D - B, linear_R20'], interactions=cnt['boost_full - boost_add, D'],
                        proxy=verdict(labs20), share_lo=min(s['d'] / s['ceil'] for s in shares),
                        share_hi=max(s['d'] / s['ceil'] for s in shares),
                        add_lo=min(s['add'] / s['ceil'] for s in shares), add_hi=max(s['add'] / s['ceil'] for s in shares)))
C = os.path.join(O, 'fit_d_models', 'compare')
pdt = pd.read_csv(os.path.join(C, 'exploratory_d_models_pd.csv'), float_precision='round_trip')
print('\n################ per model: partial dependence change >= 0.05 (D additive; value -> mean prediction at the '
      '10/30/50/70/90 % quantiles)')
for m in MODELS:
    print(f'== {m}')
    q = pdt[(pdt.model == m) & pdt.change_at_least_0_05]
    for c in CONDS:
        z = q[q.condition == c].sort_values('change', ascending=False)
        print(f'  {c}: ' + ('none' if not len(z) else ''))
        for _, r in z.iterrows():
            print(f'    {r.cue:26s} {r.change:.3f} {r.direction}{"" if r.monotone else " (not monotone)"}: ' + ', '.join(
                f'{r[f"value_q{k}"]:.4g}->{r[f"pd_q{k}"]:.3f}' for k in (10, 30, 50, 70, 90)))
sm = pd.read_csv(os.path.join(C, 'exploratory_d_models_pd_summary.csv'))
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 20)
print('\n################ cue x model: conditions with a change >= 0.05 (directions)')
print(sm[sm.models_with_any > 0].to_string(index=False))
sp = pd.read_csv(os.path.join(C, 'exploratory_d_models_spearman.csv'))
print('\n################ Spearman of D additive predictions between models (rank scores in brackets)')
for c in CONDS:
    q = sp[sp.condition == c]
    print(f'  {c}: additive {q.spearman_additive_predictions.min():.3f}-{q.spearman_additive_predictions.max():.3f} '
          f'(median {q.spearman_additive_predictions.median():.3f}); rank scores {q.spearman_rank_scores.min():.3f}-'
          f'{q.spearman_rank_scores.max():.3f} (median {q.spearman_rank_scores.median():.3f})')
    for _, r in q.iterrows():
        print(f'    {r.model_1} ~ {r.model_2}: {r.spearman_additive_predictions:.3f} ({r.spearman_rank_scores:.3f})')
print('\n################ summary')
print(pd.DataFrame(summary).to_string(index=False))
