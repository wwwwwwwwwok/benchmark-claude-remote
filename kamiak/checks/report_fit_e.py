"""Read-only (user, 2026-11-06, exploratory; E's cues were chosen after these frames' residuals were seen): the numbers
of new_cues_fit_e.py and residual_clusters_e.py for the report, with the reading rules written before the results:
  E useful (linear, 20 m main): E - D wholly above 0 in at least 4 of the 5 inputs;
  proxy (20 m main): the E - D difference of the ego-speed coefficient: containing 0 = no change seen; not containing 0
    and smaller in size: E's interval containing 0 = mostly a proxy, not = partly; larger = reported as it is; a
    conclusion only when at least 4 of 5 inputs agree, otherwise "inconsistent";
  shares of the ceiling: the highest of linear R20, the tree and boost_full; the earlier K+S+N20+H (1,084 frames) in
    both units (out-of-sample R2; share of the 1,084-frame ceiling).
usage: python -I report_fit_e.py <fit_e folder> <residual_e folder> <fit_d folder (fit_d_pred)> <rule_explore folder>
       <the fake run's out folder (the noise inputs' first cuts: the reference)>"""
import os
import sys

import numpy as np
import pandas as pd

E, RE, D, R, NOISE = sys.argv[1:6]
CONDS = ['grid', 'sep_time', 'sep_camera', 'ctrl_grid', 'ctrl_sep']
MODELS = ['linear_R10', 'linear_R20', 'linear_R30', 'linear_R50', 'tree', 'boost_full', 'boost_add']
f3 = lambda v: 'nan' if pd.isna(v) else f'{v:+.3f}'  # noqa: E731
iv = lambda p, lo, hi: f'{f3(p)} [{f3(lo)}, {f3(hi)}]'  # noqa: E731
rd = lambda d, n: pd.read_csv(os.path.join(d, n), float_precision='round_trip')  # noqa: E731
print(open(os.path.join(E, 'exploratory_e_params.txt'), encoding='utf-8').read())
oos = rd(E, 'exploratory_e_oos.csv').set_index(['condition', 'fit', 'cues'])
df = rd(E, 'exploratory_e_differences.csv').set_index(['condition', 'comparison'])
print('== out-of-sample R2, D -> E [E interval]')
for c in CONDS:
    print(f'  {c:10s} ' + '  '.join(f'{m} {oos.loc[(c, m, "D"), "r2_oos"]:.3f} -> {oos.loc[(c, m, "E"), "r2_oos"]:.3f} '
                                     f'[{oos.loc[(c, m, "E"), "r2_oos_lo"]:.3f}, {oos.loc[(c, m, "E"), "r2_oos_hi"]:.3f}]'
                                     for m in ('linear_R20', 'tree', 'boost_full', 'boost_add')))
cnt = {}
print('\n== differences (* = interval wholly above 0, - = wholly below)')
for comp in [f'E - D, {m}' for m in MODELS] + [f'boost_full - {x}, E' for x in ('tree', 'linear_R20', 'boost_add')]:
    rs = [df.loc[(c, comp)] for c in CONDS]
    cnt[comp] = sum(bool(r.above_0) for r in rs)
    print(f'  {comp:24s} {cnt[comp]}/5: ' + '; '.join(f'{c} {iv(r["diff"], r.lo, r.hi)}'
                                                     f'{"*" if r.above_0 else ("-" if r.below_0 else "")}'
                                                     for c, r in zip(CONDS, rs)))
print(f'  READING: linear R20 E - D above 0 in {cnt["E - D, linear_R20"]}/5 -> '
      f'{"useful" if cnt["E - D, linear_R20"] >= 4 else "not shown useful"} (E\'s cues chosen after these residuals: '
      f'may be optimistic); full - additive on E above 0 in {cnt["boost_full - boost_add, E"]}/5')
ce = rd(E, 'exploratory_e_ceiling.csv').set_index('condition')
oo = rd(os.path.join(R, 'main_1084'), 'exploratory_oos_r2.csv')
oo = oo[(oo.target == 'risk') & (oo.model == 'qwen3.6-27b') & (oo.fit == 'K+S+N20+H')].set_index('condition')
c84 = rd(os.path.join(R, 'tree_27b', 'after_trees', 'main_1084'), 'exploratory_ceiling.csv')
c84 = c84[c84.model == 'qwen3.6-27b'].set_index('condition')
print('\n== the ceiling (1,065 fit frames) and the best of three (linear R20, tree, boost_full); H on 1,084 frames')
for c in CONDS:
    cl = ce.loc[c, 'value']
    bd = max(('linear_R20', 'tree', 'boost_full'), key=lambda m: oos.loc[(c, m, 'D'), 'r2_oos'])
    be = max(('linear_R20', 'tree', 'boost_full'), key=lambda m: oos.loc[(c, m, 'E'), 'r2_oos'])
    vd, ve = oos.loc[(c, bd, 'D'), 'r2_oos'], oos.loc[(c, be, 'E'), 'r2_oos']
    va = oos.loc[(c, 'boost_add', 'E'), 'r2_oos']
    h = oo.loc[c, 'r2_oos']
    print(f'  {c:10s} ceiling {cl:.3f}; out-of-sample R2: D {vd:.3f} ({bd}), E {ve:.3f} ({be}), E additive {va:.3f}, '
          f'H {h:.3f} | share: D {vd / cl:.3f}, E {ve / cl:.3f}, E additive {va / cl:.3f}, H {h / c84.loc[c, "ceiling"]:.3f}'
          f' (1,084-frame ceiling {c84.loc[c, "ceiling"]:.3f}) | E best below H: R2 {ve < h}, share '
          f'{ve / cl < h / c84.loc[c, "ceiling"]}')
eg = rd(E, 'exploratory_e_ego_coefficient.csv')


def label(r):
    if r.diff_lo <= 0 <= r.diff_hi:
        return 'no change seen'
    if abs(r.ego_E) < abs(r.ego_D):
        return 'mostly a proxy' if r.ego_E_lo <= 0 <= r.ego_E_hi else 'partly a proxy'
    return 'larger in E'


print('\n== proxy test: ego-speed coefficient D and E, E - D (R20 rows, then verdicts)')
for R_ in (20, 10, 30, 50):
    labs = []
    for c in CONDS:
        r = eg[(eg.condition == c) & (eg.R == R_)].iloc[0]
        labs.append(label(r))
        if R_ == 20:
            print(f'  {c:10s} D {iv(r.ego_D, r.ego_D_lo, r.ego_D_hi)}, E {iv(r.ego_E, r.ego_E_lo, r.ego_E_hi)}, '
                  f'E - D {iv(r.diff_E_minus_D, r.diff_lo, r.diff_hi)} -> {labs[-1]}')
    vc = pd.Series(labs).value_counts()
    print(f'  R{R_}: ' + (vc.index[0] if vc.iloc[0] >= 4 else f'inconsistent ({dict(vc)})') + f'  ({labs})')
fc = rd(E, 'exploratory_e_first_cuts.csv')
fz = rd(NOISE, 'exploratory_e_first_cuts.csv')
nz = [fz[(fz.condition == c) & (fz.cues == s) & (fz.first_cue == 'true_ego_speed')].share.sum()
      for c in ('sep_time', 'ctrl_sep') for s in ('D', 'E')]
print(f'  first cut on true_ego_speed, D / E (pure-noise fake reference {min(nz):.3f}-{max(nz):.3f}): ' + '; '.join(
    f'{c} ' + ' / '.join(f'{fc[(fc.condition == c) & (fc.cues == s) & (fc.first_cue == "true_ego_speed")].share.sum():.3f}'
                         for s in ('D', 'E')) for c in CONDS))
for c in CONDS:
    q = fc[(fc.condition == c) & (fc.cues == 'E')]
    print(f'    {c} E most frequent first cuts: ' + ', '.join(f'{k} {v:.3f}' for k, v in zip(q.first_cue[:3], q.share[:3])))
tb = rd(E, 'exploratory_e_tau_b.csv')
print('\n== single-cue tau-b (all / segment; * = interval excludes 0)')
for cue in tb.cue.unique():
    q = tb[tb.cue == cue]
    parts = []
    for c in CONDS:
        a_ = q[(q.condition == c) & (q.grouping == 'all')].iloc[0]
        s_ = q[(q.condition == c) & (q.grouping == 'segment')].iloc[0]
        st = lambda r: '*' if (r.tau_b_lo > 0 or r.tau_b_hi < 0) else ''  # noqa: E731
        parts.append(f'{c} {a_.tau_b:+.3f}{st(a_)}/{s_.tau_b:+.3f}{st(s_)}')
    print(f'  {cue:34s} ({int(q.frames.iloc[0])} fr) ' + '  '.join(parts))
pdp = rd(E, 'exploratory_e_partial_dependence.csv')
print('\n== E additive partial dependence: cues with a change of 0.05 or more (value -> mean prediction); '
      'max_vehicle_length_30m always')
for c in CONDS:
    q = pdp[pdp.condition == c]
    rng = q.groupby('cue').partial_dependence.agg(lambda v: v.max() - v.min()).sort_values(ascending=False)
    print(f'  {c}:')
    for cue, rg in rng.items():
        if rg < 0.05 and cue != 'max_vehicle_length_30m':
            continue
        z = q[q.cue == cue]
        z = pd.concat([z[z.point == 'quantile'], z[z.point == 'value']])
        print(f'    {cue:32s} change {rg:.3f}: ' + ', '.join(
            f'{"q" + str(int(round(r.quantile * 100))) + " " if r.point == "quantile" else ""}{r.value:.4g}->'
            f'{r.partial_dependence:.3f}' for r in z.itertuples()))
print('\n== residual clusters (D -> E mean residual; E - D [interval]; * = E - D interval excludes 0; D-flag = D\'s '
      'interval excluded 0)')
cl = rd(RE, 'exploratory_e_residual_clusters.csv')
for c in CONDS:
    print(f'  {c}:')
    for r in cl[cl.condition == c].itertuples():
        flag = 'D-flag ' if (r.lo_D > 0 or r.hi_D < 0) else '       '
        ex = '*' if (r.lo_diff > 0 or r.hi_diff < 0) else ''
        print(f'    {flag}{int(r.cluster):2d} {int(r.frames):4d} fr  D {r.mean_residual_D:+.3f} [{r.lo_D:+.3f}, '
              f'{r.hi_D:+.3f}] -> E {r.mean_residual_E:+.3f} [{r.lo_E:+.3f}, {r.hi_E:+.3f}]; E - D '
              f'{iv(r.E_minus_D, r.lo_diff, r.hi_diff)}{ex} | {"; ".join(r.top_words.split("; ")[:6])}')
bw = rd(RE, 'exploratory_e_residual_between.csv')
print('  between-segment share D -> E: ' + '; '.join(
    f'{c} {bw[(bw.condition == c) & (bw.of == "D residual")].between_share.iloc[0]:.3f} -> '
    f'{bw[(bw.condition == c) & (bw.of == "E residual")].between_share.iloc[0]:.3f}' for c in CONDS))
print(open(os.path.join(RE, 'exploratory_e_residual_params.txt'), encoding='utf-8').read())
_ = (np, D)
