"""Read-only (user, 2026-11-05, item 2, exploratory): code/residual_words.py's outputs put together for the report. Only
counts and words; no reply text. usage: python -I report_residual.py <residual_words.py out_dir>"""
import os
import sys

import pandas as pd

O = sys.argv[1]
CONDS = ['grid', 'sep_time', 'sep_camera', 'ctrl_grid', 'ctrl_sep']
rd = lambda n: pd.read_csv(os.path.join(O, f'exploratory_residual_{n}.csv'), float_precision='round_trip')  # noqa: E731
f3 = lambda v: 'nan' if pd.isna(v) else f'{v:+.3f}'  # noqa: E731
iv = lambda p, lo, hi: f'{f3(p)} [{f3(lo)}, {f3(hi)}]'  # noqa: E731
par = open(os.path.join(O, 'exploratory_residual_params.txt'), encoding='utf-8').read()
print(par)
bw = rd('between')
print('== between-segment share (point [95 % interval]); random reference ' + f'{bw.random_reference.iloc[0]:.3f}')
for c in CONDS:
    q = bw[bw.condition == c].set_index('of')
    print(f'  {c}: residual {iv(*q.loc["residual", ["between_share", "lo", "hi"]])}; rank score '
          f'{iv(*q.loc["rank score", ["between_share", "lo", "hi"]])}')
tr = rd('tiers')
print('\n== tiers (frames low / middle / high; residual: low tier up to, high tier from)')
for _, r in tr.iterrows():
    print(f'  {r.condition} {r.half}: {r.frames} frames in {r.segments} segments; {r.frames_low}/{r.frames_middle}/'
          f'{r.frames_high}; low <= {r.residual_low_max:+.4f}, high >= {r.residual_high_min:+.4f}'
          + (f'; words in 30+ frames {int(r.words_in_min_frames)}' if r.half == 'find' else ''))
wf = rd('words_find')
print('\n== find half: the 25 words with high > low and the 25 with low > high (difference in share points)')
for c in CONDS:
    q = wf[wf.condition == c]
    hi = q[q.rank_high.notna()].sort_values('rank_high')
    lo = q[q.rank_low.notna()].sort_values('rank_low')
    print(f'  {c} high: ' + ', '.join(f'{w} {d:+.3f}' for w, d in zip(hi.word, hi.difference)))
    print(f'  {c} low:  ' + ', '.join(f'{w} {d:+.3f}' for w, d in zip(lo.word, lo.difference)))
hz = rd('hazard_find')
print('\n== find half: hazard types, mean share high - low')
for k in hz.hazard_type.unique():
    q = hz[hz.hazard_type == k].set_index('condition')
    print(f'  {k:26s} ' + '  '.join(f'{c} {q.loc[c, "difference"]:+.3f} ({q.loc[c, "mean_share_high"]:.3f} / '
                                      f'{q.loc[c, "mean_share_low"]:.3f})' for c in CONDS))
cand = rd('candidates')
print(f'\n== candidates ({len(cand)}): word, side, conditions, rank per condition')
for _, r in cand.iterrows():
    print(f'  {r.word:18s} {r.side:4s} {r.conditions}/5  ' + ' '.join(
        f'{c} {"-" if pd.isna(r[f"rank_{c}"]) else int(r[f"rank_{c}"])}' for c in CONDS))
va = rd('validate')
print('\n== validate half (* = validated: interval excludes 0 on the find side)')
for kind in ('word', 'hazard_type'):
    q = va[va.kind == kind]
    for item in q['item'].unique():
        z = q[q['item'] == item].set_index('condition')
        nv = int(z.validated.sum())
        print(f'  {item:26s} {nv}/5: ' + '; '.join(
            f'{c} find {z.loc[c, "find_difference"]:+.3f}, validate '
            f'{iv(z.loc[c, "validate_difference"], z.loc[c, "lo"], z.loc[c, "hi"])}{"*" if z.loc[c, "validated"] else ""}'
            for c in CONDS))
cl = rd('clusters')
ari = rd('cluster_ari')
print('\n== clusters (frames; mean residual [interval], * = excludes 0; the 10 words in most frames; centre terms)')
for c in CONDS:
    a = ari[ari.condition == c]
    print(f'  {c}: adjusted Rand against seeds {a.other_seed.tolist()}: {[round(x, 3) for x in a.adjusted_rand_index]}')
    for _, r in cl[cl.condition == c].iterrows():
        ex = '*' if (r.lo > 0 or r.hi < 0) else ''
        print(f'    {int(r.cluster):2d} {int(r.frames):4d} fr  {iv(r.mean_residual, r.lo, r.hi)}{ex}  | {r.top_words}')
        print(f'                centre: {r.centre_terms}')
