"""Compact tables of an analyze_vlm_ssm.py output folder (formal experiment: rankings only, no AUC), for the report:
repeat stability and answers per condition; Kendall tau-b (frames with a value; = overall tau-b here) with 95 % intervals
per condition x SSM for each grouping, and the SSMs in order; which SSMs are clearly behind the top one (paired
bootstrap interval of the difference excludes 0); Spearman and pairwise agreement for 'all'.
usage: python -I summarize.py <analysis folder> [<second folder to compare the SSM order with>]
"""
import os
import sys

import pandas as pd

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)
d = sys.argv[1]
ag = pd.read_csv(os.path.join(d, 'agreement.csv'))
cs = pd.read_csv(os.path.join(d, 'condition_summary.csv'))
dif = pd.read_csv(os.path.join(d, 'ssm_differences.csv'))
SSMS = ['TTC', 'THW', 'DRAC', 'MTTC', 'CI', 'PICUD', 'PSD', 'TTC2D', 'ACT']
CONDS = ['grid', 'sep_time', 'sep_camera', 'ctrl_grid', 'ctrl_sep']

print('=== answers and repeat stability per condition')
cols = ['model', 'condition', 'requests', 'answered', 'no_answer', 'hit_max_tokens', 'score_min', 'score_max',
        'distinct_scores', 'sd_repeats_mean', 'retest_tau_b_mean', 'icc_single', 'icc_mean_of_k', 'frames_all_k',
        'latency_median_s', 'completion_tokens_median', 'camera_written_differently',
        'hazard_type_written_differently']
print(cs[cols].round(3).to_string(index=False))


def order(sub):
    s = sub.set_index('ssm')['tau_b_conflict'].sort_values(ascending=False)
    return ' > '.join(f'{k} {v:.3f}' for k, v in s.items())


for model in ag.model.unique():
    for g in ['all', 'time_of_day', 'segment']:
        sub = ag[(ag.model == model) & (ag.grouping == g)]
        if sub.empty:
            continue
        print(f'\n=== {model}, grouping {g}: Kendall tau-b (frames n_conflict {sorted(sub.n_conflict.unique())}), '
              f'95 % interval')
        t = sub.assign(cell=sub.apply(lambda r: f'{r.tau_b_conflict:.3f} [{r.tau_b_conflict_lo:.3f}, '
                                                f'{r.tau_b_conflict_hi:.3f}]', axis=1))
        print(t.pivot(index='condition', columns='ssm', values='cell').reindex(index=CONDS, columns=SSMS)
              .to_string())
        print('  order:')
        for c in CONDS:
            print(f'    {c:10s} {order(sub[sub.condition == c])}')
        print('  clearly behind the top SSM (difference in overall tau-b, interval excludes 0):')
        for c in CONDS:
            s = sub[sub.condition == c].set_index('ssm')['tau_b_all']
            top = s.idxmax()
            dd = dif[(dif.model == model) & (dif.condition == c) & (dif.grouping == g) & (dif.stat == 'tau_b_all') &
                     ((dif.ssm_1 == top) | (dif.ssm_2 == top))]
            behind, tied = [], []
            for r in dd.itertuples():
                other = r.ssm_2 if r.ssm_1 == top else r.ssm_1
                (behind if r.clear else tied).append(other)
            print(f'    {c:10s} top {top}; clearly behind: {", ".join(sorted(behind, key=SSMS.index)) or "none"}; '
                  f'not clearly different: {", ".join(sorted(tied, key=SSMS.index)) or "none"}')
    sub = ag[(ag.model == model) & (ag.grouping == 'all')]
    for stat in ['spearman_conflict', 'agreement_conflict']:
        print(f'\n=== {model}, grouping all: {stat}')
        print(sub.pivot(index='condition', columns='ssm', values=stat).reindex(index=CONDS, columns=SSMS)
              .round(3).to_string())

if len(sys.argv) > 2:
    other = pd.read_csv(os.path.join(sys.argv[2], 'agreement.csv'))
    print(f'\n=== SSM order, {os.path.basename(d.rstrip("/"))} vs {os.path.basename(sys.argv[2].rstrip("/"))}')
    for model in ag.model.unique():
        for g in ['all', 'time_of_day', 'segment']:
            for c in CONDS:
                a = ag[(ag.model == model) & (ag.grouping == g) & (ag.condition == c)].set_index('ssm')['tau_b_conflict']
                b = other[(other.model == model) & (other.grouping == g) & (other.condition == c)].set_index('ssm')[
                    'tau_b_conflict']
                oa, ob = list(a.sort_values(ascending=False).index), list(b.sort_values(ascending=False).index)
                print(f'  {model} {g:11s} {c:10s} same order: {oa == ob}; max |tau difference| '
                      f'{(a - b.reindex(a.index)).abs().max():.3f}' + ('' if oa == ob else f'; {oa} vs {ob}'))
