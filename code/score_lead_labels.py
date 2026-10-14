"""Score the three lead-pairing methods against the hand labels.

Usage:
    python -I score_lead_labels.py <key.csv> <labels.csv> <frame_ssm.csv>

labels.csv needs columns item, answer_final:
    a number  = that numbered car is the lead
    0         = no lead
    beyond80  = the lead is a car beyond the ~80 m range of Waymo's 3D labels (no method can find it)
    ?         = unclear (skipped)

Two scorings are printed:
    strict          beyond80 counts as a lead every method misses
    labelled range  beyond80 counts as "no lead within the labelled range" (truth = 0)
The overall estimate weights the agreement frames by the accuracy on the random agreement sample.
"""
import sys

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
METHODS = ('map', 'path', 'straight')


def main(key_csv, labels_csv, frame_csv):
    key = pd.read_csv(key_csv)
    lab = pd.read_csv(labels_csv, dtype={'answer_final': str})
    d = key.merge(lab[['item', 'answer_final']], on='item')
    fr = pd.read_csv(frame_csv)
    lead = {m: fr[f'lead_{m}'].fillna('') for m in METHODS}
    n_all = len(fr)
    n_dis = int(((lead['map'] != lead['path']) | (lead['map'] != lead['straight']) |
                 (lead['path'] != lead['straight'])).sum())
    rows = []
    for scoring, beyond in (('strict', None), ('labelled range', 0)):
        for sample in ('disagree', 'random'):
            g = d[d['sample'] == sample]
            truth = g['answer_final'].map(lambda a: beyond if a == 'beyond80' else (None if a == '?' else int(a)))
            ok = truth.notna() | (g['answer_final'] == 'beyond80')
            for m in METHODS:
                hit = sum(t is not None and not pd.isna(t) and int(p) == int(t)
                          for p, t in zip(g[f'{m}_pick'], truth))
                rows.append(dict(scoring=scoring, sample=sample, method=m, correct=hit, n=int(ok.sum())))
    t = pd.DataFrame(rows)
    out = []
    for scoring, g in t.groupby('scoring', sort=False):
        acc_rand = g[g['sample'] == 'random'].set_index('method').eval('correct / n')
        for m in METHODS:
            dis = g[(g['sample'] == 'disagree') & (g['method'] == m)].iloc[0]
            rnd = g[(g['sample'] == 'random') & (g['method'] == m)].iloc[0]
            overall = (dis.correct + (n_all - n_dis) * acc_rand[m]) / n_all
            out.append(dict(scoring=scoring, method=m, disagree=f'{dis.correct}/{dis.n}',
                            random=f'{rnd.correct}/{rnd.n}', overall_estimate=f'{overall:.1%}'))
    print(pd.DataFrame(out).to_string(index=False))
    print(f'frames: {n_all} total, {n_dis} where the methods disagree')


if __name__ == '__main__':
    main(*sys.argv[1:4])
