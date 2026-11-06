"""EXPLORATORY, decided after Qwen3.6-27B's D results were seen (user, 2026-11-05, item 3.4-3.5; descriptive only): D for
every model of the risk run, compared across the models, from new_cues_fit.py's outputs (run with --model, --predictions).
  exploratory_d_models_spearman.csv: per condition and pair of models, the Spearman correlation of D's additive boosting
    leave-one-segment-out predictions (pred_boost_add_D, ranked: scipy spearmanr); beside it (Claude's addition) the
    Spearman of the two models' rank scores themselves (y_rank) on the same frames;
  exploratory_d_models_pd.csv: per model x condition x cue of D's additive boosting, the partial dependence at the cue's
    10/30/50/70/90 % quantiles (value and mean prediction), its change (largest minus smallest of the five), the
    direction (sign of the 90 % one minus the 10 % one: 'up', 'down', 'flat') and whether the five are monotone; every
    cue, the column change_at_least_0_05 marks the ones of the user's tables;
  exploratory_d_models_pd_summary.csv: per cue x model, the conditions whose change is 0.05 or more, as 'n/5' with the
    directions.
The frames must be the same in every model's predictions (they are: the fit frames do not depend on the model).
usage: python -I compare_d_models.py <out_dir> <model>=<new_cues_fit.py out_dir> [<model>=<dir> ...]
"""
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_vlm_ssm as av  # noqa: E402

PD_MIN = 0.05
PRED_COL = 'pred_boost_add_D'


def pd_rows(t):
    """One model's exploratory_d_partial_dependence.csv -> one row per condition x cue."""
    rows = []
    for (c, cue), q in t.groupby(['condition', 'cue'], sort=False):
        q = q.sort_values('quantile')
        p = q.partial_dependence.to_numpy(float)
        d = p[-1] - p[0]
        st = np.diff(p)
        row = dict(condition=c, cue=cue, change=float(p.max() - p.min()),
                   direction='up' if d > 0 else ('down' if d < 0 else 'flat'),
                   monotone=bool((st >= 0).all() or (st <= 0).all()))
        for qq, v, pp in zip(q['quantile'], q.value, p):
            row[f'value_q{int(round(qq * 100))}'] = float(v)
            row[f'pd_q{int(round(qq * 100))}'] = float(pp)
        row['change_at_least_0_05'] = bool(row['change'] >= PD_MIN)
        rows.append(row)
    return rows


def main():
    out = sys.argv[1]
    pairs = [a.split('=', 1) for a in sys.argv[2:]]
    if len(pairs) < 2 or any(len(p) != 2 for p in pairs):
        sys.exit('give at least two <model>=<dir>')
    if os.path.exists(os.path.join(out, 'exploratory_d_models_params.txt')):
        sys.exit(f'{out} has exploratory_d_models_params.txt already: give a new folder')
    t0 = time.time()
    models = [m for m, _ in pairs]
    pr, pdt = {}, {}
    for m, d in pairs:
        p = pd.read_csv(os.path.join(d, 'exploratory_d_predictions.csv'), float_precision='round_trip')
        if set(p.model) != {m}:
            sys.exit(f'{d}: predictions of {sorted(set(p.model))}, not {m}')
        pr[m] = p
        t = pd.read_csv(os.path.join(d, 'exploratory_d_partial_dependence.csv'), float_precision='round_trip')
        if set(t.model) != {m}:
            sys.exit(f'{d}: partial dependence of {sorted(set(t.model))}, not {m}')
        pdt[m] = t
    key = pr[models[0]][['condition', 'segment', 'frame_idx']].reset_index(drop=True)
    for m in models[1:]:
        if not pr[m][['condition', 'segment', 'frame_idx']].reset_index(drop=True).equals(key):
            sys.exit(f'{m}: not the same frames in the same order as {models[0]}')
    srow = []
    for c in av.CONDITIONS:
        sel = (key.condition == c).to_numpy()
        for m1, m2 in itertools.combinations(models, 2):
            a1, a2 = pr[m1][PRED_COL].to_numpy(float)[sel], pr[m2][PRED_COL].to_numpy(float)[sel]
            y1, y2 = pr[m1].y_rank.to_numpy(float)[sel], pr[m2].y_rank.to_numpy(float)[sel]
            srow.append(dict(condition=c, model_1=m1, model_2=m2, frames=int(sel.sum()),
                             spearman_additive_predictions=float(spearmanr(a1, a2)[0]),
                             spearman_rank_scores=float(spearmanr(y1, y2)[0])))
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(srow).to_csv(os.path.join(out, 'exploratory_d_models_spearman.csv'), index=False)
    prow = []
    for m in models:
        for r in pd_rows(pdt[m]):
            prow.append(dict(model=m, **r))
    pt = pd.DataFrame(prow)
    pt.to_csv(os.path.join(out, 'exploratory_d_models_pd.csv'), index=False)
    big = pt[pt.change_at_least_0_05]
    cues = list(dict.fromkeys(pt.cue))
    summ = []
    for cue in cues:
        row = dict(cue=cue)
        for m in models:
            q = big[(big.model == m) & (big.cue == cue)]
            dirs = ', '.join(f'{d} {n}' for d, n in q.direction.value_counts().sort_index().items())
            row[m] = f'{len(q)}/{len(av.CONDITIONS)}' + (f' ({dirs})' if len(q) else '')
        row['models_with_any'] = int(sum(1 for m in models if ((big.model == m) & (big.cue == cue)).any()))
        summ.append(row)
    sm = pd.DataFrame(summ).sort_values(['models_with_any', 'cue'], ascending=[False, True])
    sm.to_csv(os.path.join(out, 'exploratory_d_models_pd_summary.csv'), index=False)
    with open(os.path.join(out, 'exploratory_d_models_params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'EXPLORATORY, decided after Qwen3.6-27B\'s D results were seen (user, 2026-11-05, item 3; descriptive '
                f'only): compare_d_models.py run {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        for m, d in pairs:
            f.write(f'{m} = {os.path.abspath(d)}\n')
        f.write(f'frames per condition {int((key.condition == av.CONDITIONS[0]).sum())}; Spearman: scipy spearmanr of '
                f'{PRED_COL} (and of y_rank); partial dependence change = largest - smallest of the five quantiles, '
                f'the user\'s tables: change >= {PD_MIN}\n')
    print(f'done in {time.time() - t0:.0f} s: {len(models)} models, {len(srow)} pairs x conditions, '
          f'{int(pt.change_at_least_0_05.sum())} model x condition x cue with a change of {PD_MIN} or more')


if __name__ == '__main__':
    main()
