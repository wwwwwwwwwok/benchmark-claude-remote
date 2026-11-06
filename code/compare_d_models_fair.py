"""EXPLORATORY, descriptive only (user, 2026-11-06, item 3, after the comparison of D across the models): fairer
comparisons of the models, from new_cues_fit.py's outputs (run with --model, --predictions) on the same fit frames.
  1 exploratory_d_models_disattenuated.csv: per condition and pair of models, the Spearman correlation of the two
    models' rank scores (y_rank) divided by the square root of the product of their ceilings (each model's own ceiling
    on the fit frames, exploratory_d_beside.csv: score_ceiling.py's 5 r / (1 + 4 r) from its 5 repeats); beside it the
    Spearman of their D additive boosting leave-one-segment-out predictions (pred_boost_add_D). Above 1 reported as is.
  2 exploratory_d_models_cross.csv: per condition and ordered pair (A predicts B): the out-of-sample R2 of A's D additive
    predictions against B's rank scores, 1 - sum (y_B - p_A)^2 / sum (y_B - mean y_B)^2 (as --oos: the predictions are
    leave-one-segment-out ones), divided by B's own additive out-of-sample R2 (the same formula with p_B; checked equal
    to B's exploratory_d_oos.csv within 1e-12).
usage: python -I compare_d_models_fair.py <out_dir> <model>=<new_cues_fit.py out_dir> [<model>=<dir> ...]
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

PRED_COL = 'pred_boost_add_D'
TOL = 1e-12


def r2_of(y, p):
    """Out-of-sample R2 of predictions p of y (leave-one-segment-out predictions), as new_cues_fit.py."""
    return 1.0 - float(((y - p) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def main():
    out = sys.argv[1]
    pairs = [a.split('=', 1) for a in sys.argv[2:]]
    if len(pairs) < 2 or any(len(p) != 2 for p in pairs):
        sys.exit('give at least two <model>=<dir>')
    if os.path.exists(os.path.join(out, 'exploratory_d_models_fair_params.txt')):
        sys.exit(f'{out} has exploratory_d_models_fair_params.txt already: give a new folder')
    t0 = time.time()
    models = [m for m, _ in pairs]
    pr, ceil, own_tab = {}, {}, {}
    for m, d in pairs:
        p = pd.read_csv(os.path.join(d, 'exploratory_d_predictions.csv'), float_precision='round_trip')
        if set(p.model) != {m}:
            sys.exit(f'{d}: predictions of {sorted(set(p.model))}, not {m}')
        pr[m] = p
        b = pd.read_csv(os.path.join(d, 'exploratory_d_beside.csv'), float_precision='round_trip')
        b = b[b.measure.str.startswith('ceiling')]
        if len(b) != len(av.CONDITIONS):
            sys.exit(f'{d}: not one ceiling per condition')
        ceil[m] = dict(zip(b.condition, b.value))
        o = pd.read_csv(os.path.join(d, 'exploratory_d_oos.csv'), float_precision='round_trip')
        own_tab[m] = o[(o.fit == 'boost_add') & (o.cues == 'D')].set_index('condition').r2_oos.to_dict()
    key = pr[models[0]][['condition', 'segment', 'frame_idx']].reset_index(drop=True)
    for m in models[1:]:
        if not pr[m][['condition', 'segment', 'frame_idx']].reset_index(drop=True).equals(key):
            sys.exit(f'{m}: not the same frames in the same order as {models[0]}')
    drow, crow = [], []
    dev = 0.0
    for c in av.CONDITIONS:
        sel = (key.condition == c).to_numpy()
        y = {m: pr[m].y_rank.to_numpy(float)[sel] for m in models}
        p = {m: pr[m][PRED_COL].to_numpy(float)[sel] for m in models}
        own = {m: r2_of(y[m], p[m]) for m in models}
        for m in models:
            dev = max(dev, abs(own[m] - own_tab[m][c]))
        for m1, m2 in itertools.combinations(models, 2):
            s = float(spearmanr(y[m1], y[m2])[0])
            drow.append(dict(condition=c, model_1=m1, model_2=m2, frames=int(sel.sum()), spearman_rank_scores=s,
                             ceiling_1=ceil[m1][c], ceiling_2=ceil[m2][c],
                             disattenuated=s / np.sqrt(ceil[m1][c] * ceil[m2][c]),
                             spearman_additive_predictions=float(spearmanr(p[m1], p[m2])[0])))
        for a, b in itertools.permutations(models, 2):
            r2c = r2_of(y[b], p[a])
            crow.append(dict(condition=c, predictor=a, target=b, frames=int(sel.sum()), r2_cross=r2c,
                             r2_own_target=own[b], ratio=r2c / own[b]))
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(drow).to_csv(os.path.join(out, 'exploratory_d_models_disattenuated.csv'), index=False)
    pd.DataFrame(crow).to_csv(os.path.join(out, 'exploratory_d_models_cross.csv'), index=False)
    failed = [] if dev <= TOL else ['own R2 against exploratory_d_oos.csv']
    with open(os.path.join(out, 'exploratory_d_models_fair_params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'EXPLORATORY, descriptive (user, 2026-11-06, item 3): compare_d_models_fair.py run '
                f'{time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        for m, d in pairs:
            f.write(f'{m} = {os.path.abspath(d)}\n')
        f.write(f'frames per condition {int((key.condition == av.CONDITIONS[0]).sum())}; disattenuated = Spearman of '
                f'y_rank / sqrt(ceiling_1 x ceiling_2), the ceilings of exploratory_d_beside.csv; cross R2 = 1 - '
                f'sum (y_target - {PRED_COL} of the predictor)^2 / sum (y_target - its mean)^2, ratio = that / the '
                f'target\'s own\n')
        f.write(f'check: own additive R2 recomputed from the predictions against exploratory_d_oos.csv: largest '
                f'difference {dev:.3g} (tolerance {TOL:g})\n')
        f.write(f'checks not met: {failed if failed else "none"}\n')
    print(f'done in {time.time() - t0:.0f} s: {len(drow)} pairs x conditions, {len(crow)} ordered pairs x conditions; '
          f'own R2 largest difference {dev:.3g}; checks not met: {failed if failed else "none"}')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
