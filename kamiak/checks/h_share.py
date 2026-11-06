"""Read-only (user, 2026-11-05, item 1): the handoff's "D 占满分" line in one unit at a time. Out-of-sample R2 and the
share of the ceiling, side by side, for Qwen3.6-27B, from the unrounded numbers:
  D: the highest of linear R20, the tree and boost_full on the 1,065 fit frames (as the reading rule), and D's additive
     boosting; B the same; their shares with the ceiling recomputed on the 1,065 fit frames (exploratory_d_beside.csv);
  K+S+N20+H (1,084 frames, H from the same reply; exploratory_oos_r2.csv), its share with the 1,084-frame ceiling
     (tree_27b/after_trees/main_1084/exploratory_ceiling.csv).
usage: python -I h_share.py <rule_explore folder>"""
import os
import sys

import pandas as pd

R = sys.argv[1]
CONDS = ['grid', 'sep_time', 'sep_camera', 'ctrl_grid', 'ctrl_sep']
M = 'qwen3.6-27b'
rd = lambda p: pd.read_csv(os.path.join(R, p), float_precision='round_trip')  # noqa: E731
oos = rd('new_cues/fit_d/main_1084/exploratory_d_oos.csv').set_index(['condition', 'fit', 'cues'])
bes = rd('new_cues/fit_d/main_1084/exploratory_d_beside.csv')
oo = rd('main_1084/exploratory_oos_r2.csv')
oo = oo[(oo.target == 'risk') & (oo.model == M) & (oo.fit == 'K+S+N20+H')].set_index('condition')
ce = rd('tree_27b/after_trees/main_1084/exploratory_ceiling.csv')
ce = ce[ce.model == M].set_index('condition')
ok = {'oos': True, 'share': True}
print('condition: ceiling 1,065 / 1,084; then each as out-of-sample R2 | share of its ceiling')
for c in CONDS:
    c65 = bes[(bes.condition == c) & bes.measure.str.startswith('ceiling')].iloc[0].value
    c84 = ce.loc[c, 'ceiling']
    best = max(('linear_R20', 'tree', 'boost_full'), key=lambda m: oos.loc[(c, m, 'D'), 'r2_oos'])
    bestb = max(('linear_R20', 'tree', 'boost_full'), key=lambda m: oos.loc[(c, m, 'B'), 'r2_oos'])
    d, b, ad = (oos.loc[(c, best, 'D'), 'r2_oos'], oos.loc[(c, bestb, 'B'), 'r2_oos'],
                oos.loc[(c, 'boost_add', 'D'), 'r2_oos'])
    hh = oo.loc[c, 'r2_oos']
    assert int(oo.loc[c, 'frames']) == 1084 and int(ce.loc[c, 'frames']) == 1084
    print(f'{c}: ceiling {c65:.6f} (1,065) / {c84:.6f} (1,084, {int(ce.loc[c, "frames_5_scores"])} frames with 5 '
          f'scores)')
    print(f'    D best ({best}) {d:.6f} | {d / c65:.6f};  D additive {ad:.6f} | {ad / c65:.6f};  B best ({bestb}) '
          f'{b:.6f} | {b / c65:.6f};  K+S+N20+H {hh:.6f} | {hh / c84:.6f}')
    ok['oos'] &= d < hh and ad < hh
    ok['share'] &= d / c65 < hh / c84 and ad / c65 < hh / c84
print(f'D best and D additive below K+S+N20+H in all five: as out-of-sample R2 {ok["oos"]}, as share of the ceiling '
      f'{ok["share"]}')
