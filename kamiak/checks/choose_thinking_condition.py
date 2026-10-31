"""The rule for the one condition of the Qwen3.6-27B quantities run with thinking (user, 2026-10-29, fixed before the
results without thinking; CLAUDE.md "正式实验怎么跑" 2026-10-29): per condition, the mean over the 5 quantities, with
equal weights, of the tau-b with the truth of Qwen3.6-27B without thinking, main result (1,084 frames), grouping
'all'; the highest mean wins; a tie goes to the user. Prints the 5 x 5 tau-b, the means in order, the winner and its
margin over the second; a tie (equal means) is reported, not decided.
usage: python -I choose_thinking_condition.py <main_1084/quantity_tau_b.csv> [--model qwen3.6-27b-nothink]
"""
import argparse

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('tau_b')
ap.add_argument('--model', default='qwen3.6-27b-nothink')
a = ap.parse_args()
pd.set_option('display.width', 200)
t = pd.read_csv(a.tau_b, float_precision='round_trip')
t = t[(t.model == a.model) & (t.grouping == 'all')]
w = t.pivot(index='condition', columns='quantity', values='tau_b')
if w.shape != (5, 5) or w.isna().any().any() or (t.frames != 1084).any():
    raise SystemExit(f'{a.tau_b}: expected 5 conditions x 5 quantities with a value on 1,084 frames for {a.model}, got '
                     f'{w.shape}, missing {int(w.isna().sum().sum())}, frames {sorted(t.frames.unique())}')
w['mean'] = w.mean(axis=1)
w = w.sort_values('mean', ascending=False)
print(f'{a.tau_b}: {a.model}, grouping all, frames 1084; tau-b with the truth and the mean over the 5 quantities:')
print(w.round(4).to_string())
print('\nmeans at full precision:', {c: repr(v) for c, v in w['mean'].items()})
top = w['mean'].iloc[0]
tied = w.index[w['mean'] == top].tolist()
if len(tied) > 1:
    print(f'TIE at the top: {tied} (mean {top!r}); the user decides')
else:
    print(f'highest mean: {w.index[0]} ({top:.4f}); second {w.index[1]} ({w["mean"].iloc[1]:.4f}); margin '
          f'{top - w["mean"].iloc[1]:.4f}')
