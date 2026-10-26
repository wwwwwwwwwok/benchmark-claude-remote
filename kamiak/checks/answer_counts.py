"""How many of the 5 repeats have a readable answer, per frame and condition (user, 2026-10-26: replies without an
answer stay missing and the frame's score for that condition is the mean of the other repeats, so check that no
frame x condition is left with none). Reads a model folder the way the analysis does (analyze_vlm_ssm.read_model:
the main file and, if there is one, its forced answers; .jsonl or .jsonl.gz).
Prints, per model: the cells (frame x condition) by number of answers, the smallest number, the cells with fewer than
5 per condition and how many of them are on the 39 frames left out of the main result, and the cells with none.
usage: python -I answer_counts.py <main list csv> <model folder> [<model folder> ...]
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.expanduser('~/benchmark-claude-remote/code'))
from analyze_vlm_ssm import read_model  # noqa: E402

lst = pd.read_csv(sys.argv[1], comment='#')
slow = set(zip(lst.loc[lst.closing_below_0_1.astype(bool), 'segment'], lst.loc[lst.closing_below_0_1.astype(bool), 'frame_idx']))
for model_dir in sys.argv[2:]:
    req, info = read_model(model_dir, 'main')
    cells = req.groupby(['condition', 'segment', 'frame_idx']).agg(n=('repeat', 'size'), answers=('risk', 'count'),
                                                                  forced=('forced', 'sum')).reset_index()
    cells['left_out'] = [(s, k) in slow for s, k in zip(cells.segment, cells.frame_idx)]
    print(f"{info['model']}: {info['lines']} requests, forced answers used {info['forced_used']}; {len(cells)} cells "
          f"(frame x condition), repeats per cell {sorted(cells.n.unique())}; answers in all {int(cells.answers.sum())}")
    print(f'  cells by number of answers: {cells.answers.value_counts().sort_index(ascending=False).to_dict()}; '
          f'smallest {cells.answers.min()}')
    few = cells[cells.answers < 5]
    for c, g in few.groupby('condition'):
        print(f'  {c:10s}: {g.answers.value_counts().sort_index(ascending=False).to_dict()} '
              f'(on the 39 left-out frames: {int(g.left_out.sum())})')
    none = cells[cells.answers == 0]
    print(f'  cells with no answer: {len(none)}')
    for r in none.itertuples():
        print(f'    {r.segment} frame {r.frame_idx} {r.condition}')
