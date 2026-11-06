"""The comparison list's 750 frames that are not in the main list (user, 2026-11-06, item 2.3), as a target list for
count_nearby.py, new_cues.py and quantity_truth.py: the rows of vlm_frames/comparison_lead7_targets.csv with in_main
False, every column kept, plus main_segment (the frame's segment is one of the main list's 241).
usage: python -I make_targets_750.py <repository root> <out csv (must not exist)>"""
import os
import sys

import pandas as pd

ROOT, OUT = sys.argv[1:3]
if os.path.exists(OUT):
    sys.exit(f'{OUT} exists')
c = pd.read_csv(os.path.join(ROOT, 'vlm_frames', 'comparison_lead7_targets.csv'), comment='#')
m = pd.read_csv(os.path.join(ROOT, 'vlm_frames', 'main_all9_targets.csv'), comment='#')
t = c[~c.in_main.astype(bool)].copy()
t['main_segment'] = t.segment.isin(set(m.segment))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
t.to_csv(OUT, index=False)
print(f'{len(t)} frames in {t.segment.nunique()} segments; in the main list\'s segments {int(t.main_segment.sum())}, '
      f'outside {int((~t.main_segment).sum())} frames in {t[~t.main_segment].segment.nunique()} segments')
