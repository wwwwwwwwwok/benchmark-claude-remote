"""Image paths that run_vlm.build('grid' / 'ctrl_grid', ...) sends for one frame, resolved exactly as run_vlm.main does
(grid targets_main.csv of the main experiment). Prints one line per grid image: index, path. Text only, opens no image.
Usage: python -I prompt_example_grid_lookup.py <repo> <vlm_inputs dir> <segment> <frame_idx>"""
import os
import sys

repo, V, seg, frame = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
sys.path.insert(0, os.path.join(repo, 'code'))
import pandas as pd
import run_vlm

gt, st = os.path.join(V, 'grid', 'targets_main.csv'), os.path.join(V, 'separate', 'targets_main.csv')
tg, ts = pd.read_csv(gt), pd.read_csv(st)
t = tg.merge(ts[['segment', 'frame_idx', 'images_by_time']], on=['segment', 'frame_idx'], validate='one_to_one')
r = t[(t.segment == seg) & (t.frame_idx == frame)]
assert len(r) == 1, len(r)
r = next(r.itertuples())
root_g, root_s = os.path.dirname(os.path.abspath(gt)), os.path.dirname(os.path.abspath(st))
grid_imgs = [os.path.join(root_g, p) for p in r.images.split(';')]
by_time = [os.path.join(root_s, p) for p in r.images_by_time.split(';')]
sep_imgs = {cam: [by_time[i * 5 + c] for i in range(5)] for c, cam in enumerate(run_vlm.CAMERAS)}
g = [x for k, x in run_vlm.build('grid', grid_imgs, sep_imgs) if k == 'image']
c = [x for k, x in run_vlm.build('ctrl_grid', grid_imgs, sep_imgs) if k == 'image']
labels = [x for k, x in run_vlm.build('grid', grid_imgs, sep_imgs) if k == 'text' and x.startswith('Picture ')]
assert len(g) == 5 and g == grid_imgs, g
assert len(c) == 1 and c[0] == g[4], c
print('targets_main.csv images column:', r.images)
print('grid labels in sending order:', ' | '.join(labels))
print('ctrl_grid image == grid image 5:', c[0] == g[4], c[0])
for i, x in enumerate(g, 1):
    print(f'GRID {i} {x}')
