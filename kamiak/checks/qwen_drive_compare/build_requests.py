"""Step 0 of the Qwen-Drive vLLM vs official-code check (user, 2026-10-26): the 5 requests (one per condition, repeat 0)
of the comparison frame, built exactly as code/run_vlm.py builds them (its build(), its SYSTEM, the same target
merge and image paths), but from the pre-resized PNGs. Written as JSON parts (text and image paths) for both sides.
usage: python -I build_requests.py <grid targets_compare.csv> <separate targets_compare.csv> <out json>
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.expanduser('~/benchmark-claude-remote/code'))
from run_vlm import CAMERAS, CONDITIONS, SYSTEM, build  # noqa: E402

grid_csv, sep_csv, out = sys.argv[1:4]
tg, ts = pd.read_csv(grid_csv), pd.read_csv(sep_csv)
root_g, root_s = os.path.dirname(os.path.abspath(grid_csv)), os.path.dirname(os.path.abspath(sep_csv))
t = tg.merge(ts[['segment', 'frame_idx', 'images_by_time']], on=['segment', 'frame_idx'], validate='one_to_one')
assert len(t) == 1, len(t)
r = next(t.itertuples())
grid_imgs = [os.path.join(root_g, p) for p in r.images.split(';')]
by_time = [os.path.join(root_s, p) for p in r.images_by_time.split(';')]
sep_imgs = {cam: [by_time[i * 5 + c] for i in range(5)] for c, cam in enumerate(CAMERAS)}
reqs = [dict(segment=r.segment, frame_idx=int(r.frame_idx), condition=c, repeat=0, system=SYSTEM,
             parts=build(c, grid_imgs, sep_imgs)) for c in CONDITIONS]
with open(out, 'w') as f:
    json.dump(reqs, f, indent=1)
print(f'{out}: {len(reqs)} requests for {r.segment} frame {r.frame_idx}: '
      + ', '.join(f"{q['condition']} {sum(k == 'image' for k, _ in q['parts'])} images" for q in reqs))
