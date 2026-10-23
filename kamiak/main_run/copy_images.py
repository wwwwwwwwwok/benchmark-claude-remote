"""Copy the every-0.5 s images (frame index divisible by 5) of the 5 cameras of the segments in the comparison list
from the scratch extraction to lab storage, same layout (user approved, 2026-10-22):
  <workspace>/extracted/<split>/<segment>/images/<CAMERA>/<frame>_<ts>.jpg
  -> /data/lab/cejli/sy/waymo_v143/tables/<split>/<segment>/images/<CAMERA>/<frame>_<ts>.jpg
Files already there with the same size are skipped. Prints per-split file counts and bytes of source and copy.
usage: python -I copy_images.py <workspace> <comparison list csv> <tables root>
"""
import os
import shutil
import sys

import pandas as pd

ws, lst, tables = sys.argv[1:4]
segs = pd.read_csv(lst, comment='#')[['split', 'segment']].drop_duplicates()
tot = {}
for r in segs.itertuples():
    src_root = os.path.join(ws, 'extracted', r.split, r.segment, 'images')
    for cam in sorted(os.listdir(src_root)):
        dst = os.path.join(tables, r.split, r.segment, 'images', cam)
        os.makedirs(dst, exist_ok=True)
        for f in sorted(os.listdir(os.path.join(src_root, cam))):
            if int(f.split('_')[0]) % 5:
                continue
            s, d = os.path.join(src_root, cam, f), os.path.join(dst, f)
            size = os.path.getsize(s)
            if not (os.path.exists(d) and os.path.getsize(d) == size):
                shutil.copy2(s, d)
            n, b, nd, bd = tot.get(r.split, (0, 0, 0, 0))
            tot[r.split] = (n + 1, b + size, nd + 1, bd + os.path.getsize(d))
for split, (n, b, nd, bd) in tot.items():
    print(f'{split}: source {n} files {b} bytes; copy {nd} files {bd} bytes')
print(f'all: source {sum(v[0] for v in tot.values())} files {sum(v[1] for v in tot.values())} bytes; '
      f'copy {sum(v[2] for v in tot.values())} files {sum(v[3] for v in tot.values())} bytes; segments {len(segs)}')
