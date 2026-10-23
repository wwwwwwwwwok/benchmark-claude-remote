"""Input images of the formal VLM experiment's main run (user's decisions, 2026-10-22), from the images copied to lab
storage, with code/make_vlm_inputs.py as it is in the repository (DejaVu Sans Bold from fonts/):
  1. <lab>/results_lead_final_excl/frame_ssm_validation_training.csv: the two splits' frame_ssm.csv in one file (the
     header once, then the validation rows, then the training rows, unchanged text); segment names are unique.
  2. <lab>/vlm_inputs/segments/<segment> -> ../../tables/<split>/<segment> for the segments of the main list, so that
     make_vlm_inputs.py sees both splits under one root.
  3. <lab>/vlm_inputs/{grid,separate}/: images of every target frame of these segments (8 processes per layout on
     disjoint segments, then one pass over all segments, which keeps the images and writes targets.csv for all of them).
  4. targets_main.csv next to targets.csv: the rows of vlm_frames/main_all9_targets.csv (checked: all there, same t_s).
  5. sha256 of every image: <lab>/vlm_inputs/sha256_{grid,separate}.txt.
usage: python -I make_main_inputs.py <repo> <lab root>
"""
import concurrent.futures as cf
import hashlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd

repo, lab = sys.argv[1:3]
MK = os.path.join(repo, 'code', 'make_vlm_inputs.py')
F = os.path.join(lab, 'results_lead_final_excl')
COMBINED = os.path.join(F, 'frame_ssm_validation_training.csv')
OUT = os.path.join(lab, 'vlm_inputs')
SEGROOT = os.path.join(OUT, 'segments')

# 1. both splits in one frame_ssm.csv
parts = [os.path.join(F, sp, 'ssm', 'frame_ssm.csv') for sp in ('validation', 'training')]
heads, bodies = [], []
for p in parts:
    with open(p, 'rb') as f:
        heads.append(f.readline())
        bodies.append(f.read())
if heads[0] != heads[1]:
    sys.exit('the two frame_ssm.csv have different headers')
if os.path.exists(COMBINED):
    sys.exit(f'{COMBINED} exists already')
with open(COMBINED, 'wb') as f:
    f.write(heads[0] + b''.join(b if b.endswith(b'\n') else b + b'\n' for b in bodies))
fr = pd.read_csv(COMBINED, usecols=['segment', 'frame_idx'])
n_parts = [len(pd.read_csv(p, usecols=['segment'])) for p in parts]
print(f'{COMBINED}: {len(fr)} rows (validation {n_parts[0]} + training {n_parts[1]}), '
      f'{fr.segment.nunique()} segments, duplicated (segment, frame_idx): {int(fr.duplicated().sum())}')
assert len(fr) == sum(n_parts) and not fr.duplicated().any()

# 2. one root with both splits
main = pd.read_csv(os.path.join(repo, 'vlm_frames', 'main_all9_targets.csv'), comment='#')
segs = main[['split', 'segment']].drop_duplicates().sort_values('segment')
os.makedirs(SEGROOT, exist_ok=True)
for r in segs.itertuples():
    link = os.path.join(SEGROOT, r.segment)
    if not os.path.islink(link):
        os.symlink(os.path.join('..', '..', 'tables', r.split, r.segment), link)
    assert os.path.isdir(os.path.join(link, 'images')), link
names = segs.segment.tolist()
print(f'main list: {len(main)} frames in {len(names)} segments '
      f'(validation {int((segs.split == "validation").sum())}, training {int((segs.split == "training").sum())})')


# 3. images, then targets.csv from one pass over all segments
def make(layout, chunk):
    cmd = [sys.executable, '-I', MK, SEGROOT, COMBINED, os.path.join(OUT, layout), '--layout', layout] + chunk
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f'failed: {layout}, {len(chunk)} segments\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}')
    return r.stdout


with cf.ThreadPoolExecutor(16) as ex:
    futs = [ex.submit(make, lay, names[i::8]) for lay in ('grid', 'separate') for i in range(8)]
    for f in futs:
        f.result()
print('images made')
for lay in ('grid', 'separate'):
    log = make(lay, names)                                     # keeps the images, writes targets.csv for all segments
    print(f'{lay}: ' + log.strip().splitlines()[-1])

# 4. the main list's frames
cols = {'grid': ['images'], 'separate': ['images_by_time', 'images_by_camera']}
for lay in ('grid', 'separate'):
    d = os.path.join(OUT, lay)
    t = pd.read_csv(os.path.join(d, 'targets.csv'))
    m = main[['segment', 'frame_idx', 't_s']].merge(t, on=['segment', 'frame_idx'], how='left', suffixes=('_list', ''),
                                                    validate='one_to_one', indicator=True)
    if (m['_merge'] != 'both').any():
        sys.exit(f'{lay}: {int((m["_merge"] != "both").sum())} frames of the main list are not in targets.csv')
    bad_t = int((np.abs(m['t_s_list'] - m['t_s']) > 1e-9).sum())
    imgs = sorted({p for c in cols[lay] for v in m[c] for p in v.split(';')})
    missing = [p for p in imgs if not os.path.exists(os.path.join(d, p))]
    m[t.columns].to_csv(os.path.join(d, 'targets_main.csv'), index=False)
    print(f'{lay}: targets.csv {len(t)} target frames; targets_main.csv {len(m)} frames, t_s differing from the list '
          f'{bad_t}, images used {len(imgs)}, missing {len(missing)}')
    assert bad_t == 0 and not missing

# 5. sha256 of every image
for lay in ('grid', 'separate'):
    d = os.path.join(OUT, lay)
    files = sorted(os.path.relpath(os.path.join(p, f), d) for p, _, fs in os.walk(d) for f in fs if f.endswith('.jpg'))
    nbytes = 0
    with open(os.path.join(OUT, f'sha256_{lay}.txt'), 'w', encoding='utf-8', newline='\n') as fh:
        for f in files:
            with open(os.path.join(d, f), 'rb') as g:
                b = g.read()
            nbytes += len(b)
            fh.write(f'{hashlib.sha256(b).hexdigest()}  {f}\n')
    print(f'{lay}: {len(files)} images, {nbytes} bytes; sha256 in {os.path.join(OUT, f"sha256_{lay}.txt")}')
