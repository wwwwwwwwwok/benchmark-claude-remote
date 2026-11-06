"""Input images of the new-segment check (user, 2026-11-06, exploratory; CLAUDE.md "二、新段检验的准备"): the comparison
list's frames from the 62 segments outside the main list's 241, made exactly as the main run's inputs
(kamiak/main_run/make_main_inputs.py): code/make_vlm_inputs.py as it is in the repository (default --tile 640,
--cam_width 960; DejaVu Sans Bold from fonts/), the same combined frame_ssm, segment links to the tables (the images
copied to lab storage on 2026-10-22).
  1. Check first (stop if not met): 3 segments of the main list (the first validation and the first two training
     segments by name) made again into <scratch>/repro/{grid,separate}: every image's sha256 must equal
     <lab>/vlm_inputs/sha256_{grid,separate}.txt, no image of theirs missing or extra, and their targets.csv rows must
     equal those of <lab>/vlm_inputs/{grid,separate}/targets.csv.
  2. The 62 segments into <lab>/vlm_inputs_new62/{grid,separate} (vlm_inputs/ is not touched): 4 processes per layout on
     disjoint segments, then one pass over all of them, which keeps the images and writes targets.csv.
  3. targets_new249.csv next to targets.csv: the comparison list's frames of these segments (checked: all there, the same
     t_s, every image there).
  4. sha256 of every image: <lab>/vlm_inputs_new62/sha256_{grid,separate}.txt.
usage: python -I make_new62_inputs.py <repo> <lab root> <scratch dir for the check (must not exist)>
"""
import concurrent.futures as cf
import hashlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd

repo, lab, scratch = sys.argv[1:4]
MK = os.path.join(repo, 'code', 'make_vlm_inputs.py')
COMBINED = os.path.join(lab, 'results_lead_final_excl', 'frame_ssm_validation_training.csv')
OLD = os.path.join(lab, 'vlm_inputs')
OUT = os.path.join(lab, 'vlm_inputs_new62')
TABLES = os.path.join(lab, 'tables')
LAYOUTS = ('grid', 'separate')
COLS = {'grid': ['images'], 'separate': ['images_by_time', 'images_by_camera']}
for p in (OUT, scratch):
    if os.path.exists(p):
        sys.exit(f'{p} exists already')
cmp_ = pd.read_csv(os.path.join(repo, 'vlm_frames', 'comparison_lead7_targets.csv'), comment='#')
main = pd.read_csv(os.path.join(repo, 'vlm_frames', 'main_all9_targets.csv'), comment='#')
new = cmp_[~cmp_.segment.isin(set(main.segment))]
segs = new[['split', 'segment']].drop_duplicates().sort_values('segment')
print(f'comparison list frames outside the main list\'s segments: {len(new)} in {len(segs)} segments '
      f'(validation {int((segs.split == "validation").sum())}, training {int((segs.split == "training").sum())})')


def make(root, out, layout, chunk):
    cmd = [sys.executable, '-I', MK, root, COMBINED, os.path.join(out, layout), '--layout', layout] + chunk
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f'failed: {layout}, {len(chunk)} segments\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}')
    return r.stdout


def links(root, rows):
    os.makedirs(root)
    for r in rows.itertuples():
        os.symlink(os.path.join(TABLES, r.split, r.segment), os.path.join(root, r.segment))
        assert os.path.isdir(os.path.join(root, r.segment, 'images')), r.segment


def sha_of(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


# 1. the check on 3 main segments
ms = main[['split', 'segment']].drop_duplicates().sort_values('segment')
pick = pd.concat([ms[ms.split == 'validation'].head(1), ms[ms.split == 'training'].head(2)])
rroot = os.path.join(scratch, 'repro', 'segments')
links(rroot, pick)
names = pick.segment.tolist()
for lay in LAYOUTS:
    make(rroot, os.path.join(scratch, 'repro'), lay, names)
bad = []
for lay in LAYOUTS:
    ref = {}
    with open(os.path.join(OLD, f'sha256_{lay}.txt'), encoding='utf-8') as f:
        for line in f:
            h_, p_ = line.rstrip('\n').split('  ', 1)
            if p_.split('/', 1)[0] in names:
                ref[p_] = h_
    d = os.path.join(scratch, 'repro', lay)
    got = {os.path.relpath(os.path.join(p, f), d): sha_of(os.path.join(p, f))
           for p, _, fs in os.walk(d) for f in fs if f.endswith('.jpg')}
    diff = [p for p in got if ref.get(p) != got[p]]
    miss = [p for p in ref if p not in got]
    t_new = pd.read_csv(os.path.join(d, 'targets.csv'), dtype=str).sort_values(['segment', 'frame_idx'])
    t_old = pd.read_csv(os.path.join(OLD, lay, 'targets.csv'), dtype=str)
    t_old = t_old[t_old.segment.isin(names)].sort_values(['segment', 'frame_idx'])
    same_t = t_new.reset_index(drop=True).equals(t_old.reset_index(drop=True))
    print(f'check {lay}: {len(got)} images made again for {names}; sha256 different or not in the list {len(diff)}, '
          f'listed but not made {len(miss)}; targets.csv rows ({len(t_new)}) the same: {same_t}')
    if diff or miss or not same_t or not got:
        bad.append(lay)
if bad:
    sys.exit(f'the check is not met ({bad}): stop, nothing made for the new segments')

# 2. the 62 segments
links(os.path.join(OUT, 'segments'), segs)
root = os.path.join(OUT, 'segments')
names = segs.segment.tolist()
with cf.ThreadPoolExecutor(8) as ex:
    futs = [ex.submit(make, root, OUT, lay, names[i::4]) for lay in LAYOUTS for i in range(4)]
    for f in futs:
        f.result()
for lay in LAYOUTS:
    log = make(root, OUT, lay, names)
    print(f'{lay}: ' + log.strip().splitlines()[-1])

# 3. the comparison list's frames there
for lay in LAYOUTS:
    d = os.path.join(OUT, lay)
    t = pd.read_csv(os.path.join(d, 'targets.csv'))
    m = new[['segment', 'frame_idx', 't_s']].merge(t, on=['segment', 'frame_idx'], how='left', suffixes=('_list', ''),
                                                   validate='one_to_one', indicator=True)
    if (m['_merge'] != 'both').any():
        sys.exit(f'{lay}: {int((m["_merge"] != "both").sum())} frames of the list are not in targets.csv')
    bad_t = int((np.abs(m['t_s_list'] - m['t_s']) > 1e-9).sum())
    imgs = sorted({p for c in COLS[lay] for v in m[c] for p in v.split(';')})
    missing = [p for p in imgs if not os.path.exists(os.path.join(d, p))]
    m[t.columns].to_csv(os.path.join(d, 'targets_new249.csv'), index=False)
    print(f'{lay}: targets.csv {len(t)} target frames; targets_new249.csv {len(m)} frames, t_s differing from the list '
          f'{bad_t}, images used {len(imgs)}, missing {len(missing)}')
    assert bad_t == 0 and not missing

# 4. sha256 of every image
for lay in LAYOUTS:
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
