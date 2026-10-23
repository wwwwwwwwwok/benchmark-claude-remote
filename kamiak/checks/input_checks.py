"""Checks of the VLM input images before the formal experiment (user's requests, 2026-10-22).
 A. The 3 demo (pilot) segments made again on Kamiak with code/make_vlm_inputs.py (DejaVu Sans Bold from fonts/,
    BASIC layout), from the scratch extraction:
    - separate layout (no text on the images): every image byte-identical to the pilot's (results/vlm_inputs_separate,
      made on the office computer); targets.csv the same rows;
    - grid layout: compared with the pilot's grid images (Arial Bold): where the decoded pixels differ (expected: only
      at the camera labels); sha256 of every grid image -> manifest for the office computer;
    - the same grid images drawn with Pillow's default layout (libraqm on Kamiak): same bytes as BASIC or not.
 B. Inputs made from the images copied to lab storage vs from the scratch images, both layouts, for 3 segments of each
    split from the main list (seed 20261022): every file byte-identical.
usage: python -I input_checks.py <repo> <workspace> <lab root> <out dir>
"""
import concurrent.futures as cf
import hashlib
import io
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageFont, features

repo, ws, lab, out = sys.argv[1:5]
MK = os.path.join(repo, 'code', 'make_vlm_inputs.py')
FSSM = {sp: os.path.join(lab, 'results_lead_final_excl', sp, 'ssm', 'frame_ssm.csv') for sp in ('validation', 'training')}
DEMO = ['13184115878756336167_1354_000_1374_000', '14486517341017504003_3406_349_3426_349',
        '16229547658178627464_380_000_400_000']
TILE = 640
print(f'Pillow raqm available: {features.check("raqm")}, freetype {features.version("freetype2")}, '
      f'libjpeg-turbo {features.version("libjpeg_turbo")}')


def make(root, split, dst, layout, segs):
    cmd = [sys.executable, '-I', MK, root, FSSM[split], dst, '--layout', layout] + segs
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f'failed: {" ".join(cmd)}\n{r.stdout}\n{r.stderr}')
    return dst


def files(d):
    return sorted(os.path.relpath(os.path.join(p, f), d).replace(os.sep, '/') for p, _, fs in os.walk(d) for f in fs)


def read(p):
    with open(p, 'rb') as f:
        return f.read()


def compare(a, b, only=None):
    fa, fb = files(a), files(b)
    if only:
        fa, fb = [f for f in fa if f.endswith(only)], [f for f in fb if f.endswith(only)]
    if fa != fb:
        return f'file lists differ: {len(fa)} vs {len(fb)}, only in one: {sorted(set(fa) ^ set(fb))[:5]}'
    diff = [f for f in fa if read(os.path.join(a, f)) != read(os.path.join(b, f))]
    nbytes = sum(os.path.getsize(os.path.join(a, f)) for f in fa)
    return f'{len(fa)} files, {nbytes} bytes, {len(fa) - len(diff)} identical, {len(diff)} different' + \
        (f' (e.g. {diff[:3]})' if diff else '')


def same_targets(a, b):
    x, y = pd.read_csv(os.path.join(a, 'targets.csv')), pd.read_csv(os.path.join(b, 'targets.csv'))
    return f'targets.csv: {len(x)} vs {len(y)} rows, ' + ('same rows' if x.equals(y) else 'DIFFERENT rows')


def label_box_diff(a, b):
    """Decoded pixels that differ, by panel: the largest x / y offset from the panel's top-left corner."""
    x = np.asarray(Image.open(a).convert('RGB'), np.int16)
    y = np.asarray(Image.open(b).convert('RGB'), np.int16)
    if x.shape != y.shape:
        return None
    m = (x != y).any(axis=2)
    h_top = round(1280 * TILE / 1920)                       # front cameras 1920 x 1280, side cameras 1920 x 886
    ys, xs = np.nonzero(m)
    y0 = np.where(ys < h_top - 16, 0, h_top)                # differences just above the bottom row belong to it (JPEG)
    return int(m.sum()), int((xs % TILE).max()) if len(xs) else 0, int((ys - y0).max()) if len(ys) else 0, \
        int((ys - y0).min()) if len(ys) else 0


os.makedirs(out, exist_ok=True)
jobs = {}
with cf.ThreadPoolExecutor(8) as ex:
    val_root = os.path.join(ws, 'extracted', 'validation')
    jobs['demo_grid'] = ex.submit(make, val_root, 'validation', os.path.join(out, 'demo_grid'), 'grid', DEMO)
    jobs['demo_separate'] = ex.submit(make, val_root, 'validation', os.path.join(out, 'demo_separate'), 'separate', DEMO)
    main = pd.read_csv(os.path.join(repo, 'vlm_frames', 'main_all9_targets.csv'), comment='#')
    rng = np.random.default_rng(20261022)
    picks = {}
    for sp in ('validation', 'training'):
        segs = sorted(main.loc[main.split == sp, 'segment'].unique())
        picks[sp] = sorted(rng.choice(segs, 3, replace=False).tolist())
        for src, root in (('lab', os.path.join(lab, 'tables', sp)), ('scratch', os.path.join(ws, 'extracted', sp))):
            for lay in ('grid', 'separate'):
                jobs[f'copy_{sp}_{src}_{lay}'] = ex.submit(make, root, sp, os.path.join(out, f'copy_{sp}_{src}_{lay}'),
                                                           lay, picks[sp])
    for k, j in jobs.items():
        j.result()
print('all inputs made')

print('\n=== A. demo segments, Kamiak (DejaVu Sans Bold, BASIC layout) vs the pilot (office computer)')
pilot_g, pilot_s = os.path.join(repo, 'results', 'vlm_inputs'), os.path.join(repo, 'results', 'vlm_inputs_separate')
dg, ds = os.path.join(out, 'demo_grid'), os.path.join(out, 'demo_separate')
print('separate images vs pilot: ' + compare(ds, pilot_s, only='.jpg'))
print('separate ' + same_targets(ds, pilot_s))
print('grid images vs pilot (font changed, so expected to differ): ' + compare(dg, pilot_g, only='.jpg'))
print('grid ' + same_targets(dg, pilot_g))
worst = [0, 0, 0, 10 ** 9]
n_px = []
for f in [f for f in files(dg) if f.endswith('.jpg')]:
    r = label_box_diff(os.path.join(dg, f), os.path.join(pilot_g, f))
    if r is None:
        print(f'  {f}: different image size')
        continue
    n_px.append(r[0])
    worst = [max(worst[0], r[0]), max(worst[1], r[1]), max(worst[2], r[2]), min(worst[3], r[3])]
print(f'grid, decoded pixels that differ from the pilot: per image {min(n_px)}-{max(n_px)} pixels '
      f'(of {1920 * 722}); all within {worst[1] + 1} px of the panel left edge and {worst[3]} to {worst[2]} px of the '
      f'panel top (the labels sit at x 10, y 8 in each panel)')
with open(os.path.join(out, 'demo_grid_dejavu_sha256.txt'), 'w', encoding='utf-8', newline='\n') as fh:
    for f in [f for f in files(dg) if f.endswith('.jpg')]:
        fh.write(f'{hashlib.sha256(read(os.path.join(dg, f))).hexdigest()}  {f}\n')
print(f'sha256 manifest: {os.path.join(out, "demo_grid_dejavu_sha256.txt")}')

# the same images with Pillow's default layout (libraqm here): does the layout engine change the bytes?
sys.path.insert(0, os.path.join(repo, 'code'))
import make_vlm_inputs as mvi  # noqa: E402

mvi.font = lambda size: ImageFont.truetype(mvi.FONT_FILE, size)
same = differ = 0
for seg in DEMO:
    fl = mvi.camera_files(os.path.join(ws, 'extracted', 'validation', seg))
    for f in [f for f in files(os.path.join(dg, seg)) if f.endswith('.jpg')]:
        buf = io.BytesIO()
        mvi.step_image(fl, int(f[5:8]), TILE).save(buf, format='JPEG', quality=90)
        if buf.getvalue() == read(os.path.join(dg, seg, f)):
            same += 1
        else:
            differ += 1
print(f'grid drawn with the default layout ({"raqm" if features.check("raqm") else "basic"}) vs BASIC: '
      f'{same} identical, {differ} different')

print('\n=== B. copied images (lab tables) vs scratch images, inputs made from each')
for sp in ('validation', 'training'):
    print(f'{sp}: segments {picks[sp]}')
    for lay in ('grid', 'separate'):
        a, b = os.path.join(out, f'copy_{sp}_lab_{lay}'), os.path.join(out, f'copy_{sp}_scratch_{lay}')
        n_t = len(pd.read_csv(os.path.join(a, 'targets.csv')))
        print(f'  {lay}: {n_t} target frames; all files incl. targets.csv: {compare(a, b)}')
