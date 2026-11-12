"""CPU step before the Gemma 4 31B-it runs (user, 2026-11-12). Nothing is sent to a model; the images are not shown.
Run in the vllm_cu129 environment (transformers 5.19.0, torchvision), from the repository root.
1. Image sizes: every picture the main and pilot target lists use (grid, and separate by camera), header only; the
   distinct sizes per kind (grid picture, front cameras FRONT_LEFT / FRONT / FRONT_RIGHT, side cameras SIDE_LEFT /
   SIDE_RIGHT).
2. Image tokens with the model's own processor (transformers AutoProcessor from the model folder): for each distinct
   size, the size the processor resizes to and its number of visual tokens (+ 2 for <|image> and <image|>) at
   max_soft_tokens 560, against the hand-computed table (534 / 534 / 546, i.e. 532 + 2, 532 + 2, 544 + 2); 280 for
   reference (272 / 262 / 266, i.e. 270 + 2, 260 + 2, 264 + 2; per request 1,360 / 6,590 / 272 / 1,318).
3. Comparison inputs (as for Qwen-Drive, user 2026-10-26: both sides get the same lossless PNGs, already at their final
   size, so neither resizes again): the first frame of the main list, its pictures for the 5 conditions resized once
   by the processor itself (its aspect_ratio_preserving_resize, captured while it processes the original JPEG at
   560: torchvision bicubic with antialias on uint8) and saved as PNG under <out>/grid/ and <out>/separate/ with the
   same relative paths; targets_compare.csv (one row) for build_requests.py, and targets_original.csv (the same row
   with the original JPEGs' absolute paths) for the counts with the originals in render_check.py.
usage: python -I prep_inputs.py <model folder> <main vlm_inputs folder> <pilot grid targets> <pilot sep targets> <out>
"""
import collections
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoProcessor

model, main_root, pilot_grid, pilot_sep, out = sys.argv[1:6]
if os.path.exists(out):
    sys.exit(f'{out} exists: stop')
HAND = {560: {'grid': 534, 'front': 534, 'side': 546}, 280: {'grid': 272, 'front': 262, 'side': 266}}
FRONT, SIDE = ('FRONT_LEFT', 'FRONT', 'FRONT_RIGHT'), ('SIDE_LEFT', 'SIDE_RIGHT')
bad = []


def kind(path, sub):
    if sub == 'grid':
        return 'grid'
    cam = os.path.basename(path).rsplit('_', 1)[0]
    return 'front' if cam in FRONT else 'side' if cam in SIDE else f'unknown camera {cam}'


print('=== 1. image sizes (header only)')
lists = {'main': (os.path.join(main_root, 'grid', 'targets_main.csv'), os.path.join(main_root, 'separate', 'targets_main.csv')),
         'pilot': (pilot_grid, pilot_sep)}
example = {}
for name, (g, s) in lists.items():
    for sub, csv, col in (('grid', g, 'images'), ('separate', s, 'images_by_time')):
        t = pd.read_csv(csv)
        root = os.path.dirname(os.path.abspath(csv))
        sizes = collections.Counter()
        for paths in t[col]:
            for p in paths.split(';'):
                full = os.path.join(root, p)
                with Image.open(full) as im:
                    k = (kind(p, sub), im.size)
                sizes[k] += 1
                example.setdefault(k, full)
        print(f'{name} {sub}: {len(t)} frames, {sum(sizes.values())} pictures; sizes (kind, (w, h)): {dict(sizes)}')
        for (k, wh) in sizes:
            if k.startswith('unknown'):
                bad.append(f'{name} {sub}: {k}')

print('\n=== 2. image tokens with the processor in the model folder')
proc = AutoProcessor.from_pretrained(model)
ip = proc.image_processor
print(f'processor {type(proc).__name__}, image processor {type(ip).__name__} (patch {ip.patch_size}, pooling '
      f'{ip.pooling_kernel_size}, default max_soft_tokens {ip.max_soft_tokens}, resample {ip.resample}, '
      f'do_rescale {ip.do_rescale}, do_normalize {ip.do_normalize})')
captured = []
orig_resize = ip.aspect_ratio_preserving_resize


def capture(*args, **kwargs):
    y = orig_resize(*args, **kwargs)
    captured.append(y)
    return y


ip.aspect_ratio_preserving_resize = capture
for budget in (560, 280):
    for (k, wh), full in sorted(example.items()):
        captured.clear()
        with Image.open(full) as im:
            res = ip(images=[im.convert('RGB')], max_soft_tokens=budget, return_tensors='pt')
        n = int(res['num_soft_tokens_per_image'][0])
        h, w = captured[0].shape[-2:]
        want = HAND[budget].get(k)
        ok = want == n + 2
        if not ok:
            bad.append(f'budget {budget} {k} {wh}: processor {n} + 2, hand {want}')
        print(f'budget {budget}: {k:5s} {wh[0]}x{wh[1]} -> {w}x{h}, {n} visual tokens + 2 = {n + 2}; hand table {want}: '
              f'{"same" if ok else "DIFFERENT"}')

print('\n=== 3. comparison inputs: the first frame of the main list, resized by the processor (560), saved as PNG')
first = pd.read_csv(lists['main'][0]).iloc[0]
for sub, csv, col in (('grid', lists['main'][0], 'images'), ('separate', lists['main'][1], 'images_by_time')):
    t = pd.read_csv(csv)
    root = os.path.dirname(os.path.abspath(csv))
    row = t[(t.segment == first.segment) & (t.frame_idx == first.frame_idx)].copy()
    assert len(row) == 1, (sub, len(row))
    new_paths = []
    for p in row.iloc[0][col].split(';'):
        captured.clear()
        with Image.open(os.path.join(root, p)) as im:
            w0, h0 = im.size
            ip(images=[im.convert('RGB')], max_soft_tokens=560, return_tensors='pt')
        y = captured[0]
        assert str(y.dtype) == 'torch.uint8' and y.shape[0] == 3, (y.dtype, y.shape)
        arr = np.ascontiguousarray(y.permute(1, 2, 0).numpy())
        q = os.path.splitext(p)[0] + '.png'
        os.makedirs(os.path.dirname(os.path.join(out, sub, q)), exist_ok=True)
        Image.fromarray(arr, 'RGB').save(os.path.join(out, sub, q), format='PNG')
        with Image.open(os.path.join(out, sub, q)) as back:          # lossless: read back equals what was saved
            assert np.array_equal(np.asarray(back.convert('RGB')), arr), q
        new_paths.append(q)
        print(f'{sub} {p}: {w0}x{h0} -> {arr.shape[1]}x{arr.shape[0]}')
    orig = row.copy()                       # the same frame's original JPEGs (absolute paths), for render_check.py
    orig[col] = ';'.join(os.path.join(root, p) for p in row.iloc[0][col].split(';'))
    orig.to_csv(os.path.join(out, sub, 'targets_original.csv'), index=False)
    row[col] = ';'.join(new_paths)
    row.to_csv(os.path.join(out, sub, 'targets_compare.csv'), index=False)
    print(f'{sub}: {row.iloc[0].segment} frame {row.iloc[0].frame_idx}, {len(new_paths)} pictures')

print(f'\nchecks not met: {len(bad)}')
for b in bad:
    print('  ', b)
sys.exit(1 if bad else 0)
