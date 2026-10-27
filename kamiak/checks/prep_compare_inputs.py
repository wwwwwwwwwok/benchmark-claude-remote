"""Inputs for the vLLM vs official-code check of Qwen-Drive-1.0-4B (user, 2026-10-26): both sides get the same lossless
PNGs, already at their final size, so neither resizes again. The frame is the first of the main list; its images for
the 5 conditions (5 grid pictures, 25 separate ones) are resized once to the size the model's preprocessor_config.json
gives (smart_resize: sides to multiples of 32, 65,536 to 16,777,216 pixels), with PIL bicubic, and saved as PNG under
<out>/grid/ and <out>/separate/ with the same relative paths as in vlm_inputs; plus targets CSVs (one row) pointing at
them, for run_vlm.build. Nothing is written into the repository; the images are not shown.
usage: python -I prep_compare_inputs.py <vlm_inputs folder> <out folder>
"""
import math
import os
import sys

import pandas as pd
from PIL import Image

FACTOR, MIN_PIXELS, MAX_PIXELS = 32, 65536, 16777216          # preprocessor_config.json: patch 16 x merge 2, size


def smart_resize(h, w):
    """As in transformers' Qwen2-VL image processor (and qwen_drive.scene.smart_resize with these limits)."""
    hb, wb = round(h / FACTOR) * FACTOR, round(w / FACTOR) * FACTOR
    if hb * wb > MAX_PIXELS:
        beta = math.sqrt(h * w / MAX_PIXELS)
        hb, wb = math.floor(h / beta / FACTOR) * FACTOR, math.floor(w / beta / FACTOR) * FACTOR
    elif hb * wb < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (h * w))
        hb, wb = math.ceil(h * beta / FACTOR) * FACTOR, math.ceil(w * beta / FACTOR) * FACTOR
    return hb, wb


src, out = sys.argv[1:3]
if os.path.exists(out):
    sys.exit(f'{out} exists: stop')
first = pd.read_csv(os.path.join(src, 'grid', 'targets_main.csv')).iloc[0]
for sub, col in (('grid', 'images'), ('separate', 'images_by_time')):
    t = pd.read_csv(os.path.join(src, sub, 'targets_main.csv'))
    row = t[(t.segment == first.segment) & (t.frame_idx == first.frame_idx)].copy()
    assert len(row) == 1, (sub, len(row))
    new_paths = []
    for p in row.iloc[0][col].split(';'):
        im = Image.open(os.path.join(src, sub, p)).convert('RGB')
        w, h = im.size
        hb, wb = smart_resize(h, w)
        if (wb, hb) != (w, h):
            im = im.resize((wb, hb), Image.BICUBIC)
        q = os.path.splitext(p)[0] + '.png'
        os.makedirs(os.path.dirname(os.path.join(out, sub, q)), exist_ok=True)
        im.save(os.path.join(out, sub, q), format='PNG')
        new_paths.append(q)
        print(f'{sub} {p}: {w}x{h} -> {wb}x{hb}')
    row[col] = ';'.join(new_paths)
    row.to_csv(os.path.join(out, sub, 'targets_compare.csv'), index=False)
    print(f'{sub}: {row.iloc[0].segment} frame {row.iloc[0].frame_idx}, {len(new_paths)} images')
