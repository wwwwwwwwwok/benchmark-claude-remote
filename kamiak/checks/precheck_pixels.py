"""Pre-check before the vLLM vs official-code comparison (2026-10-26): do the official Qwen-Drive image code and the
Hugging Face processor vLLM uses turn the same PNG into the same pixel values?
  1. all 256 byte values through both normalisations: official (x / 255 - 0.5) / 0.5, and the processor's own
     (run on an image holding every value), in float32 and after the cast to bfloat16 the vision tower does;
  2. the 30 comparison PNGs through QwenDriveProcessor._patchify (budget 16,777,216, as preprocessor_config.json) and
     through the processor: grids, then pixel values in float32 and in bfloat16.
Runs on CPU; reads the official source from <src> (not installed anywhere).
usage: python -I precheck_pixels.py <official src folder> <original model folder> <vlm folder> <compare_inputs folder>
"""
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

src, orig, vlm, inputs = sys.argv[1:5]
sys.path.insert(0, src)
import transformers  # noqa: E402
from transformers import AutoImageProcessor, AutoTokenizer  # noqa: E402

from qwen_drive.configuration_qwen_drive import QwenDriveConfig  # noqa: E402
from qwen_drive.scene import CameraFrame, QwenDriveProcessor  # noqa: E402

ip = AutoImageProcessor.from_pretrained(vlm)
proc = QwenDriveProcessor(AutoTokenizer.from_pretrained(orig), QwenDriveConfig.from_pretrained(orig))
print(f'transformers {transformers.__version__}, torch {torch.__version__}; processor {type(ip).__name__}')

# 1. every byte value: a 256 x 256 image (65,536 pixels, the processor's minimum, so neither side resizes) whose
#    column j holds the value j in all three channels
strip = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
img = Image.fromarray(np.stack([strip] * 3, axis=-1))
p_off, (rows, cols) = proc._patchify(CameraFrame(img), 16777216)
out = ip(images=[img], return_tensors='pt')
hf = out['pixel_values']
print(f'1. all byte values: official patches {tuple(p_off.shape)} grid {(1, rows, cols)}, processor {tuple(hf.shape)} '
      f'grid {out["image_grid_thw"].tolist()}')
if p_off.shape == hf.shape:
    diff = p_off != hf
    bytes_diff = sorted({int(round((v + 1) / 2 * 255)) for v in p_off[diff].tolist()})
    print(f'   float32 equal: {torch.equal(p_off, hf)}; positions differing {int(diff.sum())} of {p_off.numel()}; '
          f'max |diff| {float((p_off - hf).abs().max()):.3g}; byte values affected {len(bytes_diff)}: {bytes_diff[:40]}')
    d16 = p_off.bfloat16() != hf.bfloat16()
    print(f'   after the bfloat16 cast equal: {not bool(d16.any())}; positions differing {int(d16.sum())}')

# 2. the comparison PNGs
n_img = n_grid = n32 = n16 = 0
worst = 0.0
for path in sorted(glob.glob(os.path.join(inputs, '*', '*', '*.png'))):
    im = Image.open(path)
    p_off, (rows, cols) = proc._patchify(CameraFrame(path), 16777216)
    out = ip(images=[im], return_tensors='pt')
    grid_ok = out['image_grid_thw'].tolist() == [[1, rows, cols]]
    n_img += 1
    n_grid += grid_ok
    n32 += bool(grid_ok and torch.equal(p_off, out['pixel_values']))
    n16 += bool(grid_ok and torch.equal(p_off.bfloat16(), out['pixel_values'].bfloat16()))
    if grid_ok:
        worst = max(worst, float((p_off - out['pixel_values']).abs().max()))
print(f'2. {n_img} PNGs: grids equal {n_grid}; pixel values equal in float32 {n32}, after bfloat16 cast {n16}; '
      f'max |diff| {worst:.3g}')
