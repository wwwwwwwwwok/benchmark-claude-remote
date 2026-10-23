"""Where the demo grid images made with DejaVu Sans Bold differ from the pilot's (Arial Bold): decoded pixels outside
the 5 camera-label corners (each panel's top-left 176 x 64 px, plus 1 px for the JPEG blocks next to them) should be 0.
usage: python -I label_regions.py <new grid dir> <pilot grid dir>
"""
import os
import sys

import numpy as np
from PIL import Image

new, old = sys.argv[1:3]
TILE, H_TOP = 640, 427
allowed = np.zeros((722, 1920), bool)
for x0, y0 in ((0, 0), (640, 0), (1280, 0), (0, H_TOP), (1280, H_TOP)):
    allowed[max(0, y0 - 16):y0 + 64, max(0, x0 - 1):x0 + 176] = True
n_img = n_out = 0
inside = []
for seg in sorted(os.listdir(new)):
    if not os.path.isdir(os.path.join(new, seg)):
        continue
    for f in sorted(os.listdir(os.path.join(new, seg))):
        a = np.asarray(Image.open(os.path.join(new, seg, f)).convert('RGB'), np.int16)
        b = np.asarray(Image.open(os.path.join(old, seg, f)).convert('RGB'), np.int16)
        m = (a != b).any(axis=2)
        n_img += 1
        n_out += int((m & ~allowed).sum())
        inside.append(int((m & allowed).sum()))
print(f'{n_img} images: differing pixels inside the label corners {min(inside)}-{max(inside)} per image, '
      f'outside them {n_out} in all')
