"""Follow-up of the second Qwen-Drive check (2026-10-27), for the report to the user:
1. vLLM's patch_embed hook saved nothing, its model.visual hook saved the float32 input; vLLM's own next step is
   IdentityInputNorm, pixel_values.to(bfloat16) (mm_device_do_normalize off). Cast the saved float32 input the same
   way, split it into images by grid_thw and compare each image's bfloat16 tensor with the official one (sha256 and
   exact equality); float32 against float32 as well (known to differ in the last bit).
2. The three positions of the long thinking generations where vLLM's top-1 differs and the official top-1 minus top-2
   is 0.1 or more: both sides' candidates and probabilities there, decoded.
usage: python -I followup.py <out folder> <first check's out folder> <vlm folder>
"""
import glob
import hashlib
import json
import math
import os
import sys

import torch
from transformers import AutoTokenizer

out, out1, vlm = sys.argv[1:4]


def sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


print('=== 1. vision input, vLLM float32 cast to bfloat16 as IdentityInputNorm does')
v16, v32 = {}, {}
for vp in sorted(glob.glob(os.path.join(out, 'vllm_vision', 'visual_*.pt'))):
    v = torch.load(vp)
    x = v['x'].reshape(v['x'].shape[0], -1)
    at = 0
    for g in v['grid_thw']:
        n = g[0] * g[1] * g[2]
        v16[sha(x[at:at + n].to(torch.bfloat16))] = x[at:at + n].to(torch.bfloat16)
        v32[sha(x[at:at + n])] = x[at:at + n]
        at += n
print(f'vLLM: dtype of the saved input {torch.load(sorted(glob.glob(os.path.join(out, "vllm_vision", "visual_*.pt")))[0])["x"].dtype}; '
      f'{len(v16)} distinct images (bfloat16), {len(v32)} (float32)')
for f in sorted(glob.glob(os.path.join(out, 'official_vision_*.pt'))):
    c = os.path.basename(f)[len('official_vision_'):-3]
    o = torch.load(f)
    off = json.load(open(os.path.join(out, f'official_{c}.json')))
    proj = o['proj_in'].reshape(o['pixel_values_f32'].shape[0], -1)
    f32 = o['pixel_values_f32']
    at = n16 = n32 = 0
    worst = 0.0
    for im, g in zip(off['per_image'], off['grids']):
        n = g[0] * g[1] * g[2]
        a16 = proj[at:at + n]
        hit = v16.get(sha(a16))
        n16 += hit is not None and torch.equal(hit, a16)
        n32 += sha(f32[at:at + n]) in v32
        at += n
    print(f'{c:10s}: images {len(off["per_image"])}; bfloat16 equal to one vLLM received {n16}; float32 equal {n32}')

print('\n=== 2. the three positions (thinking generations) with vLLM top-1 different and official p1 - p2 >= 0.1')
tok = AutoTokenizer.from_pretrained(vlm)
for c in ('grid', 'sep_time', 'sep_camera', 'ctrl_grid', 'ctrl_sep'):
    off = json.load(open(os.path.join(out1, f'official_{c}.json')))
    sco = json.load(open(os.path.join(out, f'vllm_scored_think_{c}.json')))
    for j, (g, row) in enumerate(zip(off['generated'], sco['top5'])):
        if row[0] and row[0][0] != g:
            (i1, i2), (p1, p2) = off['top2'][j]
            if p1 - p2 >= 0.1:
                vp = [round(math.exp(l), 4) for l in row[1]]
                ctx = tok.decode(off['generated'][max(0, j - 12):j])
                print(f'{c} position {j} (prompt {len(off["prompt_ids"])} + {j}): context ...{ctx!r}')
                print(f'   official: {tok.decode([i1])!r} {p1:.4f}, {tok.decode([i2])!r} {p2:.4f} (gap {p1 - p2:.4f})')
                print(f'   vLLM    : ' + ', '.join(f'{tok.decode([t])!r} {p}' for t, p in zip(row[0], vp)))
