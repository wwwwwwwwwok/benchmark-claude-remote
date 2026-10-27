"""Follow-up 2 of the second Qwen-Drive check (2026-10-27). vLLM 0.32 normalises on the GPU (mm_device_do_normalize
defaults to True): its vision encoder receives raw uint8 pixels and FusedMMInputNorm computes x * w + b in float32
(w = rescale / std, b = -mean / std, built as in its __init__) with a Triton kernel, then stores bfloat16. Run on one
GPU with vLLM's own kernel (fused_mm_input_norm_triton) and its native path:
1. every byte value 0-255: official (x / 255 - 0.5) / 0.5 then bfloat16, against vLLM's kernel and native path;
2. the raw pixels vLLM's encoder received (saved by the model.visual hook) against the official pixels (bytes
   recovered from the official float32 values), image by image;
3. vLLM's kernel applied to those received bytes against the bfloat16 tensor the official encoder received.
usage: python -I followup2.py <out folder>
"""
import glob
import hashlib
import json
import os
import sys

import torch
from vllm.model_executor.layers.fusion.mm_input_norm import fused_mm_input_norm_triton

out = sys.argv[1]
mean, std, rescale = [0.5] * 3, [0.5] * 3, 1 / 255          # preprocessor_config.json; rescale_factor default 1/255
m = torch.tensor(mean, dtype=torch.float32)
s = torch.tensor(std, dtype=torch.float32)
w, b = (rescale / s).cuda(), (-m / s).cuda()                  # as FusedMMInputNorm.__init__


def kernel(x_u8):
    """vLLM's forward_cuda: (patches, 3 * L) uint8 -> bfloat16."""
    p, size = x_u8.shape
    x3 = x_u8.reshape(p, 3, size // 3)
    y = torch.empty((p, size), dtype=torch.bfloat16, device=x_u8.device)
    fused_mm_input_norm_triton(x3, y.view(p, 3, size // 3), w, b)
    return y


def native(x_u8):
    p, size = x_u8.shape
    x = x_u8.reshape(p, 3, size // 3) * w.view(1, 3, 1) + b.view(1, 3, 1)
    return x.view(p, size).to(torch.bfloat16)


def sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


print(f'GPU {torch.cuda.get_device_name(0)}; torch {torch.__version__}')
v = torch.arange(256, dtype=torch.uint8).repeat_interleave(1536).reshape(256, 1536).cuda()
off = torch.arange(256, dtype=torch.float32).div(255.0).sub(0.5).div(0.5).to(torch.bfloat16)
k, nat = kernel(v)[:, 0].cpu(), native(v)[:, 0].cpu()
print(f'1. byte values: vLLM kernel = official bfloat16 for {int((k == off).sum())}/256, vLLM native = official '
      f'{int((nat == off).sum())}/256, kernel = native {int((k == nat).sum())}/256')

vl = {}
for vp in sorted(glob.glob(os.path.join(out, 'vllm_vision', 'visual_*.pt'))):
    d = torch.load(vp)
    x = d['x'].reshape(d['x'].shape[0], -1)
    at = 0
    for g in d['grid_thw']:
        n = g[0] * g[1] * g[2]
        u8 = x[at:at + n].contiguous()
        vl[sha(u8)] = u8
        at += n
print(f'vLLM encoder inputs: dtype {next(iter(vl.values())).dtype}, {len(vl)} distinct images')
for f in sorted(glob.glob(os.path.join(out, 'official_vision_*.pt'))):
    c = os.path.basename(f)[len('official_vision_'):-3]
    o = torch.load(f)
    offj = json.load(open(os.path.join(out, f'official_{c}.json')))
    f32 = o['pixel_values_f32']
    proj = o['proj_in'].reshape(f32.shape[0], -1)
    at = same_u8 = same_bf = 0
    for g in offj['grids']:
        n = g[0] * g[1] * g[2]
        u8 = torch.round((f32[at:at + n] + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        hit = vl.get(sha(u8))
        same_u8 += hit is not None and torch.equal(hit, u8)
        if hit is not None:
            same_bf += torch.equal(kernel(hit.cuda()).cpu(), proj[at:at + n])
        at += n
    print(f'{c:10s}: {len(offj["grids"])} images; 2. raw pixels equal {same_u8}; 3. bfloat16 equal {same_bf}')
