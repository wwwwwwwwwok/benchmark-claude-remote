"""Does the qwen_drive environment (torch 2.8.0+cu128) run on Kamiak's H100s with driver 535 (CUDA 12.2)? (user,
2026-10-26: if not, stop and ask.) torch on the GPU, a bfloat16 matmul; causal-conv1d's and flash-linear-attention's
GPU kernels on small random inputs, against plain torch; the official Qwen-Drive code: load the model and greedily
generate 16 tokens from a short text-only prompt (which gated-delta-net path transformers takes is printed).
usage: python -I gpu_env_test.py <official src folder> <original model folder>
"""
import logging
import subprocess
import sys
import time
import warnings

import torch
import torch.nn.functional as F

src, orig = sys.argv[1:3]
sys.path.insert(0, src)
print(subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip())
print(f'torch {torch.__version__}, built with CUDA {torch.version.cuda}; cuda available {torch.cuda.is_available()}; '
      f'{torch.cuda.device_count()} device(s): {torch.cuda.get_device_name(0)}, capability {torch.cuda.get_device_capability(0)}')

a = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
c = a @ a
torch.cuda.synchronize()
print(f'bf16 matmul: finite {bool(torch.isfinite(c).all())}')

from causal_conv1d import causal_conv1d_fn  # noqa: E402
x = torch.randn(1, 64, 128, device='cuda', dtype=torch.bfloat16)
w = torch.randn(64, 4, device='cuda', dtype=torch.bfloat16)
y = causal_conv1d_fn(x, w, None, activation=None)
ref = F.conv1d(F.pad(x.float(), (3, 0)), w.float().unsqueeze(1), groups=64)
print(f'causal_conv1d: max |diff| to torch {float((y.float() - ref).abs().max()):.3g}')

from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: E402
B, T, H, K, V = 1, 256, 4, 64, 64
q, k = (F.normalize(torch.randn(B, T, H, K, device='cuda'), dim=-1).bfloat16() for _ in range(2))
v = torch.randn(B, T, H, V, device='cuda', dtype=torch.bfloat16)
g = F.logsigmoid(torch.randn(B, T, H, device='cuda'))
beta = torch.rand(B, T, H, device='cuda').sigmoid()
o, _ = chunk_gated_delta_rule(q, k, v, g, beta)
torch.cuda.synchronize()
print(f'fla chunk_gated_delta_rule (triton): output {tuple(o.shape)}, finite {bool(torch.isfinite(o).all())}')

logging.basicConfig(level=logging.WARNING)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    from qwen_drive import QwenDriveForPlanning  # noqa: E402
    t0 = time.time()
    model = QwenDriveForPlanning.from_pretrained(orig, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    print(f'official model loaded in {time.time() - t0:.0f} s; vlm class {type(model.vlm).__name__}, dtype {model.vlm.dtype}')
    tok = model.processor.tokenizer
    ids = tok.encode('<|im_start|>user\nWhat is a traffic light?<|im_end|>\n<|im_start|>assistant\n', add_special_tokens=False)
    inp = torch.tensor([ids], device='cuda')
    out = model.vlm.generate(input_ids=inp, attention_mask=torch.ones_like(inp), do_sample=False, max_new_tokens=16)
    print(f'greedy 16 tokens: {tok.decode(out[0, len(ids):])!r}')
for w_ in caught:
    if 'fast path' in str(w_.message).lower() or 'fla' in str(w_.message).lower() or 'causal' in str(w_.message).lower():
        print('warning:', str(w_.message)[:300])
import transformers.models.qwen3_5.modeling_qwen3_5 as m  # noqa: E402
print('transformers qwen3_5 fast path pieces:', {n: getattr(m, n, None) is not None for n in
      ('causal_conv1d_fn', 'causal_conv1d_update', 'chunk_gated_delta_rule', 'fused_recurrent_gated_delta_rule')})
