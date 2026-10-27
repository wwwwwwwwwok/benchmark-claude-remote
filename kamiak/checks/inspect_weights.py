"""Tensor names in a safetensors file (header only, no tensors read) next to a reference checkpoint's names
(2026-10-26, for Qwen-Drive-1.0-4B against Qwen3.6-27B, the same Qwen3.5 architecture): count by dtype, the name
prefixes (first 3 parts), and how the names differ after the layer numbers are taken out.
usage: python -I inspect_weights.py <model.safetensors> <reference model folder>
"""
import collections
import json
import os
import re
import struct
import sys


def header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n))


h = header(sys.argv[1])
meta = h.pop('__metadata__', None)
print(f'{sys.argv[1]}: {len(h)} tensors; metadata {meta}')
print('  dtypes:', dict(collections.Counter(v['dtype'] for v in h.values())))
print('  parameters:', f"{sum(eval('*'.join(map(str, v['shape'])) or '1') for v in h.values()):,}")
pre = collections.Counter('.'.join(k.split('.')[:3]) for k in h)
for p, n in sorted(pre.items()):
    print(f'  {p}: {n}')

ref = sys.argv[2]
idx = os.path.join(ref, 'model.safetensors.index.json')
names = set(json.load(open(idx))['weight_map']) if os.path.exists(idx) else set(header(os.path.join(ref, 'model.safetensors')))
pre_r = collections.Counter('.'.join(k.split('.')[:3]) for k in names)
print(f'reference {ref}: {len(names)} tensors')
for p, n in sorted(pre_r.items()):
    print(f'  {p}: {n}')
strip = lambda k: re.sub(r'\.\d+\.', '.N.', k)
a, b = {strip(k) for k in h}, {strip(k) for k in names}
print(f'name patterns (layer numbers as N): this {len(a)}, reference {len(b)}; in both {len(a & b)}')
print('  only here:', sorted(a - b)[:40])
print('  only in the reference:', sorted(b - a)[:40])
