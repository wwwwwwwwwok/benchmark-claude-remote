"""Check the folder made by make_vlm_folder.py against the original Qwen-Drive-1.0-4B folder (user, 2026-10-26:
every tensor the same as in the original):
  1. headers: the same tensors, each named as in the original without 'vlm.', with the same dtype, shape and offsets;
  2. bytes: each tensor's bytes in the new file equal those in the original (sha256 of each), and the whole data parts
     are equal;
  3. with --load (needs safetensors and torch): each tensor read with safetensors.safe_open from both files,
     torch.equal and the same dtype;
  4. config.json is the original's vlm_config except dtype; the copied files are byte for byte the same.
usage: python -I check_vlm_folder.py <original folder> <new folder> [--load]
"""
import hashlib
import json
import os
import struct
import sys

src, dst = sys.argv[1:3]
load = '--load' in sys.argv[3:]
PREFIX = 'vlm.'


def header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop('__metadata__', None)
    return h, 8 + n


src_st, dst_st = os.path.join(src, 'model.safetensors'), os.path.join(dst, 'model.safetensors')
ha, sa = header(src_st)
hb, sb = header(dst_st)
names_ok = sorted(PREFIX + k for k in hb) == sorted(ha)
meta_ok = all(hb[k] == ha[PREFIX + k] for k in hb) if names_ok else False
print(f'1. headers: original {len(ha)} tensors, new {len(hb)}; names = original without "{PREFIX}": {names_ok}; '
      f'dtype, shape and offsets all the same: {meta_ok}')

same = 0
with open(src_st, 'rb') as fa, open(dst_st, 'rb') as fb:
    for k, v in hb.items():
        lo, hi = v['data_offsets']
        fa.seek(sa + lo)
        fb.seek(sb + lo)
        same += hashlib.sha256(fa.read(hi - lo)).digest() == hashlib.sha256(fb.read(hi - lo)).digest()
    fa.seek(sa)
    fb.seek(sb)
    whole = True
    while True:
        a, b = fa.read(64 << 20), fb.read(64 << 20)
        if a != b:
            whole = False
            break
        if not a:
            break
print(f'2. bytes: tensors with the same bytes {same}/{len(hb)}; data parts equal as a whole: {whole} '
      f'({os.path.getsize(src_st) - sa:,} and {os.path.getsize(dst_st) - sb:,} bytes)')

if load:
    import torch
    from safetensors import safe_open
    eq = 0
    dtypes = {}
    with safe_open(src_st, 'pt') as fa, safe_open(dst_st, 'pt') as fb:
        for k in fb.keys():
            a, b = fa.get_tensor(PREFIX + k), fb.get_tensor(k)
            eq += a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
            dtypes[str(b.dtype)] = dtypes.get(str(b.dtype), 0) + 1
    print(f'3. loaded with safetensors (torch {torch.__version__}): torch.equal and the same dtype {eq}/{len(hb)}; '
          f'dtypes {dtypes}')

vlm = json.load(open(os.path.join(src, 'config.json')))['vlm_config']
new = json.load(open(os.path.join(dst, 'config.json')))
diff = sorted(k for k in set(vlm) | set(new) if vlm.get(k) != new.get(k))
copied = sorted(n for n in os.listdir(dst) if n not in ('config.json', 'model.safetensors'))
same_files = [n for n in copied if open(os.path.join(src, n), 'rb').read() == open(os.path.join(dst, n), 'rb').read()]
print(f'4. config.json differs from the original vlm_config only in {diff} '
      f'({vlm.get("dtype")!r} -> {new.get("dtype")!r}); copied files the same: {len(same_files)}/{len(copied)} {copied}')
