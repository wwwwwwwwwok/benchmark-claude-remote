"""Make a folder that holds only the vision-language model of Qwen-Drive-1.0-4B, in the plain Qwen3.5 layout vLLM
loads (user, 2026-10-26; the original wraps it as QwenDriveForPlanning and names every tensor 'vlm.<name>').
  - config.json: the original's vlm_config as it is (Qwen3_5ForConditionalGeneration), except its top-level dtype,
    None in the original, set to bfloat16, the dtype of the weights and of its text_config;
  - model.safetensors: the original's tensors with the 'vlm.' prefix taken off each name; the header is written
    anew, the data part (every tensor's bytes, in the same order and at the same offsets) is copied byte for byte;
  - tokenizer, chat template, preprocessor and generation config: copied unchanged.
The original folder is not changed. Check the result with check_vlm_folder.py.
usage: python -I make_vlm_folder.py <original folder> <new folder>
"""
import json
import os
import shutil
import struct
import sys

COPY = ['chat_template.jinja', 'chat_template.json', 'generation_config.json', 'merges.txt',
        'preprocessor_config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json']
PREFIX = 'vlm.'

src, dst = sys.argv[1:3]
if os.path.exists(dst):
    sys.exit(f'{dst} exists: stop')
os.makedirs(dst)

vlm = dict(json.load(open(os.path.join(src, 'config.json')))['vlm_config'])
assert vlm['architectures'] == ['Qwen3_5ForConditionalGeneration'] and vlm['model_type'] == 'qwen3_5', vlm
assert vlm['dtype'] is None and vlm['text_config']['dtype'] == 'bfloat16'
vlm['dtype'] = 'bfloat16'
with open(os.path.join(dst, 'config.json'), 'w') as f:
    json.dump(vlm, f, indent=2)
    f.write('\n')
for name in COPY:
    shutil.copy2(os.path.join(src, name), os.path.join(dst, name))

src_st, dst_st = os.path.join(src, 'model.safetensors'), os.path.join(dst, 'model.safetensors')
with open(src_st, 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(n))
data_start = 8 + n
new = {}
if '__metadata__' in header:
    new['__metadata__'] = header.pop('__metadata__')
assert all(k.startswith(PREFIX) for k in header), [k for k in header if not k.startswith(PREFIX)][:5]
for k, v in header.items():
    new[k[len(PREFIX):]] = v
assert len(new) - ('__metadata__' in new) == len(header)
raw = json.dumps(new, separators=(',', ':')).encode()
raw += b' ' * (-len(raw) % 8)             # pad with spaces so the data part starts at a multiple of 8, as safetensors does
with open(dst_st, 'wb') as out, open(src_st, 'rb') as f:
    out.write(struct.pack('<Q', len(raw)))
    out.write(raw)
    f.seek(data_start)
    shutil.copyfileobj(f, out, 64 << 20)
print(f'{dst}: config.json, {len(COPY)} files copied, model.safetensors with {len(header)} tensors renamed; '
      f'header {n:,} -> {len(raw):,} bytes; data part {os.path.getsize(src_st) - data_start:,} bytes copied; '
      f'file {os.path.getsize(src_st):,} -> {os.path.getsize(dst_st):,} bytes')
