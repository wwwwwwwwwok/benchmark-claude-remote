"""Reference for the Gemma 4 31B-it check (user, 2026-11-12): the official transformers code against itself. The same
model, the same bfloat16 pixel values (cast as vLLM casts them), the same 5 official greedy sequences of the diagnosis
(compare_out_bf16/official_<c>.json: prompt + generated tokens), only the attention implementation changed: sdpa (the
one the official runs used) and eager. Each scores the sequences token by token (one forward pass over the prompt and
the generated tokens, logits of the positions that predict the generated tokens: logits_to_keep), saving every
position's top-2 ids and probabilities.
Eager here is "eager_qchunk": transformers' Gemma 4 eager_attention_forward step for step (repeat_kv, q k^T * scaling,
softcap if given, + mask, softmax in float32 then the input dtype, @ v) computed for 1,024 query rows at a time. The
plain eager would need about 55 GB on one GPU for the two sep sequences (~14,700 tokens: 32 heads x L^2 in bfloat16
and float32 at once). It is registered for the text and the vision attention alike, with eager's mask. On the three
short sequences (grid, ctrl_grid, ctrl_sep) the plain eager is run too and compared with eager_qchunk.
Report: top-1 agreement between the implementations, the differing positions, both sides' top-2 log-probability
difference in steps of 0.125, exact ties; also each against the official generated tokens (as vLLM was compared).
usage: python -I official_two_attn.py <model folder> <requests json> <official folder> <out folder>
           --extra-path <folder with accelerate> [--chunk 1024]
"""
import argparse
import collections
import json
import math
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemma4_common import MAX_SOFT_TOKENS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('model')
ap.add_argument('req_json')
ap.add_argument('official')
ap.add_argument('out')
ap.add_argument('--extra-path', default=None)
ap.add_argument('--chunk', type=int, default=1024)
args = ap.parse_args()
if args.extra_path:
    sys.path.insert(0, args.extra_path)
import transformers  # noqa: E402
from torch import nn  # noqa: E402
from transformers import AttentionInterface, AutoModelForImageTextToText, AutoProcessor  # noqa: E402
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface  # noqa: E402
from transformers.models.gemma4 import modeling_gemma4 as mg  # noqa: E402

CONDS = ['grid', 'sep_time', 'sep_camera', 'ctrl_grid', 'ctrl_sep']
SHORT = ['grid', 'ctrl_grid', 'ctrl_sep']


def eager_qchunk(module, query, key, value, attention_mask, dropout=0.0, scaling=None, softcap=None, **kwargs):
    """mg.eager_attention_forward, 1,024 query rows at a time (every step is row-wise or a matmul over keys)."""
    if scaling is None:
        scaling = module.head_dim ** -0.5
    key_states = mg.repeat_kv(key, module.num_key_value_groups)
    value_states = mg.repeat_kv(value, module.num_key_value_groups)
    q_len = query.shape[2]
    out = torch.empty(query.shape[:3] + (value_states.shape[-1],), dtype=query.dtype, device=query.device)
    for s in range(0, q_len, args.chunk):
        e = min(q_len, s + args.chunk)
        w = torch.matmul(query[:, :, s:e], key_states.transpose(2, 3)) * scaling
        if softcap is not None:
            w = w / softcap
            w = torch.tanh(w)
            w = w * softcap
        if attention_mask is not None:
            w = w + attention_mask[:, :, s:e]
        w = nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        w = nn.functional.dropout(w, p=dropout, training=module.training)
        out[:, :, s:e] = torch.matmul(w, value_states)
    return out.transpose(1, 2).contiguous(), None


AttentionInterface.register('eager_qchunk', eager_qchunk)
AttentionMaskInterface.register('eager_qchunk', ALL_MASK_ATTENTION_FUNCTIONS['eager'])

proc = AutoProcessor.from_pretrained(args.model)
model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map='auto',
                                                    attn_implementation='sdpa').eval()
dev = model.get_input_embeddings().weight.device
print(f'transformers {transformers.__version__}, torch {torch.__version__}; model {type(model).__name__}', flush=True)
reqs = {r['condition']: r for r in json.load(open(args.req_json))}


def impl_names():
    names = {'model': model.config._attn_implementation}
    for k in ('text_config', 'vision_config'):
        sub = getattr(model.config, k, None)
        if sub is not None:
            names[k] = sub._attn_implementation
    return names


def score(cond):
    off = json.load(open(os.path.join(args.official, f'official_{cond}.json')))
    r = reqs[cond]
    imgs = [Image.open(v).convert('RGB') for k, v in r['parts'] if k == 'image']
    enc = proc(text=[off['text_prompt']], images=[imgs], add_special_tokens=False, return_tensors='pt',
               images_kwargs={'max_soft_tokens': MAX_SOFT_TOKENS})
    assert enc['input_ids'][0].tolist() == off['prompt_ids'], f'{cond}: prompt ids differ from the official run'
    gen = off['generated']
    extra = torch.tensor([gen[:-1]], dtype=enc['input_ids'].dtype)
    enc['input_ids'] = torch.cat([enc['input_ids'], extra], dim=1)
    enc['attention_mask'] = torch.cat([enc['attention_mask'], torch.ones_like(extra)], dim=1)
    if 'mm_token_type_ids' in enc:
        enc['mm_token_type_ids'] = torch.cat([enc['mm_token_type_ids'], torch.zeros_like(extra)], dim=1)
    enc = {k: v.to(dev) for k, v in enc.items() if torch.is_tensor(v)}
    enc = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in enc.items()}   # as vLLM
    with torch.inference_mode():
        logits = model(**enc, logits_to_keep=len(gen), use_cache=False).logits[0].float()
    p, i = torch.softmax(logits, dim=-1).topk(2, dim=-1)
    return [[ii, pp] for ii, pp in zip(i.tolist(), p.tolist())], gen


res = {c: {} for c in CONDS}
gens = {}
for impl, conds in (('sdpa', CONDS), ('eager_qchunk', CONDS), ('eager', SHORT)):
    if model.config._attn_implementation != impl:
        model.set_attn_implementation(impl)
    print(f'=== {impl}: implementations {impl_names()}', flush=True)
    for c in conds:
        t0 = time.time()
        try:
            res[c][impl], gens[c] = score(c)
            print(f'  {c}: {len(gens[c])} positions scored, {time.time() - t0:.0f} s', flush=True)
        except torch.OutOfMemoryError as e:
            print(f'  {c}: out of memory ({str(e)[:120]})', flush=True)
        torch.cuda.empty_cache()
for c in CONDS:
    with open(os.path.join(args.out, f'two_attn_{c}.json'), 'w') as f:
        json.dump(dict(condition=c, generated=gens.get(c), **res[c]), f)


def compare(a, b, label, conds):
    """a, b: implementation names or 'generated'. Steps: ln(p1/p2) / 0.125 of each side at the differing positions."""
    same = tot = ties_a = ties_b = 0
    rows = []
    for c in conds:
        if (a != 'generated' and a not in res[c]) or b not in res[c]:
            continue
        for j, g in enumerate(gens[c]):
            ta = (g, None) if a == 'generated' else res[c][a][j]
            tb = res[c][b][j]
            top_a = ta[0] if a == 'generated' else ta[0][0]
            tot += 1
            if a != 'generated':
                ties_a += ta[1][0] == ta[1][1]
            ties_b += tb[1][0] == tb[1][1]
            if top_a == tb[0][0]:
                same += 1
                continue
            sa = math.log(ta[1][0] / ta[1][1]) / 0.125 if a != 'generated' else float('nan')
            sb = math.log(tb[1][0] / tb[1][1]) / 0.125
            rows.append(dict(cond=c, pos=j, a_steps=sa, b_steps=sb, a_gap=(ta[1][0] - ta[1][1]) if a != 'generated'
                             else float('nan'), b_gap=tb[1][0] - tb[1][1],
                             a_tie=a != 'generated' and ta[1][0] == ta[1][1], b_tie=tb[1][0] == tb[1][1]))
    print(f'\n### {label}: {same}/{tot} same top-1 ({same / max(tot, 1):.4%}); differing positions {len(rows)}; exact '
          f'ties at all positions: {a} {ties_a if a != "generated" else "-"}, {b} {ties_b}; among the differing: {a} '
          f'{sum(r["a_tie"] for r in rows)}, {b} {sum(r["b_tie"] for r in rows)}; {a} top-2 gap >= 0.1 at '
          f'{sum(r["a_gap"] >= 0.1 for r in rows)}, {b} at {sum(r["b_gap"] >= 0.1 for r in rows)}')
    for c in conds:
        n_c = len(gens.get(c) or [])
        d_c = sum(r['cond'] == c for r in rows)
        if n_c and (a == 'generated' or a in res[c]) and b in res[c]:
            print(f'    {c:10s}: {n_c - d_c}/{n_c} ({(n_c - d_c) / n_c:.4%})')
    if a != 'generated':
        cross = collections.Counter((round(r['a_steps'], 3), round(r['b_steps'], 3)) for r in rows)
        print(f'    steps of 0.125 at the differing positions ({a} steps, {b} steps): count')
        for k_ in sorted(cross):
            print(f'      {k_}: {cross[k_]}')
    else:
        print(f'    {b} steps at the differing positions: '
              + str(dict(sorted(collections.Counter(round(r['b_steps'], 3) for r in rows).items()))))
    for r in rows:
        print(f"      {r['cond']:10s} {r['pos']:4d}  {a} {r['a_steps']:.3f} steps (gap {r['a_gap']:.4f}), {b} "
              f"{r['b_steps']:.3f} steps (gap {r['b_gap']:.4f})")
    return same, tot


compare('sdpa', 'eager_qchunk', 'official sdpa vs official eager (both scoring the same sequences)', CONDS)
compare('generated', 'eager_qchunk', 'official generated tokens vs official eager scoring (as vLLM vs official)', CONDS)
compare('generated', 'sdpa', 'official generated tokens vs official sdpa scoring (generation with a cache vs one pass)',
        CONDS)
compare('eager', 'eager_qchunk', 'check: plain eager vs eager_qchunk (short sequences)', SHORT)
