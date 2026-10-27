"""Step 4 of the Qwen-Drive check: the user's pass criteria (2026-10-26), per request and overall.
(a) the prompt token ids, the image grids and the pixel values are the same on both sides: the official expanded
    prompt equals the ids the vLLM server used and the ids vLLM expanded when scoring; every image's grid; every
    image's tensor as the vision encoder receives it, bfloat16 (user: compared in bfloat16, what both models get),
    by sha256 (official per request against all tensors vLLM's workers received); float32 reported as well.
(b) vLLM scoring the official generation: at least 99% of positions with the same top-1 on both sides, and at every
    position where they differ the official top-1 minus top-2 probability below 0.1.
(c) both sides generating on their own: where they first part, the probability gaps there, the final scores.
usage: python -I report.py <requests json> <out folder>
"""
import glob
import hashlib
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.expanduser('~/benchmark-claude-remote/code'))
from run_vlm import parse  # noqa: E402

req_json, out = sys.argv[1:3]


def sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


# what vLLM's vision encoder received, split into images by grid_thw
v_bf16, v_f32, v_grids = set(), set(), []
for vp in sorted(glob.glob(os.path.join(out, 'vllm_vision', 'visual_*.pt'))):
    v = torch.load(vp)
    pp = vp.replace('visual_', 'patch_')
    patch = torch.load(pp) if os.path.exists(pp) else None
    x = v['x'].reshape(v['x'].shape[0], -1)
    at = 0
    for g in v['grid_thw']:
        n = g[0] * g[1] * g[2]
        v_f32.add(sha(x[at:at + n].float()) if x.dtype != torch.float32 else sha(x[at:at + n]))
        if patch is not None:
            v_bf16.add(sha(patch.reshape(patch.shape[0], -1)[at:at + n]))
        v_grids.append(list(g))
        at += n
print(f'vLLM vision encoder calls: {len(glob.glob(os.path.join(out, "vllm_vision", "visual_*.pt")))}, images '
      f'{len(v_grids)}, distinct bfloat16 tensors {len(v_bf16)}')

ok_all = True
agree_all = n_all = agree_1k = n_1k = 0
for r in json.load(open(req_json)):
    c = r['condition']
    off = json.load(open(os.path.join(out, f'official_{c}.json')))
    srv = json.load(open(os.path.join(out, f'vllm_server_{c}.json')))
    sco = json.load(open(os.path.join(out, f'vllm_scored_{c}.json')))
    print(f'\n== {c}')
    # (a)
    a_ids = off['prompt_ids'] == srv['prompt_ids'] == sco['prompt_ids']
    a_cont = sco['continuation'] == off['generated']
    a_grid = all(list(g) in v_grids for g in off['grids'])
    a_bf16 = sum(im['sha_bf16_proj_in'] in v_bf16 for im in off['per_image'])
    a_vis = all(im['sha_bf16_proj_in'] == im['sha_bf16_visual_in'] for im in off['per_image'])
    a_f32 = sum(im['sha_f32'] in v_f32 for im in off['per_image'])
    n_img = len(off['per_image'])
    a = a_ids and a_cont and a_grid and a_bf16 == n_img
    print(f"(a) prompt ids equal (official {len(off['prompt_ids'])}, server {len(srv['prompt_ids'] or [])}, scoring "
          f"{len(sco['prompt_ids'])}): {a_ids}; scored continuation is the official generation: {a_cont}; grids "
          f"{off['grids'][:2]}... all among vLLM's: {a_grid}; images whose bfloat16 vision input equals one vLLM "
          f"received: {a_bf16}/{n_img} (official visual input = conv input: {a_vis}); float32 equal: {a_f32}/{n_img} "
          f"-> {'PASS' if a else 'FAIL'}")
    # (b)
    gen, top2 = off['generated'], off['top2']
    v_top1 = [row[0][0] if row[0] else None for row in sco['top5']]
    agree = [v == g for v, g in zip(v_top1, gen)]
    gaps = [top2[j][1][0] - top2[j][1][1] for j, ok in enumerate(agree) if not ok]
    rate = sum(agree) / len(agree) if agree else float('nan')
    b = rate >= 0.99 and all(gp < 0.1 for gp in gaps)
    agree_all += sum(agree)
    n_all += len(agree)
    head = agree[:1000]                        # user, 2026-10-26: also the first 1,000 tokens, for reference only
    agree_1k += sum(head)
    n_1k += len(head)
    print(f"(b) positions {len(agree)}; same top-1 {sum(agree)} ({rate:.4%}); differing {len(gaps)}, official "
          f"top-1 minus top-2 there: max {max(gaps) if gaps else 0:.4f} -> {'PASS' if b else 'FAIL'}; first 1,000 "
          f"positions (reference only): {sum(head)}/{len(head)} ({sum(head) / len(head):.4%})")
    if gaps:
        print('    differing positions (official p1 - p2):',
              [(j, round(top2[j][1][0] - top2[j][1][1], 4)) for j, ok in enumerate(agree) if not ok][:20])
    # (c)
    sg = srv['generated'] or []
    first = next((j for j, (x, y) in enumerate(zip(gen, sg)) if x != y), None)
    if first is None and len(gen) != len(sg):
        first = min(len(gen), len(sg))
    off_text = off['text'].split('</think>')[-1]
    srv_ans = parse(srv['content'])
    off_ans = parse(off_text.replace('<|im_end|>', '').replace('<|endoftext|>', ''))
    if first is None:
        where = 'never: identical generations'
    else:
        og = top2[first][1][0] - top2[first][1][1] if first < len(top2) else float('nan')
        st = srv['top5'][first] if first < len(srv['top5']) else None
        sgap = math.exp(st[1][0]) - math.exp(st[1][1]) if st and len(st[1]) > 1 else float('nan')
        where = (f'token {first} (official {gen[first] if first < len(gen) else "end"}, vLLM '
                 f'{sg[first] if first < len(sg) else "end"}); official p1 - p2 there {og:.4f}, vLLM {sgap:.4f}')
    print(f"(c) official {len(gen)} tokens ({off['finish']}), vLLM {len(sg)} ({srv['finish_reason']}); first parting: "
          f"{where}; scores: official {off_ans.get('risk') if off_ans else None}, vLLM "
          f"{srv_ans.get('risk') if srv_ans else None}")
    ok_all = ok_all and a and b
print(f"\noverall (b): same top-1 at {agree_all}/{n_all} positions ({agree_all / n_all:.4%}); first 1,000 of each "
      f"request (reference only): {agree_1k}/{n_1k} ({agree_1k / n_1k:.4%})" if n_all else '')
print(f"ALL CRITERIA (a) AND (b) MET: {ok_all}")
