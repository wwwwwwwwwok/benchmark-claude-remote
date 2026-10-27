"""Second Qwen-Drive check (user, 2026-10-27: both 4B models without thinking), against the user's criteria:
(a) prompt token ids, image token counts and pixel values (bfloat16, as each vision encoder receives them) the same
    on both sides; the prompt must end with the empty thinking block;
(b) vLLM scoring the official greedy answers: over the 5 requests together at least 99% of positions with the same
    top-1, and at every position that differs the official top-1 minus top-2 probability below 0.1;
4.  vLLM scoring the 5 earlier official generations with thinking (their own prompts ending in <think>\\n): no
    position where vLLM's top-1 differs from the official one while the official top-1 minus top-2 is 0.1 or more;
    agreement over the first 1,000 tokens for reference only;
(c) both sides answering on their own: where they first part, the probability gaps there, the final scores;
5.  the empty thinking block the model wrote itself in the first check's vLLM answers against the tokens the
    template adds with enable_thinking false: the same token ids?
and the two servers' default sampling / stop tokens (check_defaults.out and the startup logs).
usage: python -I report.py <requests json> <out folder> <first check's out folder>
"""
import glob
import hashlib
import json
import math
import os
import re
import sys

import torch

sys.path.insert(0, os.path.expanduser('~/benchmark-claude-remote/code'))
from run_vlm import parse  # noqa: E402

req_json, out, out1 = sys.argv[1:4]
reqs = [r['condition'] for r in json.load(open(req_json))]
HEADER = [248045, 74455, 198]          # <|im_start|> assistant \n (decoded in the first check)


def added_by_template(ids):
    """The tokens after the last <|im_start|>assistant\\n: what the template puts there (official_greedy.py checked
    that the rendered text ends with <think>\\n\\n</think>\\n\\n)."""
    for i in range(len(ids) - len(HEADER), -1, -1):
        if ids[i:i + len(HEADER)] == HEADER:
            return ids[i + len(HEADER):]
    return None


def sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def load(folder, name):
    return json.load(open(os.path.join(folder, name)))


v_bf16, v_f32, v_grids = set(), set(), []
for vp in sorted(glob.glob(os.path.join(out, 'vllm_vision', 'visual_*.pt'))):
    v = torch.load(vp)
    pp = vp.replace('visual_', 'patch_')
    patch = torch.load(pp) if os.path.exists(pp) else None
    x = v['x'].reshape(v['x'].shape[0], -1)
    at = 0
    for g in v['grid_thw']:
        n = g[0] * g[1] * g[2]
        v_f32.add(sha(x[at:at + n]))
        if patch is not None:
            v_bf16.add(sha(patch.reshape(patch.shape[0], -1)[at:at + n]))
        v_grids.append(list(g))
        at += n
print(f'vLLM vision encoder: {len(glob.glob(os.path.join(out, "vllm_vision", "visual_*.pt")))} calls, '
      f'{len(v_grids)} images, {len(v_bf16)} distinct bfloat16 inputs')

ok = {}
print('\n=== (a) prompt, image tokens, vision input')
a_all = True
for c in reqs:
    off, srv, sco = load(out, f'official_{c}.json'), load(out, f'vllm_server_{c}.json'), load(out, f'vllm_scored_new_{c}.json')
    ids_eq = off['prompt_ids'] == srv['prompt_ids'] == sco['prompt_ids']
    added = added_by_template(off['prompt_ids'])
    tail_ok = bool(added) and off['text_prompt'].endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n')
    grid_ok = all(list(g) in v_grids for g in off['grids'])
    bf = sum(im['sha_bf16_proj_in'] in v_bf16 for im in off['per_image'])
    f32 = sum(im['sha_f32'] in v_f32 for im in off['per_image'])
    n = len(off['per_image'])
    a = ids_eq and tail_ok and grid_ok and bf == n
    a_all &= a
    print(f"{c:10s}: prompt ids equal (official {len(off['prompt_ids'])}, server {len(srv['prompt_ids'] or [])}, scoring "
          f"{len(sco['prompt_ids'])}): {ids_eq}; ends with the empty thinking block: {tail_ok}; image tokens "
          f"{sorted(set(off['image_tokens']))}, grids among vLLM's: {grid_ok}; bfloat16 vision input equal {bf}/{n}; "
          f"float32 equal {f32}/{n} -> {'PASS' if a else 'FAIL'}")
ok['a'] = a_all

print('\n=== (b) vLLM scoring the official answers (5 requests together)')
same = tot = 0
bad, diffs = [], []
for c in reqs:
    off, sco = load(out, f'official_{c}.json'), load(out, f'vllm_scored_new_{c}.json')
    for j, (g, row) in enumerate(zip(off['generated'], sco['top5'])):
        tot += 1
        if row[0] and row[0][0] == g:
            same += 1
        else:
            gap = off['top2'][j][1][0] - off['top2'][j][1][1]
            diffs.append((c, j, round(gap, 4)))
            if gap >= 0.1:
                bad.append((c, j, round(gap, 4)))
ok['b'] = tot > 0 and same / tot >= 0.99 and not bad
print(f'positions {tot}; same top-1 {same} ({same / tot:.4%}); differing {len(diffs)} {diffs[:20]}; of them with '
      f"official p1 - p2 >= 0.1: {len(bad)} -> {'PASS' if ok['b'] else 'FAIL'}")

print('\n=== 4. vLLM scoring the earlier official generations with thinking (long sequences)')
same = tot = same1k = tot1k = 0
big = []
for c in reqs:
    off, sco = load(out1, f'official_{c}.json'), load(out, f'vllm_scored_think_{c}.json')
    s_c = n_c = 0
    for j, (g, row) in enumerate(zip(off['generated'], sco['top5'])):
        hit = bool(row[0]) and row[0][0] == g
        n_c += 1
        s_c += hit
        if j < 1000:
            tot1k += 1
            same1k += hit
        if not hit:
            gap = off['top2'][j][1][0] - off['top2'][j][1][1]
            if gap >= 0.1:
                big.append((c, j, round(gap, 4)))
    same += s_c
    tot += n_c
    print(f'{c:10s}: {n_c} positions scored ({sco["dropped"]} at the end left out), same top-1 {s_c} ({s_c / n_c:.4%})')
ok['4'] = not big
print(f'all: {same}/{tot} ({same / tot:.4%}); first 1,000 of each (reference only): {same1k}/{tot1k} '
      f'({same1k / tot1k:.4%}); differing positions with official p1 - p2 >= 0.1: {len(big)} {big[:20]} '
      f"-> {'PASS' if ok['4'] else 'FAIL'}")

print('\n=== (c) both sides answering on their own (greedy)')
for c in reqs:
    off, srv = load(out, f'official_{c}.json'), load(out, f'vllm_server_{c}.json')
    gen, sg = off['generated'], srv['generated'] or []
    first = next((j for j, (x, y) in enumerate(zip(gen, sg)) if x != y), None)
    if first is None and len(gen) != len(sg):
        first = min(len(gen), len(sg))
    o_ans = parse(off['text'].replace('<|im_end|>', '').replace('<|endoftext|>', ''))
    s_ans = parse(srv['content'])
    if first is None:
        where = 'never: identical answers'
    else:
        og = off['top2'][first][1][0] - off['top2'][first][1][1] if first < len(off['top2']) else float('nan')
        st = srv['top5'][first] if first < len(srv['top5']) else None
        sgap = math.exp(st[1][0]) - math.exp(st[1][1]) if st and len(st[1]) > 1 else float('nan')
        where = f'token {first}; official p1 - p2 there {og:.4f}, vLLM {sgap:.4f}'
    print(f"{c:10s}: official {len(gen)} tokens ({off['finish']}), vLLM {len(sg)} ({srv['finish_reason']}); first "
          f"parting: {where}; scores: official {o_ans.get('risk') if o_ans else None}, vLLM "
          f"{s_ans.get('risk') if s_ans else None}")

print('\n=== 5. the empty thinking block: the model\'s own (first check, vLLM) against the template\'s')
eq5 = True
for c in reqs:
    added = added_by_template(load(out, f'official_{c}.json')['prompt_ids'])
    old = load(out1, f'vllm_server_{c}.json')['generated'] or []
    e = old[:len(added)] == added
    eq5 &= e
    print(f'{c:10s}: template adds {added}; the model wrote {old[:len(added)]} -> {"same" if e else "DIFFERENT"}')
ok['5'] = eq5

print('\n=== default sampling and stop tokens of the two servers')
cd = open(os.path.join(out, 'check_defaults.out')).read()
ok['defaults'] = 'SAME SAMPLING AND STOP TOKENS: True' in cd
print(cd.strip().splitlines()[-1])
for name in ('vllm_server.log', 'vllm_server_qwen3.5-4b.log'):
    log = open(os.path.join(out, name)).read()
    dflt = re.findall(r'Default vLLM sampling parameters have been overridden by ([^:]*): `([^`]*)`', log)
    spec = sorted(set(re.findall(r'speculative_config=([^,]*)', log)))
    args = re.findall(r"non-default args: (\{.*\})", log)
    print(f'{name}: default sampling override {dflt or "none"}; speculative_config {spec}; non-default args '
          f'{args[0][:400] if args else None}')
print(f'\nRESULT: {ok}; ALL MET: {all(ok.values())}')
