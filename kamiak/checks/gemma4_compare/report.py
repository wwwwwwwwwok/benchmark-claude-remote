"""Report of the Gemma 4 31B-it check (user, 2026-11-12) against the standard of the Qwen-Drive check (user, 2026-10-26;
docs/handoff.md "Qwen-Drive-1.0-4B 和 Qwen3.5-4B 的设置"):
(a) the prompt token ids the same in the official code, the vLLM server and vLLM scoring (the prompt opens the system
    turn with <|think|> and ends with <|turn>model\\n); the image tokens per request equal to the table; the pixel
    values the same: per picture, the bfloat16 tensor the patch embedder's input projection gets (what the model
    computes with), real patches only; the float32 values before it and the patch positions are compared too;
(b) vLLM scoring the official greedy generations (thinking included): over the 5 requests together at least 99% of the
    positions with the same top-1, and at every position that differs the official top-1 minus top-2 probability
    below 0.1;
(c) both sides generating on their own (greedy): where they first part, the probability gaps there, the final
    scores, reported as they are.
Also reported: lengths and finish, whether the thinking was closed (<channel|>), agreement over the first 1,000
positions (reference only), and whether a generation fell into repetition (count_cut_replies.py's rule: in its last
1,000 tokens at least 95% of the steps have a top-1 probability of 0.99 or more). Exit code 1 unless (a) and (b) met.
usage: python -I report.py <requests json> <out folder>
"""
import glob
import hashlib
import json
import math
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.expanduser('~/benchmark-claude-remote/code'))
from gemma4_common import TABLE  # noqa: E402
from run_vlm import parse  # noqa: E402

req_json, out = sys.argv[1:3]
reqs = [r['condition'] for r in json.load(open(req_json))]


def sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def load(name):
    return json.load(open(os.path.join(out, name)))


# what vLLM's vision encoder received (rank 0): per picture, real patches only
v_pv, v_proj, v_pos, n_pe = set(), set(), set(), 0
pv_dtypes, proj_dtypes = set(), set()
for pe_path in sorted(glob.glob(os.path.join(out, 'vllm_vision', 'pe_*.pt'))):
    n_pe += 1
    d = torch.load(pe_path)
    proj_path = pe_path.replace('pe_', 'proj_')
    proj = torch.load(proj_path) if os.path.exists(proj_path) else None
    pvs = d['pv'] if isinstance(d['pv'], list) else list(d['pv'])
    poss = d['pos'] if isinstance(d['pos'], list) else list(d['pos'])
    for b, (pv, pos) in enumerate(zip(pvs, poss)):
        real = ~(pos == -1).all(dim=-1)
        v_pv.add(sha(pv[real]))
        v_pos.add(sha(pos[real].to(torch.int64)))
        pv_dtypes.add(str(pv.dtype))
        if proj is not None:
            v_proj.add(sha(proj[b][real]))
            proj_dtypes.add(str(proj.dtype))
n_entry = len(glob.glob(os.path.join(out, 'vllm_vision', 'entry_*.pt')))
print(f'vLLM vision encoder (rank 0): {n_pe} patch-embedder calls, {len(v_pv)} distinct pictures (pixel values '
      f'{sorted(pv_dtypes)}, input projection {sorted(proj_dtypes)}); {n_entry} records at _process_image_input')

ok = {}
print('\n=== (a) prompt, image tokens, pixel values')
a_all = True
for c in reqs:
    off, srv, sco = load(f'official_{c}.json'), load(f'vllm_server_{c}.json'), load(f'vllm_scored_new_{c}.json')
    ids_eq = off['prompt_ids'] == srv['prompt_ids'] == sco['prompt_ids']
    head_tail = off['text_prompt'].startswith('<bos><|turn>system\n<|think|>\n') and \
        off['text_prompt'].endswith('<|turn>model\n')
    n_img = sum(n + 2 for n in off['image_tokens'])
    n = len(off['per_image'])
    proj = sum(im['sha_proj'] in v_proj for im in off['per_image'])
    pv = sum(im['sha_pv'] in v_pv for im in off['per_image'])
    pos = sum(im['sha_pos'] in v_pos for im in off['per_image'])
    a = ids_eq and head_tail and n_img == TABLE[c] and proj == n and pos == n
    a_all &= a
    print(f"{c:10s}: prompt ids equal (official {len(off['prompt_ids'])}, server {len(srv['prompt_ids'] or [])}, scoring "
          f"{len(sco['prompt_ids'])}): {ids_eq}; <|think|> at the top and ends with <|turn>model\\n: {head_tail}; image "
          f"tokens {n_img} (table {TABLE[c]}; per picture {sorted(set(off['image_tokens']))}); bfloat16 at the input "
          f"projection equal {proj}/{n}; float32 pixel values equal {pv}/{n}; positions equal {pos}/{n} -> "
          f"{'PASS' if a else 'FAIL'}")
ok['a'] = a_all

print('\n=== (b) vLLM scoring the official greedy generations (5 requests together)')
same = tot = same1k = tot1k = 0
bad, diffs = [], []
for c in reqs:
    off, sco = load(f'official_{c}.json'), load(f'vllm_scored_new_{c}.json')
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
            vl = [round(math.exp(x), 4) for x in row[1][:2]]
            diffs.append((c, j, round(gap, 4), vl))
            if gap >= 0.1:
                bad.append((c, j, round(gap, 4), vl))
    same += s_c
    tot += n_c
    print(f'{c:10s}: {n_c} positions scored ({sco["dropped"]} at the end left out), same top-1 {s_c} '
          f'({s_c / max(n_c, 1):.4%})')
ok['b'] = tot > 0 and same / tot >= 0.99 and not bad
print(f'all: {same}/{tot} ({same / max(tot, 1):.4%}); first 1,000 of each (reference only): {same1k}/{tot1k} '
      f'({same1k / max(tot1k, 1):.4%}); differing positions {len(diffs)} (condition, position, official p1 - p2, vLLM '
      f'top-2 probabilities): {diffs[:30]}; of them with official p1 - p2 >= 0.1: {len(bad)} {bad[:30]} '
      f"-> {'PASS' if ok['b'] else 'FAIL'}")


def answer(text):
    """The answer part of a raw official generation: after the last <channel|>, without <turn|> / <eos>."""
    a = text.rsplit('<channel|>', 1)[-1] if '<channel|>' in text else text
    return re.sub(r'<turn\|>|<eos>', '', a)


def repetition(top2):
    tail = top2[-1000:]
    return len(top2) >= 1000 and sum(r[1][0] >= 0.99 for r in tail) >= 0.95 * len(tail)


print('\n=== (c) both sides generating on their own (greedy)')
for c in reqs:
    off, srv = load(f'official_{c}.json'), load(f'vllm_server_{c}.json')
    gen, sg = off['generated'], srv['generated'] or []
    first = next((j for j, (x, y) in enumerate(zip(gen, sg)) if x != y), None)
    if first is None and len(gen) != len(sg):
        first = min(len(gen), len(sg))
    o_ans, s_ans = parse(answer(off['text'])), parse(srv['content'])
    if first is None:
        where = 'never: identical'
    else:
        og = off['top2'][first][1][0] - off['top2'][first][1][1] if first < len(off['top2']) else float('nan')
        st = srv['top5'][first] if first < len(srv['top5']) else None
        sgap = math.exp(st[1][0]) - math.exp(st[1][1]) if st and len(st[1]) > 1 else float('nan')
        where = f'token {first}; official p1 - p2 there {og:.4f}, vLLM {sgap:.4f}'
    print(f"{c:10s}: official {len(gen)} tokens ({off['finish']}, thinking closed: {'<channel|>' in off['text']}, "
          f"repetition at the end: {repetition(off['top2'])}, {off['seconds']} s), vLLM {len(sg)} "
          f"({srv['finish_reason']}, thinking {len(srv.get('reasoning') or ''):,} characters, {srv['seconds']} s); first "
          f"parting: {where}; scores: official {o_ans.get('risk') if o_ans else None}, vLLM "
          f"{s_ans.get('risk') if s_ans else None}")
print(f'\nRESULT: {ok}; ALL MET: {all(ok.values())}')
sys.exit(0 if all(ok.values()) else 1)
