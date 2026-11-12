"""Diagnosis of the Gemma 4 31B-it check that stopped (job 31701268, 2026-11-12; (a) pixel values and (b) not met).
Only reads the saved outputs; CPU.
(a) vLLM casts every floating-point output of the HF processor to the model dtype before the model sees it
    (vllm/multimodal/processing/context.py, _postprocess_output, "mimics transformers.BatchFeature"), so its patch
    embedder gets bfloat16 pixel values and computes 2 * (x - 0.5) in bfloat16; the official code (as the model card's
    example, inputs .to(model.device) only) keeps them float32 and casts after 2 * (x - 0.5). Here, per picture:
    the official float32 values cast to bfloat16 and put through 2 * (x - 0.5) in bfloat16 against what vLLM's
    encoder received (pixel values and input projection, real patches); how many elements differ between the two
    paths at the input projection and by how much; the same per distinct pixel value (k / 255).
(b) at every position where vLLM's top-1 differs from the official token: the official top-2 probabilities and
    log-probability difference ln(p1 / p2) = the logit difference, in steps of 2^-5 (the bfloat16 spacing of logits
    between 4 and 8; 2^-4 between 8 and 16, 2^-3 between 16 and 32: differences of bfloat16 logits of 4 and above are
    whole multiples of 2^-5); vLLM's top-2 there; how many of all scored positions have an exact official top-2 tie.
usage: python -I diag_pixels_logits.py <compare_out folder>
"""
import glob
import hashlib
import json
import math
import os
import sys

import torch

C = sys.argv[1]
CONDS = ['grid', 'sep_time', 'sep_camera', 'ctrl_grid', 'ctrl_sep']


def sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


v_pv, v_proj = {}, {}
for pe in sorted(glob.glob(os.path.join(C, 'vllm_vision', 'pe_*.pt'))):
    d = torch.load(pe)
    proj = torch.load(pe.replace('pe_', 'proj_'))
    for b in range(d['pv'].shape[0]):
        real = ~(d['pos'][b] == -1).all(dim=-1)
        v_pv[sha(d['pv'][b][real])] = d['pv'][b][real]
        v_proj[sha(proj[b][real])] = proj[b][real]
print(f'(a) vLLM encoder inputs: {len(v_pv)} distinct pictures; pixel values {next(iter(v_pv.values())).dtype}')
tot = n_pv = n_proj = 0
diff_el = all_el = 0
max_d = 0.0
table = {}
for c in CONDS:
    for im in torch.load(os.path.join(C, f'official_vision_{c}.pt')):
        tot += 1
        f32, proj_off = im['pv'], im['proj']
        sim_pv = f32.to(torch.bfloat16)
        sim_proj = 2 * (sim_pv - 0.5)                     # in bfloat16, as vLLM's patch embedder computes it
        n_pv += sha(sim_pv) in v_pv
        n_proj += sha(sim_proj) in v_proj
        dd = (sim_proj.float() - proj_off.float()).abs()
        diff_el += int((dd > 0).sum())
        all_el += dd.numel()
        max_d = max(max_d, float(dd.max()))
        for x, a_, b_ in zip(f32.reshape(-1)[:200000].tolist(), proj_off.reshape(-1)[:200000].float().tolist(),
                             sim_proj.reshape(-1)[:200000].float().tolist()):
            table.setdefault(round(x * 255), (x, a_, b_))
print(f'  official float32 -> bfloat16 -> 2 * (x - 0.5) in bfloat16 equals what vLLM received: pixel values '
      f'{n_pv}/{tot}, input projection {n_proj}/{tot}')
print(f'  input projection, official path vs vLLM path: {diff_el:,} of {all_el:,} elements differ '
      f'({diff_el / all_el:.4%}); largest difference {max_d!r}')
same_k = sum(abs(a_ - b_) == 0 for _, a_, b_ in table.values())
print(f'  per pixel value k/255 seen ({len(table)} of 256): the two paths give the same bfloat16 for {same_k}; differing '
      f'k (official, vLLM): ' + ', '.join(f'{k}: {a_!r} / {b_!r}' for k, (x, a_, b_) in sorted(table.items())
                                          if a_ != b_)[:1500])

print('\n(b) positions where vLLM\'s top-1 differs from the official token')
rows, ties_all, n_all = [], 0, 0
for c in CONDS:
    off = json.load(open(os.path.join(C, f'official_{c}.json')))
    sco = json.load(open(os.path.join(C, f'vllm_scored_new_{c}.json')))
    for j, (g, row) in enumerate(zip(off['generated'], sco['top5'])):
        p1, p2 = off['top2'][j][1]
        n_all += 1
        ties_all += p1 == p2
        if row[0] and row[0][0] == g:
            continue
        lr = math.log(p1 / p2) if p2 > 0 else float('inf')
        k = lr / 2 ** -5
        rows.append(dict(cond=c, pos=j, p1=p1, p2=p2, gap=p1 - p2, ln=lr, steps=k, whole=abs(k - round(k)) < 1e-3,
                         v1=math.exp(row[1][0]), v2=math.exp(row[1][1]) if len(row[1]) > 1 else float('nan'),
                         vln=row[1][0] - row[1][1] if len(row[1]) > 1 else float('nan'),
                         official_is_vllm_2nd=len(row[0]) > 1 and row[0][1] == g))
print(f'  {len(rows)} positions of {n_all}; exact official top-2 ties at all scored positions: {ties_all}')
big = [r for r in rows if r['gap'] >= 0.1]
print(f'  official ln(p1/p2) a whole number of 2^-5 steps: {sum(r["whole"] for r in rows)} of {len(rows)}; the official '
      f"token is vLLM's 2nd: {sum(r['official_is_vllm_2nd'] for r in rows)} of {len(rows)}; vLLM top-2 exactly tied: "
      f"{sum(r['vln'] == 0 for r in rows)} of {len(rows)}")
print(f'  official ln(p1/p2) in 2^-5 steps, all {len(rows)}: '
      + str(sorted(round(r['steps'], 3) for r in rows)))
print(f'  the {len(big)} with official p1 - p2 >= 0.1 (condition, position, p1, p2, ln(p1/p2), steps of 2^-5, vLLM '
      'top-2 probabilities, vLLM ln difference):')
for r in big:
    print(f"    {r['cond']:10s} {r['pos']:4d}: {r['p1']:.4f} {r['p2']:.4f} ln {r['ln']:.6f} = {r['steps']:.3f} steps; "
          f"vLLM {r['v1']:.4f} {r['v2']:.4f} (ln {r['vln']:.6f})")
