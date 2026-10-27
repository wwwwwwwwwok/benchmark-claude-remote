"""Step 3 of the Qwen-Drive check (user, 2026-10-26; second run 2026-10-27): vLLM, with the server's model settings
(tensor parallel 2, context 32,768, up to 25 images, GDN prefill on Triton), scores official generations token by
token: for each official_<condition>.json in each given folder, the official prompt (one <|image_pad|> per image;
vLLM expands them) followed by what the official code generated, with the same PNGs, prompt_logprobs=5, so every
generated position gets vLLM's top-5 given the official prefix. Scoring needs one free position after the
sequence (max_tokens 1), so a sequence of 32,768 tokens loses its last generated position(s); how many is recorded.
Hooks put into the workers (LLM.apply_model) save what the vision encoder receives: the input of model.visual
(float32 pixel values, grid_thw) and the input of model.visual.patch_embed (after the cast to bfloat16); rank 0.
usage: VLLM_ALLOW_INSECURE_SERIALIZATION=1 python -I vllm_offline_score.py <vlm folder> <requests json> <out folder>
           <official folder>:<tag> [<official folder>:<tag> ...]
"""
import json
import os
import sys
import time

from PIL import Image

MAX_MODEL_LEN = 32768


def install(model):
    """Runs inside each worker: pre-hooks that save the vision encoder's inputs (rank 0)."""
    import os
    import torch
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    d, n = os.environ['CAPTURE_DIR'], [0]

    def pre_visual(mod, args, kwargs):
        if rank == 0:
            n[0] += 1
            g = kwargs.get('grid_thw', args[1] if len(args) > 1 else None)
            torch.save(dict(x=args[0].detach().cpu(), grid_thw=g if isinstance(g, list) else g.detach().cpu().tolist()),
                       os.path.join(d, f'visual_{n[0]:03d}.pt'))

    def pre_patch(mod, args):
        if rank == 0:
            torch.save(args[0].detach().cpu(), os.path.join(d, f'patch_{n[0]:03d}.pt'))

    model.visual.register_forward_pre_hook(pre_visual, with_kwargs=True)
    model.visual.patch_embed.register_forward_pre_hook(pre_patch)
    return rank


def main():
    # inside main: vLLM starts its engine with the spawn method, which imports this file again in the new processes
    vlm, req_json, out = sys.argv[1:4]
    sets = [s.rsplit(':', 1) for s in sys.argv[4:]]
    os.environ['CAPTURE_DIR'] = os.path.join(out, 'vllm_vision')
    os.makedirs(os.environ['CAPTURE_DIR'], exist_ok=True)
    os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model=vlm, tensor_parallel_size=2, max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.85,
              limit_mm_per_prompt={'image': 25}, gdn_prefill_backend='triton', max_num_batched_tokens=4096)
    print('hooks in ranks', llm.apply_model(install), flush=True)
    sp = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=5)
    reqs = json.load(open(req_json))
    for folder, tag in sets:
        for r in reqs:
            cond = r['condition']
            off = json.load(open(os.path.join(folder, f'official_{cond}.json')))
            imgs = [Image.open(v).convert('RGB') for k, v in r['parts'] if k == 'image']
            n_prompt = len(off['prompt_ids'])
            keep = min(len(off['generated']), MAX_MODEL_LEN - 1 - n_prompt)
            t0 = time.time()
            res = llm.generate([TokensPrompt(prompt_token_ids=off['prompt_ids_unexpanded'] + off['generated'][:keep],
                                             multi_modal_data={'image': imgs})], sp, use_tqdm=False)[0]
            rows = []
            for lp in res.prompt_logprobs[n_prompt:]:
                top = sorted(((v.rank, t, v.logprob) for t, v in lp.items() if v.rank is not None and v.rank <= 5))
                rows.append([[t for _, t, _ in top], [l for _, _, l in top]])
            with open(os.path.join(out, f'vllm_scored_{tag}_{cond}.json'), 'w') as f:
                json.dump(dict(condition=cond, tag=tag, prompt_ids=list(res.prompt_token_ids[:n_prompt]),
                               continuation=list(res.prompt_token_ids[n_prompt:]), top5=rows,
                               dropped=len(off['generated']) - keep, seconds=round(time.time() - t0, 1)), f)
            print(f'{tag} {cond}: {n_prompt} prompt + {keep} generated scored (last {len(off["generated"]) - keep} '
                  f'left out), {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
