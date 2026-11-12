"""vLLM side, scoring (Gemma 4 31B-it check, user 2026-11-12; as step 3 of the Qwen-Drive check): vLLM's offline engine
with the server's model settings (tensor parallel 2, context 40,960, up to 25 pictures, max_soft_tokens 560) scores
the official greedy generations token by token: for each official_<condition>.json, the official prompt before the
expansion (one <|image|> per picture; vLLM expands them) followed by what the official code generated, with the same
PNGs, prompt_logprobs=5, so every generated position gets vLLM's top-5 given the official prefix. Scoring needs one
free position after the sequence (max_tokens 1), so a sequence filling 40,960 tokens loses its last position(s); how
many is recorded. Prefix caching off (only so no prompt position is skipped; the numbers do not depend on it); 4,096
tokens per step as in the Qwen-Drive check. vision_hooks.install saves what the vision encoder receives (rank 0).
usage: VLLM_ALLOW_INSECURE_SERIALIZATION=1 python -I vllm_offline_score.py <model folder> <requests json> <out folder>
           <official folder>:<tag> [...]
"""
import json
import os
import sys
import time

from PIL import Image

MAX_MODEL_LEN = 40960
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # spawned workers take the parent's sys.path
from gemma4_common import MAX_SOFT_TOKENS  # noqa: E402
from vision_hooks import install  # noqa: E402


def main():
    # inside main: vLLM starts its engine with the spawn method, which imports this file again in the new processes
    model, req_json, out = sys.argv[1:4]
    sets = [s.rsplit(':', 1) for s in sys.argv[4:]]
    os.environ['CAPTURE_DIR'] = os.path.join(out, 'vllm_vision')
    os.makedirs(os.environ['CAPTURE_DIR'], exist_ok=True)
    os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model=model, tensor_parallel_size=2, max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.75,
              limit_mm_per_prompt={'image': 25}, mm_processor_kwargs={'max_soft_tokens': MAX_SOFT_TOKENS},
              max_num_batched_tokens=4096, enable_prefix_caching=False)
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
                rows.append([[t for _, t, _ in top], [lg for _, _, lg in top]])
            with open(os.path.join(out, f'vllm_scored_{tag}_{cond}.json'), 'w') as f:
                json.dump(dict(condition=cond, tag=tag, prompt_ids=list(res.prompt_token_ids[:n_prompt]),
                               continuation=list(res.prompt_token_ids[n_prompt:]), top5=rows,
                               dropped=len(off['generated']) - keep, seconds=round(time.time() - t0, 1)), f)
            print(f'{tag} {cond}: {n_prompt} prompt + {keep} generated scored (last {len(off["generated"]) - keep} '
                  f'left out), {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
