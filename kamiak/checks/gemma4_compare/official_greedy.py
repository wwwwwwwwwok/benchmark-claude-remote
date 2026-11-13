"""Official side of the Gemma 4 31B-it check (user, 2026-11-12; the standard of the Qwen-Drive check): transformers
5.19.0's model class (AutoModelForImageTextToText -> Gemma4ForConditionalGeneration) and the processor from the model
folder answer the 5 requests greedily with thinking on.
The conversation is the one vLLM hands to the chat template (gemma4_common.vllm_conversation; render_check.py checked
the rendering on the CPU), rendered with add_generation_prompt=True and enable_thinking=True and tokenized without
adding special tokens, as vLLM does; the processor expands the pictures at max_soft_tokens 560. The pictures are the
PNGs prep_inputs.py made (already at their final size, so neither side resizes). Greedy: do_sample=False;
max_new_tokens = min(24,576, 40,960 - prompt length); eos ids from generation_config.json; bfloat16, sdpa attention,
the model spread over the visible GPUs (device_map auto).
Saved per request (official_<condition>.json): prompt ids before and after the expansion, visual tokens per picture,
generated ids, each step's top-2 ids and probabilities, finish, seconds, the decoded text; the vision input, real
patches only, per picture: sha256 of the float32 pixel values the patch embedder gets and of the bfloat16 tensor its
input projection gets (after 2 * (x - 0.5)), and of the patch positions; the tensors in official_vision_<condition>.pt.
--extra-path: a folder put first on sys.path before transformers is imported. The vllm_cu129 environment has no
accelerate, which device_map needs; the folder holds only links to the accelerate package (1.14.0) of the benchmark
environment (same Python 3.11), so no environment is changed and nothing is installed.
--cast-pixels-bf16 (diagnosis, user 2026-11-12): the floating-point processor outputs (pixel_values) cast to bfloat16
before the model sees them, as vLLM does (vllm/multimodal/processing/context.py _postprocess_output), so both sides'
vision input is the same; without the option nothing changes.
usage: python -I official_greedy.py <model folder> <requests json> <out folder> [--max-tokens 24576]
           [--extra-path <folder>] [--cast-pixels-bf16]
"""
import argparse
import hashlib
import json
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemma4_common import EOS, MAX_SOFT_TOKENS, image_runs, openai_messages, vllm_conversation  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('model')
ap.add_argument('req_json')
ap.add_argument('out')
ap.add_argument('--max-tokens', type=int, default=24576)
ap.add_argument('--extra-path', default=None)
ap.add_argument('--cast-pixels-bf16', action='store_true')
args = ap.parse_args()
if args.extra_path:
    sys.path.insert(0, args.extra_path)
import transformers  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor, LogitsProcessor, LogitsProcessorList  # noqa: E402
from transformers.utils import is_accelerate_available  # noqa: E402

import accelerate  # noqa: E402
print(f'transformers {transformers.__version__} ({os.path.dirname(transformers.__file__)}); accelerate '
      f'{accelerate.__version__} ({os.path.dirname(accelerate.__file__)}), available to transformers: '
      f'{is_accelerate_available()}', flush=True)
MAX_MODEL_LEN = 40960


def sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


class Top2(LogitsProcessor):
    """Records the two most likely next tokens and their probabilities at every step; changes nothing."""

    def __init__(self):
        self.rows = []

    def __call__(self, input_ids, scores):
        p, i = torch.softmax(scores.float(), dim=-1).topk(2, dim=-1)
        self.rows.append([i[0].tolist(), p[0].tolist()])
        return scores


proc = AutoProcessor.from_pretrained(args.model)
tok = proc.tokenizer
model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map='auto',
                                                    attn_implementation='sdpa').eval()
print(f'model class {type(model).__name__}; devices {sorted(set(str(p.device) for p in model.parameters()))}; '
      f'torch {torch.__version__}', flush=True)
seen = {}
pe = [m for m in model.modules() if type(m).__name__ == 'Gemma4VisionPatchEmbedder']
assert len(pe) == 1, len(pe)
pe[0].register_forward_pre_hook(
    lambda m, a: seen.setdefault('calls', []).append(dict(pv=a[0].detach().cpu(), pos=a[1].detach().cpu())))
pe[0].input_proj.register_forward_pre_hook(lambda m, a: seen.setdefault('proj', []).append(a[0].detach().cpu()))
dev = model.get_input_embeddings().weight.device

for r in json.load(open(args.req_json)):
    cond = r['condition']
    conv = vllm_conversation(openai_messages([tuple(p) for p in r['parts']], r['system']))
    text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    ids = tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=True, enable_thinking=True,
                                  return_dict=False)
    assert text.startswith('<bos><|turn>system\n<|think|>\n') and text.endswith('<|turn>model\n'), (text[:60], text[-40:])
    paths = [v for k, v in r['parts'] if k == 'image']
    imgs = [Image.open(v).convert('RGB') for v in paths]
    enc = proc(text=[text], images=[imgs], add_special_tokens=False, return_tensors='pt',
               images_kwargs={'max_soft_tokens': MAX_SOFT_TOKENS})
    exp = enc['input_ids'][0].tolist()
    runs = image_runs(exp)
    assert len(runs) == len(imgs), (len(runs), len(imgs))
    dropped = sorted(k for k, v in enc.items() if not torch.is_tensor(v))
    enc = {k: v.to(dev) for k, v in enc.items() if torch.is_tensor(v)}
    if args.cast_pixels_bf16:                     # as vLLM: every floating-point processor output to the model dtype
        enc = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in enc.items()}
    print(f'{cond}: model inputs {sorted(enc)} (not tensors, left out: {dropped})'
          + (f'; dtypes { {k: str(v.dtype) for k, v in enc.items()} }' if args.cast_pixels_bf16 else ''), flush=True)
    rec = Top2()
    seen.clear()
    t0 = time.time()
    with torch.inference_mode():
        gen = model.generate(**enc, do_sample=False, max_new_tokens=min(args.max_tokens, MAX_MODEL_LEN - len(exp)),
                             eos_token_id=EOS, logits_processor=LogitsProcessorList([rec]))
    secs = time.time() - t0
    new = gen[0, len(exp):].tolist()
    # vision input, real patches only (the processor pads every picture to 560 x 9 patches; position -1 = padding)
    calls = seen.get('calls', [])
    assert calls and len(calls) == len(seen.get('proj', [])), (len(calls), len(seen.get('proj', [])))
    pv = torch.cat([c['pv'] for c in calls])
    pos = torch.cat([c['pos'] for c in calls])
    proj = torch.cat(seen['proj'])
    assert pv.shape[0] == len(imgs), (pv.shape, len(imgs))
    per_image, keep = [], []
    for b in range(pv.shape[0]):
        real = ~(pos[b] == -1).all(dim=-1)
        f32, bf, ps = pv[b][real], proj[b][real], pos[b][real]
        keep.append(dict(pv=f32, proj=bf, pos=ps))
        per_image.append(dict(image=os.path.basename(paths[b]), patches=int(real.sum()), dtype_pv=str(f32.dtype),
                              dtype_proj=str(bf.dtype), sha_pv=sha(f32), sha_proj=sha(bf),
                              sha_pos=sha(ps.to(torch.int64))))
    torch.save(keep, os.path.join(args.out, f'official_vision_{cond}.pt'))
    res = dict(condition=cond, text_prompt=text, prompt_ids_unexpanded=ids, prompt_ids=exp, image_tokens=runs,
               per_image=per_image, vision_calls=len(calls), generated=new, finish='stop' if new and new[-1] in EOS else 'length',
               top2=rec.rows, seconds=round(secs, 1), text=tok.decode(new, skip_special_tokens=False))
    with open(os.path.join(args.out, f'official_{cond}.json'), 'w') as f:
        json.dump(res, f)
    print(f'{cond}: prompt {len(exp)} tokens ({len(imgs)} pictures), generated {len(new)} tokens, {res["finish"]}, '
          f'{secs:.0f} s ({len(new) / max(secs, 1e-9):.1f} tokens/s); vision input {per_image[0]["dtype_pv"]} -> '
          f'{per_image[0]["dtype_proj"]}', flush=True)
