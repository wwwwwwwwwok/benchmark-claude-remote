"""Step 1 of the Qwen-Drive check (user, 2026-10-26): the official code answers the 5 requests greedily.
The official model class and image code (QwenDriveForPlanning, QwenDriveProcessor._patchify) and the generation call
generate_text makes (self.vlm.generate with input_ids, pixel_values, image_grid_thw, mm_token_type_ids), but with
our full prompt instead of encode_vqa's fixed layout (which cannot hold a system prompt, text between images or the
template's <think>): the model's own chat template renders system + user parts with add_generation_prompt (thinking
on, its default), each <|image_pad|> is expanded to its image's token count, images are capped at 16,777,216 pixels
as in preprocessor_config.json (user's choice). Greedy: do_sample=False. Saved per request: the prompt ids (before
and after the expansion), the grids, the generated ids, each step's top-2 probabilities, the decoded text; and the
vision input: the float32 pixel values given to the model and the tensor the vision encoder receives (bfloat16).
2026-10-27 (user: both 4B models without thinking): --enable-thinking false renders the template with
enable_thinking=False (the prompt ends with an empty thinking block); --max-tokens is the reply cap (1024 now).
usage: python -I official_greedy.py <official src> <original model folder> <requests json> <out folder>
           --enable-thinking true|false --max-tokens N
"""
import argparse
import hashlib
import json
import os
import sys
import time

import torch
from transformers import LogitsProcessor, LogitsProcessorList

ap = argparse.ArgumentParser()
ap.add_argument('src')
ap.add_argument('orig')
ap.add_argument('req_json')
ap.add_argument('out')
ap.add_argument('--enable-thinking', choices=['true', 'false'], required=True)
ap.add_argument('--max-tokens', type=int, required=True)
args = ap.parse_args()
src, orig, req_json, out = args.src, args.orig, args.req_json, args.out
THINK = args.enable_thinking == 'true'
TAIL = '<|im_start|>assistant\n<think>\n' if THINK else '<|im_start|>assistant\n<think>\n\n</think>\n\n'
sys.path.insert(0, src)
from qwen_drive import QwenDriveForPlanning  # noqa: E402
from qwen_drive.scene import CameraFrame  # noqa: E402

MAX_PIXELS = 16777216                    # preprocessor_config.json longest_edge
MAX_MODEL_LEN, MAX_TOKENS = 32768, args.max_tokens  # the settings of the runs (user)
EOS = [248046, 248044]                   # generation_config.json eos_token_id


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


model = QwenDriveForPlanning.from_pretrained(orig, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
proc = model.processor
tok = proc.tokenizer
image_id = model.config.vlm_config.image_token_id
seen = {}
model.vlm.model.visual.register_forward_pre_hook(
    lambda m, a, k: seen.update(visual_in=a[0].detach().cpu(), grid_thw=k['grid_thw'].detach().cpu().tolist()),
    with_kwargs=True)
model.vlm.model.visual.patch_embed.proj.register_forward_pre_hook(
    lambda m, a: seen.update(proj_in=a[0].detach().cpu()))

for r in json.load(open(req_json)):
    cond = r['condition']
    msgs = [{'role': 'system', 'content': r['system']},
            {'role': 'user', 'content': [{'type': 'text', 'text': v} if k == 'text' else {'type': 'image'}
                                         for k, v in r['parts']]}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=THINK)
    assert text.endswith(TAIL), text[-80:]
    ids = tok.encode(text, add_special_tokens=False)
    imgs = [v for k, v in r['parts'] if k == 'image']
    patches, grids = [], []
    for p in imgs:
        pt, (rows, cols) = proc._patchify(CameraFrame(p), MAX_PIXELS)
        patches.append(pt)
        grids.append([1, rows, cols])
    counts = [g[1] * g[2] // proc.merge_size ** 2 for g in grids]
    assert ids.count(image_id) == len(imgs), (ids.count(image_id), len(imgs))
    exp, k = [], 0
    for t in ids:
        if t == image_id:
            exp += [image_id] * counts[k]
            k += 1
        else:
            exp.append(t)
    input_ids = torch.tensor([exp], device='cuda')
    pixel_values = torch.cat(patches).to('cuda')
    rec = Top2()
    seen.clear()
    torch.manual_seed(3407)                      # as generate_text; greedy does not use it
    t0 = time.time()
    gen = model.vlm.generate(input_ids=input_ids, pixel_values=pixel_values,
                             image_grid_thw=torch.tensor(grids, device='cuda'),
                             mm_token_type_ids=(input_ids == image_id).long(), do_sample=False,
                             max_new_tokens=min(MAX_TOKENS, MAX_MODEL_LEN - len(exp)), eos_token_id=EOS,
                             logits_processor=LogitsProcessorList([rec]))
    secs = time.time() - t0
    new = gen[0, len(exp):].tolist()
    n_patch = [g[0] * g[1] * g[2] for g in grids]
    f32 = torch.cat(patches)
    proj = seen['proj_in'].reshape(f32.shape[0], -1)
    vis = seen['visual_in'].reshape(f32.shape[0], -1)
    per_image, at = [], 0
    for n, p in zip(n_patch, imgs):
        per_image.append(dict(image=os.path.basename(p), patches=n, sha_f32=sha(f32[at:at + n]),
                              sha_bf16_visual_in=sha(vis[at:at + n]), sha_bf16_proj_in=sha(proj[at:at + n])))
        at += n
    torch.save(dict(pixel_values_f32=f32, visual_in=seen['visual_in'], proj_in=seen['proj_in'], grids=grids),
               os.path.join(out, f'official_vision_{cond}.pt'))
    res = dict(condition=cond, text_prompt=text, prompt_ids_unexpanded=ids, prompt_ids=exp, grids=grids,
               image_tokens=counts, visual_grid_thw=seen['grid_thw'], visual_in_dtype=str(seen['visual_in'].dtype),
               proj_in_dtype=str(seen['proj_in'].dtype), per_image=per_image, generated=new,
               finish='stop' if new and new[-1] in EOS else 'length', top2=rec.rows, seconds=round(secs, 1),
               text=tok.decode(new, skip_special_tokens=False))
    with open(os.path.join(out, f'official_{cond}.json'), 'w') as f:
        json.dump(res, f)
    print(f'{cond}: prompt {len(exp)} tokens ({len(imgs)} images), generated {len(new)} tokens, {res["finish"]}, '
          f'{secs:.0f} s; vision input {res["visual_in_dtype"]}', flush=True)
