"""CPU step 2 before the Gemma 4 31B-it GPU runs (user, 2026-11-12). Nothing is sent to a model; the images are not
shown. vllm_cu129 environment (transformers 5.19.0, vLLM 0.32.0), from the repository root.
4. The content format vLLM 0.32 detects for the model's chat template (vllm.renderers.hf._detect_content_format, the
   function its renderer uses); gemma4_common.vllm_conversation builds the conversation in that format.
5. The 5 requests with the official tokenizer and processor (AutoProcessor from the model folder): the conversation
   rendered by tokenizer.apply_chat_template(add_generation_prompt=True, enable_thinking=True), the call vLLM makes
   (it tokenizes the rendered text without adding special tokens); then the processor expands each <|image|> at
   max_soft_tokens 560. Per request: prompt length, visual tokens per picture and in all against the table (2,670 /
   13,470 / 13,470 / 534 / 2,694); that the expansion changed nothing but the pictures; the rendered text with
   image runs shortened (written to rendered_prompts.txt). Then the same with the original JPEGs of that frame:
   the counts must be equal. Writes official_prompts.json (condition -> unexpanded and expanded ids) for the GPU step.
6. The forced answer: the request code/force_answer.py --thinking_format gemma4 builds (its own forced_messages), as
   vLLM hands it to the template, rendered with add_generation_prompt False and continue_final_message True: the
   text must be the normal prompt + '<|channel>thought\\n' + thinking + the sentence + '\\n<channel|>', so the
   thinking, the sentence and <channel|> are all there and the prompt ends with <channel|>. Two test thinkings (one
   with blank lines and markdown, one cut mid-word, as a reply cut at max_tokens is).
usage: python -I render_check.py <model folder> <requests json> <requests_original json> <out folder>
"""
import json
import os
import sys

from PIL import Image
from transformers import AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.expanduser('~/benchmark-claude-remote/code'))
from gemma4_common import (MAX_SOFT_TOKENS, TABLE, image_runs, openai_messages, shorten,  # noqa: E402
                           vllm_conversation)
import force_answer  # noqa: E402

model, req_json, orig_json, out = sys.argv[1:5]
bad = []
proc = AutoProcessor.from_pretrained(model)
tok = proc.tokenizer

print('=== 4. content format vLLM 0.32 detects for the chat template')
from vllm.renderers.hf import _detect_content_format  # noqa: E402
template = open(os.path.join(model, 'chat_template.jinja')).read()
fmt = _detect_content_format(template, default='string')
same_tpl = tok.chat_template == template
print(f"detected: {fmt}; tokenizer.chat_template is chat_template.jinja: {same_tpl}")
if fmt != 'openai':
    bad.append(f'content format {fmt}, gemma4_common.vllm_conversation assumes openai')
if not same_tpl:
    bad.append('tokenizer.chat_template differs from chat_template.jinja')


def render(r):
    conv = vllm_conversation(openai_messages([tuple(p) for p in r['parts']], r['system']))
    text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    ids = tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=True, enable_thinking=True,
                                  return_dict=False)
    imgs = [Image.open(v).convert('RGB') for k, v in r['parts'] if k == 'image']
    enc = proc(text=[text], images=[imgs], add_special_tokens=False, return_tensors='pt',
               images_kwargs={'max_soft_tokens': MAX_SOFT_TOKENS})
    return conv, text, ids, enc, imgs


print('\n=== 5. the 5 requests with the official tokenizer and processor (max_soft_tokens 560)')
results, rendered = {}, []
for label, path in (('PNG inputs', req_json), ('original JPEGs', orig_json)):
    for r in json.load(open(path)):
        c = r['condition']
        conv, text, ids, enc, imgs = render(r)
        exp = enc['input_ids'][0].tolist()
        runs = image_runs(exp)
        n_img = sum(n + 2 for n in runs)
        # the expansion only replaces each <|image|> by <|image> + N x <|image|> + <image|>
        back, i, k = [], 0, 0
        while i < len(exp):
            if exp[i] == 255999:
                back.append(258880)
                i += runs[k] + 2
                k += 1
            else:
                back.append(exp[i])
                i += 1
        unexp_ok = back == ids
        ok = n_img == TABLE[c] and unexp_ok and len(runs) == len(imgs) and ids[0] == 2 and ids[1] != 2
        print(f'{label:14s} {c:10s}: {len(imgs)} pictures, prompt {len(ids)} tokens before and {len(exp)} after the '
              f'expansion; visual tokens per picture {sorted(set(runs))}, with <|image>/<image|> {n_img} (table '
              f'{TABLE[c]}); only the pictures changed: {unexp_ok}; one <bos> at the start: '
              f'{ids[0] == 2 and ids[1] != 2} -> {"ok" if ok else "NOT MET"}')
        if not ok:
            bad.append(f'{label} {c}')
        if label == 'PNG inputs':
            results[c] = dict(prompt_ids_unexpanded=ids, prompt_ids=exp, image_tokens=runs, text_prompt=text)
            rendered.append(f'##### {c} ({len(exp)} tokens)\n{shorten(tok.decode(exp, skip_special_tokens=False))}')
        else:
            same = exp == results[c]['prompt_ids']
            print(f'{"":14s} {"":10s}  expanded ids equal to the PNG inputs\' : {same}')
            if not same:
                bad.append(f'original JPEGs {c}: ids differ from the PNG inputs')
with open(os.path.join(out, 'official_prompts.json'), 'w') as f:
    json.dump(results, f)
with open(os.path.join(out, 'rendered_prompts.txt'), 'w') as f:
    f.write('\n\n'.join(rendered) + '\n')
print(f'rendered prompts (image runs shortened): {os.path.join(out, "rendered_prompts.txt")}')
print('--- grid prompt as rendered (image runs shortened):')
print(rendered[0])

print('\n=== 6. the forced answer, rendered by the template (add_generation_prompt False, continue_final_message True)')
r0 = json.load(open(req_json))[0]
normal_text = tok.apply_chat_template(vllm_conversation(openai_messages([tuple(p) for p in r0['parts']], r0['system'])),
                                      tokenize=False, add_generation_prompt=True, enable_thinking=True)
tests = ['The user wants a risk score.\n\n*   **Picture 1:** a car ahead, about 20 m.\n*   It is braking.\n\nSo the',
         'Let me look at picture 5 again. The pedestrian on the right is walk']
for th in tests:
    msgs, early = force_answer.forced_messages([tuple(p) for p in r0['parts']], th, 'gemma4', 'test')
    conv = vllm_conversation(msgs)
    forced = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False, continue_final_message=True,
                                     enable_thinking=True)
    want = normal_text + '<|channel>thought\n' + th + early
    ok = forced == want
    tail = forced[len(normal_text) - 40:] if forced.startswith(normal_text[:-40]) else forced[-300:]
    print(f'thinking {len(th)} characters: rendered = normal prompt + <|channel>thought\\n + thinking + sentence + '
          f'\\n<channel|>: {ok}; ends with <channel|>: {forced.endswith("<channel|>")}; tail {tail!r}')
    if not ok:
        bad.append(f'forced render for test thinking of {len(th)} characters')

print(f'\nchecks not met: {len(bad)}')
for b in bad:
    print('  ', b)
sys.exit(1 if bad else 0)
