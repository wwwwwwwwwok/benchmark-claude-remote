"""Render requests with Qwen3.6-27B's own chat template (user, 2026-11-05, item 5.2): transformers'
apply_chat_template (tokenize False, add_generation_prompt True, the template's defaults as vLLM uses them), CPU only:
  1 the described situation as drafted on 2026-11-05 (three text parts) - to see how the parts are joined;
  2 the same as run_text_vlm.py writes it now (one text part, the pieces joined by newlines);
  3 the original ctrl_sep request of run_vlm.py, the images as the image_url parts run_vlm.py sends (no picture read).
Also: the content format vLLM detects for this template (vllm.renderers.hf._detect_content_format), if it imports.
usage: python -I render_template.py <repository code folder> <model folder> <truth_main.csv> <ssm_target_frames.csv>"""
import sys
import types

CODE, MODEL, TRUTH, FRAMES = sys.argv[1:5]
sys.path.insert(0, CODE)
import run_text_vlm as rt  # noqa: E402
import run_vlm as rv  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
import transformers  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL)
print(f'transformers {transformers.__version__}; chat template from {MODEL} ({len(tok.chat_template)} characters)')
a = types.SimpleNamespace(system='S1', camera_field='keep', distance_ref='bumper')
d, lv, n = rt.design(TRUTH, FRAMES, 'bumper', 1)
r = next(d.itertuples())


def show(title, msgs):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(f'\n########## {title} ({len(text)} characters; between the >>> and <<< lines, verbatim)\n>>>')
    print(text, end='')
    print('<<<')
    return text


t1 = show('1 the draft: three text parts', rt.three_parts(r, a))
t2 = show('2 run_text_vlm.py now: one text part, joined by newlines', rt.messages_of(r, a))
print('\njoins in 1: ' + ', '.join(repr(t1[t1.find(k) - 12:t1.find(k) + len(k)]) for k in ('Your car is', 'How risky')))
print('joins in 2: ' + ', '.join(repr(t2[t2.find(k) - 12:t2.find(k) + len(k)]) for k in ('Your car is', 'How risky')))
parts = rv.build('ctrl_sep', ['g'] * 5, {c: ['img'] * 5 for c in rv.CAMERAS})
content = [{'type': 'text', 'text': v} if k == 'text' else
           {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,(picture not read here)'}} for k, v in parts]
show('3 the original ctrl_sep request of run_vlm.py (images as sent, no picture read)',
     [{'role': 'system', 'content': rv.SYSTEM}, {'role': 'user', 'content': content}])
try:
    from vllm.renderers.hf import _detect_content_format
    print(f'\nvLLM content format detected for this template: {_detect_content_format(tok.chat_template, default="string")}')
except Exception as e:
    print(f'\nvLLM content-format detection not run here: {type(e).__name__}: {e}')
