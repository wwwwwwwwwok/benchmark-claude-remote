"""Render one ctrl_sep risk request of run_vlm.py with each chat template the earlier runs used (user, 2026-11-05, item 3;
CPU only), as vLLM 0.32 hands it over: every log of those runs says the content format was 'openai', so the content
list goes to the template as it is, an image part as {"type": "image"} (vllm/renderers/chat_utils.py, wrap_dicts).
The templates are applied with transformers' apply_chat_template(chat_template=<the file>, add_generation_prompt=True,
plus the run's --default-chat-template-kwargs), through Qwen3.6-27B's tokenizer (the templates use no tokenizer
attribute). No picture is read.
usage: python -I render_all.py <repository root> <models folder>"""
import os
import sys

ROOT, MODELS = sys.argv[1:3]
sys.path.insert(0, os.path.join(ROOT, 'code'))
import run_vlm as rv  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(os.path.join(MODELS, 'Qwen3.6-27B'))
runs = [('qwen3.6-27b (thinking on: risk run, quantities on sep_camera and grid)', 'Qwen3.6-27B/chat_template.jinja', {}),
        ('qwen3.6-27b-nothink (quantities, thinking off)', 'Qwen3.6-27B/chat_template.jinja', {'enable_thinking': False}),
        ('internvl3.5-38b', 'InternVL3_5-38B/chat_template.jinja', {}),
        ('internvl3.5-38b-thinking (--chat-template kamiak/internvl_thinking_chat_template.jinja)',
         os.path.join(ROOT, 'kamiak/internvl_thinking_chat_template.jinja'), {}),
        ('qwen-drive-1.0-4b and qwen3.5-4b (Qwen-Drive-1.0-4B/chat_template.jinja, enable_thinking false)',
         'Qwen-Drive-1.0-4B/chat_template.jinja', {'enable_thinking': False})]
parts = rv.build('ctrl_sep', ['g'] * 5, {c: ['img'] * 5 for c in rv.CAMERAS})
content = [{'type': 'text', 'text': v} if k == 'text' else {'type': 'image'} for k, v in parts]
msgs = [{'role': 'system', 'content': rv.SYSTEM}, {'role': 'user', 'content': content}]
for name, path, kw in runs:
    p = path if os.path.isabs(path) else os.path.join(MODELS, path)
    tpl = open(p, encoding='utf-8').read()
    text = tok.apply_chat_template(msgs, chat_template=tpl, tokenize=False, add_generation_prompt=True, **kw)
    print(f'\n########## {name}: {p} ({len(tpl)} characters of template; {len(text)} rendered; verbatim between >>> and <<<)')
    print('>>>')
    print(text, end='')
    print('<<<')
    for k in ('taken now.', 'camera:', 'How risky'):
        i = text.find(k)
        print(f'  around {k!r}: {text[max(0, i - 30):i + len(k) + 40]!r}')
print('\nnot rendered: qwen3-vl-32b, qwen3-vl-32b-thinking (their model folders are no longer in ' + MODELS + ')')
