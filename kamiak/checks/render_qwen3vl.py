"""Render one ctrl_sep risk request of run_vlm.py with the chat templates of Qwen3-VL-32B-Instruct and -Thinking (user,
2026-11-05, item 4): their model folders are gone; only chat_template.json and tokenizer_config.json were downloaded
at the recorded revisions (kamiak/model_files/, git blob ids checked). As render_all.py: the content list as vLLM 0.32
hands it over (content format 'openai'; an image part as {"type": "image"}), transformers' apply_chat_template with
the template string, add_generation_prompt=True and no template kwargs (both were served with the model's own template
and no --default-chat-template-kwargs), through Qwen3.6-27B's tokenizer. Also: is the template in tokenizer_config.json
the same as chat_template.json's (vLLM takes the processor's, i.e. chat_template.json's, first), and vLLM's own content
format detection. No picture is read.
usage: python -I render_qwen3vl.py <repository root> <models folder> <downloaded files folder>"""
import json
import os
import sys

ROOT, MODELS, DL = sys.argv[1:4]
sys.path.insert(0, os.path.join(ROOT, 'code'))
import run_vlm as rv  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm.renderers.hf import _detect_content_format  # noqa: E402

tok = AutoTokenizer.from_pretrained(os.path.join(MODELS, 'Qwen3.6-27B'))
parts = rv.build('ctrl_sep', ['g'] * 5, {c: ['img'] * 5 for c in rv.CAMERAS})
content = [{'type': 'text', 'text': v} if k == 'text' else {'type': 'image'} for k, v in parts]
msgs = [{'role': 'system', 'content': rv.SYSTEM}, {'role': 'user', 'content': content}]
for name in ('Qwen3-VL-32B-Instruct', 'Qwen3-VL-32B-Thinking'):
    d = os.path.join(DL, name)
    tpl = json.load(open(os.path.join(d, 'chat_template.json'), encoding='utf-8'))['chat_template']
    tc = json.load(open(os.path.join(d, 'tokenizer_config.json'), encoding='utf-8')).get('chat_template')
    text = tok.apply_chat_template(msgs, chat_template=tpl, tokenize=False, add_generation_prompt=True)
    print(f'\n########## {name}: chat_template.json ({len(tpl)} characters); tokenizer_config.json has '
          f'{"the same template" if tc == tpl else ("a different template" if tc else "no template")}; vLLM content '
          f'format {_detect_content_format(tpl, default="string")!r}; {len(text)} characters rendered, verbatim between '
          f'>>> and <<<')
    print('>>>')
    print(text, end='')
    print('<<<')
    for k in ('taken now.', 'camera:', 'How risky'):
        i = text.find(k)
        print(f'  around {k!r}: {text[max(0, i - 30):i + len(k) + 40]!r}')
