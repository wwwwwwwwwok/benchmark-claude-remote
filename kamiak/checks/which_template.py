"""Which chat template vLLM 0.32 took for the runs served with the model's own template (no --chat-template):
resolve_chat_template (vllm/renderers/hf.py) takes the AutoProcessor's chat_template first, then the AutoTokenizer's.
Load both as the server did (trust_remote_code as in qwen_vlm.sbatch) and compare them with the chat_template.jinja
that render_all.py rendered; also vLLM's own content-format detection on each. No picture is read; CPU only.
usage: python -I which_template.py <models folder> <repository root>"""
import hashlib
import os
import sys

MODELS, ROOT = sys.argv[1:3]
from transformers import AutoProcessor, AutoTokenizer  # noqa: E402
from vllm.renderers.hf import _detect_content_format  # noqa: E402

sha = lambda s: hashlib.sha256(s.encode('utf-8')).hexdigest()[:16] if isinstance(s, str) else repr(s)  # noqa: E731
for name, trust in (('Qwen3.6-27B', False), ('InternVL3_5-38B', True), ('Qwen-Drive-1.0-4B', False)):
    d = os.path.join(MODELS, name)
    jinja = open(os.path.join(d, 'chat_template.jinja'), encoding='utf-8').read()
    print(f'== {name} (trust_remote_code {trust}); chat_template.jinja sha256 {sha(jinja)}')
    for label, cls in (('AutoProcessor', AutoProcessor), ('AutoTokenizer', AutoTokenizer)):
        try:
            obj = cls.from_pretrained(d, trust_remote_code=trust)
            ct = getattr(obj, 'chat_template', None)
            print(f'  {label}: {type(obj).__name__}; chat_template {sha(ct)}; same as chat_template.jinja: {ct == jinja}'
                  + (f'; content format {_detect_content_format(ct, default="string")!r}' if isinstance(ct, str) else ''))
        except Exception as e:  # noqa: BLE001
            print(f'  {label}: not loaded ({type(e).__name__}: {str(e)[:200]})')
t = open(os.path.join(ROOT, 'kamiak/internvl_thinking_chat_template.jinja'), encoding='utf-8').read()
print(f'== kamiak/internvl_thinking_chat_template.jinja: content format {_detect_content_format(t, default="string")!r}')
