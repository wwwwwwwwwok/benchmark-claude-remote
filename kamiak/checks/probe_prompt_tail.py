"""What the vLLM server actually uses and generates for run_vlm.py's requests (user, 2026-10-29: Qwen3.6-27B without
thinking; the smoke test must show that every prompt ends with an empty thinking block and that no reply thinks).
Runs code/run_vlm.py's own main() with --send, with a client that passes every request on to the server unchanged
except that extra_body also asks for return_token_ids (vLLM: the prompt ids it used and the ids it generated). One
worker, so the requests go in the job order. Saved per request (with the key of run_vlm.py's output line): the number
of prompt ids, the last 8 and their text, whether the prompt ends with the ids of '<think>\n\n</think>\n\n', the
number of generated ids and whether <think> or </think> is among them. run_vlm.py's own lines go to the probe's --out.
usage: python -I probe_prompt_tail.py <run_vlm.py> <model folder> <out json> <run_vlm arguments ...>
       (the arguments as the submission script passes them, with --send, --workers 1 and the probe's own --out)
"""
import importlib.util
import json
import sys
import types

import openai
from transformers import AutoTokenizer

run_vlm_py, model_dir, out_json = sys.argv[1:4]
args = sys.argv[4:]
tok = AutoTokenizer.from_pretrained(model_dir)
EMPTY = tok.encode('<think>\n\n</think>\n\n', add_special_tokens=False)
THINK = set(tok.convert_tokens_to_ids(['<think>', '</think>']))
print(f'empty thinking block: ids {EMPTY}; thinking tokens {sorted(THINK)}')
real_client = openai.OpenAI
records = []


class Probe:
    def __init__(self, **kw):
        self.client = real_client(**kw)
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **kw):
        resp = self.client.chat.completions.create(**{**kw, 'extra_body': {**(kw.get('extra_body') or {}),
                                                                            'return_token_ids': True}})
        p = (resp.model_extra or {}).get('prompt_token_ids') or []
        g = (resp.choices[0].model_extra or {}).get('token_ids') or []
        records.append(dict(prompt_tokens=len(p), tail_ids=p[-8:], tail_text=tok.decode(p[-8:]),
                            ends_with_empty_thinking=len(p) > len(EMPTY) and p[-len(EMPTY):] == EMPTY,
                            generated_tokens=len(g), thinking_token_generated=bool(THINK & set(g))))
        return resp


openai.OpenAI = Probe                     # run_vlm.py imports it inside main()
spec = importlib.util.spec_from_file_location('run_vlm_probe', run_vlm_py)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.argv = [run_vlm_py] + args
mod.main()
out = args[args.index('--out') + 1]
with open(out, encoding='utf-8') as f:
    lines = [json.loads(x) for x in f]
if len(lines) != len(records):
    sys.exit(f'{len(lines)} output lines but {len(records)} replies recorded: stop')
res = [dict(segment=d['segment'], frame_idx=d['frame_idx'], condition=d['condition'], quantity=d.get('quantity'),
            repeat=d['repeat'], **r) for d, r in zip(lines, records)]
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
for r in res:
    print(f"{r['condition']:10s} {r['quantity']:10s} prompt {r['prompt_tokens']:6d} ends with the empty block "
          f"{r['ends_with_empty_thinking']}; generated {r['generated_tokens']}, thinking token "
          f"{r['thinking_token_generated']}; tail {r['tail_text']!r}")
print(f'{len(res)} requests: all end with the empty thinking block: {all(r["ends_with_empty_thinking"] for r in res)}; '
      f'none generates a thinking token: {not any(r["thinking_token_generated"] for r in res)}')
