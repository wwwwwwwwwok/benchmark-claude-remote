"""Requests of two run_vlm.py argument lists that should differ only in the model name and sampling (user, 2026-10-29:
Qwen3.6-27B without thinking gets the same images, prompts and max_tokens as with thinking). Runs code/run_vlm.py's
main() with --send against a stand-in for the OpenAI client that sends nothing (as compare_risk_requests.py; image
data URLs replaced by their sha256), once with each list, one worker (sending order = job order), and compares the two
request lists in order: everything but the model name and the sampling keys (temperature, top_p, extra_body with
top_k, presence_penalty) must be the same; the values of those keys are listed.
usage: python -I compare_sampling_only.py <run_vlm.py> <work folder> <label> <arguments A> --vs <arguments B>
       (the arguments as the submission script passes them, without --send, --out and --workers)
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import types
from collections import Counter

sys.dont_write_bytecode = True
py, work, label = sys.argv[1:4]
rest = sys.argv[4:]
args = {'A': rest[:rest.index('--vs')], 'B': rest[rest.index('--vs') + 1:]}
SAMPLING = ['model', 'temperature', 'top_p', 'extra_body', 'presence_penalty']


class Stand:
    """Stand-in for openai.OpenAI: keeps a hash of each request without the sampling keys, and those keys."""
    requests = []

    def __init__(self, **kw):
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **kw):
        msgs = []
        for m in kw['messages']:
            c = m['content']
            if isinstance(c, list):
                c = [{'type': 'image_url', 'image_url': {'url': 'sha256:' + hashlib.sha256(
                    x['image_url']['url'].encode()).hexdigest()}} if x.get('type') == 'image_url' else x for x in c]
            msgs.append({**m, 'content': c})
        other = {k: v for k, v in kw.items() if k not in SAMPLING and k != 'messages'}
        s = json.dumps({**other, 'messages': msgs}, ensure_ascii=False, sort_keys=True)
        Stand.requests.append((hashlib.sha256(s.encode()).hexdigest(),
                               json.dumps({k: kw.get(k) for k in SAMPLING}, sort_keys=True), sorted(other)))
        msg = types.SimpleNamespace(content='{"risk": 50}', reasoning=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason='stop', logprobs=None)],
                                     usage=None)


sys.modules['openai'] = types.SimpleNamespace(OpenAI=Stand)
spec = importlib.util.spec_from_file_location('run_vlm_cmp', py)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
res = {}
for tag, a in args.items():
    out = os.path.join(work, f'{label}_{tag}.jsonl')
    if os.path.exists(out):
        sys.exit(f'{out} exists: stop')
    Stand.requests = []
    sys.argv = [py] + a + ['--workers', '1', '--out', out, '--send']
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        mod.main()
    res[tag] = (list(Stand.requests), log.getvalue().strip().splitlines()[-1])
    print(f'{tag}: {len(Stand.requests)} requests; run log "{res[tag][1]}"')
(ra, _), (rb, _) = res['A'], res['B']
same = len(ra) == len(rb) and all(x[0] == y[0] for x, y in zip(ra, rb))
print(f'{label}: everything but the model name and sampling identical in order: {same}')
print(f'  other keys sent: A {sorted(Counter(tuple(x[2]) for x in ra))}, B {sorted(Counter(tuple(x[2]) for x in rb))}')
print(f'  model name and sampling: A {dict(Counter(x[1] for x in ra))}')
print(f'                           B {dict(Counter(x[1] for x in rb))}')
print(f'SAME APART FROM SAMPLING: {same}')
sys.exit(0 if same else 1)
