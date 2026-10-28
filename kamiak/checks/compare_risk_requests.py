"""Risk requests before and after a change to run_vlm.py (user, 2026-10-28: after adding --task quantities, every
risk request must be byte for byte the same). Runs each version's main() with --send against a stand-in for the
OpenAI client that sends nothing and connects nowhere: every request it would send (the whole keyword set: model,
messages with the prompt text and the images in order, sampling parameters, log-probability settings) is written as
JSON with each image's data URL replaced by the sha256 of that URL (the base64 of the file), and the sha256 of that
JSON is kept in sending order (one worker, so the order is the job order). The output lines written for the stand-in
answer {"risk": 50} are compared too (without latency_s), and the dry-run printout (no --send).
usage: python -I compare_risk_requests.py <old run_vlm.py> <new run_vlm.py> <work folder> <label> <run_vlm arguments>
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

sys.dont_write_bytecode = True
old_py, new_py, work, label = sys.argv[1:5]
args = sys.argv[5:]


class Stand:
    """Stand-in for openai.OpenAI: records the requests, answers {"risk": 50}."""
    requests = []
    first = []

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
        s = json.dumps({**kw, 'messages': msgs}, ensure_ascii=False)
        Stand.requests.append(hashlib.sha256(s.encode()).hexdigest())
        if len(Stand.first) < 1:
            Stand.first.append(s)
        msg = types.SimpleNamespace(content='{"risk": 50}', reasoning=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason='stop', logprobs=None)],
                                     usage=None)


sys.modules['openai'] = types.SimpleNamespace(OpenAI=Stand)


def run(py, tag):
    spec = importlib.util.spec_from_file_location(f'run_vlm_{tag}', py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = os.path.join(work, f'{label}_{tag}.jsonl')
    if os.path.exists(out):
        sys.exit(f'{out} exists: stop')
    Stand.requests, Stand.first = [], []
    dry = io.StringIO()
    sys.argv = [py] + args + ['--show', '3', '--out', out]           # nothing is written without --send
    with contextlib.redirect_stdout(dry):
        mod.main()
    log = io.StringIO()
    sys.argv = [py] + args + ['--workers', '1', '--out', out, '--send']
    with contextlib.redirect_stdout(log):
        mod.main()
    lines = []
    with open(out, encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            d.pop('latency_s')
            lines.append(json.dumps(d, ensure_ascii=False))
    return dict(requests=Stand.requests, first=Stand.first[0] if Stand.first else '', dry=dry.getvalue(),
                log=log.getvalue().strip().splitlines()[-1], lines=lines)


o, n = run(old_py, 'old'), run(new_py, 'new')
same_req = o['requests'] == n['requests']
same_lines = o['lines'] == n['lines']
same_dry = o['dry'] == n['dry']
print(f'{label}: requests old {len(o["requests"])}, new {len(n["requests"])}; identical in order: {same_req}; '
      f'output lines identical (without latency_s): {same_lines} ({len(o["lines"])} / {len(n["lines"])}); '
      f'dry-run printout identical: {same_dry}')
print(f'  run logs: old "{o["log"]}" / new "{n["log"]}"')
print(f'  sha256 of the whole request list: {hashlib.sha256("".join(n["requests"]).encode()).hexdigest()}')
if not same_req:
    i = next((i for i, (x, y) in enumerate(zip(o['requests'], n['requests'])) if x != y),
             min(len(o['requests']), len(n['requests'])))
    print(f'  first different request: number {i + 1}')
first = json.loads(n['first'])
print('  first request: ' + json.dumps({k: v for k, v in first.items() if k != 'messages'}, ensure_ascii=False))
print('    system: ' + json.dumps(first['messages'][0]['content'], ensure_ascii=False)[:300])
texts = [x['text'] for x in first['messages'][1]['content'] if x['type'] == 'text']
print(f'    user: {len(texts)} text parts, '
      f'{sum(x["type"] == "image_url" for x in first["messages"][1]["content"])} images; last text: '
      + json.dumps(texts[-1], ensure_ascii=False)[:200])
