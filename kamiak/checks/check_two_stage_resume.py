"""The quantities task run in two stages on one output file (user, 2026-10-29, for the Qwen3.6-27B main run): first
--repeats 1, then --repeats 3, which must send repeats 1 and 2 only, in the right order and number. run_vlm.py without
--send does not look at the output file, so both stages run run_vlm.main() with --send against a stand-in for the
OpenAI client that sends nothing and connects nowhere (as compare_risk_requests.py): it records every request (sha256
of the whole keyword set, each image's data URL replaced by its sha256) and answers {"<key>": 1.0}. One worker, so the
file order is the sending order. Checks: stage 1 writes repeat 0 of every target x condition x quantity, targets in
the order of the targets file, then conditions, then quantities, in the order given; stage 2 prints the right counts,
sends exactly repeats 1 and 2 of each, in that order (repeat 1 before 2), each request the same as stage 1's for that
target x condition x quantity; the file then holds every request of --repeats 3 exactly once.
usage: python -I check_two_stage_resume.py <run_vlm.py> <work folder> <run_vlm arguments without --repeats, --out,
       --workers and --send>
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import sys
import types

import pandas as pd

sys.dont_write_bytecode = True
py, work = sys.argv[1:3]
args = sys.argv[3:]


class Stand:
    """Stand-in for openai.OpenAI: records the requests, answers {"<key>": 1.0}."""
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
        s = json.dumps({**kw, 'messages': msgs}, ensure_ascii=False)
        Stand.requests.append(hashlib.sha256(s.encode()).hexdigest())
        key = re.findall(r'\{"(\w+)": <number>\}', msgs[-1]['content'][-1]['text'])[-1]
        msg = types.SimpleNamespace(content=json.dumps({key: 1.0}), reasoning=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason='stop', logprobs=None)],
                                     usage=None)


sys.modules['openai'] = types.SimpleNamespace(OpenAI=Stand)
spec = importlib.util.spec_from_file_location('run_vlm_checked', py)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
os.makedirs(work, exist_ok=True)
out = os.path.join(work, 'two_stage.jsonl')
if os.path.exists(out):
    sys.exit(f'{out} exists: stop')


def stage(repeats):
    Stand.requests = []
    log = io.StringIO()
    sys.argv = [py] + args + ['--repeats', str(repeats), '--workers', '1', '--out', out, '--send']
    with contextlib.redirect_stdout(log):
        mod.main()
    lines = [json.loads(x) for x in open(out, encoding='utf-8')]
    return log.getvalue().strip().splitlines(), Stand.requests, lines


def opt(name):
    return args[len(args) - 1 - args[::-1].index(name) + 1]


log1, req1, lines1 = stage(1)
log2, req2, lines2 = stage(3)
conds = opt('--conditions').split(',')
quants = opt('--quantities').split(',') if '--quantities' in args else list(mod.QUANTITIES)
tg = pd.read_csv(opt('--grid_targets'))
ts = pd.read_csv(opt('--sep_targets'))
t = tg.merge(ts[['segment', 'frame_idx']], on=['segment', 'frame_idx'], validate='one_to_one')
n_cell = len(t) * len(conds) * len(quants)
key = lambda d: (d['segment'], d['frame_idx'], d['condition'], d['quantity'], d['repeat'])   # noqa: E731
want1 = [(r.segment, int(r.frame_idx), c, q, 0) for r in t.itertuples() for c in conds for q in quants]
want2 = [(s, k, c, q, rep) for s, k, c, q, _ in want1 for rep in (1, 2)]
got1 = [key(d) for d in lines1]
got2 = [key(d) for d in lines2[len(lines1):]]
bad = 0


def report(ok, text):
    global bad
    bad += not ok
    print(f'  [{"ok" if ok else "NOT OK"}] {text}')


print(f'{len(t)} targets x {len(conds)} conditions x {len(quants)} quantities = {n_cell} cells; arguments {args}')
print(f'stage 1 (--repeats 1): {log1[0]} / {log1[-1]}')
print(f'stage 2 (--repeats 3): {log2[0]} / {log2[-1]}')
report(got1 == want1 and len(req1) == n_cell,
       f'stage 1: {len(req1)} requests sent, {len(got1)} lines, repeat 0 of every cell in the order targets -> '
       f'conditions -> quantities: {got1 == want1}')
report(log2[0] == f'{3 * n_cell} requests, {n_cell} already done, {2 * n_cell} to send',
       f'stage 2 counts: "{log2[0]}"')
report(got2 == want2 and len(req2) == 2 * n_cell,
       f'stage 2: {len(req2)} requests sent, {len(got2)} new lines, only repeats {sorted({k[4] for k in got2})}, in the '
       f'order targets -> conditions -> quantities -> repeat: {got2 == want2}')
report(req2 == [h for h in req1 for _ in (1, 2)],
       'stage 2: each request is the same as stage 1\'s for that target x condition x quantity')
allk = got1 + got2
report(len(allk) == len(set(allk)) == 3 * n_cell and set(allk) == {(s, k, c, q, rep) for s, k, c, q, _ in want1
                                                                   for rep in (0, 1, 2)},
       f'the file: {len(allk)} lines, every request of --repeats 3 exactly once')
s0 = {k: v for k, v in lines1[0].items() if k not in ('latency_s', 'text', 'parsed', 'reasoning', 'finish_reason',
                                                    'usage')}
print(f'first line (without the stand-in answer): {s0}')
print(f'\nchecks not passed: {bad}')
