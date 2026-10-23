"""Forced answers of a formal-experiment main run (2026-10-23) against the main file and the same model's pilot forced
answers: the forced lines are exactly the main replies that stopped at max_tokens without an answer; the thinking and
usage sent back are the main reply's; fields, sampling, max_tokens and the early-stopping text are the same as in the
pilot's forced answers; finish_reason, parsing, prompt and answer tokens.
usage: python -I check_forced.py <main jsonl> <main forced jsonl> <pilot forced jsonl>
"""
import collections
import json
import sys

KEY = ('segment', 'frame_idx', 'condition', 'repeat')


def light(raw):
    """A line without its 'logprobs' list (as check_run.read)."""
    i = raw.find(b', "logprobs": ')
    if i >= 0:
        j = raw.find(b', "latency_s": ', i)
        raw = raw[:i] + (raw[j:] if j > 0 else b'}')
    return json.loads(raw)


def read(path):
    with open(path, 'rb') as f:
        return [light(raw) for raw in f]


main_path, forced_path, pilot_forced_path = sys.argv[1:4]
unparsable = {}
with open(main_path, 'rb') as f:
    for raw in f:
        d = light(raw)
        if d.get('parsed') is None:
            unparsable[tuple(d[k] for k in KEY)] = d
cut = {k for k, d in unparsable.items() if d.get('finish_reason') == 'length'}
print(f'main: {len(unparsable)} unparsable, finish_reason {dict(collections.Counter(d.get("finish_reason") for d in unparsable.values()))}; '
      f'{len(cut)} to be answered (length and no answer)')

fm, fp = read(forced_path), read(pilot_forced_path)
keys = [tuple(d[k] for k in KEY) for d in fm]
print(f'forced: {len(fm)} lines, {len(set(keys))} distinct; same keys as the cut replies: {set(keys) == cut}')
print(f'  per condition: {dict(collections.Counter(k[2] for k in keys))}')
print(f'  fields same as the pilot forced answers: '
      f'{set(tuple(sorted(d)) for d in fm) == set(tuple(sorted(d)) for d in fp)}')
for f in ('model', 'temperature', 'top_p', 'top_k', 'max_tokens', 'original_max_tokens', 'forced_answer',
          'early_stopping_text'):
    print(f'  {f}: main {dict(collections.Counter(repr(d.get(f)) for d in fm))}; '
          f'pilot {dict(collections.Counter(repr(d.get(f)) for d in fp))}')
same_think = sum(d['reasoning'] == unparsable[k]['reasoning'] for d, k in zip(fm, keys))
same_usage = sum(d['original_usage'] == unparsable[k]['usage'] for d, k in zip(fm, keys))
print(f'  thinking sent back equal to the main reply: {same_think}/{len(fm)}; original usage equal: {same_usage}/{len(fm)}')
for name, rs in (('main', fm), ('pilot', fp)):
    fr = collections.Counter(d.get('finish_reason') for d in rs)
    extra = sorted(d['usage']['prompt_tokens'] - d['original_usage']['prompt_tokens']
                   - d['original_usage']['completion_tokens'] for d in rs)
    ct = sorted(d['usage']['completion_tokens'] for d in rs)
    risk = sorted(d['parsed']['risk'] for d in rs if d.get('parsed'))
    print(f'  {name}: finish_reason {dict(fr)}; parsed {sum(bool(d.get("parsed")) for d in rs)}/{len(rs)}; '
          f'original completion tokens {sorted(set(d["original_usage"]["completion_tokens"] for d in rs))}; '
          f'prompt now minus (prompt + thinking) {extra}; answer tokens {ct}; risk {risk}; '
          f'latency {min(d["latency_s"] for d in rs)}-{max(d["latency_s"] for d in rs)} s')
cams = collections.Counter((d['condition'], (d.get('parsed') or {}).get('main_hazard_camera')) for d in fm)
print(f'  main camera per condition: {dict(sorted(cams.items()))}')
