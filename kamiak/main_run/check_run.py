"""Checks of a finished formal-experiment run against the main list and the same model's pilot run:
requests (every frame x condition x repeat once), the settings and fields recorded in every line, the prompt tokens per
condition (the same text and image sizes as the pilot give the same counts), finish_reason, parsing, answer values.
usage: python -I check_run.py <main jsonl[.gz]> <pilot jsonl> <main list csv>
"""
import collections
import gzip
import json
import sys

import numpy as np
import pandas as pd

CONDS = ['grid', 'sep_time', 'sep_camera', 'ctrl_grid', 'ctrl_sep']


def read(path):
    """Lines without their 'logprobs' list (as analyze_vlm_ssm.load_light), with a flag whether it was there and how
    many tokens it had."""
    out = []
    with (gzip.open if path.endswith('.gz') else open)(path, 'rb') as f:
        for raw in f:
            i = raw.find(b', "logprobs": ')
            n_lp = None
            if i >= 0:
                j = raw.find(b', "latency_s": ', i)
                n_lp = len(json.loads(raw[i + 14:j])) if j > 0 else None
                raw = raw[:i] + (raw[j:] if j > 0 else b'}')
            d = json.loads(raw)
            d['_n_logprobs'] = n_lp
            out.append(d)
    return out


main_path, pilot_path, lst = sys.argv[1:4]
m, p = read(main_path), read(pilot_path)
frames = pd.read_csv(lst, comment='#')[['segment', 'frame_idx']]
keys = [(d['segment'], d['frame_idx'], d['condition'], d['repeat']) for d in m]
want = {(s, int(k), c, r) for s, k in frames.itertuples(index=False) for c in CONDS for r in range(5)}
print(f'main run: {len(m)} lines, {len(set(keys))} distinct requests; expected {len(want)} '
      f'({len(frames)} frames x 5 conditions x 5 repeats): missing {len(want - set(keys))}, '
      f'not in the list {len(set(keys) - want)}')
print(f'pilot run: {len(p)} lines')

print('\nfields recorded per line (main / pilot):')
fm = collections.Counter(tuple(sorted(k for k in d if k != '_n_logprobs')) for d in m)
fp = collections.Counter(tuple(sorted(k for k in d if k != '_n_logprobs')) for d in p)
print(f'  main {len(fm)} field set(s), pilot {len(fp)}; same sets: {set(fm) == set(fp)}')
for s, n in fm.items():
    print(f'  main {n}: {", ".join(s)}')
lp_m = sum(d['_n_logprobs'] is not None for d in m)
lp_p = sum(d['_n_logprobs'] is not None for d in p)
tok_ok = sum(d['_n_logprobs'] == (d.get('usage') or {}).get('completion_tokens') for d in m)
print(f'  log-probabilities present: main {lp_m}/{len(m)}, pilot {lp_p}/{len(p)}; main lines with one per generated '
      f'token: {tok_ok}')

print('\nsettings (main / pilot):')
for f in ('model', 'temperature', 'top_p', 'top_k', 'max_tokens', 'system_prefix_file', 'reasoning'):
    vm = collections.Counter(str(d.get(f)) if f != 'reasoning' else str(d.get(f) is None) for d in m)
    vp = collections.Counter(str(d.get(f)) if f != 'reasoning' else str(d.get(f) is None) for d in p)
    print(f'  {f if f != "reasoning" else "reasoning is None"}: main {dict(vm)}; pilot {dict(vp)}')

print('\nprompt tokens per condition (value: lines), main / pilot:')
for c in CONDS:
    pm = collections.Counter(d['usage']['prompt_tokens'] for d in m if d['condition'] == c)
    pp = collections.Counter(d['usage']['prompt_tokens'] for d in p if d['condition'] == c)
    print(f'  {c:10s}: main {dict(pm)}; pilot {dict(pp)}')

print('\nreplies, main / pilot:')
for name, rs in (('main', m), ('pilot', p)):
    fr = collections.Counter(d.get('finish_reason') for d in rs)
    risk = np.array([d['parsed']['risk'] for d in rs if d.get('parsed')], float)
    ct = np.array([d['usage']['completion_tokens'] for d in rs], float)
    lat = np.array([d['latency_s'] for d in rs], float)
    print(f'  {name}: finish_reason {dict(fr)}; unparsable {sum(not d.get("parsed") for d in rs)}; '
          f'risk {risk.min():.0f}-{risk.max():.0f}, {len(np.unique(risk))} distinct values, mean {risk.mean():.2f}; '
          f'completion tokens median {np.median(ct):.0f}, max {ct.max():.0f}; latency median {np.median(lat):.2f} s, '
          f'max {lat.max():.2f} s')
for f in ('main_hazard_camera', 'hazard_type'):
    v = collections.Counter(str(d['parsed'].get(f)) for d in m if d.get('parsed'))
    print(f'  main {f}: {dict(v.most_common())}')
bad = [d for d in m if not d.get('parsed')]
for d in bad[:5]:
    print(f'  unparsable: {d["segment"][:6]} {d["frame_idx"]} {d["condition"]} {d["repeat"]}: {str(d.get("text"))[:300]!r}')
