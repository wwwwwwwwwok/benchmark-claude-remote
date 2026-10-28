"""A quantities-task run file (user, 2026-10-28): every cell (frame x condition x quantity) has all its repeats and no
request is there twice; the settings of every line; per quantity x condition the share read and the replies cut at
max_tokens; the replies without a readable number by kind, with a few of each; reply length (and thinking length when
recorded) and time.
Kinds of replies without a readable number (Claude's, for the report):
  cut_thinking  finish_reason length, nothing after the thinking (a thinking model cut while thinking)
  cut           finish_reason length, cut in the answer
  key_null      finished; a JSON object has the quantity's key, but its value is null
  key_text      finished; the key holds text that is not a number
  key_other     finished; the key holds something else (list, object, true / false, NaN, infinity)
  other_json    finished; JSON objects, none with the key
  bare_number   finished; the whole reply is one number, no JSON
  prose_number  finished; no JSON object, but a number in the text
  no_number     finished; no JSON object and no number (or no text at all)
usage: python -I check_quantities_run.py <run jsonl[.gz]> <target list csv> [--repeats 5] [--examples 3]
"""
import argparse
import gzip
import json
import re

import pandas as pd

CONDS = ['grid', 'sep_time', 'sep_camera', 'ctrl_grid', 'ctrl_sep']
KEYS = {'distance': 'distance_m', 'ego_speed': 'ego_speed_mps', 'lead_speed': 'lead_speed_mps',
        'ego_accel': 'ego_accel_mps2', 'lead_accel': 'lead_accel_mps2'}
KINDS = ['cut_thinking', 'cut', 'key_null', 'key_text', 'key_other', 'other_json', 'bare_number', 'prose_number',
         'no_number']


def kind(d, key):
    text = d.get('text') or ''
    if d.get('finish_reason') == 'length':
        return 'cut_thinking' if not text.strip() and d.get('reasoning') else 'cut'
    objs, i = [], 0
    while (i := text.find('{', i)) >= 0:
        try:
            o, _ = json.JSONDecoder().raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(o, dict):
            objs.append(o)
        i += 1
    with_key = [o for o in objs if key in o]
    if with_key:
        v = with_key[0][key]
        return 'key_null' if v is None else 'key_text' if isinstance(v, str) else 'key_other'
    if objs:
        return 'other_json'
    if re.fullmatch(r'\s*[-+]?\d+(\.\d*)?\s*', text):
        return 'bare_number'
    return 'prose_number' if re.search(r'\d', text) else 'no_number'


ap = argparse.ArgumentParser()
ap.add_argument('run')
ap.add_argument('targets')
ap.add_argument('--repeats', type=int, default=5)
ap.add_argument('--examples', type=int, default=3)
a = ap.parse_args()
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)

rows, settings, bad = [], {}, []
n_logprobs = n_prefix = 0
with (gzip.open(a.run, 'rt', encoding='utf-8') if a.run.endswith('.gz') else open(a.run, encoding='utf-8')) as f:
    for line in f:
        d = json.loads(line)
        q = d['quantity']
        u = d.get('usage') or {}
        det = u.get('completion_tokens_details') or {}
        s = (d.get('model'), d.get('temperature'), d.get('top_p'), d.get('top_k'), d.get('max_tokens'))
        settings[s] = settings.get(s, 0) + 1
        n_logprobs += 'logprobs' in d
        n_prefix += 'system_prefix_file' in d
        k = None if d.get('parsed') else kind(d, KEYS[q])
        rows.append(dict(segment=d['segment'], frame_idx=int(d['frame_idx']), condition=d['condition'], quantity=q,
                         repeat=int(d['repeat']), read=d.get('parsed') is not None, kind=k,
                         finish=d.get('finish_reason'), completion=u.get('completion_tokens'),
                         reasoning=det.get('reasoning_tokens'), latency=d.get('latency_s')))
        if k:
            bad.append(dict(kind=k, quantity=q, condition=d['condition'], segment=d['segment'][:6],
                            frame_idx=d['frame_idx'], repeat=d['repeat'], tokens=u.get('completion_tokens'),
                            text=(d.get('text') or '')[-240:]))
r = pd.DataFrame(rows)
tg = pd.read_csv(a.targets, comment='#')[['segment', 'frame_idx']].drop_duplicates()
key = ['segment', 'frame_idx', 'condition', 'quantity', 'repeat']
dup = int(r.duplicated(key).sum())
exp = pd.MultiIndex.from_tuples([(s, int(k), c, q, p) for s, k in tg.itertuples(index=False) for c in CONDS
                                 for q in KEYS for p in range(a.repeats)], names=key)
have = pd.MultiIndex.from_frame(r[key])
print(f'{a.run}: {len(r)} lines, duplicates {dup}; expected {len(exp)} ({len(tg)} frames x {len(CONDS)} conditions x '
      f'{len(KEYS)} quantities x {a.repeats} repeats): missing {int((~exp.isin(have)).sum())}, '
      f'not in the list {int((~have.isin(exp)).sum())}')
cells = r.groupby(['segment', 'frame_idx', 'condition', 'quantity']).repeat.nunique()
print(f'cells (frame x condition x quantity): {len(cells)}; repeats per cell {sorted(cells.unique())}')
print(f'settings (model, temperature, top_p, top_k, max_tokens): {settings}; lines with log-probabilities {n_logprobs}, '
      f'with a system prefix {n_prefix}')
print(f'finish_reason: {r.finish.value_counts().to_dict()}')

print('\n=== read and cut at max_tokens, per quantity x condition')
r['cut'] = r.finish == 'length'
t = r.groupby(['quantity', 'condition'], sort=False).agg(requests=('read', 'size'), read=('read', 'sum'),
                                                         cut=('cut', 'sum'))
t['read_rate'] = (t.read / t.requests).round(5)
t = t.reindex(pd.MultiIndex.from_product([list(KEYS), CONDS], names=['quantity', 'condition']))
print(t.to_string())
print(f'total: read {int(r.read.sum())} of {len(r)} ({r.read.mean():.5f}); cut {int(r.cut.sum())}; finished without a '
      f'readable number {int((~r.read & ~r.cut).sum())}')

print('\n=== replies without a readable number, by kind (rows: quantity; columns: kind)')
nb = r[~r.read]
if len(nb):
    print(pd.crosstab(nb.quantity, nb.kind).reindex(columns=[k for k in KINDS if k in set(nb.kind)]).to_string())
    print(pd.crosstab(nb.condition, nb.kind).reindex(columns=[k for k in KINDS if k in set(nb.kind)]).to_string())
    b = pd.DataFrame(bad)
    for k in KINDS:
        sub = b[b.kind == k]
        if len(sub):
            print(f'\n  {k}: {len(sub)}; {min(a.examples, len(sub))} examples (the last 240 characters of the reply)')
            for x in sub.head(a.examples).itertuples():
                print(f'    {x.quantity} {x.condition} {x.segment} {x.frame_idx} r{x.repeat} ({x.tokens} tokens): '
                      f'{x.text!r}')
else:
    print('  none')

print('\n=== reply length (completion tokens), thinking length (when recorded) and time, per condition')
agg = dict(tokens_median=('completion', 'median'), tokens_max=('completion', 'max'),
           latency_median_s=('latency', 'median'), latency_max_s=('latency', 'max'))
if r.reasoning.notna().any():
    agg.update(thinking_median=('reasoning', 'median'), thinking_max=('reasoning', 'max'))
print(r.groupby('condition', sort=False).agg(**agg).reindex(CONDS).to_string())
print(f'all: tokens median {r.completion.median()}, max {r.completion.max()}; latency median {r.latency.median()} s')
