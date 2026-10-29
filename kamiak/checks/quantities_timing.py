"""Time of a quantities run by condition, by the measured speed (user, 2026-10-29), and what a run on other targets
would take at that speed. The requests run in parallel (--workers 32), so a condition's own time is not measured; its
share of the request time is taken as its share of the summed latency of its requests. Request time: from the first
to the last request, given as two times (from the vLLM log). Projection: requests x request time / requests of this
run, per condition and in all, plus the server start-up given.
usage: python -I quantities_timing.py <jsonl> <first request 'YYYY-MM-DD HH:MM:SS'> <last request ...>
                                      <targets of the other run> <repeats> [<repeats> ...] [--startup_min 2.5]
"""
import argparse
import json

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('jsonl')
ap.add_argument('start')
ap.add_argument('end')
ap.add_argument('targets')
ap.add_argument('repeats', type=int, nargs='+')
ap.add_argument('--startup_min', type=float, default=2.5)
a = ap.parse_args()
rows = []
for line in open(a.jsonl, encoding='utf-8'):
    d = json.loads(line)
    u = d.get('usage') or {}
    rows.append(dict(condition=d['condition'], quantity=d['quantity'], latency=d['latency_s'],
                     tokens=u.get('completion_tokens'), prompt=u.get('prompt_tokens')))
r = pd.DataFrame(rows)
hours = (pd.Timestamp(a.end) - pd.Timestamp(a.start)).total_seconds() / 3600
n_targets = len(pd.read_csv(a.targets))
per = r.groupby('condition', sort=False).agg(requests=('latency', 'size'), latency_sum_h=('latency', lambda s: s.sum() / 3600),
                                              latency_median_s=('latency', 'median'), tokens_median=('tokens', 'median'),
                                              tokens_mean=('tokens', 'mean'), prompt_median=('prompt', 'median'))
per['share'] = per.latency_sum_h / per.latency_sum_h.sum()
per['hours_in_this_run'] = per.share * hours
per['seconds_per_request'] = per.hours_in_this_run * 3600 / per.requests
print(f'{a.jsonl}: {len(r)} requests; request time {hours:.3f} h ({a.start} to {a.end}), '
      f'{hours * 3600 / len(r):.3f} s per request in all; summed latency {per.latency_sum_h.sum():.1f} h '
      f'(= {per.latency_sum_h.sum() / hours:.1f} requests at a time on average)')
print(per.round(3).to_string())
print(f'\nprojection to {n_targets} targets ({a.targets}), at {hours * 3600 / len(r):.3f} s per request, plus '
      f'{a.startup_min} min start-up per job:')
for rep in a.repeats:
    k = n_targets * 25 * rep / len(r)                             # 5 conditions x 5 quantities per target
    by_c = (per.hours_in_this_run * k).round(2).to_dict()
    tot = hours * k + a.startup_min / 60
    print(f'  {rep} repeat(s): {n_targets * 25 * rep:,} requests, {tot:.1f} h = {tot / 24:.2f} days; by condition (h) '
          f'{by_c}')
