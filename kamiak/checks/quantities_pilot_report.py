"""Report of a quantities-task pilot (user, 2026-10-28), per quantity x condition: answers read, reply length, time,
replies cut at max_tokens; the answers (median, range, negative, implausible); spread over the 5 repeats of a frame;
the truth on these frames; and, as a reference only, the frame median of the repeats against the truth (Kendall tau-b,
median absolute error, median error) and, for the accelerations, speeding up / steady / slowing down at +-0.5 m/s2.
"Implausible" (Claude's thresholds for this report only): distance above 200 m, a speed above 60 m/s (216 km/h), an
acceleration beyond +-10 m/s2 (about 1 g). Negative values are counted on their own (for the accelerations they are
normal).
usage: python -I quantities_pilot_report.py <pilot_quantities jsonl> <truth csv from code/quantity_truth.py>
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

QUANT = {'distance': ('distance_m', 'distance'), 'ego_speed': ('ego_speed_mps', 'v_ego'),
         'lead_speed': ('lead_speed_mps', 'v_lead'), 'ego_accel': ('ego_accel_mps2', 'a_ego'),
         'lead_accel': ('lead_accel_mps2', 'a_lead')}
CONDS = ['grid', 'sep_time', 'sep_camera', 'ctrl_grid', 'ctrl_sep']
IMPLAUSIBLE = {'distance': lambda v: v > 200, 'ego_speed': lambda v: v > 60, 'lead_speed': lambda v: v > 60,
               'ego_accel': lambda v: v.abs() > 10, 'lead_accel': lambda v: v.abs() > 10}

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)
path, truth_csv = sys.argv[1:3]
rows = []
with open(path, encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        q = d['quantity']
        p = d.get('parsed')
        u = d.get('usage') or {}
        det = u.get('completion_tokens_details') or {}
        rows.append(dict(segment=d['segment'], frame_idx=d['frame_idx'], condition=d['condition'], quantity=q,
                         repeat=d['repeat'], value=float(p[QUANT[q][0]]) if p else np.nan,
                         finish=d.get('finish_reason'), max_tokens=d.get('max_tokens'),
                         completion=u.get('completion_tokens'), reasoning_tokens=det.get('reasoning_tokens'),
                         latency=d.get('latency_s'), text=d.get('text')))
r = pd.DataFrame(rows)
keys = r[['segment', 'frame_idx', 'condition', 'quantity', 'repeat']]
print(f'{path}: {len(r)} lines, {len(keys.drop_duplicates())} distinct requests; '
      f'frames {len(r[["segment", "frame_idx"]].drop_duplicates())}; settings max_tokens {sorted(r.max_tokens.unique())}')

print('\n=== answers read, reply length, time (per quantity x condition)')
g = r.groupby(['quantity', 'condition'], sort=False)
t = g.agg(requests=('value', 'size'), read=('value', 'count'), at_max_tokens=('finish', lambda s: int((s == 'length').sum())),
          tokens_median=('completion', 'median'), tokens_max=('completion', 'max'),
          latency_median_s=('latency', 'median'), latency_max_s=('latency', 'max')).reset_index()
t['read_rate'] = (t.read / t.requests).round(4)
if r.reasoning_tokens.notna().any():
    t = t.merge(g.reasoning_tokens.median().rename('reasoning_median').reset_index(), on=['quantity', 'condition'])
print(t.to_string(index=False))
print(f'finish_reason overall: {r.finish.value_counts().to_dict()}')

bad = r[r.value.isna()]
if len(bad):
    print(f'\n=== replies without a readable number: {len(bad)} (up to 3 per quantity shown, text cut at 300 characters)')
    for q, sub in bad.groupby('quantity', sort=False):
        for x in sub.head(3).itertuples():
            print(f'  {q} {x.segment[:6]} {x.frame_idx} {x.condition} r{x.repeat} {x.finish}: {str(x.text)[:300]!r}')

print('\n=== the answers (per quantity x condition)')
out = []
for (q, c), sub in r.groupby(['quantity', 'condition'], sort=False):
    v = sub.value.dropna()
    out.append(dict(quantity=q, condition=c, n=len(v), median=v.median(), min=v.min(), max=v.max(),
                    distinct=v.nunique(), negative=int((v < 0).sum()), zero=int((v == 0).sum()),
                    implausible=int(IMPLAUSIBLE[q](v).sum())))
print(pd.DataFrame(out).round(3).to_string(index=False))

print('\n=== spread over the repeats of a frame (frames with at least 2 answers; SD with ddof 0)')
fr = r.dropna(subset=['value']).groupby(['quantity', 'condition', 'segment', 'frame_idx'], sort=False).value
sp = pd.DataFrame({'n': fr.size(), 'sd': fr.std(ddof=0), 'iqr': fr.quantile(0.75) - fr.quantile(0.25),
                   'all_same': fr.nunique() == 1}).reset_index()
sp = sp[sp.n >= 2]
print(sp.groupby(['quantity', 'condition'], sort=False).agg(frames=('sd', 'size'), sd_mean=('sd', 'mean'),
                                                            sd_median=('sd', 'median'), iqr_median=('iqr', 'median'),
                                                            all_5_same=('all_same', 'mean')).round(3).to_string())

tr = pd.read_csv(truth_csv)
print(f'\n=== truth on these frames ({truth_csv}; {len(tr)} frames, {int(tr.has_lead.sum())} with a lead)')
for q, (_, col) in QUANT.items():
    v = tr[col].dropna()
    print(f'  {q:10s} ({col}): {len(v)} frames, median {v.median():.3f}, range {v.min():.3f} to {v.max():.3f}')

print('\n=== reference only: frame median of the repeats against the truth (frames with a truth value and an answer)')
med = fr.median().rename('pred').reset_index()
ref = []
for (q, c), sub in med.groupby(['quantity', 'condition'], sort=False):
    m = sub.merge(tr[['segment', 'frame_idx', QUANT[q][1]]], on=['segment', 'frame_idx']).dropna()
    x, y = m.pred.to_numpy(float), m[QUANT[q][1]].to_numpy(float)
    tau = kendalltau(x, y, variant='b').statistic if len(m) > 1 and np.unique(x).size > 1 else np.nan
    row = dict(quantity=q, condition=c, frames=len(m), tau_b=tau, median_abs_error=np.median(np.abs(x - y)),
               median_error=np.median(x - y))
    if q.endswith('accel'):
        cls = lambda a: np.where(a > 0.5, 1, np.where(a < -0.5, -1, 0))     # noqa: E731
        cp, ct = cls(x), cls(y)
        row.update(class_right=float(np.mean(cp == ct)),
                   truth_up_steady_down=f'{int((ct == 1).sum())}/{int((ct == 0).sum())}/{int((ct == -1).sum())}',
                   answer_up_steady_down=f'{int((cp == 1).sum())}/{int((cp == 0).sum())}/{int((cp == -1).sum())}')
    ref.append(row)
print(pd.DataFrame(ref).round(3).to_string(index=False))
