"""Gemma 4 31B-it pilot report (user, 2026-11-12: report the pilot's time estimate; the user confirms before the main
run). Reads the pilot file only. Prints:
- lines, distinct (segment, frame, condition, repeat), lines with an error;
- the thinking: lines with a non-empty `reasoning`, lines whose answer text or reasoning still holds Gemma's channel
  markers (<|channel>, <channel|>), and the number of thinking tokens: the generated tokens up to and including the
  first '<channel|>' in the per-token log-probabilities (lines without it are counted apart);
- per condition: prompt tokens (distinct values), generated tokens, thinking tokens, answer tokens;
- time: request time from the server being ready to the last line written (two times given), each condition's share
  by its summed latency (the requests run 32 at a time, so a condition's own time is not measured), the same split by
  whether the frame has a lead vehicle (truth csv with has_lead), and the projection to the main run (targets x
  5 conditions x 5 repeats) at the speed of all frames and of the frames with a lead, plus a start-up per job;
- size: bytes per line (all, frames with a lead) and the projected main-run file.
usage: python -I pilot_report.py <pilot jsonl> <server ready 'YYYY-MM-DD HH:MM:SS'> <last line 'YYYY-MM-DD HH:MM:SS'>
                                 <truth csv with has_lead> <main targets csv> [--startup_min 2.5]
"""
import argparse
import json

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('jsonl')
ap.add_argument('start')
ap.add_argument('end')
ap.add_argument('truth')
ap.add_argument('targets')
ap.add_argument('--startup_min', type=float, default=2.5)
a = ap.parse_args()
ORDER = ['grid', 'sep_time', 'sep_camera', 'ctrl_grid', 'ctrl_sep']
MARKS = ('<|channel>', '<channel|>')

rows, keys, n_err = [], set(), 0
with open(a.jsonl, 'rb') as f:
    for raw in f:
        d = json.loads(raw)
        n_err += 'error' in d
        keys.add((d['segment'], d['frame_idx'], d['condition'], d['repeat']))
        lp = d.get('logprobs') or []
        toks = [t['token'] for t in lp]
        think = toks.index('<channel|>') + 1 if '<channel|>' in toks else None
        u = d.get('usage') or {}
        rows.append(dict(segment=d['segment'], frame_idx=d['frame_idx'], condition=d['condition'],
                         latency=d['latency_s'], finish=d.get('finish_reason'), parsed=d.get('parsed') is not None,
                         prompt=u.get('prompt_tokens'), completion=u.get('completion_tokens'),
                         reasoning_tokens=(u.get('completion_tokens_details') or {}).get('reasoning_tokens'),
                         n_logprobs=len(lp), think=think, bytes=len(raw),
                         has_reasoning=bool((d.get('reasoning') or '').strip()),
                         marks_text=any(m in (d.get('text') or '') for m in MARKS),
                         marks_reasoning=any(m in (d.get('reasoning') or '') for m in MARKS),
                         text_empty=not (d.get('text') or '').strip()))
r = pd.DataFrame(rows)
print(f'{a.jsonl}: {len(r)} lines, {len(keys)} distinct (segment, frame, condition, repeat), {n_err} with an error')
print(f"finish_reason {r.finish.value_counts().to_dict()}; parsed {int(r.parsed.sum())}; answer text empty "
      f"{int(r.text_empty.sum())}")
print(f'thinking: non-empty reasoning {int(r.has_reasoning.sum())}/{len(r)}; channel markers left in the answer text '
      f'{int(r.marks_text.sum())}, in the reasoning {int(r.marks_reasoning.sum())}; usage reasoning_tokens given in '
      f'{int(r.reasoning_tokens.notna().sum())} lines')
print(f'log-probabilities: one per generated token in {int((r.n_logprobs == r.completion).sum())}/{len(r)} lines; '
      f"'<channel|>' found in {int(r.think.notna().sum())} lines")
r['answer'] = r.completion - r.think

print(f"\n{'condition':11s}{'n':>5s}  {'prompt tokens':>14s}  {'generated min/med/mean/max':>27s}  "
      f"{'thinking min/med/max':>21s}  {'answer med/max':>14s}  {'latency med/max s':>18s}")
for c in ORDER:
    g = r[r.condition == c]
    pr = sorted(g.prompt.unique().tolist())
    print(f"{c:11s}{len(g):5d}  {','.join(map(str, pr)):>14s}  "
          f"{g.completion.min():>6d}/{g.completion.median():>6.0f}/{g.completion.mean():>6.0f}/{g.completion.max():>6d}  "
          f"{g.think.min():>6.0f}/{g.think.median():>6.0f}/{g.think.max():>6.0f}  "
          f"{g.answer.median():>6.0f}/{g.answer.max():>6.0f}  {g.latency.median():>8.1f}/{g.latency.max():>8.1f}")
print(f'all: generated tokens median {r.completion.median():.0f}, mean {r.completion.mean():.1f}, max '
      f'{r.completion.max()}; thinking median {r.think.median():.0f}, max {r.think.max():.0f}')

hours = (pd.Timestamp(a.end) - pd.Timestamp(a.start)).total_seconds() / 3600
per = r.groupby('condition', sort=False).agg(requests=('latency', 'size'),
                                              latency_sum_h=('latency', lambda s: s.sum() / 3600),
                                              latency_median_s=('latency', 'median'),
                                              tokens_mean=('completion', 'mean')).reindex(ORDER)
per['share'] = per.latency_sum_h / per.latency_sum_h.sum()
per['hours_in_this_run'] = per.share * hours
per['seconds_per_request'] = per.hours_in_this_run * 3600 / per.requests
print(f'\ntime: request time {hours:.4f} h ({a.start} to {a.end}), {hours * 3600 / len(r):.3f} s per request; summed '
      f'latency {per.latency_sum_h.sum():.2f} h (= {per.latency_sum_h.sum() / hours:.1f} requests at a time on '
      f'average); generated tokens per second {r.completion.sum() / (hours * 3600):.1f}')
print(per.round(4).to_string())

tr = pd.read_csv(a.truth, usecols=['segment', 'frame_idx', 'has_lead'])
r = r.merge(tr, on=['segment', 'frame_idx'], how='left', validate='many_to_one')
if r.has_lead.isna().any():
    raise SystemExit(f'{a.truth}: frames of the pilot missing')
g = r.groupby('has_lead').agg(frames=('frame_idx', lambda s: len(set(zip(r.loc[s.index, 'segment'], s)))),
                              requests=('latency', 'size'), latency_sum_h=('latency', lambda s: s.sum() / 3600),
                              tokens_mean=('completion', 'mean'), bytes_mean=('bytes', 'mean'))
g['seconds_per_request'] = g.latency_sum_h / g.latency_sum_h.sum() * hours * 3600 / g.requests
print(f'\nby whether the frame has a lead vehicle ({a.truth}; time shared by the summed latency):')
print(g.round(3).to_string())

tg = pd.read_csv(a.targets, comment='#')
n_main = len(tg) * 5 * 5
spr_all = hours * 3600 / len(r)
spr_lead = float(g.loc[True, 'seconds_per_request'])
print(f'\nmain run: {len(tg)} targets ({a.targets}) x 5 conditions x 5 repeats = {n_main:,} requests '
      f'(= {n_main / len(r):.4f} x the pilot); plus {a.startup_min} min start-up')
for lab, spr in (('all frames', spr_all), ('frames with a lead', spr_lead)):
    tot = n_main * spr / 3600 + a.startup_min / 60
    print(f'  at the speed of {lab} ({spr:.3f} s per request): {tot:.2f} h')
print(f'  by condition at the speed of all frames (h): '
      f'{(per.hours_in_this_run * n_main / len(r)).round(2).to_dict()}')
by_all, by_lead = r.bytes.mean(), float(g.loc[True, 'bytes_mean'])
print(f'size: pilot {int(r.bytes.sum()):,} bytes, {by_all:,.0f} per line (frames with a lead {by_lead:,.0f}); main run '
      f'{n_main * by_all:,.0f} bytes at all frames, {n_main * by_lead:,.0f} at frames with a lead')
