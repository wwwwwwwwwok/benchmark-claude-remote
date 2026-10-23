"""Quick data-quality summary of a run_vlm.py JSONL file (not part of the analysis)."""
import json
import statistics as st
import sys
from collections import Counter, defaultdict

ORDER = ['grid', 'sep_time', 'sep_camera', 'ctrl_grid', 'ctrl_sep']


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else None


rows = []
for line in open(sys.argv[1], encoding='utf-8'):
    d = json.loads(line)
    d.pop('logprobs', None)
    rows.append(d)
print(f'{len(rows)} lines in {sys.argv[1]}')
by = defaultdict(list)
for r in rows:
    by[r['condition']].append(r)

print(f"\n{'condition':11s}{'n':>5s}{'unparsed':>9s}{'length':>7s}  {'prompt med':>10s}  {'think min/med/max':>18s}"
      f"  {'answer med':>10s}  {'latency med/max s':>18s}  {'risk mean/sd/min/max':>21s}")
for c in ORDER:
    rs = by.get(c)
    if not rs:
        continue
    use = [r['usage'] for r in rs if r.get('usage')]
    think = [(u.get('completion_tokens_details') or {}).get('reasoning_tokens') or 0 for u in use]
    answer = [u['completion_tokens'] - t for u, t in zip(use, think)]
    prompt = [u['prompt_tokens'] for u in use]
    lat = [r['latency_s'] for r in rs]
    risk = [r['parsed']['risk'] for r in rs if r.get('parsed')]
    sd = st.pstdev(risk) if len(risk) > 1 else 0.0
    print(f"{c:11s}{len(rs):5d}{sum(r.get('parsed') is None for r in rs):9d}"
          f"{sum(r.get('finish_reason') == 'length' for r in rs):7d}  {q(prompt, .5):10d}"
          f"  {min(think):>5d}/{q(think, .5):>5d}/{max(think):>5d}  {q(answer, .5):10d}"
          f"  {q(lat, .5):>8.1f}/{max(lat):>8.1f}"
          f"  {st.mean(risk) if risk else float('nan'):>6.1f}/{sd:>4.1f}/{min(risk, default=-1):>4}/{max(risk, default=-1):>4}")

groups = defaultdict(list)
for r in rows:
    if r.get('parsed'):
        groups[(r['segment'], r['frame_idx'], r['condition'])].append(r['parsed']['risk'])
multi = {k: v for k, v in groups.items() if len(v) > 1}
if multi:
    print('\nrepeat stability (frames with >1 parsed answer): mean within-frame SD, mean range')
    for c in ORDER:
        vs = [v for k, v in multi.items() if k[2] == c]
        if vs:
            print(f'  {c:11s} frames={len(vs):4d}  SD={st.mean(st.pstdev(v) for v in vs):5.2f}'
                  f'  range={st.mean(max(v) - min(v) for v in vs):5.2f}')

print('\nfinish_reason:', dict(Counter(r.get('finish_reason') for r in rows)))
print('hazard_type:', dict(Counter((r.get('parsed') or {}).get('hazard_type') for r in rows)))
print('main_hazard_camera:', dict(Counter((r.get('parsed') or {}).get('main_hazard_camera') for r in rows)))
bad = [r for r in rows if r.get('parsed') is None]
for r in bad[:3]:
    print(f"\nUNPARSED {r['segment'][:6]} {r['frame_idx']} {r['condition']} finish={r.get('finish_reason')}: "
          f"{(r.get('text') or '')[:300]!r}")
