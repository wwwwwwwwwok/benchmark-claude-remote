"""The replies cut off by length in a run file, counted for the user (2026-10-26, for the two 4B models' pilots):
per condition, those that stopped at max_tokens and those whose prompt + reply filled the context (completion
tokens below max_tokens: vLLM stopped at the context; a forced answer cannot fit its closing text there), and of
each kind how many fell into repetition. Repetition, as the user put it (every step's top-1 near certain at the
end): in the last 1,000 generated tokens at least 95% of the steps have a top-1 probability of 0.99 or more. As a
second sign, whether those 1,000 tokens repeat one block exactly (period up to 500 tokens). Also how many of them
have no readable answer (only those get a forced answer), and, to see whether the rule calls ordinary text
repetition, the same test on the replies that stopped normally with at least 1,000 tokens. Settings not touched.
usage: python -I count_cut_replies.py <run jsonl[.gz]> [--tail 1000] [--p 0.99] [--share 0.95]
"""
import argparse
import collections
import gzip
import json
import math

ap = argparse.ArgumentParser()
ap.add_argument('path')
ap.add_argument('--tail', type=int, default=1000)
ap.add_argument('--p', type=float, default=0.99)
ap.add_argument('--share', type=float, default=0.95)
a = ap.parse_args()


def period(ids, max_p=500):
    """The smallest p <= max_p with ids[i] == ids[i + p] all through, or None."""
    for p in range(1, min(max_p, len(ids) // 2) + 1):
        if all(ids[i] == ids[i + p] for i in range(len(ids) - p)):
            return p
    return None


def tail_share(d):
    """Share of the last --tail steps whose top-1 probability is >= --p, and those steps' tokens."""
    tail = (d.get('logprobs') or [])[-a.tail:]
    p1 = [math.exp(max(x[1] for x in s['top'])) if s.get('top') else math.exp(s['logprob']) for s in tail]
    return (sum(p >= a.p for p in p1) / len(p1) if p1 else float('nan')), [s['token'] for s in tail]


n_lines = 0
counts = collections.Counter()
examples = collections.defaultdict(list)
calib = collections.Counter()
shares_stop, shares_cut = [], []
with (gzip.open if a.path.endswith('.gz') else open)(a.path, 'rb') as f:
    for raw in f:
        n_lines += 1
        if b'"finish_reason": "length"' not in raw:
            if b'"finish_reason": "stop"' in raw:
                d = json.loads(raw)
                if d['usage']['completion_tokens'] >= a.tail:
                    share, _ = tail_share(d)
                    calib['stop, at least tail tokens'] += 1
                    calib['stop, called repetition'] += share >= a.share
                    shares_stop.append(share)
            continue
        d = json.loads(raw)
        ct, mt = d['usage']['completion_tokens'], d['max_tokens']
        kind = 'max_tokens' if ct >= mt else 'context full'
        share, toks = tail_share(d)
        shares_cut.append(share)
        rep = share >= a.share
        per = period(toks)
        c = d['condition']
        counts[(kind, c, 'all')] += 1
        counts[(kind, c, 'no answer')] += not d.get('parsed')
        counts[(kind, c, 'repetition')] += rep
        counts[(kind, c, 'no answer, repetition')] += rep and not d.get('parsed')
        counts[(kind, c, 'exact period')] += per is not None
        examples[(kind, c)].append(f"{d['segment'][:6]} {d['frame_idx']} r{d['repeat']}: prompt {d['usage']['prompt_tokens']} "
                                   f"+ {ct}; answer {'yes' if d.get('parsed') else 'no'}; last {a.tail} steps with "
                                   f"p1 >= {a.p}: {share:.1%}; period {per}")
print(f'{a.path}: {n_lines} lines; cut by length {sum(v for (k, c, t), v in counts.items() if t == "all")}')
print(f'repetition = in the last {a.tail} tokens at least {a.share:.0%} of steps with top-1 probability >= {a.p}')
for kind in ('max_tokens', 'context full'):
    conds = sorted({c for (k, c, t) in counts if k == kind})
    tot = sum(counts[(kind, c, 'all')] for c in conds)
    print(f'{kind}: {tot}')
    for c in conds:
        print(f"  {c:10s}: {counts[(kind, c, 'all')]} (no answer {counts[(kind, c, 'no answer')]}); repetition "
              f"{counts[(kind, c, 'repetition')]} (no answer {counts[(kind, c, 'no answer, repetition')]}); "
              f"exact period {counts[(kind, c, 'exact period')]}")
print(f"calibration: replies that stopped normally with at least {a.tail} tokens: {calib['stop, at least tail tokens']}, "
      f"of them called repetition by the same rule: {calib['stop, called repetition']}")


def quantiles(v):
    v = sorted(x for x in v if x == x)
    return {q: round(v[min(len(v) - 1, int(q * len(v)))], 3) for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)} if v else {}


print(f'share of the last {a.tail} steps with top-1 >= {a.p}, quantiles: stopped normally {quantiles(shares_stop)}; '
      f'cut by length {quantiles(shares_cut)}')
for (kind, c), ex in sorted(examples.items()):
    print(f'-- {kind} {c}')
    for e in ex[:40]:
        print(f'    {e}')
