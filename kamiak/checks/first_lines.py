"""Why the first lines of a run file give a size estimate that is off (2026-10-23): for the first N lines and for the
whole file, the number of frames they come from, the mean generated tokens and the mean bytes per line.
usage: python -I first_lines.py <run jsonl[.gz]> 300 3000
"""
import gzip
import json
import sys

marks = sorted(int(x) for x in sys.argv[2:])
n = size = tok = 0
frames = set()
with (gzip.open if sys.argv[1].endswith('.gz') else open)(sys.argv[1], 'rb') as f:
    for raw in f:
        n += 1
        size += len(raw)
        i = raw.find(b', "logprobs": ')
        if i >= 0:
            j = raw.find(b', "latency_s": ', i)
            raw = raw[:i] + (raw[j:] if j > 0 else b'}')
        d = json.loads(raw)
        tok += d['usage']['completion_tokens']
        frames.add((d['segment'], d['frame_idx']))
        if n in marks:
            print(f'first {n} lines: {len(frames)} frames, mean generated tokens {tok / n:,.0f}, mean {size / n:,.0f} bytes per line')
print(f'all {n} lines: {len(frames)} frames, mean generated tokens {tok / n:,.0f}, mean {size / n:,.0f} bytes per line')
