"""Estimate the final size of a run file from its first complete lines (user, 2026-10-22: for Qwen3.6-27B's main run,
from its first few hundred lines, so that the lab folder's free space can be checked before it fills up).
Prints: the average bytes per line of the first --lines complete lines, that times --total, the gzip size of the same
lines (level 6, as compress.sh's gzip -c), and the free space of the file's folder now. Lines are written in the
order the replies finish, so the first ones lean towards short replies; check how far off that is on a finished file.
usage: python -I estimate_size.py <run jsonl> [--lines 300] [--total 28075]
"""
import argparse
import os
import shutil
import zlib

ap = argparse.ArgumentParser()
ap.add_argument('path')
ap.add_argument('--lines', type=int, default=300, help='use the first N complete lines')
ap.add_argument('--total', type=int, default=28075, help='lines the finished file will have')
a = ap.parse_args()

n = size = 0
comp = zlib.compressobj(6, zlib.DEFLATED, 31)            # wbits 31: gzip format, as the gzip command writes
gz = 0
with open(a.path, 'rb') as f:
    for line in f:
        if not line.endswith(b'\n') or n == a.lines:      # stop at a line still being written
            break
        n += 1
        size += len(line)
        gz += len(comp.compress(line))
gz += len(comp.flush())
now = os.path.getsize(a.path)
free = shutil.disk_usage(os.path.dirname(os.path.abspath(a.path))).free
est, est_gz = size / n * a.total, gz / size * size / n * a.total
print(f'{a.path}: {now:,} bytes now; first {n} complete lines {size:,} bytes, {size / n:,.0f} per line')
print(f'  estimate for {a.total:,} lines: {est:,.0f} bytes; gzip of these lines {gz / size:.1%} -> about {est_gz:,.0f} '
      f'bytes compressed')
print(f'  still to be written {max(est - now, 0):,.0f} bytes, plus the compressed copy {est_gz:,.0f}: '
      f'{max(est - now, 0) + est_gz:,.0f} bytes; free in the folder now {free:,} bytes')
