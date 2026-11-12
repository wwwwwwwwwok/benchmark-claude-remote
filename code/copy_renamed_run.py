"""Copy a finished run's replies (main file and forced answers) into a new model folder with only the model field
renamed (user, 2026-11-11: the speed run 31628444 becomes the model qwen3.6-27b-speed, so the analysis programs read it
as a model of its own; the acceleration run 31681661 likewise). The source files are only read.

Byte for byte: in every line the one occurrence of '"model": "<old>"' becomes '"model": "<new>"' and nothing else
changes (inside a JSON string a quote is escaped, so the text can only be the key and its value); a line that holds it
other than exactly once, or already holds the new text, stops the copy. Written as <dest>/main_<new>.jsonl.gz and
<dest>/main_<new>_forced_answers.jsonl.gz (gzip level 6, first as .part); then read back from disk, renamed back and
compared with the source: the same number of lines, bytes and sha256 (of the decompressed text). Only then the .part is
renamed and set read-only; when a check is not met it stays as .part and the program ends with exit code 1.
usage: python -I copy_renamed_run.py <source main jsonl[.gz]> <source forced jsonl[.gz]> <old name> <new name> <dest>
"""
import gzip
import hashlib
import os
import stat
import sys
import time


def opener(path):
    return gzip.open(path, 'rb') if path.endswith('.gz') else open(path, 'rb')


src_main, src_forced, old, new, dest = sys.argv[1:6]
OLD, NEW = f'"model": "{old}"'.encode(), f'"model": "{new}"'.encode()
if OLD in NEW:
    sys.exit(f'the new text {NEW!r} contains the old one: renaming back would not be unique')
os.makedirs(dest, exist_ok=True)
t0 = time.time()
failed = []
for src, stem in ((src_main, f'main_{new}'), (src_forced, f'main_{new}_forced_answers')):
    out = os.path.join(dest, stem + '.jsonl.gz')
    part = out + '.part'
    for p in (out, part, os.path.join(dest, stem + '.jsonl')):
        if os.path.exists(p):
            sys.exit(f'{p} exists already: stop')
    h_src, n, size = hashlib.sha256(), 0, 0
    with opener(src) as f, gzip.open(part, 'wb', compresslevel=6) as g:
        for raw in f:
            n += 1
            size += len(raw)
            h_src.update(raw)
            if raw.count(OLD) != 1 or NEW in raw:
                sys.exit(f'{src} line {n}: {raw.count(OLD)} times {OLD!r}, the new text there already: '
                         f'{NEW in raw}; stop ({part} left as it is)')
            g.write(raw.replace(OLD, NEW, 1))
    h_new, h_back, m, size_back, bad = hashlib.sha256(), hashlib.sha256(), 0, 0, 0
    with gzip.open(part, 'rb') as g:
        for raw in g:
            m += 1
            h_new.update(raw)
            bad += raw.count(NEW) != 1 or OLD in raw
            back = raw.replace(NEW, OLD, 1)
            size_back += len(back)
            h_back.update(back)
    ok = m == n and size_back == size and h_back.hexdigest() == h_src.hexdigest() and bad == 0
    print(f'{os.path.basename(src)} -> {os.path.basename(out)}: {n} lines, {size} bytes, sha256 {h_src.hexdigest()}; '
          f'copy: {m} lines with the new name once each (lines not so: {bad}), sha256 of its text {h_new.hexdigest()}; '
          f'renamed back: {size_back} bytes, sha256 {h_back.hexdigest()}, the same as the source: {ok} '
          f'({time.time() - t0:.0f} s)', flush=True)
    if ok:
        os.rename(part, out)
        os.chmod(out, os.stat(out).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        print(f'  written {out} ({os.path.getsize(out)} bytes, read-only)', flush=True)
    else:
        failed.append(stem)
print(f'done in {time.time() - t0:.0f} s; checks not met: {failed if failed else "none"}')
sys.exit(1 if failed else 0)
