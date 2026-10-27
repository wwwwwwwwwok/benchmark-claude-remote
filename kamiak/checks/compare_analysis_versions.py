"""Two runs of analyze_vlm_ssm.py on the same replies, one without and one with an addition to the program (user,
2026-10-27: the model differences table; every table that was there before must stay the same). For each analysis
subfolder: the files of both runs; every CSV of the run without the addition compared with the other run's as text,
line by line, and byte for byte; params.txt and the run log <subfolder>.log next to the subfolder: the lines that
differ, to read by eye.
usage: python -I compare_analysis_versions.py <analysis folder without> <analysis folder with> [<subfolder> ...]
       (main_1084 and sensitivity_1123 if no subfolder is given)
"""
import difflib
import os
import sys

old, new = sys.argv[1:3]
subs = sys.argv[3:] or ['main_1084', 'sensitivity_1123']
bad = 0


def lines(path):
    return open(path, encoding='utf-8').read().splitlines()


def differing(a_path, b_path):
    return [l for l in difflib.unified_diff(lines(a_path), lines(b_path), lineterm='', n=0)
            if l[:1] in '+-' and l[:3] not in ('+++', '---')]


for sub in subs:
    a_names, b_names = set(os.listdir(os.path.join(old, sub))), set(os.listdir(os.path.join(new, sub)))
    print(f'== {sub}: {len(a_names)} files without, {len(b_names)} with; only in the run with: '
          f'{sorted(b_names - a_names) or "none"}; only in the run without: {sorted(a_names - b_names) or "none"}')
    bad += len(a_names - b_names)
    for name in sorted(a_names & b_names):
        a_path, b_path = os.path.join(old, sub, name), os.path.join(new, sub, name)
        if name.endswith('.csv'):
            a, b = lines(a_path), lines(b_path)
            same = a == b
            same_bytes = open(a_path, 'rb').read() == open(b_path, 'rb').read()
            bad += not (same and same_bytes)
            print(f'  {name}: {len(a)} lines without, {len(b)} with; identical line by line: {same}; '
                  f'bytes identical: {same_bytes}')
            if not same:
                i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                print(f'    first difference at line {i + 1}:\n      without {a[i][:300] if i < len(a) else "(end)"}\n'
                      f'      with    {b[i][:300] if i < len(b) else "(end)"}')
        else:
            d = differing(a_path, b_path)
            print(f'  {name}: {len(d)} differing lines (without -, with +)')
            for l in d[:60]:
                print(f'    {l[:250]}')
    log = sub + '.log'
    if os.path.exists(os.path.join(old, log)) and os.path.exists(os.path.join(new, log)):
        d = differing(os.path.join(old, log), os.path.join(new, log))
        print(f'  {log}: {len(d)} differing lines (without -, with +)')
        for l in d[:80]:
            print(f'    {l[:250]}')
print(f'\nCSV files missing from the run with the addition, or not identical: {bad}')
