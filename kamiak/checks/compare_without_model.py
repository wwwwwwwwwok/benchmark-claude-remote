"""The quantities analysis with one model more against the analysis without it (Claude, 2026-10-30: Qwen3.6-27B without
thinking added to the two 4B models; same program, same replies of the others, and the resamples are shared by all
models, so nothing of the other models may change).
For main_1084/ and sensitivity_1123/: every CSV of the old run against the new run's lines without the lines that name
the added model (one of the comma-separated fields is the model name), as text, line by line in file order, header
included; how many lines name the added model; files only in one of the two runs; the other files (params.txt,
exploratory_params.txt) and the run logs: the lines that differ, to read by eye.
usage: python -I compare_without_model.py <old analysis folder> <new analysis folder> <added model>
"""
import difflib
import os
import sys


def diff_lines(a_path, b_path, n_max):
    a, b = open(a_path).read().splitlines(), open(b_path).read().splitlines()
    d = [l for l in difflib.unified_diff(a, b, lineterm='', n=0) if l[:1] in '+-' and l[:3] not in ('+++', '---')]
    return d, d[:n_max]


old, new, model = sys.argv[1:4]
bad = 0
for sub in ('main_1084', 'sensitivity_1123'):
    a_names, b_names = sorted(os.listdir(os.path.join(old, sub))), sorted(os.listdir(os.path.join(new, sub)))
    only_old = [n for n in a_names if n not in b_names]
    only_new = [n for n in b_names if n not in a_names]
    bad += len(only_old) + len(only_new)
    print(f'== {sub}: {len(a_names)} files old, {len(b_names)} new; only in the old run {only_old}, only in the new '
          f'{only_new}')
    for name in a_names:
        if name in only_old:
            continue
        a_path, b_path = os.path.join(old, sub, name), os.path.join(new, sub, name)
        if not name.endswith('.csv'):
            d, show = diff_lines(a_path, b_path, 60)
            print(f'  {name}: {len(d)} differing lines (old -, new +)')
            for l in show:
                print(f'    {l[:250]}')
            continue
        a, b = open(a_path).read().splitlines(), open(b_path).read().splitlines()
        named = [model in l.split(',') for l in b]
        rest = [l for l, n in zip(b, named) if not n]
        same = a == rest
        bad += not same
        print(f'  {name}: old {len(a)} lines, new {len(b)}, of which {sum(named)} name {model}; the other {len(rest)} '
              f'identical to the old, in order: {same}')
        if not same:
            for i, (x, y) in enumerate(zip(a, rest)):
                if x != y:
                    print(f'    first difference at line {i + 1}:\n      old {x[:300]}\n      new {y[:300]}')
                    break
for name in ('main_1084.log', 'sensitivity_1123.log'):
    d, show = diff_lines(os.path.join(old, name), os.path.join(new, name), 40)
    print(f'== {name}: {len(d)} differing lines (old -, new +)')
    for l in show:
        print(f'    {l[:250]}')
print(f'\nCSV comparisons not identical, or files in one run only: {bad}')
