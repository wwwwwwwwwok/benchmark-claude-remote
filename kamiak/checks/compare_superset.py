"""An analysis with one cell more against the analysis without it (user, 2026-11-03: the exploratory version with
qwen3.6-27b x grid against the formal analysis, job 31542131). For main_1084/ and sensitivity_1123/: every CSV of the
old run must be in the new one line by line, character for character and in the same order (the old lines a
subsequence of the new), and every line the new one has more must name the model and the condition (both among its
comma-separated fields); the files of both runs; params.txt, exploratory_params.txt and the run logs: the lines that
differ, to read by eye.
usage: python -I compare_superset.py <old analysis folder> <new analysis folder> <model> <condition>
"""
import difflib
import os
import sys


def differing(a_path, b_path):
    a, b = open(a_path).read().splitlines(), open(b_path).read().splitlines()
    return [l for l in difflib.unified_diff(a, b, lineterm='', n=0) if l[:1] in '+-' and l[:3] not in ('+++', '---')]


old, new, model, cond = sys.argv[1:5]
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
            d = differing(a_path, b_path)
            print(f'  {name}: {len(d)} differing lines (old -, new +)')
            for l in d[:40]:
                print(f'    {l[:300]}')
            continue
        a, b = open(a_path).read().splitlines(), open(b_path).read().splitlines()
        j, more = 0, []
        for l in b:                                # the old lines, in order, within the new ones
            if j < len(a) and l == a[j]:
                j += 1
            else:
                more.append(l)
        cell = [model in l.split(',') and cond in l.split(',') for l in more]
        ok = j == len(a) and all(cell)
        bad += not ok
        print(f'  {name}: old {len(a)} lines, all in the new in order: {j == len(a)} ({j} found); new {len(b)}, '
              f'{len(more)} more, all of them {model} x {cond}: {all(cell)}')
        if not ok:
            if j < len(a):
                print(f'    first old line not found in order: {a[j][:300]}')
            for l, c in zip(more, cell):
                if not c:
                    print(f'    a line more that is not {model} x {cond}: {l[:300]}')
                    break
for name in ('main_1084.log', 'sensitivity_1123.log'):
    d = differing(os.path.join(old, name), os.path.join(new, name))
    print(f'== {name}: {len(d)} differing lines (old -, new +)')
    for l in d[:40]:
        if 'bootstrap' not in l:
            print(f'    {l[:300]}')
print(f'\nCSV comparisons not met, or files in one run only: {bad}')
