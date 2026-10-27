"""One model's rows in two runs of the analysis (user, 2026-10-26: Qwen3-VL-32B-Instruct after all 5 models against
its own analysis of 2026-10-22; the resamples are shared by all models, so its numbers must not change).
For main_1084/ and sensitivity_1123/: every CSV with a model column, that model's lines in file order, compared as
text, line by line; ssm_target_frames.csv (no model column) compared whole; params.txt and the run logs: the lines
that differ, to read by eye.
usage: python -I compare_model_rows.py <old analysis folder> <new analysis folder> <model>
"""
import difflib
import os
import sys

old, new, model = sys.argv[1:4]
bad = 0
for sub in ('main_1084', 'sensitivity_1123'):
    print(f'== {sub}')
    for name in sorted(os.listdir(os.path.join(old, sub))):
        a_path, b_path = os.path.join(old, sub, name), os.path.join(new, sub, name)
        if not name.endswith('.csv'):
            continue
        if not os.path.exists(b_path):
            print(f'  {name}: MISSING in the new run')
            bad += 1
            continue
        a, b = open(a_path).read().splitlines(), open(b_path).read().splitlines()
        if a[0] != b[0]:
            print(f'  {name}: headers differ')
            bad += 1
            continue
        if a[0].startswith('model,'):
            a_rows = [r for r in a[1:] if r.startswith(model + ',')]
            b_rows = [r for r in b[1:] if r.startswith(model + ',')]
            models = sorted({r.split(',', 1)[0] for r in b[1:]})
            what = f'{model} lines: old {len(a_rows)} of {len(a) - 1}, new {len(b_rows)} of {len(b) - 1} (models in new: {models})'
        else:
            a_rows, b_rows = a[1:], b[1:]
            what = f'whole file: old {len(a_rows)} lines, new {len(b_rows)}'
        same = a_rows == b_rows
        bad += not same
        print(f'  {name}: {what}; identical: {same}')
        if not same:
            for i, (x, y) in enumerate(zip(a_rows, b_rows)):
                if x != y:
                    print(f'    first difference at line {i + 1}:\n      old {x[:300]}\n      new {y[:300]}')
                    break
    for name in ('params.txt',):
        a, b = open(os.path.join(old, sub, name)).read().splitlines(), open(os.path.join(new, sub, name)).read().splitlines()
        d = [l for l in difflib.unified_diff(a, b, lineterm='', n=0) if l[:1] in '+-' and l[:3] not in ('+++', '---')]
        print(f'  {name}: {len(d)} differing lines (old -, new +)')
        for l in d[:60]:
            print(f'    {l[:250]}')
for name in ('main_1084.log', 'sensitivity_1123.log'):
    a, b = open(os.path.join(old, name)).read().splitlines(), open(os.path.join(new, name)).read().splitlines()
    d = [l for l in difflib.unified_diff(a, b, lineterm='', n=0) if l[:1] in '+-' and l[:3] not in ('+++', '---')]
    print(f'== {name}: {len(d)} differing lines (old -, new +)')
    for l in d[:40]:
        print(f'    {l[:250]}')
print(f'\nCSV comparisons that are not identical: {bad}')
