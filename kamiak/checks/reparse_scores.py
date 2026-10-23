"""Recompute `parsed` from `text` with the current code/run_vlm.py parse() in ONE result file (2026-10-20: the
InternVL3.5-38B-Thinking pilot started 45 s before the first-JSON parser was saved). Backs the file up first and
refuses to write if any row that parsed before would now parse differently or not at all.

    python -I reparse_scores.py <result.jsonl> <backup.jsonl>"""
import json
import os
import shutil
import sys

sys.path.insert(0, '/home/shuyi.wang/benchmark-claude-remote/code')
from run_vlm import parse  # noqa: E402

src, backup = sys.argv[1], sys.argv[2]
assert not os.path.exists(backup), f'backup {backup} already exists'
rows = [json.loads(line) for line in open(src, encoding='utf-8')]
n = {'same': 0, 'newly parsed': 0, 'lost': 0, 'changed': 0}
for d in rows:
    old, new = d.get('parsed'), parse(d.get('text'))
    key = 'same' if old == new else 'newly parsed' if old is None else 'lost' if new is None else 'changed'
    n[key] += 1
    d['parsed'] = new
print(len(rows), 'rows:', n)
if n['lost'] or n['changed']:
    sys.exit('not written: some rows would change')
shutil.copy2(src, backup)
tmp = src + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    for d in rows:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')
os.replace(tmp, src)
print('backup:', backup, '| rewritten:', src)
