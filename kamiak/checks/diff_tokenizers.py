"""Where two tokenizer.json files differ (2026-10-26: Qwen3.5-4B's is 12 bytes longer than Qwen-Drive-1.0-4B's).
As JSON: the top-level keys whose values differ; added_tokens entry by entry; the vocabulary and merges as wholes.
As text: the common start and end, and what lies between them on each side (one pass, so fast on 12.8 MB).
usage: python -I diff_tokenizers.py <tokenizer.json A> <tokenizer.json B>
"""
import json
import sys

pa, pb = sys.argv[1:3]
ra, rb = open(pa, 'rb').read(), open(pb, 'rb').read()
a, b = json.loads(ra), json.loads(rb)
print(f'A {pa}: {len(ra):,} bytes; B {pb}: {len(rb):,} bytes', flush=True)
print(f'top-level keys the same: {list(a) == list(b)} {list(a)}')
for k in a:
    if a.get(k) != b.get(k):
        print(f'  value differs: {k}')
ta, tb = {t['id']: t for t in a['added_tokens']}, {t['id']: t for t in b['added_tokens']}
print(f'added_tokens: A {len(ta)}, B {len(tb)}')
for i in sorted(set(ta) | set(tb)):
    if ta.get(i) != tb.get(i):
        print(f'  token {i}: A {ta.get(i)}\n            B {tb.get(i)}')
ma, mb = a.get('model', {}), b.get('model', {})
print(f"model: keys {list(ma) == list(mb)}; vocab equal {ma.get('vocab') == mb.get('vocab')}; merges equal "
      f"{ma.get('merges') == mb.get('merges')}; other model fields equal "
      f"{ {k: ma.get(k) == mb.get(k) for k in ma if k not in ('vocab', 'merges')} }")
n = min(len(ra), len(rb))
lo = next((i for i in range(n) if ra[i] != rb[i]), n)
hi = 0
while hi < n - lo and ra[len(ra) - 1 - hi] == rb[len(rb) - 1 - hi]:
    hi += 1
print(f'text: same first {lo:,} bytes and last {hi:,} bytes; between them A has {len(ra) - lo - hi} bytes, '
      f'B {len(rb) - lo - hi}')
print(f'  A: ...{ra[max(0, lo - 80):len(ra) - hi + 80]!r}...')
print(f'  B: ...{rb[max(0, lo - 80):len(rb) - hi + 80]!r}...')
