"""Recompute the checksum of every file in a local HF snapshot and compare with the official tree listing."""
import concurrent.futures as cf
import hashlib
import json
import os
import sys

snap, tree_json = sys.argv[1], sys.argv[2]
official = {t['path']: t for t in json.load(open(tree_json)) if t['type'] == 'file'}


def digest(path, git_blob):
    size = os.path.getsize(path)
    h = hashlib.sha1(f'blob {size}\0'.encode()) if git_blob else hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(64 << 20):
            h.update(chunk)
    return size, h.hexdigest()


def check(name):
    t = official.get(name)
    if t is None:
        return name, 'NOT IN OFFICIAL REPO', '', ''
    lfs = t.get('lfs')
    want_size, want = (lfs['size'], lfs['oid']) if lfs else (t['size'], t['oid'])
    size, got = digest(os.path.join(snap, name), git_blob=lfs is None)
    ok = size == want_size and got == want
    return name, 'OK' if ok else 'MISMATCH', f'{size:,}', got if ok else f'got {got} want {want}'


local = sorted(n for n in os.listdir(snap) if n != '.cache')  # .cache = hf download bookkeeping
with cf.ThreadPoolExecutor(8) as ex:
    res = list(ex.map(check, local))
for name, status, size, note in res:
    print(f'{status:9s} {name:36s} {size:>15s}  {"sha256" if name.endswith(".safetensors") else "git-oid"} {note}')
print(f'\n{sum(r[1] == "OK" for r in res)}/{len(res)} local files match the official checksums')
print('official files not present locally:', sorted(set(official) - set(local)))
