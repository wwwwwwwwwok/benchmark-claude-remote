"""Candidate parser (first JSON object with a valid risk) vs the `parsed` field run_vlm.py stored, on every result file."""
import glob, json

DEC = json.JSONDecoder()


def parse_first(text):
    text = text or ''
    i = 0
    while (i := text.find('{', i)) >= 0:
        try:
            d, _ = DEC.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        r = d.get('risk') if isinstance(d, dict) else None
        if isinstance(r, str) and r.strip().isdigit():
            d['risk'] = int(r)
        if isinstance(d, dict) and isinstance(d.get('risk'), (int, float)) and 0 <= d['risk'] <= 100:
            return d
        i += 1
    return None


for f in sorted(glob.glob('/home/shuyi.wang/benchmark-claude-remote/results/vlm_scores/*/*.jsonl')):
    n = same = newly = lost = changed = 0
    for line in open(f, encoding='utf-8'):
        d = json.loads(line)
        if 'parsed' not in d:
            continue
        n += 1
        old, new = d['parsed'], parse_first(d.get('text'))
        if old == new:
            same += 1
        elif old is None:
            newly += 1
        elif new is None:
            lost += 1
        else:
            changed += 1
    print(f"{f.split('vlm_scores/')[1]:75s} rows {n:5d} same {same:5d} | newly parsed {newly} | lost {lost} | changed {changed}")
