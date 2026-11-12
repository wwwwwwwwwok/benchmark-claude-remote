"""The 'main_hazard' and 'reason' of every answer in one model folder's run (user, 2026-11-11, for
analyze_told_quantity.py and paper_sec43_more.py): the answer analyze_vlm_ssm.read_model uses (the reply's own parsed
answer; for a reply without one, the forced answer's when it has one), lines read with analyze_vlm_ssm.load_light
(log-probabilities skipped; the thinking is not used). Writes <out csv>: model, segment, frame_idx, condition, repeat,
forced, risk ('' without an answer), main_hazard, reason ('' when the answer has none or it is not text). The model
folder is only read.
usage: python -I extract_reasons.py <model folder> <run, e.g. main> <out csv>
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_vlm_ssm as av  # noqa: E402

folder, run, out = sys.argv[1:4]
if os.path.exists(out):
    sys.exit(f'{out} exists already: stop')
t0 = time.time()
name = os.path.basename(folder.rstrip('/\\'))
main = av.jsonl_file(folder, f'{run}_{name}')
forced = av.jsonl_file(folder, f'{run}_{name}_forced_answers')
if main is None:
    sys.exit(f'no {run}_{name}.jsonl(.gz) in {folder}')
rows = {}
with av.open_jsonl(main) as f:
    for raw in f:
        d = av.load_light(raw)
        k = tuple(d[c] for c in av.KEY)
        if k in rows:
            sys.exit(f'{main}: duplicate request {k}')
        p = d.get('parsed')
        rows[k] = [False, float(p['risk']) if p else None, p.get('main_hazard') if p else None,
                   p.get('reason') if p else None]
n_forced = 0
if forced:
    with av.open_jsonl(forced) as f:
        for raw in f:
            d = av.load_light(raw)
            k = tuple(d[c] for c in av.KEY)
            if rows[k][1] is not None:
                sys.exit(f'{forced}: forced answer for {k}, but the original reply has one')
            p = d.get('parsed')
            if not p:
                continue                                  # forced, still no readable answer
            rows[k] = [True, float(p['risk']), p.get('main_hazard'), p.get('reason')]
            n_forced += 1
n_text = n_haz = 0
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['model', 'segment', 'frame_idx', 'condition', 'repeat', 'forced', 'risk', 'main_hazard', 'reason'])
    for k, (fz, risk, hazard, reason) in rows.items():
        n_text += isinstance(reason, str)
        n_haz += isinstance(hazard, str)
        w.writerow([name, *k, fz, '' if risk is None else repr(risk), hazard if isinstance(hazard, str) else '',
                    reason if isinstance(reason, str) else ''])
print(f'{name}: {len(rows)} replies ({os.path.basename(main)}), forced answers used {n_forced}, answers '
      f'{sum(v[1] is not None for v in rows.values())}, with a text main_hazard {n_haz}, with a text reason {n_text}; '
      f'written {out} ({time.time() - t0:.0f} s)')
