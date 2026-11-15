"""EXPLORATORY (user, 2026-11-15; plan in CLAUDE.md "理由分析" -> "LLM 标注理由", recorded as given before this program
was written): LLM annotation of Qwen3.6-27B's 5,420 stated reasons (sep_camera, main set) as PHYSICAL / ENVIRONMENT.
Step 1 here: the validation sample for the user (no model needed).
  sample <round-1 work folder> <to_office folder> <key out_dir>: 200 reasons drawn at random without replacement,
    numpy default_rng(20261115).choice(5420, 200, replace=False) over round 1's reasons.jsonl rows (the answers_base.csv
    order; reasons sha256 c718132a... checked), items 1..200 in draw order. Writes <to_office>/reason_labels_sample.csv
    (item, reason, physical, environment; the last two empty; no score, no theme, no other label) and
    <key out_dir>/exploratory_llm_validation_key.csv (item, row, segment, frame_idx, repeat) to match the user's labels
    later (kept out of the download folder).
Step 2, run <round-1 work folder> <base_url> <out jsonl> (vLLM environment, server running): every reason once, one
  request at a time, in the order default_rng(20261115).permutation(5420); system = code/llm_labels_system_prompt.txt (the
  plan's text as given), user = "Explanation: <reason>", temperature 0, max_tokens 300, nothing else sent. Each reply: the
  first JSON object; a label fails when a key is missing, a label is not true/false, a quote is not an exact substring
  (case-sensitive), true has an empty quote or false a non-empty one. The failed ones are asked once more with the same
  request (<out>_reask.jsonl).
Step 3, analyze <waymo_v143> <round-1 work folder> <run jsonl> <out_dir> (waymo environment): labels (first answer if it
  passes, else the re-ask if it passes, else none = class failed), 4 classes, shares, mean score per class minus the mean
  of all 5,420 answers with the segment-bootstrap 95% interval, the 4-class composition of each k-means theme (k 7);
  answers matched to round 1's cluster table by segment, frame_idx, repeat.
usage: python -I llm_labels.py sample <round-1 work folder> <to_office folder> <key out_dir>
       python -I llm_labels.py run <round-1 work folder> <base_url> <out jsonl>
       python -I llm_labels.py analyze <waymo_v143> <round-1 work folder> <run jsonl> <out_dir>
"""
import csv
import hashlib
import json
import os
import sys

import numpy as np

REASONS_SHA = 'c718132acf962585cb1463f0fc1736b77d984b598808498f2bd8e5a3c7345373'
SEED, N_SAMPLE = 20261115, 200


def read_reasons(W1):
    recs = [json.loads(x) for x in open(os.path.join(W1, 'reasons.jsonl'), encoding='utf-8')]
    h = hashlib.sha256()
    for r in recs:
        h.update(r['reason'].encode('utf-8') + b'\n')
    if h.hexdigest() != REASONS_SHA or [r['row'] for r in recs] != list(range(len(recs))) or len(recs) != 5420:
        sys.exit(f'reasons.jsonl is not round 1\'s (sha256 {h.hexdigest()}, {len(recs)} rows)')
    return recs


def sample(W1, T, K):
    recs = read_reasons(W1)
    os.makedirs(T, exist_ok=True)
    os.makedirs(K, exist_ok=True)
    p_s, p_k = os.path.join(T, 'reason_labels_sample.csv'), os.path.join(K, 'exploratory_llm_validation_key.csv')
    for p in (p_s, p_k):
        if os.path.exists(p):
            sys.exit(f'{p} exists: stopped (not overwritten)')
    idx = np.random.default_rng(SEED).choice(len(recs), N_SAMPLE, replace=False)
    with open(p_s, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['item', 'reason', 'physical', 'environment'])
        for i, j in enumerate(idx, 1):
            w.writerow([i, recs[j]['reason'], '', ''])
    with open(p_k, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['item', 'row', 'segment', 'frame_idx', 'repeat'])
        for i, j in enumerate(idx, 1):
            r = recs[j]
            w.writerow([i, int(j), r['segment'], r['frame_idx'], r['repeat']])
    print(f'EXPLORATORY validation sample: {N_SAMPLE} of {len(recs)} reasons, default_rng({SEED}); {p_s} '
          f'({os.path.getsize(p_s)} bytes, {len(set(idx.tolist()))} distinct rows, {len({recs[j]["segment"] for j in idx})} '
          f'segments); key {p_k}')




# ----------------------------------------------------------------------------------------------- step 2: the model run
SYSTEM_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'llm_labels_system_prompt.txt')
MODEL_NAME = 'llama-3.1-8b-instruct'
SEND = dict(temperature=0, max_tokens=300)
KEYS = ('physical', 'physical_quote', 'environment', 'environment_quote')


def first_json(text):
    """The first JSON object in the reply (each '{' tried left to right with raw_decode), or None."""
    dec = json.JSONDecoder()
    for i, ch in enumerate(text or ''):
        if ch == '{':
            try:
                obj, _ = dec.raw_decode(text[i:])
            except ValueError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def check_label(reason, obj):
    """'' when the label is usable, else why not (the plan's checks)."""
    if obj is None:
        return 'no JSON object'
    if any(k not in obj for k in KEYS):
        return 'missing key ' + ','.join(k for k in KEYS if k not in obj)
    why = []
    for f in ('physical', 'environment'):
        lab, q = obj[f], obj[f + '_quote']
        if not isinstance(lab, bool):
            why.append(f'{f} not true/false')
            continue
        if not isinstance(q, str):
            why.append(f'{f}_quote not a string')
            continue
        if lab and q == '':
            why.append(f'{f} true with an empty quote')
        if not lab and q != '':
            why.append(f'{f} false with a non-empty quote')
        if q != '' and q not in reason:
            why.append(f'{f}_quote not a substring')
    return '; '.join(why)


def ask(client, system, reason):
    r = client.chat.completions.create(model=MODEL_NAME, messages=[
        {'role': 'system', 'content': system}, {'role': 'user', 'content': f'Explanation: {reason}'}], **SEND)
    ch = r.choices[0]
    text = ch.message.content
    obj = first_json(text)
    return dict(text=text, finish_reason=ch.finish_reason, prompt_tokens=r.usage.prompt_tokens,
                completion_tokens=r.usage.completion_tokens, parsed=obj, fail=check_label(reason, obj))


def run(W1, base_url, out):
    from openai import OpenAI
    recs = read_reasons(W1)
    system = open(SYSTEM_FILE, encoding='utf-8').read()
    if os.path.exists(out):
        sys.exit(f'{out} exists: stopped')
    client = OpenAI(base_url=base_url, api_key='EMPTY', timeout=600, max_retries=0)
    order = np.random.default_rng(SEED).permutation(len(recs))
    n_fail = 0
    with open(out, 'w', encoding='utf-8') as f:
        for k, j in enumerate(order):
            r = recs[j]
            d = dict(row=int(j), send_order=k, segment=r['segment'], frame_idx=r['frame_idx'], repeat=r['repeat'],
                     attempt=1, **ask(client, system, r['reason']))
            n_fail += d['fail'] != ''
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    fails = [json.loads(x) for x in open(out, encoding='utf-8') if json.loads(x)['fail']]
    p_re = out.replace('.jsonl', '_reask.jsonl')
    with open(p_re, 'w', encoding='utf-8') as f:
        for d0 in fails:
            r = recs[d0['row']]
            d = dict(row=d0['row'], send_order=d0['send_order'], segment=r['segment'], frame_idx=r['frame_idx'],
                     repeat=r['repeat'], attempt=2, **ask(client, system, r['reason']))
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    n_still = sum(json.loads(x)['fail'] != '' for x in open(p_re, encoding='utf-8'))
    print(f'{len(recs)} reasons sent one at a time in the order default_rng({SEED}).permutation; system prompt sha256 '
          f'{hashlib.sha256(system.encode()).hexdigest()}; sampling {SEND}; failures {n_fail}; re-asked {len(fails)}, still '
          f'failing {n_still}; {out}, {p_re}')


# ----------------------------------------------------------------------------------------------- step 3: the tables
CLASSES = ['physical only', 'environment only', 'both', 'neither']
N_BOOT, BOOT_SEED = 2000, 20261021
HASH = '48bc1f413f9d5b99da57ff4edde7bbb4eb378c5a9343c44ccad4e0165466604c'


def analyze(L, W1, run_file, O):
    import pandas as pd
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import analyze_rules as ar
    recs = read_reasons(W1)
    first = {d['row']: d for d in map(json.loads, open(run_file, encoding='utf-8'))}
    re_f = run_file.replace('.jsonl', '_reask.jsonl')
    second = {d['row']: d for d in map(json.loads, open(re_f, encoding='utf-8'))}
    if sorted(first) != list(range(len(recs))) or set(second) != {r for r, d in first.items() if d['fail']}:
        sys.exit('the run file does not have every reason once, or the re-asked rows are not the failed ones')
    ca = pd.read_csv(os.path.join(L, 'analysis', 'reason_27b', 'main_1084', 'exploratory_reason_answers.csv'),
                     float_precision='round_trip')
    key = pd.MultiIndex.from_frame(ca[['segment', 'frame_idx', 'repeat']])
    rows = []
    for j, r in enumerate(recs):
        d = second.get(j, first[j])
        ok = d['fail'] == ''
        p = d['parsed'] if ok else None
        cls = ('failed' if not ok else 'both' if p['physical'] and p['environment'] else 'physical only'
               if p['physical'] else 'environment only' if p['environment'] else 'neither')
        rows.append(dict(segment=r['segment'], frame_idx=int(r['frame_idx']), repeat=int(r['repeat']),
                         physical=(p['physical'] if ok else ''), environment=(p['environment'] if ok else ''),
                         **{'class': cls}, physical_quote=(p['physical_quote'] if ok else ''),
                         environment_quote=(p['environment_quote'] if ok else ''), attempt=d['attempt'],
                         fail_first=first[j]['fail'], fail_reask=(second[j]['fail'] if j in second else '')))
    t = pd.DataFrame(rows)
    ix = pd.MultiIndex.from_frame(t[['segment', 'frame_idx', 'repeat']])
    if not (ix.is_unique and set(ix) == set(key)):
        sys.exit('answers do not match the cluster table by segment, frame_idx, repeat')
    cm = ca.set_index(['segment', 'frame_idx', 'repeat']).reindex(ix)
    t.insert(3, 'kmeans7', cm.cluster.to_numpy(int))
    t.insert(4, 'score', cm.risk.to_numpy(float))
    os.makedirs(O, exist_ok=True)
    t.to_csv(os.path.join(O, 'exploratory_llm_labels.csv'), index=False)
    usable = t['class'] != 'failed'
    sh = [dict(**{'class': c}, answers=int((t['class'] == c).sum()), share_of_usable=float((t['class'] == c).sum() / usable.sum()),
               share_of_all=float((t['class'] == c).mean())) for c in CLASSES + ['failed']]
    pd.DataFrame(sh).to_csv(os.path.join(O, 'exploratory_llm_class_shares.csv'), index=False)
    # mean score per class minus the mean of all answers, segment bootstrap
    score, seg, cls = t.score.to_numpy(float), t.segment.to_numpy(), t['class'].to_numpy()
    seg_list = np.array(sorted(set(seg)))
    A = os.path.join(L, 'analysis', 'after_gemma-4-31b-it', 'main_1084')
    nb, sd, nf, segs = ar.read_params_risk(os.path.join(A, 'params.txt'))
    if (nb, sd, nf) != (N_BOOT, BOOT_SEED, 1084) or list(seg_list) != segs:
        sys.exit('resampling set-up differs from the main analysis')
    rng = np.random.default_rng(BOOT_SEED)
    draws = [rng.choice(seg_list, len(seg_list)) for _ in range(N_BOOT)]
    hh = hashlib.sha256()
    for dr in draws:
        hh.update(np.searchsorted(seg_list, dr).astype(np.int32).tobytes())
    if hh.hexdigest() != HASH:
        sys.exit("drawn segments' sha256 differs from the main analysis")
    seg_rows = {s: np.flatnonzero(seg == s) for s in seg_list}
    bd = np.full((N_BOOT, len(CLASSES)), np.nan)
    for b, dr in enumerate(draws):
        idx = np.concatenate([seg_rows[s] for s in dr])
        x, c_ = score[idx], cls[idx]
        ov = x.mean()
        for k, c in enumerate(CLASSES):
            m = c_ == c
            if m.any():
                bd[b, k] = x[m].mean() - ov
    overall = float(score.mean())
    rs = []
    for k, c in enumerate(CLASSES):
        m = cls == c
        v = bd[:, k]
        ok = ~np.isnan(v)
        lo, hi = (np.percentile(v[ok], [2.5, 97.5]) if ok.any() else (np.nan, np.nan))
        rs.append(dict(**{'class': c}, answers=int(m.sum()), frames=int(t[m][['segment', 'frame_idx']].drop_duplicates().shape[0]),
                       segments=int(t[m].segment.nunique()), mean_score=float(score[m].mean()) if m.any() else np.nan,
                       mean_all_answers=overall, mean_minus_all=float(score[m].mean() - overall) if m.any() else np.nan,
                       lo95=float(lo), hi95=float(hi), nboot=int(ok.sum()), missing=int((~ok).sum()),
                       side='no interval' if not ok.any() else 'above 0' if lo > 0 else 'below 0' if hi < 0 else 'contains 0'))
    pd.DataFrame(rs).to_csv(os.path.join(O, 'exploratory_llm_class_scores.csv'), index=False)
    comp = []
    for k7 in range(7):
        g = t[t.kmeans7 == k7]
        for c in CLASSES + ['failed']:
            comp.append(dict(kmeans7=k7, answers_in_theme=len(g), **{'class': c}, answers=int((g['class'] == c).sum()),
                             share_of_theme=float((g['class'] == c).mean()),
                             share_of_usable_in_theme=float((g['class'] == c).sum() / (g['class'] != 'failed').sum())))
    pd.DataFrame(comp).to_csv(os.path.join(O, 'exploratory_llm_theme_composition.csv'), index=False)
    with open(os.path.join(O, 'exploratory_llm_params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'EXPLORATORY: LLM annotation of the 5,420 reasons (user, 2026-11-15; plan in CLAUDE.md "LLM 标注理由")\n'
                f'annotator {MODEL_NAME} = meta-llama/Llama-3.1-8B-Instruct rev 0e9e39f249a16976918f6564b8830bc894c89659, bf16, '
                f'vLLM, official chat template; system prompt {SYSTEM_FILE} sha256 '
                f'{hashlib.sha256(open(SYSTEM_FILE, "rb").read()).hexdigest()}; user "Explanation: <reason>"; sampling {SEND}; '
                f'sent one at a time in the order default_rng({SEED}).permutation(5420)\nrun file {run_file}; re-asks {re_f}\n'
                f'labels used: the first answer when it passes the checks, else the re-ask when it passes, else none (class failed)\n'
                f'95% intervals: {N_BOOT} resamples of whole segments, default_rng({BOOT_SEED}), drawn segments\' sha256 '
                f'{hh.hexdigest()}; per resample class mean minus the mean of all drawn answers; percentiles 2.5 / 97.5 (numpy '
                f'linear) over the resamples where the class is present\nprogram {os.path.abspath(__file__)} sha256 '
                f'{hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()}\n')
    print(pd.DataFrame(sh).to_string(index=False))
    print(pd.DataFrame(rs).to_string(index=False))


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == 'sample':
        sample(*sys.argv[2:5])
    elif len(sys.argv) == 5 and sys.argv[1] == 'run':
        run(*sys.argv[2:5])
    elif len(sys.argv) == 6 and sys.argv[1] == 'analyze':
        analyze(*sys.argv[2:6])
    else:
        sys.exit(__doc__)
