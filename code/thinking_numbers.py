"""EXPLORATORY, post-hoc (user, 2026-11-16; plan verbatim in CLAUDE.md "理由分析" -> "思考文字里的数", recorded before this
program was written): does Qwen3.6-27B's thinking text (risk main run, sep_camera, main set 1,084 frames, 5,420 answers)
state NUMBERS for SSM-related quantities (distance, speed, acceleration, time-to-X / headway), and for which road user.
Llama-3.1-8B-Instruct labels single sentences that contain a number.
  prepare <waymo_v143> <work dir> <to_office dir> <key dir> (waymo environment): thinking = the main reply's `reasoning`
    (the 26 forced answers keep the same text, as in reason_keywords.py). Sentences: split at line breaks, then after
    . ! ? followed by white space and an upper-case letter, digit, quote, bracket, * or - (the splitter of the size count
    of 2026-11-16). A sentence passes when it has a digit or a number word (NUMBER_WORDS, whole word, any case) outside the
    image time labels (TIME_LABEL). Writes <work>/sentences_pass.jsonl (every passing sentence, in answer order),
    <work>/pilot.jsonl (default_rng(20261116).choice(n_pass, 2000, replace=False), in drawn order), the user's labelling
    file <to_office>/thinking_labels_sample.csv (item, sentence, distance, speed, acceleration, time, subject; 200 of the
    pilot: default_rng(20261116).choice(2000, 200, replace=False)) and its key in <key dir>.
  run <sentences jsonl> <base_url> <out jsonl> (vLLM environment, server running): every sentence once, one request at a
    time, in file order; system = code/thinking_numbers_system_prompt.txt (the plan's text), user = "Sentence: <text>",
    official chat template, temperature 0, max_tokens 100, nothing else sent. A reply is valid when its first JSON object
    has distance, speed, acceleration, time as true/false and subject one of EGO, LEAD, OTHER, UNCLEAR, NONE. Invalid
    replies are asked once more (<out>_reask.jsonl); still invalid = not used. No quote check.
  compare <run A> <run B>: replies with different text and final label values that differ; exit 1 when anything differs.
  summary <run jsonl>: time per sentence, failures, label shares (valid final labels).
usage: python -I thinking_numbers.py prepare <waymo_v143> <work dir> <to_office dir> <key dir>
       python -I thinking_numbers.py run <sentences jsonl> <base_url> <out jsonl>
       python -I thinking_numbers.py compare <run A jsonl> <run B jsonl>
       python -I thinking_numbers.py summary <run jsonl>
"""
import collections
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SYSTEM_FILE = os.path.join(HERE, 'thinking_numbers_system_prompt.txt')
MODEL_NAME = 'llama-3.1-8b-instruct'
SEND = dict(temperature=0, max_tokens=100)
SEED = 20261116
N_PILOT, N_LABEL = 2000, 200
QTY = ('distance', 'speed', 'acceleration', 'time')
SUBJECTS = ('EGO', 'LEAD', 'OTHER', 'UNCLEAR', 'NONE')
NUMBER_WORDS = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
                'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty',
                'forty', 'fifty', 'sixty', 'hundred', 'half', 'quarter', 'couple', 'dozen']
NUMBER = re.compile(r'\d|\b(?:' + '|'.join(NUMBER_WORDS) + r')\b', re.IGNORECASE)
SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(*\-])')
_NL = r'(?:2\.0|1\.5|1\.0|0\.5)'
TIME_LABEL = re.compile('|'.join([   # the image time labels of the size count (2026-11-16), unchanged
    _NL + r'\s*(?:s|sec|secs|seconds?)?\s*ago\b',
    r'(?<![\d.])' + _NL + r'\s*(?:s|sec|secs|seconds?)\b(?!\s*(?:to|until|before|after)\b)',
    r'\b(?:moment|picture|image|frame|photo|shot|pic|img)s?\s*#?\s*\d\b',
    r'\b\d(?:st|nd|rd|th)\s+(?:moment|picture|image|frame|photo|shot)s?\b',
    r'\b(?:5|five)\s+(?:moments|pictures|images|frames|photos|shots|timestamps)\b',
    r'\b0\.5\s*(?:s|sec|secs|seconds?)\s+(?:apart|interval|intervals|between)\b',
    r'\b(?:first|last|final|latest|earliest)\s+(?:picture|image|frame|moment|photo)s?\b',
    r'\bnow\b']), re.IGNORECASE)


def sentences(t):
    return [s.strip() for line in t.split('\n') for s in SPLIT.split(line) if s.strip()]


def passes(s):
    lab = [m.span() for m in TIME_LABEL.finditer(s)]
    return any(not any(a <= m.start() and m.end() <= b for a, b in lab) for m in NUMBER.finditer(s))


def prepare(L, W, T, K):
    import pandas as pd
    import analyze_vlm_ssm as av
    A = os.path.join(L, 'analysis', 'after_gemma-4-31b-it', 'main_1084')
    MD = os.path.join(L, 'vlm_scores', 'qwen3.6-27b')
    lab_csv = os.path.join(T, 'thinking_labels_sample.csv')
    key_csv = os.path.join(K, 'exploratory_thinking_numbers_label_key.csv')
    for p in (os.path.join(W, 'sentences_pass.jsonl'), lab_csv, key_csv):
        if os.path.exists(p):
            sys.exit(f'{p} exists: stopped')
    td = pd.read_csv(os.path.join(A, 'vlm_requests_tidy.csv'), low_memory=False,
                     usecols=['model', 'condition', 'segment', 'frame_idx', 'repeat'])
    td = td[(td.model == 'qwen3.6-27b') & (td.condition == 'sep_camera')]
    keys = set(zip(td.segment, td.frame_idx.astype(int), td['repeat'].astype(int)))
    if not (len(keys) == len(td) == 5420 and td.segment.nunique() == 238):
        sys.exit('the tidy table does not have 5,420 answers in 238 segments')
    think = {}
    with av.open_jsonl(av.jsonl_file(MD, 'main_qwen3.6-27b')) as f:
        for raw in f:
            d = av.load_light(raw)
            k = (d['segment'], int(d['frame_idx']), int(d['repeat']))
            if d['condition'] == 'sep_camera' and k in keys:
                if k in think:
                    sys.exit(f'duplicate reply {k}')
                think[k] = d.get('reasoning') or ''
    if len(think) != 5420 or any(v == '' for v in think.values()):
        sys.exit('not every answer has a thinking text')
    n_sent, rows = 0, []
    for k in sorted(think):
        for j, s in enumerate(sentences(think[k])):
            n_sent += 1
            if passes(s):
                rows.append(dict(id=len(rows), segment=k[0], frame_idx=k[1], repeat=k[2], sent=j, text=s))
    os.makedirs(W, exist_ok=True)
    with open(os.path.join(W, 'sentences_pass.jsonl'), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    pick = np.random.default_rng(SEED).choice(len(rows), N_PILOT, replace=False)
    with open(os.path.join(W, 'pilot.jsonl'), 'w', encoding='utf-8') as f:
        for i in pick:
            f.write(json.dumps(rows[i], ensure_ascii=False) + '\n')
    lab = np.random.default_rng(SEED).choice(N_PILOT, N_LABEL, replace=False)
    os.makedirs(T, exist_ok=True)
    os.makedirs(K, exist_ok=True)
    ls = pd.DataFrame([dict(item=n + 1, sentence=rows[pick[i]]['text'], distance='', speed='', acceleration='', time='',
                            subject='') for n, i in enumerate(lab)])
    ls.to_csv(lab_csv, index=False, encoding='utf-8')
    pd.DataFrame([dict(item=n + 1, pilot_position=int(i), id=rows[pick[i]]['id'], segment=rows[pick[i]]['segment'],
                       frame_idx=rows[pick[i]]['frame_idx'], repeat=rows[pick[i]]['repeat'], sent=rows[pick[i]]['sent'])
                  for n, i in enumerate(lab)]).to_csv(key_csv, index=False)
    per = collections.Counter((r['segment'], r['frame_idx'], r['repeat']) for r in rows)
    pa = np.array([per[k] for k in sorted(think)])
    print(f'{len(think)} answers, {n_sent} sentences; passing (digit or number word outside the image time labels): '
          f'{len(rows)} ({len(rows) / n_sent:.6f}); per answer median {np.median(pa)}, 90th pct {np.percentile(pa, 90)}, '
          f'max {pa.max()}, answers with none {int((pa == 0).sum())}')
    print(f'pilot {N_PILOT} -> {W}/pilot.jsonl; labelling file {lab_csv} ({N_LABEL} rows, sha256 '
          f'{hashlib.sha256(open(lab_csv, "rb").read()).hexdigest()}); key {key_csv}')


def check(obj):
    if not isinstance(obj, dict):
        return 'no JSON object'
    why = [f'{q} not true/false' for q in QTY if not isinstance(obj.get(q), bool)]
    if obj.get('subject') not in SUBJECTS:
        why.append('subject not one of ' + '/'.join(SUBJECTS))
    return '; '.join(why)


def ask(client, system, text):
    from llm_labels import first_json
    t0 = time.monotonic()
    r = client.chat.completions.create(model=MODEL_NAME, messages=[
        {'role': 'system', 'content': system}, {'role': 'user', 'content': f'Sentence: {text}'}], **SEND)
    dt = time.monotonic() - t0
    ch = r.choices[0]
    obj = first_json(ch.message.content)
    return dict(text_reply=ch.message.content, finish_reason=ch.finish_reason, prompt_tokens=r.usage.prompt_tokens,
                completion_tokens=r.usage.completion_tokens, parsed=obj, fail=check(obj)), dt


def run(sent_file, base_url, out, concurrency=1):
    if concurrency > 1:
        return run_concurrent(sent_file, base_url, out, concurrency)
    from openai import OpenAI
    if os.path.exists(out):
        sys.exit(f'{out} exists: stopped')
    system = open(SYSTEM_FILE, encoding='utf-8').read()
    client = OpenAI(base_url=base_url, api_key='EMPTY', timeout=600, max_retries=0)
    recs = [json.loads(x) for x in open(sent_file, encoding='utf-8')]
    secs = []
    p_re = out.replace('.jsonl', '_reask.jsonl')
    with open(out, 'w', encoding='utf-8') as f:   # timings go to a side file so that the run files stay comparable
        for k, r in enumerate(recs):
            d, dt = ask(client, system, r['text'])
            secs.append(dt)
            f.write(json.dumps(dict(id=r['id'], send_order=k, attempt=1, **d), ensure_ascii=False) + '\n')
    first = [json.loads(x) for x in open(out, encoding='utf-8')]
    with open(p_re, 'w', encoding='utf-8') as f:
        for d0, r in zip(first, recs):
            if d0['fail']:
                d, dt = ask(client, system, r['text'])
                secs.append(dt)
                f.write(json.dumps(dict(id=r['id'], send_order=d0['send_order'], attempt=2, **d), ensure_ascii=False)
                        + '\n')
    with open(out.replace('.jsonl', '_seconds.json'), 'w') as f:
        json.dump(dict(requests=len(secs), seconds=secs), f)
    n1 = sum(bool(d['fail']) for d in first)
    n2 = sum(bool(json.loads(x)['fail']) for x in open(p_re, encoding='utf-8'))
    print(f'{len(recs)} sentences sent one at a time in file order; system prompt sha256 '
          f'{hashlib.sha256(system.encode()).hexdigest()}; sampling {SEND}; invalid {n1}, still invalid after re-ask {n2}; '
          f'request time {sum(secs):.1f} s for {len(secs)} requests ({np.mean(secs):.4f} s each)')


def run_concurrent(sent_file, base_url, out, concurrency):
    """User, 2026-11-16: <concurrency> requests in flight (threads); replies written in file order (same fields as the
    sequential run); re-asks also concurrent; _seconds.json has per-request times and the wall-clock time."""
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI
    if os.path.exists(out):
        sys.exit(f'{out} exists: stopped')
    system = open(SYSTEM_FILE, encoding='utf-8').read()
    client = OpenAI(base_url=base_url, api_key='EMPTY', timeout=600, max_retries=2)
    recs = [json.loads(x) for x in open(sent_file, encoding='utf-8')]
    p_re = out.replace('.jsonl', '_reask.jsonl')

    def many(items):
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            return list(ex.map(lambda r: ask(client, system, r['text']), items))
    t0 = time.monotonic()
    res = many(recs)
    t1 = time.monotonic()
    first = [dict(id=r['id'], send_order=k, attempt=1, **d) for k, (r, (d, _)) in enumerate(zip(recs, res))]
    with open(out, 'w', encoding='utf-8') as f:
        for d in first:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    redo = [(d0, r) for d0, r in zip(first, recs) if d0['fail']]
    res2 = many([r for _, r in redo])
    t2 = time.monotonic()
    with open(p_re, 'w', encoding='utf-8') as f:
        for (d0, r), (d, _) in zip(redo, res2):
            f.write(json.dumps(dict(id=r['id'], send_order=d0['send_order'], attempt=2, **d), ensure_ascii=False) + '\n')
    secs = [dt for _, dt in res] + [dt for _, dt in res2]
    with open(out.replace('.jsonl', '_seconds.json'), 'w') as f:
        json.dump(dict(requests=len(secs), seconds=secs, concurrency=concurrency, wall_first=t1 - t0, wall_reask=t2 - t1,
                       wall=t2 - t0), f)
    n2 = sum(bool(d['fail']) for d, _ in res2)
    print(f'{len(recs)} sentences, {concurrency} requests in flight, written in file order; system prompt sha256 '
          f'{hashlib.sha256(system.encode()).hexdigest()}; sampling {SEND}; invalid {len(redo)}, still invalid after re-ask '
          f'{n2}; wall-clock {t2 - t0:.1f} s ({len(recs) / (t2 - t0):.3f} sentences per second)', flush=True)


def split(src, out1, out2):
    """The passing sentences in id order: the first ceil(n/2) to out1, the rest to out2."""
    for p in (out1, out2):
        if os.path.exists(p):
            sys.exit(f'{p} exists: stopped')
    lines = open(src, encoding='utf-8').read().splitlines(keepends=True)
    h = (len(lines) + 1) // 2
    open(out1, 'w', encoding='utf-8').writelines(lines[:h])
    open(out2, 'w', encoding='utf-8').writelines(lines[h:])
    print(f'{len(lines)} sentences: {h} -> {out1}, {len(lines) - h} -> {out2}')


SUBJ2 = ('EGO', 'OTHER_ROAD_USER', 'UNCLEAR')


def rule_subject(lab):
    """The two subject rules (user, 2026-11-16): all four quantities false -> NONE; LEAD and OTHER -> OTHER_ROAD_USER."""
    if not any(lab[:4]):
        return 'NONE'
    return 'OTHER_ROAD_USER' if lab[4] in ('LEAD', 'OTHER') else lab[4]


def aggregate(L, sent_file, out_csv, *runs):
    """Per sentence (valid final labels, subject rules applied), per answer (any sentence) and per frame (1+ and 3+ of
    the 5 answers), by quantity x subject. Answers: all 5,420 of the tidy table (no passing sentence = no)."""
    import pandas as pd
    sents = {r['id']: r for r in map(json.loads, open(sent_file, encoding='utf-8'))}
    lab = {}
    for p in runs:
        f, s = load(p)
        for k in f:
            if k in lab:
                sys.exit(f'sentence {k} in two runs')
            lab[k] = final(f, s, k)
    if set(lab) != set(sents):
        sys.exit(f'runs cover {len(lab)} sentences, the file has {len(sents)}')
    A = os.path.join(L, 'analysis', 'after_gemma-4-31b-it', 'main_1084')
    td = pd.read_csv(os.path.join(A, 'vlm_requests_tidy.csv'), low_memory=False,
                     usecols=['model', 'condition', 'segment', 'frame_idx', 'repeat'])
    td = td[(td.model == 'qwen3.6-27b') & (td.condition == 'sep_camera')]
    answers = sorted(set(zip(td.segment, td.frame_idx.astype(int), td['repeat'].astype(int))))
    frames = sorted({a[:2] for a in answers})
    if len(answers) != 5420 or len(frames) != 1084:
        sys.exit('not 5,420 answers / 1,084 frames')
    QS = list(QTY) + ['any quantity']
    SS = list(SUBJ2) + ['any subject']
    hit = collections.defaultdict(set)   # (q, s) -> set of answer keys
    n_valid = n_fail = 0
    sent_cnt = collections.Counter()
    for k, l in lab.items():
        if l[0] == 'failed':
            n_fail += 1
            continue
        n_valid += 1
        sj = rule_subject(l)
        a = (sents[k]['segment'], int(sents[k]['frame_idx']), int(sents[k]['repeat']))
        for q in QS:
            qt = any(l[:4]) if q == 'any quantity' else l[QTY.index(q)]
            if not qt:
                continue
            for s in SS:
                if s == 'any subject' or s == sj:
                    hit[(q, s)].add(a)
                    sent_cnt[(q, s)] += 1
    rows = []
    for q in QS:
        for s in SS:
            ans = hit[(q, s)]
            per_frame = collections.Counter(a[:2] for a in ans)
            f1 = sum(1 for fr in frames if per_frame[fr] >= 1)
            f3 = sum(1 for fr in frames if per_frame[fr] >= 3)
            rows.append(dict(quantity=q, subject=s, sentences=sent_cnt[(q, s)], share_of_valid_sentences=sent_cnt[(q, s)] / n_valid,
                             answers=len(ans), share_of_answers=len(ans) / len(answers), frames_1of5=f1,
                             share_frames_1of5=f1 / len(frames), frames_3of5=f3, share_frames_3of5=f3 / len(frames)))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    sj_all = collections.Counter(rule_subject(l) for l in lab.values() if l[0] != 'failed')
    print(f'sentences {len(lab)}: valid {n_valid}, still invalid (not used) {n_fail}; subject after the rules: '
          + ', '.join(f'{s} {sj_all[s]}' for s in ('NONE',) + SUBJ2) + f'; answers {len(answers)}, frames {len(frames)}; {out_csv}')
    print(pd.DataFrame(rows).to_string(index=False))


def compare_pilot(seq_run, conc_run):
    """Concurrent run vs the sequential pilot on the same sentences: differing reply texts and label values (raw labels,
    before the subject rules); does not exit 1."""
    (fa, sa), (fb, sb) = load(seq_run), load(conc_run)
    if set(fa) != set(fb):
        sys.exit('not the same sentences')
    d1 = sum(fa[k]['text_reply'] != fb[k]['text_reply'] for k in fa)
    rows = [k for k in fa if final(fa, sa, k) != final(fb, sb, k)]
    vals = sum(1 if 'failed' in (final(fa, sa, k)[0], final(fb, sb, k)[0]) else
               sum(x != y for x, y in zip(final(fa, sa, k), final(fb, sb, k))) for k in rows)
    per = collections.Counter()
    for k in rows:
        a, b = final(fa, sa, k), final(fb, sb, k)
        if 'failed' not in (a[0], b[0]):
            for j, name in enumerate(list(QTY) + ['subject']):
                per[name] += a[j] != b[j]
    ra = [rule_subject(final(fa, sa, k)) for k in fa if final(fa, sa, k)[0] != 'failed' and final(fb, sb, k)[0] != 'failed']
    rb = [rule_subject(final(fb, sb, k)) for k in fa if final(fa, sa, k)[0] != 'failed' and final(fb, sb, k)[0] != 'failed']
    print(f'sequential vs concurrent, {len(fa)} sentences: replies with different text {d1}; sentences whose labels differ '
          f'{len(rows)}; label values differing {vals} of {5 * len(fa)} (' + ', '.join(f'{n} {per[n]}' for n in list(QTY) + ['subject'])
          + f'); subject after the two rules differs in {sum(x != y for x, y in zip(ra, rb))}')


def load(p):
    first = {d['id']: d for d in map(json.loads, open(p, encoding='utf-8'))}
    second = {d['id']: d for d in map(json.loads, open(p.replace('.jsonl', '_reask.jsonl'), encoding='utf-8'))}
    return first, second


def final(first, second, k):
    d = second.get(k, first[k])
    return tuple(d['parsed'][q] for q in QTY) + (d['parsed']['subject'],) if d['fail'] == '' else ('failed',)


def compare(a, b):
    (fa, sa), (fb, sb) = load(a), load(b)
    if set(fa) != set(fb):
        sys.exit('the two runs do not have the same sentences')
    d1 = sum(fa[k]['text_reply'] != fb[k]['text_reply'] for k in fa)
    d2 = sum(sa[k]['text_reply'] != sb[k]['text_reply'] for k in set(sa) & set(sb)) + len(set(sa) ^ set(sb))
    rows = [k for k in fa if final(fa, sa, k) != final(fb, sb, k)]
    vals = sum(1 if 'failed' in (final(fa, sa, k)[0], final(fb, sb, k)[0]) else
               sum(x != y for x, y in zip(final(fa, sa, k), final(fb, sb, k))) for k in rows)
    print(f'replies with different text: {d1} of {len(fa)}; re-asks different or asked in one run only: {d2}; sentences '
          f'whose final labels differ: {len(rows)} (label values differing: {vals} of 5 per sentence; a sentence valid in '
          f'only one run counted as 1)')
    sys.exit(1 if d1 or d2 or rows else 0)


def summary(p):
    first, second = load(p)
    sec = json.load(open(p.replace('.jsonl', '_seconds.json')))
    n = len(first)
    print(f'sentences {n}; requests {sec["requests"]}; seconds per request mean {np.mean(sec["seconds"]):.4f}, median '
          f'{np.median(sec["seconds"]):.4f}; per sentence (re-asks included) {sec.get("wall", sum(sec["seconds"])) / n:.4f}'
          f' (concurrency {sec.get("concurrency", 1)}); '
          f'completion tokens mean {np.mean([d["completion_tokens"] for d in first.values()]):.2f}, prompt tokens mean '
          f'{np.mean([d["prompt_tokens"] for d in first.values()]):.2f}; finish length {sum(d["finish_reason"] == "length" for d in first.values())}')
    for name, ds in (('first answer', first), ('after re-ask (not used)', second)):
        c = collections.Counter(d['fail'] for d in ds.values() if d['fail'])
        print(f'invalid, {name}: {sum(c.values())}' + ''.join(f'\n    {v:5d}  {k}' for k, v in sorted(c.items())))
    lab = [final(first, second, k) for k in first]
    ok = [x for x in lab if x[0] != 'failed']
    print(f'valid final labels: {len(ok)} of {n}')
    for j, q in enumerate(QTY):
        m = sum(x[j] for x in ok)
        print(f'  {q:12s} true {m:5d}  share {m / len(ok):.6f}')
    anyq = sum(any(x[:4]) for x in ok)
    print(f'  any quantity true {anyq:5d}  share {anyq / len(ok):.6f}')
    cs = collections.Counter(x[4] for x in ok)
    for s in SUBJECTS:
        print(f'  subject {s:8s} {cs[s]:5d}  share {cs[s] / len(ok):.6f}  (with any quantity true: '
              f'{sum(1 for x in ok if x[4] == s and any(x[:4]))})')
    print(f'  all four false but subject not NONE: {sum(1 for x in ok if not any(x[:4]) and x[4] != "NONE")}; a quantity '
          f'true but subject NONE: {sum(1 for x in ok if any(x[:4]) and x[4] == "NONE")}')


if __name__ == '__main__':
    a = sys.argv[1:]
    if len(a) == 5 and a[0] == 'prepare':
        prepare(*a[1:])
    elif len(a) == 4 and a[0] == 'run':
        run(*a[1:])
    elif len(a) == 6 and a[0] == 'run' and a[4] == '--concurrency':
        run(*a[1:4], concurrency=int(a[5]))
    elif len(a) == 4 and a[0] == 'split':
        split(*a[1:])
    elif len(a) >= 5 and a[0] == 'aggregate':
        aggregate(*a[1:])
    elif len(a) == 3 and a[0] == 'compare_pilot':
        compare_pilot(*a[1:])
    elif len(a) == 3 and a[0] == 'compare':
        compare(*a[1:])
    elif len(a) == 2 and a[0] == 'summary':
        summary(a[1])
    else:
        sys.exit(__doc__)
