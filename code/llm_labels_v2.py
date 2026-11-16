"""EXPLORATORY (user, 2026-11-15; plan in CLAUDE.md "理由分析" -> "LLM 标注理由", prompt v2 recorded as given before this
program was written and before any agreement was computed): Llama-3.1-8B-Instruct labels each of Qwen3.6-27B's 5,420
stated reasons (sep_camera, main set) with three yes/no labels: LEAD, OTHER, IMAGE_ONLY (prompt v2; replaces the
PHYSICAL / ENVIRONMENT prompt v1 and its rule 5, whose runs are kept in tmp_checks/reason_27b/llm_labels/prompt_v1*/ and
not reported; the first draft of v2 run 31820857 is kept in prompt_v2_draft/, also not reported; the prompt file is the
final v2 of 2026-11-15, extracted from the CLAUDE.md record). Two schemes: three-way (the three labels; class_3way = their combination) and two-way (TRAJECTORY = LEAD
or OTHER, IMAGE_ONLY; class_2way = trajectory only / image only / both / neither).
  run <round-1 work folder> <base_url> <out jsonl> (vLLM environment, server running): every reason once, one request at
    a time, in the order default_rng(20261115).permutation(5420); system = code/llm_labels_v2_system_prompt.txt (the
    plan's text as given), user = "Explanation: <reason>", official chat template, temperature 0, max_tokens 300,
    nothing else sent. Each reply: the first JSON object; a label fails when a key is missing, a label is not
    true/false, a quote is not a string or is a paraphrase under quote rule C (user, 2026-11-15; see quote_kind: exact,
    case only and joined pieces pass), true has an empty quote or false a non-empty one. The failed ones are asked once
    more with the same request (<out>_reask.jsonl).
  analyze <waymo_v143> <round-1 work folder> <run jsonl> <out_dir> (waymo environment): labels = the first answer if it
    passes, else the re-ask if it passes, else none (class failed; not used). Answers matched to round 1's cluster table
    by segment, frame_idx, repeat. Writes exploratory_llm_v2_labels.csv (one row per answer), _classes.csv (counts and
    shares, both schemes), _class_scores.csv (mean score per class minus the mean of all 5,420 answers, segment-bootstrap
    95% interval: the main analysis's 2,000 resamples, default_rng(20261021), drawn segments' sha256 48bc1f41...; per
    resample the class mean minus the mean of all drawn answers; 2.5 / 97.5 % numpy linear over the resamples where the
    class is present), _label_scores.csv (each of the three labels on its own: mentioned and not mentioned, each minus
    the mean of all answers, and mentioned minus not mentioned, same intervals), _theme_composition.csv (k-means k 7
    themes under both schemes), _failures.csv (failed replies by type, first answer and re-ask), _quote_kinds.csv
    (exact / case / pieces / paraphrase per label, first answers and the used labels), _label_rates.csv (replies with
    all three labels false and the share of each label true: usable labels, and all first answers with three
    true/false labels), _params.txt.
  compare <run A jsonl> <run B jsonl>: replies with different text (first answers, re-asks) and final label values
    that differ; exit 1 when anything differs.
usage: python -I llm_labels_v2.py run <round-1 work folder> <base_url> <out jsonl>
       python -I llm_labels_v2.py analyze <waymo_v143> <round-1 work folder> <run jsonl> <out_dir>
       python -I llm_labels_v2.py compare <run A jsonl> <run B jsonl>
"""
import collections
import hashlib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_labels as v1  # noqa: E402  (read_reasons, first_json: unchanged)

SYSTEM_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'llm_labels_v2_system_prompt.txt')
MODEL_NAME = 'llama-3.1-8b-instruct'
SEND = dict(temperature=0, max_tokens=300)
SEED = 20261115
LABELS = ('lead', 'other', 'image_only')
KEYS = tuple(k for f in LABELS for k in (f, f + '_quote'))
CL2 = ['trajectory only', 'image only', 'both', 'neither']
N_BOOT, BOOT_SEED = 2000, 20261021
HASH = '48bc1f413f9d5b99da57ff4edde7bbb4eb378c5a9343c44ccad4e0165466604c'


QUOTE_SPLIT = re.compile(r'[,;]|\band\b', flags=re.IGNORECASE)


def quote_kind(q, reason):
    """Quote rule C (user, 2026-11-15, recorded in CLAUDE.md before the run): 'exact' (a case-sensitive substring), 'case'
    (a substring when case is ignored), 'pieces' (not as a whole, but split on commas, semicolons and the word "and",
    every non-empty piece, stripped of white space only, is a case-insensitive substring; at least one piece), or
    'paraphrase' (a piece is not, or no piece at all). Only 'paraphrase' fails."""
    if q in reason:
        return 'exact'
    if q.lower() in reason.lower():
        return 'case'
    pieces = [p.strip() for p in QUOTE_SPLIT.split(q)]
    pieces = [p for p in pieces if p != '']
    if pieces and all(p.lower() in reason.lower() for p in pieces):
        return 'pieces'
    return 'paraphrase'


def check_label(reason, obj):
    """'' when the label is usable, else why not (the plan's checks, quote rule C)."""
    if obj is None:
        return 'no JSON object'
    if any(k not in obj for k in KEYS):
        return 'missing key ' + ','.join(k for k in KEYS if k not in obj)
    why = []
    for f in LABELS:
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
        if q != '' and quote_kind(q, reason) == 'paraphrase':
            why.append(f'{f}_quote paraphrase')
    return '; '.join(why)


def quote_kinds(reason, obj):
    if not isinstance(obj, dict):
        return {}
    return {f: quote_kind(obj[f + '_quote'], reason) for f in LABELS
            if isinstance(obj.get(f + '_quote'), str) and obj[f + '_quote'] != ''}


def fail_type(d):
    """One type per failed reply: length (cut at max_tokens), no JSON, missing key, label or quote type, paraphrase,
    consistency (true with an empty quote / false with a non-empty one); several reasons are joined in that order."""
    if d['fail'] == '':
        return ''
    t = []
    if d.get('finish_reason') == 'length':
        t.append('length')
    w = d['fail']
    if w == 'no JSON object':
        t.append('no JSON')
    elif w.startswith('missing key'):
        t.append('missing key')
    else:
        if 'not true/false' in w or 'not a string' in w:
            t.append('label or quote type')
        if 'paraphrase' in w:
            t.append('paraphrase')
        if 'empty quote' in w or 'non-empty quote' in w:
            t.append('consistency')
    return '+'.join(dict.fromkeys(t))


def ask(client, system, reason):
    r = client.chat.completions.create(model=MODEL_NAME, messages=[
        {'role': 'system', 'content': system}, {'role': 'user', 'content': f'Explanation: {reason}'}], **SEND)
    ch = r.choices[0]
    text = ch.message.content
    obj = v1.first_json(text)
    return dict(text=text, finish_reason=ch.finish_reason, prompt_tokens=r.usage.prompt_tokens,
                completion_tokens=r.usage.completion_tokens, parsed=obj, fail=check_label(reason, obj),
                quote_kinds=quote_kinds(reason, obj))


def run(W1, base_url, out):
    from openai import OpenAI
    recs = v1.read_reasons(W1)
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
    fails = [d for d in map(json.loads, open(out, encoding='utf-8')) if d['fail']]
    p_re = out.replace('.jsonl', '_reask.jsonl')
    with open(p_re, 'w', encoding='utf-8') as f:
        for d0 in fails:
            r = recs[d0['row']]
            d = dict(row=d0['row'], send_order=d0['send_order'], segment=r['segment'], frame_idx=r['frame_idx'],
                     repeat=r['repeat'], attempt=2, **ask(client, system, r['reason']))
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    n_still = sum(d['fail'] != '' for d in map(json.loads, open(p_re, encoding='utf-8')))
    print(f'{len(recs)} reasons sent one at a time, order default_rng({SEED}).permutation; system prompt sha256 '
          f'{hashlib.sha256(system.encode()).hexdigest()}; sampling {SEND}; failures {n_fail}; re-asked {len(fails)}, still '
          f'failing {n_still}; {out}, {p_re}')


def class3(p):
    on = [f for f in LABELS if p[f]]
    return '+'.join(on) if on else 'none'


def class2(traj, img):
    return 'both' if traj and img else 'trajectory only' if traj else 'image only' if img else 'neither'


def pct(v):
    ok = ~np.isnan(v)
    if not ok.any():
        return np.nan, np.nan, 0, int((~ok).sum())
    lo, hi = np.percentile(v[ok], [2.5, 97.5])
    return float(lo), float(hi), int(ok.sum()), int((~ok).sum())


def side(lo, hi):
    return 'no interval' if np.isnan(lo) else 'above 0' if lo > 0 else 'below 0' if hi < 0 else 'contains 0'


def analyze(L, W1, run_file, O):
    import pandas as pd
    import analyze_rules as ar
    recs = v1.read_reasons(W1)
    first = {d['row']: d for d in map(json.loads, open(run_file, encoding='utf-8'))}
    re_f = run_file.replace('.jsonl', '_reask.jsonl')
    second = {d['row']: d for d in map(json.loads, open(re_f, encoding='utf-8'))}
    if sorted(first) != list(range(len(recs))) or set(second) != {r for r, d in first.items() if d['fail']}:
        sys.exit('the run file does not have every reason once, or the re-asked rows are not the failed ones')
    ca = pd.read_csv(os.path.join(L, 'analysis', 'reason_27b', 'main_1084', 'exploratory_reason_answers.csv'),
                     float_precision='round_trip')
    rows = []
    for j, r in enumerate(recs):
        d = second.get(j, first[j])
        ok = d['fail'] == ''
        p = d['parsed'] if ok else None
        row = dict(segment=r['segment'], frame_idx=int(r['frame_idx']), repeat=int(r['repeat']))
        if ok:
            traj = p['lead'] or p['other']
            row.update(lead=p['lead'], other=p['other'], image_only=p['image_only'], trajectory=traj,
                       lead_quote=p['lead_quote'], other_quote=p['other_quote'], image_only_quote=p['image_only_quote'],
                       class_3way=class3(p), class_2way=class2(traj, p['image_only']))
        else:
            row.update(lead='', other='', image_only='', trajectory='', lead_quote='', other_quote='',
                       image_only_quote='', class_3way='failed', class_2way='failed')
        row.update(attempt=d['attempt'], fail_first=first[j]['fail'], fail_reask=(second[j]['fail'] if j in second else ''))
        rows.append(row)
    t = pd.DataFrame(rows)
    ix = pd.MultiIndex.from_frame(t[['segment', 'frame_idx', 'repeat']])
    key = pd.MultiIndex.from_frame(ca[['segment', 'frame_idx', 'repeat']])
    if not (ix.is_unique and set(ix) == set(key)):
        sys.exit('answers do not match the cluster table by segment, frame_idx, repeat')
    cm = ca.set_index(['segment', 'frame_idx', 'repeat']).reindex(ix)
    t.insert(3, 'kmeans7', cm.cluster.to_numpy(int))
    t.insert(4, 'score', cm.risk.to_numpy(float))
    os.makedirs(O, exist_ok=True)
    pfx = os.path.join(O, 'exploratory_llm_v2_')
    t.to_csv(pfx + 'labels.csv', index=False)
    usable = t.class_2way != 'failed'
    c3_all = ['none'] + ['+'.join(c) for n in (1, 2, 3) for c in __import__('itertools').combinations(LABELS, n)]
    schemes = {'3way': ('class_3way', c3_all), '2way': ('class_2way', CL2)}
    rows_c = []
    for sch, (col, cls) in schemes.items():
        for c in cls + ['failed']:
            n = int((t[col] == c).sum())
            rows_c.append(dict(scheme=sch, **{'class': c}, answers=n, share_of_usable=n / int(usable.sum()),
                               share_of_all=n / len(t)))
    pd.DataFrame(rows_c).to_csv(pfx + 'classes.csv', index=False)

    seg = t.segment.to_numpy()
    score = t.score.to_numpy(float)
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
    overall = float(score.mean())
    # groups: classes of both schemes, and each label mentioned / not mentioned (usable answers only)
    groups = [(sch, c, (t[col] == c).to_numpy()) for sch, (col, cls) in schemes.items() for c in cls]
    lab_groups = []
    for f in LABELS:
        lab_groups.append((f, 'mentioned', (usable & (t[f] == True)).to_numpy()))  # noqa: E712
        lab_groups.append((f, 'not mentioned', (usable & (t[f] == False)).to_numpy()))  # noqa: E712
    allg = groups + lab_groups
    bmean = np.full((N_BOOT, len(allg)), np.nan)
    bdiff = np.full((N_BOOT, len(allg)), np.nan)
    for b, dr in enumerate(draws):
        idx = np.concatenate([seg_rows[s] for s in dr])
        x = score[idx]
        ov = x.mean()
        for k, (_, _, m) in enumerate(allg):
            mm = m[idx]
            if mm.any():
                bmean[b, k] = x[mm].mean()
                bdiff[b, k] = bmean[b, k] - ov

    def grow(k, m):
        lo, hi, nbt, miss = pct(bdiff[:, k])
        return dict(answers=int(m.sum()), frames=int(t[m][['segment', 'frame_idx']].drop_duplicates().shape[0]),
                    segments=int(t[m].segment.nunique()), mean_score=float(score[m].mean()) if m.any() else np.nan,
                    mean_all_answers=overall, mean_minus_all=float(score[m].mean() - overall) if m.any() else np.nan,
                    lo95=lo, hi95=hi, nboot=nbt, missing=miss, side=side(lo, hi))

    rows_s = [dict(scheme=sch, **{'class': c}, **grow(k, m)) for k, (sch, c, m) in enumerate(groups)]
    pd.DataFrame(rows_s).to_csv(pfx + 'class_scores.csv', index=False)
    rows_l = []
    for i, f in enumerate(LABELS):
        k_on, k_off = len(groups) + 2 * i, len(groups) + 2 * i + 1
        for kk, nm in ((k_on, 'mentioned'), (k_off, 'not mentioned')):
            rows_l.append(dict(label=f, group=nm, **grow(kk, allg[kk][2])))
        d = bmean[:, k_on] - bmean[:, k_off]
        lo, hi, nbt, miss = pct(d)
        m_on, m_off = allg[k_on][2], allg[k_off][2]
        rows_l.append(dict(label=f, group='mentioned minus not mentioned', answers=int(m_on.sum() + m_off.sum()),
                           frames=np.nan, segments=np.nan, mean_score=np.nan, mean_all_answers=overall,
                           mean_minus_all=float(score[m_on].mean() - score[m_off].mean()), lo95=lo, hi95=hi, nboot=nbt,
                           missing=miss, side=side(lo, hi)))
    pd.DataFrame(rows_l).to_csv(pfx + 'label_scores.csv', index=False)
    rows_t = []
    for sch, (col, cls) in schemes.items():
        for k7 in range(7):
            g = t[t.kmeans7 == k7]
            nu = int((g[col] != 'failed').sum())
            for c in cls + ['failed']:
                n = int((g[col] == c).sum())
                rows_t.append(dict(scheme=sch, kmeans7=k7, answers_in_theme=len(g), **{'class': c}, answers=n,
                                   share_of_theme=n / len(g), share_of_usable_in_theme=(n / nu if nu else np.nan)))
    pd.DataFrame(rows_t).to_csv(pfx + 'theme_composition.csv', index=False)
    # failures by type, quote kinds, label rates (user, 2026-11-15)
    final = {j: second.get(j, first[j]) for j in first}
    rows_f = []
    for stage, ds in (('first answer', first), ('re-ask (still failing = not used)', second)):
        cnt = collections.Counter(fail_type(d) for d in ds.values() if d['fail'])
        for k, v in sorted(cnt.items()):
            rows_f.append(dict(stage=stage, failure_type=k, replies=v))
        rows_f.append(dict(stage=stage, failure_type='all failures', replies=sum(cnt.values())))
    pd.DataFrame(rows_f).to_csv(pfx + 'failures.csv', index=False)
    rows_q = []
    for stage, ds in (('first answer', first), ('final, used labels', {j: d for j, d in final.items() if not d['fail']})):
        cnt = collections.Counter()
        for j, d in ds.items():
            for f, k in quote_kinds(recs[j]['reason'], d['parsed']).items():
                cnt[(f, k)] += 1
        for f in LABELS:
            for k in ('exact', 'case', 'pieces', 'paraphrase'):
                rows_q.append(dict(stage=stage, label=f, quote_kind=k, quotes=cnt[(f, k)]))
    pd.DataFrame(rows_q).to_csv(pfx + 'quote_kinds.csv', index=False)
    used = t[usable]
    rows_r = [dict(group='usable labels', replies=len(used),
                   all_three_false=int(((used.lead == False) & (used.other == False) & (used.image_only == False)).sum()),  # noqa: E712
                   **{f'share_{f}_true': float((used[f] == True).mean()) for f in LABELS})]  # noqa: E712
    fp = [d['parsed'] for d in first.values() if isinstance(d['parsed'], dict) and all(isinstance(d['parsed'].get(f), bool)
                                                                                         for f in LABELS)]
    rows_r.append(dict(group='first answers with three true/false labels (any check result)', replies=len(fp),
                       all_three_false=int(sum(not any(p[f] for f in LABELS) for p in fp)),
                       **{f'share_{f}_true': float(np.mean([p[f] for p in fp])) for f in LABELS}))
    pd.DataFrame(rows_r).to_csv(pfx + 'label_rates.csv', index=False)
    n1 = sum(d['fail'] != '' for d in first.values())
    n2 = sum(d['fail'] != '' for d in second.values())
    with open(pfx + 'params.txt', 'w', encoding='utf-8') as f:
        f.write(f'EXPLORATORY: LLM annotation of the 5,420 reasons, prompt v2 (LEAD / OTHER / IMAGE_ONLY; user, '
                f'2026-11-15, plan in CLAUDE.md "LLM 标注理由")\nannotator {MODEL_NAME} = meta-llama/Llama-3.1-8B-Instruct rev '
                f'0e9e39f249a16976918f6564b8830bc894c89659, bf16, vLLM, official chat template; system prompt {SYSTEM_FILE} '
                f'sha256 {hashlib.sha256(open(SYSTEM_FILE, "rb").read()).hexdigest()}; user "Explanation: <reason>"; sampling '
                f'{SEND}; one request at a time, order default_rng({SEED}).permutation(5420)\nrun file {run_file}; re-asks '
                f'{re_f}; first-answer failures {n1}, still failing after the re-ask {n2} (class failed, not used)\n'
                f'two-way: TRAJECTORY = LEAD or OTHER\n95% intervals: {N_BOOT} resamples of whole segments, '
                f'default_rng({BOOT_SEED}), drawn segments\' sha256 {hh.hexdigest()}; group mean minus the mean of all drawn '
                f'answers; percentiles 2.5 / 97.5 (numpy linear) over the resamples where the group is present\nprogram '
                f'{os.path.abspath(__file__)} sha256 {hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()}\n')
    print(f'first-answer failures {n1}; still failing after re-ask {n2}')
    print(pd.DataFrame(rows_c).to_string(index=False))


def compare(run_a, run_b):
    """Two runs (each with its _reask file): replies whose text differs (first answers and re-asks) and final label values
    (3 labels, after the re-ask rule) that differ; exit 1 when anything differs."""
    out = {}
    for name, p in (('a', run_a), ('b', run_b)):
        first = {d['row']: d for d in map(json.loads, open(p, encoding='utf-8'))}
        second = {d['row']: d for d in map(json.loads, open(p.replace('.jsonl', '_reask.jsonl'), encoding='utf-8'))}
        out[name] = (first, second)
    (fa, sa), (fb, sb) = out['a'], out['b']
    d_first = sum(fa[k]['text'] != fb[k]['text'] for k in fa)
    common_re = set(sa) & set(sb)
    d_re = sum(sa[k]['text'] != sb[k]['text'] for k in common_re) + len(set(sa) ^ set(sb))

    def lab(f, s, k):
        d = s.get(k, f[k])
        return tuple(d['parsed'][x] for x in LABELS) if d['fail'] == '' else ('failed',)
    d_rows = [k for k in fa if lab(fa, sa, k) != lab(fb, sb, k)]
    d_vals = sum(1 if ('failed',) in (lab(fa, sa, k), lab(fb, sb, k)) else
                 sum(x != y for x, y in zip(lab(fa, sa, k), lab(fb, sb, k))) for k in d_rows)
    print(f'first answers with different text: {d_first} of {len(fa)}; re-asks with different text or asked in one run '
          f'only: {d_re}; answers whose final labels differ: {len(d_rows)} (label values differing: {d_vals}, a label set '
          f'that failed in one run counted as 1)')
    sys.exit(1 if d_first or d_re or d_rows else 0)


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == 'run':
        run(*sys.argv[2:5])
    elif len(sys.argv) == 6 and sys.argv[1] == 'analyze':
        analyze(*sys.argv[2:6])
    elif len(sys.argv) == 4 and sys.argv[1] == 'compare':
        compare(*sys.argv[2:4])
    else:
        sys.exit(__doc__)
