"""Checks of the Qwen3.6-27B quantities runs without thinking (user, 2026-10-29), used by
kamiak/main_run/quantities_27b_nothink.sbatch to decide whether to go on.
  smoke <smoke jsonl> <smoke jsonl with thinking> <probe json> <expected settings json>
      The 25 smoke requests: each once, every line with the expected settings, no thinking (no `reasoning` text, no
      <think> or </think> in the reply, no reasoning tokens counted), a readable answer. The probe
      (probe_prompt_tail.py): every prompt ends with the empty thinking block, no thinking token generated, as many
      prompt tokens as the smoke line and 2 more than the same request with thinking (that prompt ends with <think>\n,
      this one with <think>\n\n</think>\n\n). Exit 0 when all hold; 3 when a reply was taken for thinking (empty
      content, the text in `reasoning`: the --reasoning-parser case); 1 otherwise.
  greedy <presence_penalty 1.5 jsonl> <presence_penalty 0 jsonl> [<1.5 again jsonl>]
      The greedy control: request by request, the answer (the number read) and the reply. Exit 0 when the first two
      give the same answer to every request, 1 otherwise. The third run (Claude's addition) shows whether greedy
      decoding repeats itself.
  gate <pilot jsonl> <truth_pilot.csv> <run log> <expected settings json> <folder of the main run>
      Before the main run: 0 errors (run_vlm.py's done line), every request once, the expected settings and no
      thinking on every line, answers read on the frames with a lead >= 98 %; and room for the main run (pilot bytes
      per line on the frames with a lead x 140,375 x 1.3, the margin of CLAUDE.md). Exit 0 when all hold, 1 otherwise.
  defaults <check_defaults.py output>
      The final sampling vLLM's code makes (qwen_drive_compare2/check_defaults.py, run for the run's request, then the
      two greedy ones): as the user set it (0.7 / 0.8 / 20, min_p 0, presence_penalty 1.5, repetition_penalty 1.0,
      frequency_penalty 0, 24,576; greedy: temperature 0, presence_penalty 1.5 and 0). Exit 0 when they are, 1 if not.
usage: python -I check_nothink_runs.py smoke|greedy|gate|defaults ...
"""
import ast
import json
import re
import shutil
import sys

import pandas as pd

KEY = ['segment', 'frame_idx', 'condition', 'quantity', 'repeat']
JKEY = {'distance': 'distance_m', 'ego_speed': 'ego_speed_mps', 'lead_speed': 'lead_speed_mps',
        'ego_accel': 'ego_accel_mps2', 'lead_accel': 'lead_accel_mps2'}       # run_vlm.py's QUANTITIES
MAIN_REQUESTS = 140375


def load(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(x) for x in f]


def key(d):
    return tuple(d.get(k) for k in KEY)


def value(d):
    p = d.get('parsed')
    return None if p is None else float(p[JKEY[d['quantity']]])


def settings_bad(lines, expect):
    return [d for d in lines if any(d.get(k) != v for k, v in expect.items())]


def thinking(d):
    """Signs of thinking in one output line."""
    rt = ((d.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens')
    text = d.get('text') or ''
    return ((['text in reasoning'] if d.get('reasoning') else [])
            + (['<think> or </think> in the reply'] if '<think>' in text or '</think>' in text else [])
            + ([f'{rt} reasoning tokens'] if rt else []))


def short(d):
    u = d.get('usage') or {}
    return (f"{d['condition']:10s} {d['quantity']:10s} r{d['repeat']} {d.get('finish_reason')} "
            f"{u.get('completion_tokens')} tokens, answer {value(d)}: {(d.get('text') or '')[:160]!r}")


def smoke(path, think_path, probe_path, expect):
    lines = load(path)
    keys = set(map(key, lines))
    print(f'{path}: {len(lines)} lines, {len(keys)} distinct requests')
    bad = settings_bad(lines, expect)
    print(f'settings {expect}: lines that differ {len(bad)}')
    print(f'finish_reason: {pd.Series([d.get("finish_reason") for d in lines]).value_counts().to_dict()}')
    taken = [d for d in lines if not (d.get('text') or '').strip() and d.get('reasoning')]
    th = [d for d in lines if thinking(d)]
    unread = [d for d in lines if d.get('parsed') is None]
    print(f'replies taken for thinking (empty content, text in reasoning): {len(taken)}; lines with signs of '
          f'thinking: {len(th)}; without a readable answer: {len(unread)}')
    for d in th:
        print(f'  thinking: {thinking(d)}: {short(d)}')
    ok = len(lines) == len(keys) == 25 and not bad and not th and not unread
    tk = {key(d): (d.get('usage') or {}).get('prompt_tokens') for d in load(think_path)}
    sm = {key(d): (d.get('usage') or {}).get('prompt_tokens') for d in lines}
    with open(probe_path, encoding='utf-8') as f:
        pr = json.load(f)
    t = pd.DataFrame([dict(condition=r['condition'], quantity=r['quantity'], probe=r['prompt_tokens'],
                           smoke=sm.get(key(r)), with_thinking=tk.get(key(r)), ends=r['ends_with_empty_thinking'],
                           thinking_token=r['thinking_token_generated'], tail=r['tail_text']) for r in pr])
    t['probe_minus_with_thinking'] = t.probe - t.with_thinking
    print('\nprompt tokens: probe (return_token_ids), smoke line, the same request with thinking')
    print(t.to_string(index=False))
    pok = (len(t) == 25 and bool(t.ends.all()) and not t.thinking_token.any() and bool((t.probe == t.smoke).all())
           and bool((t.probe_minus_with_thinking == 2).all()))
    print(f'probe: 25 requests, every prompt ends with the empty thinking block, no thinking token generated, '
          f'prompt tokens = smoke line = with thinking + 2: {pok}')
    print('\nreplies:')
    for d in lines:
        print('  ' + short(d))
    if taken:
        print('SMOKE: replies taken for thinking -> start the server without --reasoning-parser')
        return 3
    print(f'SMOKE PASSED: {ok and pok}')
    return 0 if ok and pok else 1


def greedy(paths):
    runs = [{key(d): d for d in load(p)} for p in paths]
    a = runs[0]
    same = None
    for p, b in zip(paths[1:], runs[1:]):
        ks = sorted(set(a) | set(b))
        da = [k for k in ks if k not in a or k not in b or value(a[k]) != value(b[k])]
        dt = [k for k in ks if k in a and k in b and a[k].get('text') != b[k].get('text')]
        print(f'{paths[0]} vs {p}: {len(a)} / {len(b)} requests; different answers {len(da)}, different replies '
              f'{len(dt)}')
        for k in ks:
            if k in da or k in dt:
                x, y = a.get(k), b.get(k)
                print(f'  {"ANSWER" if k in da else "reply "} {k[2]} {k[3]} r{k[4]}: {value(x) if x else None} vs '
                      f'{value(y) if y else None}')
                print(f'    {(x or {}).get("text")!r}\n    {(y or {}).get("text")!r}')
        if same is None:
            same = not da and len(a) == len(b) == 25
    print(f'GREEDY SAME ANSWERS (presence_penalty 1.5 vs 0): {same}')
    return 0 if same else 1


def gate(path, truth_csv, log_path, expect, main_dir):
    with open(path, 'rb') as f:
        raw = f.read().splitlines()
    lines = [json.loads(x) for x in raw]
    keys = [key(d) for d in lines]
    tr = pd.read_csv(truth_csv)
    frames = set(zip(tr.segment, tr.frame_idx))
    lead = set(zip(tr.segment[tr.has_lead], tr.frame_idx[tr.has_lead]))
    with open(log_path, encoding='utf-8') as f:
        done = [x.strip() for x in f if x.startswith('done:')]
    m = re.match(r'done: (\d+) ok \((\d+) unparsable\), (\d+) errors', done[-1]) if done else None
    bad = settings_bad(lines, expect)
    th = [d for d in lines if thinking(d)]
    on_lead = [i for i, d in enumerate(lines) if (d['segment'], d['frame_idx']) in lead]
    read = sum(lines[i].get('parsed') is not None for i in on_lead)
    per_line = sum(len(raw[i]) + 1 for i in on_lead) / len(on_lead)
    need = per_line * MAIN_REQUESTS * 1.3
    free = shutil.disk_usage(main_dir).free
    checks = {
        f'0 errors ({done[-1] if done else "no done line"})': m is not None and int(m.group(3)) == 0,
        f'every request once ({len(keys)} lines, {len(set(keys))} distinct, expected {len(frames) * 125}; every frame '
        f'in the truth file)': len(keys) == len(set(keys)) == len(frames) * 125
        and all(k[:2] in frames for k in keys),
        f'settings {expect} on every line (lines that differ: {len(bad)})': not bad,
        f'no thinking on any line (lines with signs: {len(th)})': not th,
        f'answers read on the frames with a lead: {read} of {len(on_lead)} = {read / len(on_lead):.4%} (>= 98 %)':
            read / len(on_lead) >= 0.98,
        f'room for the main run: {per_line:.0f} bytes per line x {MAIN_REQUESTS:,} x 1.3 = {need / 1e9:.2f} GB; free in '
        f'{main_dir}: {free / 1e9:.1f} GB': free > need,
    }
    for c, v in checks.items():
        print(f'  [{"ok" if v else "NOT MET"}] {c}')
    for d in th[:10]:
        print(f'  thinking: {thinking(d)}: {short(d)}')
    print(f'all frames: read {sum(d.get("parsed") is not None for d in lines)} of {len(lines)}; finish_reason '
          f'{pd.Series([d.get("finish_reason") for d in lines]).value_counts().to_dict()}')
    ok = all(checks.values())
    print(f'GATE PASSED: {ok}')
    return 0 if ok else 1


def defaults(path):
    base = dict(min_p=0.0, repetition_penalty=1.0, frequency_penalty=0.0, max_tokens=24576)
    want = [dict(base, temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5),
            dict(base, temperature=0.0, presence_penalty=1.5), dict(base, temperature=0.0, presence_penalty=0.0)]
    with open(path, encoding='utf-8') as f:
        finals = [ast.literal_eval(m.group(1)) for m in re.finditer(r'final sampling: (\{.*\})$', f.read(), re.M)]
    ok = len(finals) >= 3
    for i, w in enumerate(want):
        got = {k: finals[i].get(k) for k in w} if i < len(finals) else {}
        print(f'  [{"ok" if got == w else "NOT MET"}] request {i + 1}: {got}')
        ok &= got == w
    print(f'DEFAULTS AS SET: {ok}')
    return 0 if ok else 1


if __name__ == '__main__':
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == 'defaults':
        sys.exit(defaults(rest[0]))
    if cmd == 'smoke':
        sys.exit(smoke(rest[0], rest[1], rest[2], json.loads(rest[3])))
    if cmd == 'greedy':
        sys.exit(greedy(rest))
    if cmd == 'gate':
        sys.exit(gate(rest[0], rest[1], rest[2], json.loads(rest[3]), rest[4]))
    sys.exit(f'unknown command {cmd}')
