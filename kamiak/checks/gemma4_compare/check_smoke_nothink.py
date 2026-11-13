"""Checks of a Gemma 4 31B-it run without thinking (user, 2026-11-13): the small test before the pilot, and the pilot
itself; exit code 1 when one is not met (check_smoke.py's checks with the thinking ones turned round).
usage: python -I check_smoke_nothink.py <jsonl> <official_prompts_nothink.json> <vLLM server log> <expected lines>
1. lines: the expected number, no key (segment, frame, condition, repeat) twice, no line with an error;
2. settings on every line: model gemma-4-31b-it-nothink, temperature 1.0, top_p 0.95, top_k 64, max_tokens 24,576;
   5 top log-probabilities on every generated token;
3. no thinking: `reasoning` empty (None or '') on every line and the reasoning token count 0 or absent; the answer
   text has none of <|channel>, <channel|>, <|think|>, <turn|>, <|turn> and does not start with 'thought'; answers
   readable (parsed) except on replies cut at max_tokens;
4. prompt tokens of each condition the same on every line and equal to the official prompt without thinking of that
   condition (render_nothink.py: 3,022 / 13,897 / 13,893 / 820 / 2,958);
5. the server log: speculative_config=None; the non-default args carry default_chat_template_kwargs
   {'enable_thinking': False}, mm_processor_kwargs {'max_soft_tokens': 560} and reasoning_parser gemma4; the chat
   template content format detected as 'openai'.
Reported, not checks: ERROR and WARNING lines by kind, finish reasons, reply lengths (completion tokens: mean, median,
maximum per condition), scores, and the replies cut at max_tokens or not readable: each one listed (condition,
segment, frame, repeat, finish reason, completion tokens, the last 200 characters).
"""
import collections
import json
import re
import statistics
import sys

path, prompts_json, log_path, n_expected = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
MODEL = 'gemma-4-31b-it-nothink'
official = {c: len(v['prompt_ids']) for c, v in json.load(open(prompts_json)).items()}
bad = []
rows = [json.loads(line) for line in open(path, encoding='utf-8')]
keys = collections.Counter((d['segment'], d['frame_idx'], d['condition'], d['repeat']) for d in rows)
errors = sum('error' in d for d in rows)
print(f'1. {len(rows)} lines (expected {n_expected}); keys twice: {sum(v > 1 for v in keys.values())}; lines with an '
      f'error: {errors}')
if len(rows) != n_expected or any(v > 1 for v in keys.values()) or errors:
    bad.append('1 lines')
rows = [d for d in rows if 'error' not in d]

settings = collections.Counter((d['model'], d['temperature'], d['top_p'], d['top_k'], d['max_tokens']) for d in rows)
tops = collections.Counter(len(t['top']) for d in rows for t in (d.get('logprobs') or []))
no_lp = sum(not d.get('logprobs') for d in rows)
print(f'2. settings (model, temperature, top_p, top_k, max_tokens): {dict(settings)}; top log-probabilities per '
      f'token: {dict(tops)}; lines without log-probabilities: {no_lp}')
if set(settings) != {(MODEL, 1.0, 0.95, 64, 24576)} or set(tops) != {5} or no_lp:
    bad.append('2 settings')

MARKS = ('<|channel>', '<channel|>', '<|think|>', '<turn|>', '<|turn>')
with_reason = [d for d in rows if (d.get('reasoning') or '') != '']
rtok = collections.Counter(((d['usage'].get('completion_tokens_details') or {}).get('reasoning_tokens')) for d in rows)
marked = [(d['condition'], m) for d in rows for m in MARKS if m in (d.get('text') or '')]
thought = sum((d.get('text') or '').lstrip().startswith('thought') for d in rows)
cut = [d for d in rows if d['finish_reason'] == 'length']
unread = [d for d in rows if d.get('parsed') is None and d['finish_reason'] != 'length']
print(f'3. lines with a non-empty reasoning: {len(with_reason)}; reasoning token counts {dict(rtok)}; answer texts with '
      f'a marker: {marked}; starting with "thought": {thought}; unreadable answers that were not cut: {len(unread)}')
if with_reason or set(rtok) - {0, None} or marked or thought or unread:
    bad.append('3 no thinking / answers')

by_cond = collections.defaultdict(set)
for d in rows:
    by_cond[d['condition']].add(d['usage']['prompt_tokens'])
print('4. prompt tokens per condition (server; official without thinking): ' + ', '.join(
    f'{c} {sorted(v)} ({official.get(c)})' for c, v in sorted(by_cond.items())))
if any(v != {official.get(c)} for c, v in by_cond.items()):
    bad.append('4 prompt tokens')

log = open(log_path, errors='replace').read()
spec = sorted(set(re.findall(r'speculative_config=([^,]*)', log)))
args = re.findall(r'non-default args: (\{.*\})', log)
a0 = args[0] if args else ''
need = ["'default_chat_template_kwargs': {'enable_thinking': False}",
        "'mm_processor_kwargs': {'max_soft_tokens': 560}", "'reasoning_parser': 'gemma4'"]
fmt = re.findall(r"Detected the chat template content format to be '([^']*)'", log)
print(f'5. speculative_config {spec}; non-default args carry {[n for n in need if n in a0]} (missing '
      f'{[n for n in need if n not in a0]}); content format detected {sorted(set(fmt))}')
if spec != ['None'] or any(n not in a0 for n in need) or sorted(set(fmt)) != ['openai']:
    bad.append('5 server log')
print(f'   non-default args: {a0[:1500]}')
kinds = collections.Counter()
for line in log.splitlines():
    m = re.search(r'\b(ERROR|WARNING)\b[^\]]*\]\s*(.*)', line)
    if m:
        kinds[(m.group(1), re.sub(r'\d+', 'N', m.group(2))[:160])] += 1
print(f'   ERROR / WARNING lines by kind ({sum(kinds.values())}):')
for (lvl, msg), n in sorted(kinds.items()):
    print(f'     {lvl} x{n}: {msg}')

print('reported (not checks):')
print(f'   finish reasons {dict(collections.Counter(d["finish_reason"] for d in rows))}')
for c in ('grid', 'sep_time', 'sep_camera', 'ctrl_grid', 'ctrl_sep'):
    ds = [d for d in rows if d['condition'] == c]
    if not ds:
        continue
    comp = [d['usage']['completion_tokens'] for d in ds]
    risks = [d['parsed']['risk'] for d in ds if d.get('parsed')]
    print(f'   {c:10s}: {len(ds)} replies; completion tokens mean {statistics.mean(comp)!r}, median '
          f'{statistics.median(comp)!r}, max {max(comp)}; scores mean '
          f'{statistics.mean(risks) if risks else None!r}, distinct {len(set(risks))}')
comp_all = [d['usage']['completion_tokens'] for d in rows]
print(f'   all: completion tokens mean {statistics.mean(comp_all)!r}, median {statistics.median(comp_all)!r}, max '
      f'{max(comp_all)}; answer characters mean {statistics.mean(len(d.get("text") or "") for d in rows)!r}')
print(f'   cut at max_tokens: {len(cut)}; of them without a readable answer: {sum(d.get("parsed") is None for d in cut)}')
for d in [d for d in rows if d['finish_reason'] == 'length' or d.get('parsed') is None]:
    print(f"     {d['condition']} {d['segment']} {d['frame_idx']} repeat {d['repeat']}: {d['finish_reason']}, "
          f"{d['usage']['completion_tokens']} tokens, parsed {d.get('parsed') is not None}; last 200 characters "
          f"{(d.get('text') or '')[-200:]!r}")
print(f'\nchecks not met: {len(bad)} {bad}')
sys.exit(1 if bad else 0)
