"""Checks of the Gemma 4 31B-it small test (user, 2026-11-12), run inside the GPU check job; exit code 1 when one is not
met. usage: python -I check_smoke.py <smoke jsonl> <official_prompts.json> <vLLM server log> <expected lines>
1. lines: the expected number, no key (segment, frame, condition, repeat) twice;
2. settings on every line: model gemma-4-31b-it, temperature 1.0, top_p 0.95, top_k 64, max_tokens 24,576; 5 top
   log-probabilities on every generated token;
3. thinking split off: `reasoning` not empty on every line; the answer text has none of <|channel>, <channel|>,
   <|think|>, <turn|>, <|turn>; answers readable (parsed) except on replies cut at max_tokens;
4. prompt tokens of each condition the same on every line and equal to the official prompt of that condition
   (render_check.py, from the official processor: 3,020 / 13,895 / 13,891 / 818 / 2,956);
5. the server log: speculative_config=None; the non-default args carry default_chat_template_kwargs
   {'enable_thinking': True}, mm_processor_kwargs {'max_soft_tokens': 560} and reasoning_parser gemma4; the chat
   template content format detected as 'openai'.
Reported, not checks: the default-sampling line, the attention backend lines, ERROR and WARNING lines by kind,
finish reasons, reply lengths, scores, hazards, latency, and the replies cut at max_tokens: how many, and how many of
them have no readable answer (only those get a forced answer).
"""
import collections
import json
import re
import sys

smoke, prompts_json, log_path, n_expected = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
official = {c: len(v['prompt_ids']) for c, v in json.load(open(prompts_json)).items()}
bad = []
rows = [json.loads(line) for line in open(smoke, encoding='utf-8')]
keys = collections.Counter((d['segment'], d['frame_idx'], d['condition'], d['repeat']) for d in rows)
print(f'1. {len(rows)} lines (expected {n_expected}); keys twice: {sum(v > 1 for v in keys.values())}')
if len(rows) != n_expected or any(v > 1 for v in keys.values()):
    bad.append('1 lines')

settings = collections.Counter((d['model'], d['temperature'], d['top_p'], d['top_k'], d['max_tokens']) for d in rows)
tops = collections.Counter(len(t['top']) for d in rows for t in (d.get('logprobs') or []))
no_lp = sum(not d.get('logprobs') for d in rows)
print(f'2. settings (model, temperature, top_p, top_k, max_tokens): {dict(settings)}; top log-probabilities per '
      f'token: {dict(tops)}; lines without log-probabilities: {no_lp}')
if set(settings) != {('gemma-4-31b-it', 1.0, 0.95, 64, 24576)} or set(tops) != {5} or no_lp:
    bad.append('2 settings')

MARKS = ('<|channel>', '<channel|>', '<|think|>', '<turn|>', '<|turn>')
no_reason = sum(not (d.get('reasoning') or '').strip() for d in rows)
marked = [(d['condition'], m) for d in rows for m in MARKS if m in (d.get('text') or '')]
cut = [d for d in rows if d['finish_reason'] == 'length']
unread = [d for d in rows if d.get('parsed') is None and d['finish_reason'] != 'length']
print(f'3. lines without thinking: {no_reason}; answer texts with a marker: {marked}; unreadable answers that were '
      f'not cut: {len(unread)}')
if no_reason or marked or unread:
    bad.append('3 thinking / answers')

by_cond = collections.defaultdict(set)
for d in rows:
    by_cond[d['condition']].add(d['usage']['prompt_tokens'])
print('4. prompt tokens per condition (server; official): ' + ', '.join(
    f'{c} {sorted(v)} ({official.get(c)})' for c, v in sorted(by_cond.items())))
if any(v != {official.get(c)} for c, v in by_cond.items()):
    bad.append('4 prompt tokens')

log = open(log_path, errors='replace').read()
spec = sorted(set(re.findall(r'speculative_config=([^,]*)', log)))
args = re.findall(r'non-default args: (\{.*\})', log)
a0 = args[0] if args else ''
need = ["'default_chat_template_kwargs': {'enable_thinking': True}", "'mm_processor_kwargs': {'max_soft_tokens': 560}",
        "'reasoning_parser': 'gemma4'"]
fmt = re.findall(r"Detected the chat template content format to be '([^']*)'", log)
print(f'5. speculative_config {spec}; non-default args carry {[n for n in need if n in a0]} (missing '
      f'{[n for n in need if n not in a0]}); content format detected {sorted(set(fmt))}')
if spec != ['None'] or any(n not in a0 for n in need) or sorted(set(fmt)) != ['openai']:
    bad.append('5 server log')
print(f'   non-default args: {a0[:1500]}')
for pat in (r'Default vLLM sampling parameters have been overridden by [^\n]*', r'Gemma4 model has heterogeneous[^\n]*',
            r'Using [A-Za-z0-9_]+ (?:attention )?backend[^\n]*', r'attention backend[^\n]*'):
    hits = sorted(set(re.findall(pat, log)))
    if hits:
        print('   log:', ' | '.join(h[:300] for h in hits[:6]))
kinds = collections.Counter()
for line in log.splitlines():
    m = re.search(r'\b(ERROR|WARNING)\b[^\]]*\]\s*(.*)', line)
    if m:
        kinds[(m.group(1), re.sub(r'\d+', 'N', m.group(2))[:160])] += 1
print(f'   ERROR / WARNING lines by kind ({sum(kinds.values())}):')
for (lvl, msg), n in sorted(kinds.items()):
    print(f'     {lvl} x{n}: {msg}')

print('reported (not checks):')
fin = collections.Counter(d['finish_reason'] for d in rows)
print(f'   finish reasons {dict(fin)}')
for c in sorted(by_cond):
    ds = [d for d in rows if d['condition'] == c]
    comp = [d['usage']['completion_tokens'] for d in ds]
    rt = [((d['usage'].get('completion_tokens_details') or {}).get('reasoning_tokens')) for d in ds]
    risks = [d['parsed']['risk'] if d.get('parsed') else None for d in ds]
    print(f"   {c:10s}: completion tokens {comp}; reasoning tokens {rt}; scores {risks}; hazard types "
          f"{[d['parsed'].get('hazard_type') if d.get('parsed') else None for d in ds]}; latency "
          f"{[d['latency_s'] for d in ds]} s")
print(f'   cut at max_tokens: {len(cut)}; of them without a readable answer (get a forced answer): '
      f'{sum(d.get("parsed") is None for d in cut)}')
print(f'\nchecks not met: {len(bad)} {bad}')
sys.exit(1 if bad else 0)
