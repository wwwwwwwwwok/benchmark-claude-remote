"""Checks of an InternVL3.5-8B run (user, 2026-11-13; symmetric with InternVL3.5-38B): the small test before the pilot
and the pilot itself; exit code 1 when one is not met.
usage: python -I check_smoke_internvl8b.py <off|on> <jsonl> <38B main jsonl.gz of the same mode> <vLLM server log>
       <expected lines>
1. lines: the expected number, no key (segment, frame, condition, repeat) twice, no line with an error;
2. settings on every line (temperature, top_p, top_k, max_tokens) = the 38B main run's of the same mode (its first
   2,000 lines), model internvl3.5-8b (off) / internvl3.5-8b-thinking (on); 5 top log-probabilities on every token;
3. off: `reasoning` empty on every line, no <think> / </think> in the answer (no leftover thinking); on: the answer
   has no <think> / </think> (thinking and answer separated), lines with a non-empty reasoning counted; answers
   readable (parsed) except on replies cut at max_tokens;
4. prompt tokens of each condition the same on every line and = the official count (render_tokens.py, srun 31777778,
   = the 38B main run's): off 14,423 / 35,022 / 35,034 / 3,082 / 7,167 (grid, sep_time, sep_camera, ctrl_grid,
   ctrl_sep), on 2 more each;
5. the server log: speculative_config=None; non-default args carry trust_remote_code True, max_model_len 40960,
   tensor_parallel_size 2, and (on) reasoning_parser deepseek_r1 and the thinking chat template.
Reported, not checks: the log's model memory, KV cache and concurrency lines, ERROR / WARNING kinds, finish reasons,
completion tokens, scores, and every reply cut at max_tokens or not readable (the last 200 characters).
"""
import collections
import gzip
import json
import re
import statistics
import sys

mode, path, g38, log_path, n_expected = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
assert mode in ('off', 'on')
MODEL = 'internvl3.5-8b' + ('-thinking' if mode == 'on' else '')
OFF = {'grid': 14423, 'sep_time': 35022, 'sep_camera': 35034, 'ctrl_grid': 3082, 'ctrl_sep': 7167}
official = {c: v + (2 if mode == 'on' else 0) for c, v in OFF.items()}
bad = []
ref = set()
with gzip.open(g38, 'rt') as f:
    for i, line in enumerate(f):
        if i >= 2000:
            break
        d = json.loads(line)
        ref.add((d['temperature'], d['top_p'], d['top_k'], d['max_tokens']))
rows = [json.loads(line) for line in open(path, encoding='utf-8')]
keys = collections.Counter((d['segment'], d['frame_idx'], d['condition'], d['repeat']) for d in rows)
errors = sum('error' in d for d in rows)
print(f'1. {len(rows)} lines (expected {n_expected}); keys twice: {sum(v > 1 for v in keys.values())}; lines with an '
      f'error: {errors}')
if len(rows) != n_expected or any(v > 1 for v in keys.values()) or errors:
    bad.append('1 lines')
rows = [d for d in rows if 'error' not in d]

settings = collections.Counter((d['model'], d['temperature'], d['top_p'], d['top_k'], d['max_tokens']) for d in rows)
want = {(MODEL,) + r for r in ref}
tops = collections.Counter(len(t['top']) for d in rows for t in (d.get('logprobs') or []))
no_lp = sum(not d.get('logprobs') for d in rows)
print(f'2. settings (model, temperature, top_p, top_k, max_tokens): {dict(settings)}; 38B main ({g38}): {sorted(ref, key=str)}; '
      f'top log-probabilities per token: {dict(tops)}; lines without log-probabilities: {no_lp}')
if len(ref) != 1 or set(settings) != want or set(tops) != {5} or no_lp:
    bad.append('2 settings')

with_reason = [d for d in rows if (d.get('reasoning') or '') != '']
tagged = [d for d in rows if '<think>' in (d.get('text') or '') or '</think>' in (d.get('text') or '')]
cut = [d for d in rows if d['finish_reason'] == 'length']
unread = [d for d in rows if d.get('parsed') is None and d['finish_reason'] != 'length']
print(f'3. lines with a non-empty reasoning: {len(with_reason)} of {len(rows)}; answers with <think> or </think>: '
      f'{len(tagged)}; unreadable answers that were not cut: {len(unread)}')
if tagged or unread or (mode == 'off' and with_reason):
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
need = ["'trust_remote_code': True", "'max_model_len': 40960", "'tensor_parallel_size': 2"]
if mode == 'on':
    need += ["'reasoning_parser': 'deepseek_r1'", "internvl_thinking_chat_template.jinja"]
print(f'5. speculative_config {spec}; non-default args carry {[n for n in need if n in a0]} (missing '
      f'{[n for n in need if n not in a0]})')
if spec != ['None'] or any(n not in a0 for n in need):
    bad.append('5 server log')
print(f'   non-default args: {a0[:1500]}')
for line in log.splitlines():
    if re.search(r'Model loading took|KV cache size|Maximum concurrency|Available KV cache memory|memory profiling', line):
        print('   log: ' + line.strip()[:300])
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
      f'{max(comp_all)}; answer characters mean {statistics.mean(len(d.get("text") or "") for d in rows)!r}; reasoning '
      f'characters mean {statistics.mean(len(d.get("reasoning") or "") for d in rows)!r}')
print(f'   cut at max_tokens: {len(cut)}; of them without a readable answer: {sum(d.get("parsed") is None for d in cut)}')
for d in [d for d in rows if d['finish_reason'] == 'length' or d.get('parsed') is None]:
    print(f"     {d['condition']} {d['segment']} {d['frame_idx']} repeat {d['repeat']}: {d['finish_reason']}, "
          f"{d['usage']['completion_tokens']} tokens, parsed {d.get('parsed') is not None}; last 200 characters "
          f"{(d.get('text') or '')[-200:]!r}")
print(f'\nchecks not met: {len(bad)} {bad}')
sys.exit(1 if bad else 0)
