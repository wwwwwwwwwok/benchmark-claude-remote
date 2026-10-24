"""Every reply without a readable answer in a run file (user, 2026-10-23, for InternVL3.5-38B's main run): condition,
repeat, finish_reason, generated tokens, the reply text, and where run_vlm.parse gives up. parse tries every '{' from
left to right: the JSON object starting there (json raw_decode) must be a dict with a risk that is a number 0-100
(or a string of digits); the first one that is, is the answer. Here each '{' is tried the same way and the reason it
was not taken is printed, so a reply cut at max_tokens and a finished reply the rule cannot read can be told apart.
usage: python -I unparsable_report.py <run jsonl[.gz]>
"""
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.expanduser('~/benchmark-claude-remote/code'))
from run_vlm import parse  # noqa: E402


def why(text):
    """One line per '{' in the text: why parse does not take the object starting there."""
    out, i = [], 0
    while (i := text.find('{', i)) >= 0:
        try:
            d, end = json.JSONDecoder().raw_decode(text, i)
        except json.JSONDecodeError as e:
            out.append(f"char {i}: not JSON: {e.msg} at char {e.pos} of {len(text)}")
            i += 1
            continue
        if not isinstance(d, dict):
            out.append(f'char {i}: JSON but not an object ({type(d).__name__})')
        elif 'risk' not in d:
            out.append(f'char {i}-{end}: object without "risk" (keys {list(d)})')
        else:
            out.append(f'char {i}-{end}: "risk" is {d["risk"]!r}, not a number 0-100 or a string of digits')
        i += 1
    return out or ["no '{' in the reply"]


n = 0
with (gzip.open if sys.argv[1].endswith('.gz') else open)(sys.argv[1], 'rb') as f:
    for raw in f:
        i = raw.find(b', "logprobs": ')
        if i >= 0:
            j = raw.find(b', "latency_s": ', i)
            raw = raw[:i] + (raw[j:] if j > 0 else b'}')
        d = json.loads(raw)
        if d.get('parsed'):
            continue
        n += 1
        text = d.get('text') or ''
        u = d.get('usage') or {}
        ct, pt = u.get('completion_tokens'), u.get('prompt_tokens')
        # finish_reason "length" is either max_tokens or the context (--max-model-len) filling up
        if d.get('finish_reason') != 'length':
            how = 'finished (not cut)'
        elif ct >= d.get('max_tokens'):
            how = 'cut at max_tokens'
        else:
            how = f'cut by the context length (prompt {pt} + generated {ct} = {pt + ct} tokens)'
        print(f"[{n}] {d['segment']} frame {d['frame_idx']} {d['condition']} repeat {d['repeat']}: finish_reason "
              f"{d.get('finish_reason')}, generated {ct} tokens (max_tokens {d.get('max_tokens')}), prompt {pt}, "
              f"text {len(text):,} characters, thinking {len(d.get('reasoning') or ''):,} characters")
        print(f'  {how}; parse gives {parse(d.get("text"))!r} (rerun with the current rule)')
        for w in why(text):
            print(f'  {w}')
        print(f'  text: {text!r}')
print(f'{n} replies without a readable answer')
