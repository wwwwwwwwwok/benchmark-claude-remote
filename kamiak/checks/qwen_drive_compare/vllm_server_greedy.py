"""Step 2 of the Qwen-Drive check (user, 2026-10-26): the vLLM server, started as kamiak/qwen_vlm.sbatch starts it for
this model, answers the same 5 requests greedily (temperature 0). The messages are built by code/run_vlm.py's
to_messages from the same parts, as in the runs. Saved per request: the prompt ids vLLM used (return_token_ids), the
generated ids, each step's top-5 (ids via return_tokens_as_token_ids, log-probabilities), reasoning and content,
finish_reason, usage.
usage: python -I vllm_server_greedy.py <requests json> <base url> <served model name> <out folder>
"""
import json
import os
import sys
import time

from openai import OpenAI

sys.path.insert(0, os.path.expanduser('~/benchmark-claude-remote/code'))
from run_vlm import to_messages  # noqa: E402

req_json, base_url, model, out = sys.argv[1:5]
client = OpenAI(base_url=base_url, api_key='EMPTY', timeout=3600, max_retries=0)


def tid(s):
    assert s.startswith('token_id:'), s
    return int(s.split(':', 1)[1])


for r in json.load(open(req_json)):
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model, messages=to_messages([tuple(p) for p in r['parts']], r['system']), temperature=0,
        max_tokens=24576, logprobs=True, top_logprobs=5,
        extra_body={'return_token_ids': True, 'return_tokens_as_token_ids': True})
    ch = resp.choices[0]
    extra_r, extra_c = resp.model_extra or {}, ch.model_extra or {}
    steps = [[[tid(x.token) for x in c.top_logprobs], [x.logprob for x in c.top_logprobs], tid(c.token)]
             for c in ch.logprobs.content]
    res = dict(condition=r['condition'], prompt_ids=extra_r.get('prompt_token_ids'), generated=extra_c.get('token_ids'),
               top5=steps, finish_reason=ch.finish_reason, usage=resp.usage.model_dump() if resp.usage else None,
               reasoning=getattr(ch.message, 'reasoning', None), content=ch.message.content,
               seconds=round(time.time() - t0, 1))
    with open(os.path.join(out, f'vllm_server_{r["condition"]}.json'), 'w') as f:
        json.dump(res, f)
    print(f"{r['condition']}: prompt {len(res['prompt_ids'] or [])} tokens, generated "
          f"{len(res['generated'] or [])} tokens, {ch.finish_reason}, {res['seconds']} s", flush=True)
