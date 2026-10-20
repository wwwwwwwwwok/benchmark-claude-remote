"""Get a final answer for every reply that ran out of tokens while still thinking (finish_reason "length").

Qwen's own way to cap the thinking ("thinking budget", Qwen quickstart docs and Qwen3 Technical Report, arXiv
2505.09388, sec. 4.3): keep the thinking the model wrote, append the early-stopping text below, which closes the
thinking, and let the model write its answer from what it has thought so far. So the answer still belongs to the same
sample; nothing is resampled (decided with the user 2026-10-20).

The request is rebuilt exactly as in run_vlm.py; the original thinking is sent back as the start of the assistant turn
and the server continues it (vLLM: add_generation_prompt False, continue_final_message True). The thinking goes back as
text, so its tokens may split slightly differently from the original ones.

Start the server WITHOUT --reasoning-parser: the reply only contains the answer (the thinking is already closed), and
the qwen3 parser would put a reply without </think> into `reasoning`.

Usage (see kamiak/force_answer.sbatch):
    python -I code/force_answer.py --pilot results/vlm_scores/<MODEL>/pilot_<MODEL>.jsonl \
        --grid_targets results/vlm_inputs/targets.csv --sep_targets results/vlm_inputs_separate/targets.csv \
        --out results/vlm_scores/<MODEL>/pilot_<MODEL>_forced_answers.jsonl \
        --base_url http://localhost:8000/v1 --model <served model name> --temperature ... --top_p ... --top_k ... [--send]

Output: one JSON line per truncated reply, with its key and sampling settings, forced_answer=true, the thinking that was
sent back (`reasoning`, as in the pilot file), the new answer (`text`, `parsed`), finish_reason, usage, latency and
log-probabilities of the answer tokens. The pilot file is not changed."""
import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I leaves the script's folder off sys.path
from run_vlm import CAMERAS, build, parse, to_messages  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')

# verbatim from the thinking-budget example in the Qwen quickstart docs (qwen.readthedocs.io, checked 2026-10-20)
EARLY_STOPPING_TEXT = ('\n\nConsidering the limited time by the user, I have to give the solution based on the thinking '
                       'directly now.\n</think>\n\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', required=True, help='run_vlm.py output whose finish_reason "length" lines get answered')
    ap.add_argument('--grid_targets', required=True)
    ap.add_argument('--sep_targets', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--base_url', default='http://localhost:8000/v1')
    ap.add_argument('--model', default='qwen')
    ap.add_argument('--api_key_env', default='VLM_API_KEY')
    ap.add_argument('--temperature', type=float, required=True, help='the same as in the pilot')
    ap.add_argument('--top_p', type=float, default=None)
    ap.add_argument('--top_k', type=int, default=None)
    ap.add_argument('--max_tokens', type=int, default=1024, help='for the answer only (pilot answers were ~80 tokens)')
    ap.add_argument('--top_logprobs', type=int, default=5)
    ap.add_argument('--timeout', type=float, default=3600)
    ap.add_argument('--send', action='store_true', help='actually send the requests (default: check only)')
    a = ap.parse_args()

    cut = []
    with open(a.pilot, encoding='utf-8') as f:
        for line in f:
            if '"length"' not in line:                # cheap filter: the Qwen3.6 file is 3.5 GB of log-probabilities
                continue
            d = json.loads(line)
            if d.get('finish_reason') == 'length':
                d.pop('logprobs', None)
                cut.append(d)
    print(f'{len(cut)} replies in {a.pilot} stopped at --max_tokens')

    tg, ts = pd.read_csv(a.grid_targets), pd.read_csv(a.sep_targets)
    root_g, root_s = os.path.dirname(os.path.abspath(a.grid_targets)), os.path.dirname(os.path.abspath(a.sep_targets))
    t = tg.merge(ts[['segment', 'frame_idx', 'images_by_time']], on=['segment', 'frame_idx'], validate='one_to_one')
    t = t.set_index(['segment', 'frame_idx'])

    done = set()
    if os.path.exists(a.out):
        with open(a.out, encoding='utf-8') as f:
            done = {(d['segment'], d['frame_idx'], d['condition'], d['repeat']) for d in map(json.loads, f)}

    if a.send:
        from openai import OpenAI
        client = OpenAI(base_url=a.base_url, api_key=os.environ.get(a.api_key_env, 'EMPTY'),
                        timeout=a.timeout, max_retries=0)
    n_ok = n_err = 0
    for old in cut:
        key = (old['segment'], old['frame_idx'], old['condition'], old['repeat'])
        if key in done:
            print('already done:', key)
            continue
        assert old['temperature'] == a.temperature and old['top_p'] == a.top_p and old['top_k'] == a.top_k, \
            ('sampling differs from the pilot', old['temperature'], old['top_p'], old['top_k'])
        r = t.loc[(old['segment'], old['frame_idx'])]
        grid_imgs = [os.path.join(root_g, p) for p in r['images'].split(';')]
        by_time = [os.path.join(root_s, p) for p in r['images_by_time'].split(';')]
        sep_imgs = {cam: [by_time[i * 5 + c] for i in range(5)] for c, cam in enumerate(CAMERAS)}
        thinking = (old['reasoning'] or '').strip()
        assert thinking and '<think>' not in thinking and '</think>' not in thinking, key
        # the chat template turns "<think>\n...</think>" in an assistant message back into the thinking block,
        # so the model sees: <think>\n{thinking}{EARLY_STOPPING_TEXT} and writes its answer next
        prefill = '<think>\n' + thinking + EARLY_STOPPING_TEXT
        messages = to_messages(build(old['condition'], grid_imgs, sep_imgs)) + [
            {'role': 'assistant', 'content': prefill}]
        print(f'{key[0][:6]} frame {key[1]} {key[2]} repeat {key[3]}: thinking {len(thinking):,} characters, '
              f'{old["usage"]["completion_tokens"]:,} tokens; prompt was {old["usage"]["prompt_tokens"]:,} tokens')
        if not a.send:
            continue

        rec = {k: old[k] for k in ('segment', 'frame_idx', 'condition', 'repeat', 'model', 'temperature', 'top_p',
                                   'top_k')}
        rec.update(forced_answer=True, early_stopping_text=EARLY_STOPPING_TEXT, max_tokens=a.max_tokens,
                   original_max_tokens=old['max_tokens'], original_usage=old['usage'], reasoning=old['reasoning'])
        t0 = time.time()
        try:
            extra = {'add_generation_prompt': False, 'continue_final_message': True}
            if a.top_k is not None:
                extra['top_k'] = a.top_k
            kw = dict(model=a.model, messages=messages, temperature=a.temperature, max_tokens=a.max_tokens,
                      extra_body=extra)
            if a.top_p is not None:
                kw['top_p'] = a.top_p
            if a.top_logprobs:
                kw.update(logprobs=True, top_logprobs=a.top_logprobs)
            resp = client.chat.completions.create(**kw)
            ch = resp.choices[0]
            rec.update(text=ch.message.content, parsed=parse(ch.message.content), finish_reason=ch.finish_reason,
                       usage=resp.usage.model_dump() if resp.usage else None)
            if ch.logprobs and ch.logprobs.content:
                rec['logprobs'] = [dict(token=c.token, logprob=c.logprob,
                                        top=[(x.token, x.logprob) for x in (c.top_logprobs or [])])
                                   for c in ch.logprobs.content]
        except Exception as e:                       # not written; resent on the next run
            rec['error'] = f'{type(e).__name__}: {e}'
        rec['latency_s'] = round(time.time() - t0, 2)
        if 'error' in rec:
            n_err += 1
            print('  error:', rec['error'][:300])
            continue
        n_ok += 1
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        u = rec['usage'] or {}
        print(f"  prompt now {u.get('prompt_tokens')} tokens, answer {u.get('completion_tokens')} tokens, "
              f"{rec['finish_reason']}, {rec['latency_s']} s: {rec['text']!r}")
    if a.send:
        print(f'done: {n_ok} answered, {n_err} errors (rerun to retry them)')
    else:
        print('nothing sent (add --send to send)')


if __name__ == '__main__':
    main()
