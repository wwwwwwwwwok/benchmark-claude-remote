"""Forced-answer rendering on the real server (Gemma 4 31B-it; user, 2026-11-12: before any forced answer, check that
the original thinking, the sentence and <channel|> are all in the prompt, not removed by the template). Against the
server started with kamiak/force_answer.sbatch's gemma-4-31b-it settings. For the thinking of the first two smoke
lines, whole and cut to its first half (as a reply cut at max_tokens is), sends
 (i) the original request (code/run_vlm.py's build and to_messages, as in the run) and
 (ii) the forced request code/force_answer.py --thinking_format gemma4 builds (its own forced_messages; extra_body
      add_generation_prompt False, continue_final_message True, as force_answer.py sends it),
both with max_tokens 1, temperature 0 and return_token_ids, and compares the prompt ids the server used: (ii) must
start with all of (i), the rest must decode to '<|channel>thought\\n' + thinking + the sentence + '\\n<channel|>'
(whether it equals the tokenizer's ids of that text is reported), the last token must be <channel|>, and (i) must
have the prompt length the run recorded for that line. Exit code 1 when not met.
usage: python -I force_render_check.py <smoke jsonl> <grid targets> <sep targets> <model folder> <base url>
           <served model name> <out json>
"""
import json
import os
import sys

import pandas as pd
from openai import OpenAI
from transformers import AutoTokenizer

sys.path.insert(0, os.path.expanduser('~/benchmark-claude-remote/code'))
import force_answer  # noqa: E402
from run_vlm import CAMERAS, build, to_messages  # noqa: E402

smoke, grid_t, sep_t, model_dir, base_url, served, out = sys.argv[1:8]
tok = AutoTokenizer.from_pretrained(model_dir)
client = OpenAI(base_url=base_url, api_key='EMPTY', timeout=3600, max_retries=0)
tg, ts = pd.read_csv(grid_t), pd.read_csv(sep_t)
root_g, root_s = os.path.dirname(os.path.abspath(grid_t)), os.path.dirname(os.path.abspath(sep_t))
t = tg.merge(ts[['segment', 'frame_idx', 'images_by_time']], on=['segment', 'frame_idx'],
             validate='one_to_one').set_index(['segment', 'frame_idx'])


def prompt_ids(messages, forced):
    extra = {'return_token_ids': True}
    if forced:
        extra.update(add_generation_prompt=False, continue_final_message=True)
    resp = client.chat.completions.create(model=served, messages=messages, temperature=0, max_tokens=1,
                                          extra_body=extra)
    return (resp.model_extra or {}).get('prompt_token_ids')


rows = [json.loads(line) for line in open(smoke, encoding='utf-8')]
rows = [d for d in rows if (d.get('reasoning') or '').strip()][:2]
bad, results = [], []
for d in rows:
    r = t.loc[(d['segment'], d['frame_idx'])]
    grid_imgs = [os.path.join(root_g, p) for p in r['images'].split(';')]
    by_time = [os.path.join(root_s, p) for p in r['images_by_time'].split(';')]
    sep_imgs = {cam: [by_time[i * 5 + c] for i in range(5)] for c, cam in enumerate(CAMERAS)}
    parts = build(d['condition'], grid_imgs, sep_imgs)
    normal = prompt_ids(to_messages(parts), forced=False)
    full = d['reasoning'].strip()
    for label, th in (('whole', full), ('first half', full[:len(full) // 2].strip())):
        msgs, early = force_answer.forced_messages(parts, th, 'gemma4', 'render check')
        ids = prompt_ids(msgs, forced=True)
        want = '<|channel>thought\n' + th + early
        tail = ids[len(normal):]
        dec = tok.decode(tail, skip_special_tokens=False)
        ok_prefix = ids[:len(normal)] == normal
        ok_text = dec == want
        ok_ids = tail == tok.encode(want, add_special_tokens=False)
        ok_last = tok.convert_ids_to_tokens(ids[-1]) == '<channel|>'
        ok_run = len(normal) == d['usage']['prompt_tokens']        # this server renders the run's prompt
        ok = ok_prefix and ok_text and ok_last and ok_run
        if not ok:
            bad.append(f'{d["condition"]} {label}')
        results.append(dict(segment=d['segment'], frame_idx=d['frame_idx'], condition=d['condition'], which=label,
                            normal_prompt=len(normal), forced_prompt=len(ids), thinking_chars=len(th),
                            starts_with_normal=ok_prefix, tail_decodes_to_expected=ok_text,
                            tail_ids_equal_tokenizer=ok_ids, last_token=tok.convert_ids_to_tokens(ids[-1]),
                            tail_start=dec[:80], tail_end=dec[-200:]))
        print(f"{d['segment'][:6]} frame {d['frame_idx']} {d['condition']} ({label}, thinking {len(th):,} characters): "
              f"normal prompt {len(normal):,} tokens (the run's {d['usage']['prompt_tokens']:,}: {ok_run}), forced "
              f"{len(ids):,}; starts with the normal prompt "
              f"{ok_prefix}; the rest is <|channel>thought\\n + thinking + sentence + \\n<channel|>: {ok_text} (ids "
              f"equal to the tokenizer's: {ok_ids}); last token {tok.convert_ids_to_tokens(ids[-1])!r} -> "
              f"{'ok' if ok else 'NOT MET'}")
        print(f'    rest starts {dec[:60]!r} ... ends {dec[-170:]!r}')
with open(out, 'w') as f:
    json.dump(results, f, indent=1)
print(f'checks not met: {len(bad)} {bad}')
sys.exit(1 if bad or len(results) < 4 else 0)
