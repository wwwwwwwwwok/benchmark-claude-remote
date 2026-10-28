"""Ask a vision-language model for the risk score of every target frame, under the pilot conditions.

Works with any OpenAI-compatible chat endpoint: a vLLM server started on Kamiak for an open-weight Qwen model
(--base_url http://localhost:8000/v1), or a hosted API such as Alibaba Cloud Model Studio. The API key, if needed, is
read from the environment variable named by --api_key_env and is never written anywhere.

Conditions (the pilot comparison groups, decided 2026-10-17; prompt wording as in vlm_prompt_template.txt):
    grid        5 images: the 2 x 3 camera grid at t-2.0, -1.5, -1.0, -0.5, 0 s
    sep_time    25 images: every camera captioned, moment by moment
    sep_camera  25 images: the same images, camera by camera
    ctrl_grid   1 image: the grid at t only
    ctrl_sep    5 images: the five cameras at t only

Usage:
    python -I run_vlm.py --grid_targets results/vlm_inputs/targets.csv \
        --sep_targets results/vlm_inputs_separate/targets.csv --out results/vlm_scores/<run>.jsonl \
        --conditions grid,sep_time,sep_camera,ctrl_grid,ctrl_sep --repeats 3 \
        --base_url http://localhost:8000/v1 --model <served model name> [--show 2] [--send]

Without --send nothing is sent: the script prints --show assembled requests and the size of the run (requests,
images, estimated image tokens). Add --send only when the endpoint and the size are right.

--task quantities (user, 2026-10-28; default --task risk): the same system prompt, conditions, pictures and captions,
but the closing question asks for one physical quantity instead of the risk score, one request per quantity
(QUANTITIES: distance to the lead, ego speed, lead speed, ego acceleration, lead acceleration; --quantities picks
some). Requests go frame -> condition -> quantity -> repeat; every output line has a `quantity` column, and `parsed`
is the first JSON object whose key for that quantity holds a number (parse_quantity).

Output: one JSON line per (segment, frame_idx, condition, repeat) with the raw reply, the parsed JSON, the sampling
settings sent, finish_reason, the model's thinking text (`reasoning`, when the server's reasoning parser splits it off),
token usage, latency and (when the server returns them) the log-probabilities of all generated tokens, thinking
included. Requests already in the output file are skipped, so an interrupted run (e.g. a preempted Slurm job) can
simply be restarted."""
import argparse
import base64
import concurrent.futures as cf
import json
import math
import os
import sys
import threading
import time

import pandas as pd
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

SYSTEM = 'You are an experienced driver. You are driving the car whose cameras took the following pictures.'
CAMERAS = ['FRONT_LEFT', 'FRONT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
CAMERA_NAME = {'FRONT_LEFT': 'Front-left camera', 'FRONT': 'Front camera', 'FRONT_RIGHT': 'Front-right camera',
               'SIDE_LEFT': 'Left-side camera', 'SIDE_RIGHT': 'Right-side camera'}
AGO = ['2.0 s ago', '1.5 s ago', '1.0 s ago', '0.5 s ago', 'now']
GRID_TEXT = ('the top row shows the front-left, front and front-right cameras; the bottom row shows the left-side '
             'and right-side cameras. Each panel is labelled with its camera.')
FORMAT = ('Give a risk score from 0 to 100, where 0 means no risk at all and 100 means a collision is about to '
          'happen.\n\n'
          'Reply with JSON only, in this format:\n'
          '{\n'
          '  "risk": <integer from 0 to 100>,\n'
          '  "main_hazard_camera": "front-left" | "front" | "front-right" | "left-side" | "right-side" | "none",\n'
          '  "main_hazard": "<the road user or situation that matters most, in a few words>",\n'
          '  "hazard_type": "rear-end" | "lane change or side" | "crossing or intersection" | "pedestrian or cyclist"'
          ' | "oncoming" | "other" | "none",\n'
          '  "reason": "<one sentence>"\n'
          '}')
CONDITIONS = ['grid', 'sep_time', 'sep_camera', 'ctrl_grid', 'ctrl_sep']
# --task quantities (user, 2026-10-28): question and JSON key per quantity; {now} is filled per condition as in the risk
# question ("right now, in picture 5" etc.)
NOW = {'grid': ', in picture 5', 'sep_time': ', at moment 5', 'sep_camera': '', 'ctrl_grid': '', 'ctrl_sep': ''}
GUESS = 'Always give a number; if you are not sure, give your best estimate.'
UNIT_SPEED = 'in meters per second (1 m/s = 3.6 km/h, about 2.24 mph)'
QUANTITIES = {
    'distance': ('Right now{now}, how far is it from your front camera to the back of the vehicle (or cyclist) directly '
                 'ahead of you in your lane? Measure along the lane, in meters. ' + GUESS, 'distance_m'),
    'ego_speed': (f'How fast is your car going right now{{now}}? Give the speed {UNIT_SPEED}. ' + GUESS,
                  'ego_speed_mps'),
    'lead_speed': ('How fast is the vehicle (or cyclist) directly ahead of you in your lane going right now{now}? Give its '
                   f'speed over the ground, not relative to your car, {UNIT_SPEED}. ' + GUESS, 'lead_speed_mps'),
    'ego_accel': ('Right now{now}, is your car speeding up or slowing down, and how strongly? Give its acceleration '
                  'along the direction of travel in meters per second squared: positive when speeding up, negative '
                  'when slowing down, 0 at a steady speed. ' + GUESS, 'ego_accel_mps2'),
    'lead_accel': ('Right now{now}, is the vehicle (or cyclist) directly ahead of you in your lane speeding up or slowing '
                   'down, and how strongly? Give its acceleration along its direction of travel in meters per second '
                   'squared: positive when speeding up, negative when slowing down, 0 at a steady speed. ' + GUESS,
                   'lead_accel_mps2'),
}


def quantity_question(q):
    """The closing text for quantity q, with {now} still to fill."""
    text, key = QUANTITIES[q]
    return text + '\nReply with JSON only, in this format:\n{"' + key + '": <number>}'


def build(cond, grid_imgs, sep_imgs, question=None):
    """User content as a list of ('text', str) and ('image', path) parts, in sending order. `question`: another closing
    text instead of the risk question (task quantities; '{now}' in it is filled for the condition)."""
    p = []
    if cond == 'grid':
        p.append(('text', 'Below are 5 pictures from your car, taken 0.5 seconds apart and shown in time order. '
                          'Picture 5 is now.\nEach picture combines your five cameras: ' + GRID_TEXT))
        for i, img in enumerate(grid_imgs):
            p += [('text', f'Picture {i + 1} ({AGO[i]}):'), ('image', img)]
        p.append(('text', 'How risky is the driving situation right now, in picture 5?\n' + FORMAT))
    elif cond == 'sep_time':
        p.append(('text', "Below are pictures from your car's five cameras at 5 moments, taken 0.5 seconds apart "
                          'and shown in time order. Moment 5 is now.'))
        for i in range(5):
            p.append(('text', f'Moment {i + 1} ({AGO[i]})'))
            for c, cam in enumerate(CAMERAS):
                p += [('text', f'{CAMERA_NAME[cam]}:'), ('image', sep_imgs[cam][i])]
        p.append(('text', 'How risky is the driving situation right now, at moment 5?\n' + FORMAT))
    elif cond == 'sep_camera':
        p.append(('text', "Below are pictures from your car's five cameras. For each camera there are 5 pictures "
                          'taken 0.5 seconds apart, in time order: the first was taken 2.0 seconds ago and the last '
                          'one is now.'))
        for cam in CAMERAS:
            p.append(('text', CAMERA_NAME[cam]))
            for i in range(5):
                p += [('text', f'{AGO[i][0].upper() + AGO[i][1:]}:'), ('image', sep_imgs[cam][i])]
        p.append(('text', 'How risky is the driving situation right now?\n' + FORMAT))
    elif cond == 'ctrl_grid':
        p.append(('text', 'Below is a picture from your car, taken now. It combines your five cameras: ' + GRID_TEXT))
        p += [('text', 'Picture (now):'), ('image', grid_imgs[-1])]
        p.append(('text', 'How risky is the driving situation right now?\n' + FORMAT))
    elif cond == 'ctrl_sep':
        p.append(('text', "Below are pictures from your car's five cameras, taken now."))
        for cam in CAMERAS:
            p += [('text', f'{CAMERA_NAME[cam]}:'), ('image', sep_imgs[cam][-1])]
        p.append(('text', 'How risky is the driving situation right now?\n' + FORMAT))
    else:
        raise ValueError(cond)
    if question is not None:
        p[-1] = ('text', question.replace('{now}', NOW[cond]))
    return p


def to_messages(parts, system=SYSTEM):
    content = []
    for kind, val in parts:
        if kind == 'text':
            content.append({'type': 'text', 'text': val})
        else:
            with open(val, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode()
            content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}})
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]


def qwen_image_tokens(path):
    """Qwen (Model Studio docs): about h x w / (32 x 32) + 2 tokens per image, sides rounded to multiples of 32."""
    w, h = Image.open(path).size
    return round(h / 32) * round(w / 32) + 2


def parse(text):
    """The first JSON object in the reply that has a valid risk. InternVL3.5 sometimes writes its answer again after a
    stray </think>; the first copy is the answer (2026-10-20, user's choice). On every earlier reply this gives the same
    result as the old "first { to last }" rule."""
    text = text or ''
    i = 0
    while (i := text.find('{', i)) >= 0:
        try:
            d, _ = json.JSONDecoder().raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(d, dict):
            r = d.get('risk')
            if isinstance(r, str) and r.strip().isdigit():
                d['risk'] = int(r)
            if isinstance(d.get('risk'), (int, float)) and 0 <= d['risk'] <= 100:
                return d
        i += 1
    return None


def parse_quantity(text, key):
    """Task quantities: the first JSON object in the reply whose `key` holds a number; a number written as a string is
    converted. Negative and implausible values are kept as they are; NaN and infinity count as no number (user,
    2026-10-28)."""
    text = text or ''
    i = 0
    while (i := text.find('{', i)) >= 0:
        try:
            d, _ = json.JSONDecoder().raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(d, dict) and key in d:
            v = d[key]
            if isinstance(v, str):
                try:
                    v = float(v.strip())
                except ValueError:
                    v = None
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                d[key] = v
                return d
        i += 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid_targets', required=True)
    ap.add_argument('--sep_targets', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--conditions', default=','.join(CONDITIONS))
    ap.add_argument('--task', choices=['risk', 'quantities'], default='risk')
    ap.add_argument('--quantities', default=','.join(QUANTITIES), help='task quantities: which ones, in this order')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--base_url', default='http://localhost:8000/v1')
    ap.add_argument('--model', default='qwen')
    ap.add_argument('--api_key_env', default='VLM_API_KEY', help='environment variable holding the key (if any)')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top_p', type=float, default=None, help='sent only when given')
    ap.add_argument('--top_k', type=int, default=None, help='sent only when given (vLLM extension, via extra_body)')
    ap.add_argument('--max_tokens', type=int, default=300)
    ap.add_argument('--top_logprobs', type=int, default=5, help='0 = do not ask for log-probabilities')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--timeout', type=float, default=3600,
                    help='seconds to wait for one reply; no automatic retry, a failed request is resent on the next run')
    ap.add_argument('--system_prefix_file', default=None,
                    help='text put before the system prompt, separated by a blank line (e.g. the prompt that turns '
                         "on InternVL3.5's thinking); its file name is recorded in every output line")
    ap.add_argument('--limit', type=int, default=0, help='only the first N targets (0 = all)')
    ap.add_argument('--show', type=int, default=1, help='print this many assembled requests')
    ap.add_argument('--send', action='store_true', help='actually send the requests (default: check only)')
    a = ap.parse_args()

    tg, ts = pd.read_csv(a.grid_targets), pd.read_csv(a.sep_targets)
    root_g, root_s = os.path.dirname(os.path.abspath(a.grid_targets)), os.path.dirname(os.path.abspath(a.sep_targets))
    t = tg.merge(ts[['segment', 'frame_idx', 'images_by_time']], on=['segment', 'frame_idx'], validate='one_to_one')
    if a.limit:
        t = t.head(a.limit)
    conds = a.conditions.split(',')
    assert all(c in CONDITIONS for c in conds), conds
    quants = [None] if a.task == 'risk' else a.quantities.split(',')
    assert a.task == 'risk' or all(q in QUANTITIES for q in quants), quants
    system = SYSTEM
    if a.system_prefix_file:
        with open(a.system_prefix_file, encoding='utf-8') as f:
            system = f.read().strip() + '\n\n' + SYSTEM

    jobs = []                                         # (segment, frame, condition, quantity or None, repeat, parts)
    for r in t.itertuples():
        grid_imgs = [os.path.join(root_g, p) for p in r.images.split(';')]
        by_time = [os.path.join(root_s, p) for p in r.images_by_time.split(';')]
        sep_imgs = {cam: [by_time[i * 5 + c] for i in range(5)] for c, cam in enumerate(CAMERAS)}
        for cond in conds:
            for q in quants:
                for rep in range(a.repeats):
                    jobs.append((r.segment, int(r.frame_idx), cond, q, rep,
                                 build(cond, grid_imgs, sep_imgs, None if q is None else quantity_question(q))))

    def key(seg, k, cond, q, rep):                    # a risk line has no quantity
        return (seg, k, cond, rep) if q is None else (seg, k, cond, q, rep)

    if not a.send:
        for seg, k, cond, q, rep, parts in jobs[:a.show]:
            print(f'===== {seg[:6]} frame {k} {cond}' + (f' {q}' if q else '') + f' repeat {rep}')
            print('[SYSTEM] ' + system)
            for kind, val in parts:
                print(val if kind == 'text' else f'<image {os.path.relpath(val)}>')
        per = {}
        for seg, k, cond, q, rep, parts in jobs:
            if cond not in per:
                imgs = [v for kind, v in parts if kind == 'image']
                per[cond] = (len(imgs), sum(qwen_image_tokens(p) for p in imgs))
        print(f'\n{len(jobs)} requests ({len(t)} targets x {len(conds)} conditions'
              + ('' if a.task == 'risk' else f' x {len(quants)} quantities') + f' x {a.repeats} repeats)')
        for cond, (n_img, tok) in per.items():
            print(f'  {cond:10s}: {n_img:2d} images, about {tok:,} image tokens per request (Qwen estimate)')
        total = sum(per[c][1] for c in conds) * len(t) * len(quants) * a.repeats
        print(f'  total image tokens about {total:,}')
        print('nothing sent (add --send to send)')
        return

    done = set()
    if os.path.exists(a.out):
        with open(a.out, encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                done.add(key(d['segment'], d['frame_idx'], d['condition'], d.get('quantity'), d['repeat']))
    todo = [j for j in jobs if key(*j[:5]) not in done]
    print(f'{len(jobs)} requests, {len(done)} already done, {len(todo)} to send')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    from openai import OpenAI
    # max_retries=0: the client's own retries would silently replace a slow answer with a new sample
    client = OpenAI(base_url=a.base_url, api_key=os.environ.get(a.api_key_env, 'EMPTY'),
                    timeout=a.timeout, max_retries=0)
    lock = threading.Lock()

    def ask(job):
        seg, k, cond, q, rep, parts = job
        rec = dict(segment=seg, frame_idx=k, condition=cond)
        if q is not None:
            rec['quantity'] = q
        rec.update(repeat=rep, model=a.model, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                   max_tokens=a.max_tokens)
        if a.system_prefix_file:
            rec['system_prefix_file'] = os.path.basename(a.system_prefix_file)
        t0 = time.time()
        try:
            kw = dict(model=a.model, messages=to_messages(parts, system), temperature=a.temperature,
                      max_tokens=a.max_tokens)
            if a.top_p is not None:
                kw['top_p'] = a.top_p
            if a.top_k is not None:
                kw['extra_body'] = {'top_k': a.top_k}
            if a.top_logprobs:
                kw.update(logprobs=True, top_logprobs=a.top_logprobs)
            resp = client.chat.completions.create(**kw)
            ch = resp.choices[0]
            # vLLM's --reasoning-parser puts the thinking text in `reasoning` (formerly `reasoning_content`)
            rec.update(text=ch.message.content, parsed=(parse(ch.message.content) if q is None else
                                                        parse_quantity(ch.message.content, QUANTITIES[q][1])),
                       reasoning=getattr(ch.message, 'reasoning', None), finish_reason=ch.finish_reason,
                       usage=resp.usage.model_dump() if resp.usage else None)
            if ch.logprobs and ch.logprobs.content:
                rec['logprobs'] = [dict(token=c.token, logprob=c.logprob,
                                        top=[(x.token, x.logprob) for x in (c.top_logprobs or [])])
                                   for c in ch.logprobs.content]
        except Exception as e:                       # keep going; failed requests are retried on the next run
            rec['error'] = f'{type(e).__name__}: {e}'
        rec['latency_s'] = round(time.time() - t0, 2)
        if 'error' not in rec:
            with lock, open(a.out, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        return rec

    n_ok = n_err = n_bad = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for i, rec in enumerate(ex.map(ask, todo), 1):
            if 'error' in rec:
                n_err += 1
                print(f"error {rec['segment'][:6]} {rec['frame_idx']} {rec['condition']}"
                      + (f" {rec['quantity']}" if 'quantity' in rec else '') + f": {rec['error'][:200]}")
            else:
                n_ok += 1
                n_bad += rec['parsed'] is None
            if i % 50 == 0:
                print(f'{i}/{len(todo)} sent: {n_ok} ok ({n_bad} unparsable), {n_err} errors')
    print(f'done: {n_ok} ok ({n_bad} unparsable), {n_err} errors (rerun to retry them)')


if __name__ == '__main__':
    main()
