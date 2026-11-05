"""DRAFT, not run (user, 2026-11-05, item 5 "第 2 步，只给文字和数字"; CLAUDE.md, the entry of 2026-11-05): ask
Qwen3.6-27B (thinking on) for the risk score of described situations instead of pictures, so that the cues vary
independently of each other. Nothing is sent without --send; the user decides whether to run it, and only after the
grid run and the jobs queued behind it are done.

The question is run_vlm.py's risk question with as few changes as possible (every change is listed by --show; the
choices still open are options, defaults marked "undecided"): the pictures are replaced by an English description,
    "Below is a description of your car's situation right now."  (instead of "Below are pictures ... taken now.")
    "Your car is going at {X} meters per second (1 m/s = 3.6 km/h, about 2.24 mph). Directly ahead of you in your lane
     there is a vehicle going in the same direction at {Z} meters per second. Measured along the lane, it is {Y}
     meters from your front camera to the back of that vehicle."   (--distance_ref camera, the quantities wording;
     --distance_ref bumper: "... the gap between your front bumper and the back of that vehicle is {Y} meters.")
    two scene sentences or none: "There are many / few other vehicles and pedestrians around you." and
     "It is raining and the road is wet." / "It is clear and the road is dry."
then the risk question and FORMAT of run_vlm.py unchanged ("How risky is the driving situation right now?\n" + FORMAT);
--camera_field drop removes FORMAT's "main_hazard_camera" line. The system prompt: --system S1 "You are an experienced
driver. You are driving the car described below.", S2 "... the car in the situation described below.", S3 "You are an
experienced driver." (run_vlm.py: "... You are driving the car whose cameras took the following pictures.").

Design (full factorial, 625 combinations x --repeats 5 = 3,125 requests): X (ego speed), Z (lead speed), Y (distance) at
the 10, 30, 50, 70, 90 % quantiles of the main result's 1,084 frames (truth_main.csv: v_ego, v_lead, distance; with
--distance_ref bumper the gap's quantiles), written with --decimals (undecided); scene: many / few x rain / dry, and not
said (5); combinations with Z >= X (not closing in) are kept and flagged (closing False).
Sampling and model as the 27B risk run (kamiak/qwen_vlm.sbatch, MODEL=qwen3.6-27b): temperature 1.0, top_p 0.95,
top_k 20, max_tokens 24576, log-probabilities 5; checked without --send against the first line of that run's replies
(--check_run). Forced answers (--forced): as code/force_answer.py (Qwen's early-stopping text after the thinking, the
server continues the assistant turn; answer max_tokens 1024, the same sampling), for the replies stopped at max_tokens
without an answer; written to <out stem>_forced_answers.jsonl.
Output: one JSON line per (combination, repeat) as run_vlm.py's (raw reply, parsed JSON, sampling, finish_reason,
thinking, usage, latency, log-probabilities) plus the design columns; requests already in the file are skipped.
usage: python -I run_text_vlm.py --truth truth_main.csv --frames <ssm_target_frames.csv of main_1084> --out <jsonl>
       [--system S1] [--camera_field keep] [--distance_ref camera] [--decimals 1] [--repeats 5] [--show 3]
       [--check_run <main_qwen3.6-27b.jsonl.gz>] [--design_csv <path>] [--base_url ...] [--model qwen3.6-27b]
       [--workers 32] [--send | --forced]
"""
import argparse
import concurrent.futures as cf
import gzip
import json
import os
import sys
import threading
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_vlm import FORMAT, SYSTEM, parse  # noqa: E402
from force_answer import EARLY_STOPPING_TEXT  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]
UNIT_SPEED_FULL = 'meters per second (1 m/s = 3.6 km/h, about 2.24 mph)'     # run_vlm.UNIT_SPEED without "in "
SYSTEMS = {'S1': 'You are an experienced driver. You are driving the car described below.',
           'S2': 'You are an experienced driver. You are driving the car in the situation described below.',
           'S3': 'You are an experienced driver.'}
INTRO = "Below is a description of your car's situation right now."
QUESTION = 'How risky is the driving situation right now?\n'
TRAFFIC = {'many': 'There are many other vehicles and pedestrians around you.',
           'few': 'There are few other vehicles and pedestrians around you.'}
WEATHER = {'rain': 'It is raining and the road is wet.', 'dry': 'It is clear and the road is dry.'}
SCENES = [('many', 'rain'), ('many', 'dry'), ('few', 'rain'), ('few', 'dry'), (None, None)]
CAMERA_LINE = ('  "main_hazard_camera": "front-left" | "front" | "front-right" | "left-side" | "right-side" | "none",\n')
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, max_tokens=24576)          # kamiak/qwen_vlm.sbatch, qwen3.6-27b
FORCED_MAX_TOKENS = 1024


def fmt(v, decimals):
    return f'{v:.{decimals}f}'


def design(truth, frames, distance_ref, decimals):
    """The 625 combinations: levels from the frames' truth, the texts of the numbers, the scene, closing in or not."""
    t = pd.read_csv(frames)[['segment', 'frame_idx']].merge(pd.read_csv(truth, float_precision='round_trip'),
                                                            on=['segment', 'frame_idx'], how='left', validate='one_to_one')
    if t[['v_ego', 'v_lead', 'distance', 'gap']].isna().any().any():
        sys.exit('truth missing for some frames')
    lv = {'x': np.quantile(t.v_ego, QUANTILES), 'z': np.quantile(t.v_lead, QUANTILES),
          'y': np.quantile(t['distance' if distance_ref == 'camera' else 'gap'], QUANTILES)}
    rows, cid = [], 0
    for i, x in enumerate(lv['x']):
        for j, z in enumerate(lv['z']):
            for k, y in enumerate(lv['y']):
                for traffic, weather in SCENES:
                    rows.append(dict(combo=cid, x_level=i + 1, z_level=j + 1, y_level=k + 1, x=float(x), z=float(z),
                                     y=float(y), x_text=fmt(x, decimals), z_text=fmt(z, decimals),
                                     y_text=fmt(y, decimals), traffic=traffic or 'not said',
                                     weather=weather or 'not said', scene_said=traffic is not None,
                                     closing=bool(float(fmt(x, decimals)) > float(fmt(z, decimals)))))
                    cid += 1
    return pd.DataFrame(rows), lv, len(t)


def description(r, distance_ref):
    s = (f'Your car is going at {r.x_text} {UNIT_SPEED_FULL}. Directly ahead of you in your lane there is a vehicle '
         f'going in the same direction at {r.z_text} meters per second. ')
    if distance_ref == 'camera':
        s += f'Measured along the lane, it is {r.y_text} meters from your front camera to the back of that vehicle.'
    else:
        s += f'Measured along the lane, the gap between your front bumper and the back of that vehicle is {r.y_text} meters.'
    if r.scene_said:
        s += f' {TRAFFIC[r.traffic]} {WEATHER[r.weather]}'
    return s


def messages_of(r, a):
    fmt_ = FORMAT if a.camera_field == 'keep' else FORMAT.replace(CAMERA_LINE, '')
    if a.camera_field == 'drop' and fmt_ == FORMAT:
        sys.exit('FORMAT: the main_hazard_camera line was not found')
    user = [{'type': 'text', 'text': INTRO}, {'type': 'text', 'text': description(r, a.distance_ref)},
            {'type': 'text', 'text': QUESTION + fmt_}]
    return [{'role': 'system', 'content': SYSTEMS[a.system]}, {'role': 'user', 'content': user}]


def first_line(path):
    with (gzip.open(path, 'rb') if path.endswith('.gz') else open(path, 'rb')) as f:
        raw = f.readline()
    i = raw.find(b', "logprobs": ')
    if i >= 0:
        j = raw.find(b', "latency_s": ', i)
        raw = raw[:i] + (raw[j:] if j > 0 else b'}')
    return json.loads(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth', required=True)
    ap.add_argument('--frames', required=True, help='ssm_target_frames.csv of the main result (1,084 frames)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--system', choices=sorted(SYSTEMS), default='S1', help='undecided (user)')
    ap.add_argument('--camera_field', choices=['keep', 'drop'], default='keep', help='undecided (user)')
    ap.add_argument('--distance_ref', choices=['camera', 'bumper'], default='camera', help='undecided (user)')
    ap.add_argument('--decimals', type=int, default=1, help='undecided (user)')
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--show', type=int, default=3)
    ap.add_argument('--check_run', default=None, help='the 27B risk run\'s replies, to check the settings against')
    ap.add_argument('--design_csv', default=None, help='write the design here')
    ap.add_argument('--base_url', default='http://localhost:8000/v1')
    ap.add_argument('--model', default='qwen3.6-27b')
    ap.add_argument('--api_key_env', default='VLM_API_KEY')
    ap.add_argument('--top_logprobs', type=int, default=5)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--timeout', type=float, default=3600)
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--send', action='store_true')
    g.add_argument('--forced', action='store_true', help='forced answers for the replies cut at max_tokens')
    a = ap.parse_args()
    d, lv, n_frames = design(a.truth, a.frames, a.distance_ref, a.decimals)
    if a.design_csv:
        d.to_csv(a.design_csv, index=False)
    jobs = [(r, rep) for r in d.itertuples() for rep in range(a.repeats)]
    if not (a.send or a.forced):
        print(f'levels from {n_frames} frames (quantiles {QUANTILES}):')
        for k, name in (('x', 'X ego speed'), ('z', 'Z lead speed'), ('y', f'Y distance ({a.distance_ref})')):
            print(f'  {name}: ' + ', '.join(f'{v!r}' for v in lv[k]) + f'  -> written {sorted(set(d[k + "_text"]), key=float)}')
        print(f'{len(d)} combinations ({int(d.scene_said.sum())} with a scene, {int((~d.scene_said).sum())} without), '
              f'{int((~d.closing).sum())} not closing in (Z >= X as written); {len(jobs)} requests')
        print(f'choices: system {a.system}, camera field {a.camera_field}, distance {a.distance_ref}, decimals {a.decimals}')
        print(f'sampling {SAMPLING}, log-probabilities {a.top_logprobs}, model {a.model}')
        if a.check_run:
            f0 = first_line(a.check_run)
            same = {k: (f0.get(k), SAMPLING[k], f0.get(k) == SAMPLING[k]) for k in SAMPLING}
            same['model'] = (f0.get('model'), a.model, f0.get('model') == a.model)
            same['presence_penalty sent'] = ('presence_penalty' in f0, False, 'presence_penalty' not in f0)
            print(f'against the first line of {a.check_run}: ' + '; '.join(f'{k} {v[0]!r} there, {v[1]!r} here: '
                                                                        f'{"same" if v[2] else "DIFFERENT"}'
                                                                        for k, v in same.items()))
            if not all(v[2] for v in same.values()):
                sys.exit('settings differ from the 27B risk run')
        shown = set()
        for r, rep in jobs:
            if len(shown) >= a.show:
                break
            if (r.scene_said, r.closing) in shown or rep:
                continue
            shown.add((r.scene_said, r.closing))
            print(f'\n===== combination {r.combo} (X{r.x_level} Z{r.z_level} Y{r.y_level}, {r.traffic}, {r.weather}, '
                  f'closing {r.closing}) repeat {rep}')
            m = messages_of(r, a)
            print('[SYSTEM] ' + m[0]['content'])
            for part in m[1]['content']:
                print(part['text'])
        print('\nnothing sent (add --send to send)')
        return
    from openai import OpenAI
    client = OpenAI(base_url=a.base_url, api_key=os.environ.get(a.api_key_env, 'EMPTY'), timeout=a.timeout, max_retries=0)
    lock = threading.Lock()
    if a.forced:
        out = a.out[:-len('.jsonl')] + '_forced_answers.jsonl' if a.out.endswith('.jsonl') else a.out + '.forced'
        cut = []
        with open(a.out, encoding='utf-8') as f:
            for line in f:
                o = json.loads(line)
                if o.get('finish_reason') == 'length' and o.get('parsed') is None:
                    o.pop('logprobs', None)
                    cut.append(o)
        done = set()
        if os.path.exists(out):
            with open(out, encoding='utf-8') as f:
                done = {(o['combo'], o['repeat']) for o in map(json.loads, f)}
        by = {r.combo: r for r in d.itertuples()}
        for old in cut:
            if (old['combo'], old['repeat']) in done:
                continue
            thinking = (old.get('reasoning') or '').strip()
            assert thinking and '<think>' not in thinking and '</think>' not in thinking, (old['combo'], old['repeat'])
            msgs = messages_of(by[old['combo']], a) + [{'role': 'assistant',
                                                       'content': '<think>\n' + thinking + EARLY_STOPPING_TEXT}]
            rec = {k: old[k] for k in ('combo', 'repeat', 'model', 'temperature', 'top_p', 'top_k')}
            rec.update(forced_answer=True, early_stopping_text=EARLY_STOPPING_TEXT, max_tokens=FORCED_MAX_TOKENS,
                       original_max_tokens=old['max_tokens'], original_usage=old['usage'], reasoning=old['reasoning'])
            t0 = time.time()
            try:
                kw = dict(model=a.model, messages=msgs, temperature=SAMPLING['temperature'], top_p=SAMPLING['top_p'],
                          max_tokens=FORCED_MAX_TOKENS, extra_body={'add_generation_prompt': False,
                                                                    'continue_final_message': True,
                                                                    'top_k': SAMPLING['top_k']})
                if a.top_logprobs:
                    kw.update(logprobs=True, top_logprobs=a.top_logprobs)
                resp = client.chat.completions.create(**kw)
                ch = resp.choices[0]
                rec.update(text=ch.message.content, parsed=parse(ch.message.content), finish_reason=ch.finish_reason,
                           usage=resp.usage.model_dump() if resp.usage else None)
            except Exception as e:
                rec['error'] = f'{type(e).__name__}: {e}'
            rec['latency_s'] = round(time.time() - t0, 2)
            if 'error' not in rec:
                with open(out, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        return
    done = set()
    if os.path.exists(a.out):
        with open(a.out, encoding='utf-8') as f:
            done = {(o['combo'], o['repeat']) for o in map(json.loads, f)}
    todo = [j for j in jobs if (j[0].combo, j[1]) not in done]
    print(f'{len(jobs)} requests, {len(done)} already done, {len(todo)} to send', flush=True)

    def ask(job):
        r, rep = job
        rec = dict(combo=int(r.combo), x_level=int(r.x_level), z_level=int(r.z_level), y_level=int(r.y_level),
                   x=r.x_text, z=r.z_text, y=r.y_text, traffic=r.traffic, weather=r.weather, closing=bool(r.closing),
                   repeat=rep, model=a.model, **SAMPLING, system=a.system, camera_field=a.camera_field,
                   distance_ref=a.distance_ref)
        t0 = time.time()
        try:
            kw = dict(model=a.model, messages=messages_of(r, a), temperature=SAMPLING['temperature'],
                      top_p=SAMPLING['top_p'], max_tokens=SAMPLING['max_tokens'], extra_body={'top_k': SAMPLING['top_k']})
            if a.top_logprobs:
                kw.update(logprobs=True, top_logprobs=a.top_logprobs)
            resp = client.chat.completions.create(**kw)
            ch = resp.choices[0]
            rec.update(text=ch.message.content, parsed=parse(ch.message.content),
                       reasoning=getattr(ch.message, 'reasoning', None), finish_reason=ch.finish_reason,
                       usage=resp.usage.model_dump() if resp.usage else None)
            if ch.logprobs and ch.logprobs.content:
                rec['logprobs'] = [dict(token=c.token, logprob=c.logprob,
                                        top=[(x.token, x.logprob) for x in (c.top_logprobs or [])])
                                   for c in ch.logprobs.content]
        except Exception as e:
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
            else:
                n_ok += 1
                n_bad += rec['parsed'] is None
            if i % 50 == 0:
                print(f'{i}/{len(todo)} sent: {n_ok} ok ({n_bad} unparsable), {n_err} errors', flush=True)
    print(f'done: {n_ok} ok ({n_bad} unparsable), {n_err} errors (rerun to retry them)')


if __name__ == '__main__':
    main()
