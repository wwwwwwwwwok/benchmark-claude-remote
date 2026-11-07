"""EXPLORATORY new experiment (user, 2026-11-06): the risk question of the main run with the ego car's speed added.
Every request is the one code/run_vlm.py builds for the risk task (same system text, images, captions and question),
except that its last text part (the risk question) starts with one more sentence:

    Your car's speed is <v_ego, 1 decimal> meters per second.

v_ego is the true ego speed of the frame (frame_ssm / quantities truth csv, column v_ego). run_vlm.py and force_answer.py
are not changed (other jobs read them): this program replaces their `build` in memory by one that adds the sentence and
then runs their own main(), so every other step (sampling, sending, resuming, output format, forced answers) is theirs.
The frame of a request is read from its "now" grid image (<segment>/step_<frame>.jpg); a frame without a speed in the
csv stops the program.
usage: python -I run_vlm_speed.py <speed csv (segment, frame_idx, v_ego)> run <run_vlm.py arguments>
       python -I run_vlm_speed.py <speed csv> forced <force_answer.py arguments>
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import force_answer as fa  # noqa: E402
import run_vlm as rv  # noqa: E402

SENTENCE = "Your car's speed is {v} meters per second."


def speed_text(v):
    return f'{max(float(v), 0.0):.1f}'


def make_build(speed, orig=rv.build):
    def build(cond, grid_imgs, sep_imgs, question=None):
        if question is not None:
            raise SystemExit('run_vlm_speed.py: risk task only')
        p = orig(cond, grid_imgs, sep_imgs)
        now = grid_imgs[-1]
        key = (os.path.basename(os.path.dirname(now)), int(os.path.basename(now)[len('step_'):-len('.jpg')]))
        if key not in speed:
            raise SystemExit(f'run_vlm_speed.py: no speed for {key}')
        kind, text = p[-1]
        assert kind == 'text' and text.startswith('How risky'), (cond, text[:40])
        p[-1] = ('text', SENTENCE.format(v=speed[key]) + '\n' + text)
        return p
    return build


def main():
    csv, mode = sys.argv[1:3]
    t = pd.read_csv(csv)
    speed = {(s, int(k)): speed_text(v) for s, k, v in zip(t.segment, t.frame_idx, t.v_ego)}
    b = make_build(speed)
    rv.build = b
    fa.build = b
    sys.argv = [sys.argv[0]] + sys.argv[3:]
    print(f'run_vlm_speed.py: {len(speed)} frames with a speed from {csv}; mode {mode}', flush=True)
    if mode == 'run':
        rv.main()
    elif mode == 'forced':
        fa.main()
    else:
        raise SystemExit('mode: run or forced')


if __name__ == '__main__':
    main()
