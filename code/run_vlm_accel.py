"""EXPLORATORY new experiment (user, 2026-11-09): the risk question of the main run with only the ego car's acceleration
added (not its speed; the speed experiment is code/run_vlm_speed.py, run separately in /data/lab/cejli/sy/speed_risk).
Every request is the one code/run_vlm.py builds for the risk task, except that its last text part (the risk question)
starts with one more sentence:

    Your car's acceleration is <a_ego, 1 decimal> meters per second squared (negative means slowing down).

a_ego is the true ego acceleration of the frame (quantities truth csv, column a_ego: the rate of change of the speed,
along the direction of travel); "-0.0" is written "0.0". run_vlm.py and force_answer.py are not changed: their `build`
is replaced in memory and their own main() runs, as in run_vlm_speed.py.
usage: python -I run_vlm_accel.py <truth csv (segment, frame_idx, a_ego)> run <run_vlm.py arguments>
       python -I run_vlm_accel.py <truth csv> forced <force_answer.py arguments>
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import force_answer as fa  # noqa: E402
import run_vlm as rv  # noqa: E402

SENTENCE = "Your car's acceleration is {a} meters per second squared (negative means slowing down)."


def accel_text(a):
    s = f'{float(a):.1f}'
    return '0.0' if s == '-0.0' else s


def make_build(accel, orig=rv.build):
    def build(cond, grid_imgs, sep_imgs, question=None):
        if question is not None:
            raise SystemExit('run_vlm_accel.py: risk task only')
        p = orig(cond, grid_imgs, sep_imgs)
        now = grid_imgs[-1]
        key = (os.path.basename(os.path.dirname(now)), int(os.path.basename(now)[len('step_'):-len('.jpg')]))
        if key not in accel:
            raise SystemExit(f'run_vlm_accel.py: no acceleration for {key}')
        kind, text = p[-1]
        assert kind == 'text' and text.startswith('How risky'), (cond, text[:40])
        p[-1] = ('text', SENTENCE.format(a=accel[key]) + '\n' + text)
        return p
    return build


def main():
    csv, mode = sys.argv[1:3]
    t = pd.read_csv(csv)
    if t.a_ego.isna().any():
        raise SystemExit(f'{csv}: frames without a_ego')
    accel = {(s, int(k)): accel_text(v) for s, k, v in zip(t.segment, t.frame_idx, t.a_ego)}
    b = make_build(accel)
    rv.build = b
    fa.build = b
    sys.argv = [sys.argv[0]] + sys.argv[3:]
    print(f'run_vlm_accel.py: {len(accel)} frames with an acceleration from {csv}; mode {mode}', flush=True)
    if mode == 'run':
        rv.main()
    elif mode == 'forced':
        fa.main()
    else:
        raise SystemExit('mode: run or forced')


if __name__ == '__main__':
    main()
