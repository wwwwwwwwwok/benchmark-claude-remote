"""The two lists of target frames for the formal VLM experiment (user's decision, 2026-10-22), from frame_ssm.csv of the
validation and training splits:
  main:        target frames where all 9 SSMs have a value (no closing-speed threshold; the main analysis later leaves
               out the frames with a closing speed below 0.1 m/s, column closing_below_0_1);
  comparison:  target frames where the 7 lead-vehicle SSMs have a value (the main list is a subset; column in_main).
Target frames as in make_vlm_inputs.py: frame_idx divisible by 5, at least 20, not an edge frame. "Has a value" as in
analyze_vlm_ssm.py: TTC, THW, MTTC, PICUD, PSD, TTC2D, ACT finite; DRAC, CI > 0; the lead-vehicle SSMs have no value
where the ego or the lead reverses. Each file starts with '#' lines naming its source and date (read with
pandas.read_csv(..., comment='#')).

usage: python -I make_vlm_frame_lists.py <results root with <split>/ssm/frame_ssm.csv> <out dir>
"""
import datetime
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
LEAD = ['TTC', 'THW', 'DRAC', 'MTTC', 'CI', 'PICUD', 'PSD']
ALL9 = LEAD + ['TTC2D', 'ACT']
CLOSING_MIN = 0.1        # m/s: the main analysis leaves out frames closing in more slowly (analyze_vlm_ssm.CLOSING_MIN)
COLS = ['split', 'segment', 'frame_idx', 't_s', 'lead_map', 'gap', 'v_ego', 'v_lead', 'closing'] + ALL9


def has_value(d, s):
    v = d[s].to_numpy(float)
    val = (np.nan_to_num(v, nan=0.0) > 0) if s in ('DRAC', 'CI') else np.isfinite(v)
    if s in LEAD:
        val &= ~(d['reversing_ego'].astype(bool) | d['reversing_lead'].astype(bool)).to_numpy()
    return val


def write(df, path, header):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(''.join(f'# {h}\n' for h in header))
        df.to_csv(f, index=False)


def main():
    root, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    parts, rules = [], []
    for split in ('validation', 'training'):
        p = os.path.join(root, split, 'ssm')
        parts.append(pd.read_csv(os.path.join(p, 'frame_ssm.csv'), dtype={'lead_map': str}).assign(split=split))
        with open(os.path.join(p, 'params.txt'), encoding='utf-8') as fh:
            rules.append([ln for ln in fh.read().splitlines() if ln.startswith(('lead ', 'slow lead', 'reversing'))])
    if rules[0] != rules[1]:
        sys.exit('the two splits were computed with different lead rules')
    f = pd.concat(parts, ignore_index=True)
    tgt = ~f['edge_frame'].astype(bool) & (f['frame_idx'] % 5 == 0) & (f['frame_idx'] >= 20)
    all9 = tgt & np.logical_and.reduce([has_value(f, s) for s in ALL9])
    all7 = tgt & np.logical_and.reduce([has_value(f, s) for s in LEAD])
    slow = (f['closing'] > 0) & (f['closing'] < CLOSING_MIN)
    today = datetime.date.today().isoformat()
    src = [f'source: {os.path.abspath(root)}/{{validation,training}}/ssm/frame_ssm.csv',
           f'generated: {today} by code/make_vlm_frame_lists.py'] + [f'lead rule (params.txt): {ln}' for ln in rules[0]]
    main_df = f.loc[all9, COLS].assign(closing_below_0_1=slow[all9].to_numpy())
    comp_df = f.loc[all7, COLS].assign(in_main=all9[all7].to_numpy(), closing_below_0_1=slow[all7].to_numpy())
    write(main_df, os.path.join(out, 'main_all9_targets.csv'), src + [
        f'rows: {len(main_df)} target frames (frame_idx % 5 == 0, >= 20, not edge) where all 9 SSMs have a value, no '
        f'closing-speed threshold; closing_below_0_1 marks {int(main_df.closing_below_0_1.sum())} frames with '
        f'0 < closing < {CLOSING_MIN} m/s, left out of the main analysis entirely (every SSM; user, 2026-10-22)'])
    write(comp_df, os.path.join(out, 'comparison_lead7_targets.csv'), src + [
        f'rows: {len(comp_df)} target frames where the 7 lead-vehicle SSMs have a value, no closing-speed threshold; '
        f'in_main marks the {int(comp_df.in_main.sum())} frames of main_all9_targets.csv '
        f'({int((~comp_df.in_main).sum())} more frames); closing_below_0_1 as in the main list '
        f'({int(comp_df.closing_below_0_1.sum())} frames)'])
    print(f'main: {len(main_df)} frames in {main_df.segment.nunique()} segments, closing < {CLOSING_MIN}: '
          f'{int(main_df.closing_below_0_1.sum())}')
    print(f'comparison: {len(comp_df)} frames in {comp_df.segment.nunique()} segments, in the main list '
          f'{int(comp_df.in_main.sum())}, closing < {CLOSING_MIN}: {int(comp_df.closing_below_0_1.sum())}')


if __name__ == '__main__':
    main()
