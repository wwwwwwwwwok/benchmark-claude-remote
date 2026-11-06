"""Read-only (user, 2026-11-05, item 3.1): the risk run's models with 5 repeats of every condition, on the 1,065 fit
frames of D (every new cue known, new_cues_min30.csv): per model x condition the frames, the requests, the repeats asked
(should be 0-4 on every frame), the answers without a risk score, the frames with fewer than 5 scores (by count), the
forced answers used. usage: python -I list_models.py <risk subfolder> <new_cues_min30.csv>"""
import os
import sys

import numpy as np
import pandas as pd

R, NEW = sys.argv[1:3]
NEED = ['speed_limit_mps', 'dist_crosswalk', 'dist_stop_sign', 'dist_lane_crossing', 'n_crossing_ped_cyc',
        'n_crossing_vehicle', 'n_oncoming_vehicle', 'n_ped_cyc_on_road', 'heading_change_deg'] + \
    [f'n_vehicle_{m}_{r}m' for r in (10, 20, 30, 50) for m in ('moving', 'stationary')]
st = pd.read_csv(os.path.join(R, 'ssm_target_frames.csv'))[['segment', 'frame_idx']]
nw = st.merge(pd.read_csv(NEW), on=['segment', 'frame_idx'], how='left', validate='one_to_one')
fit = st[~nw[NEED].isna().any(axis=1).to_numpy()]
print(f'frames {len(st)}; fit frames {len(fit)} in {fit.segment.nunique()} segments')
rq = pd.read_csv(os.path.join(R, 'vlm_requests_tidy.csv'), low_memory=False,
                 usecols=['model', 'segment', 'frame_idx', 'condition', 'repeat', 'forced', 'risk'])
rq = rq.merge(fit, on=['segment', 'frame_idx'], how='inner')
rows = []
for (m, c), g in rq.groupby(['model', 'condition'], sort=False):
    per = g.groupby(['segment', 'frame_idx'])
    reps_ok = bool(per['repeat'].apply(lambda v: sorted(v) == [0, 1, 2, 3, 4]).all())
    n_sc = per['risk'].apply(lambda v: int(v.notna().sum()))
    dist = '/'.join(str(int((n_sc == k).sum())) for k in (4, 3, 2, 1, 0))
    forced = g['forced'].astype(str).str.lower().isin(['true', '1'])
    rows.append(dict(model=m, condition=c, frames=int(per.ngroups), requests=len(g), repeats_0_to_4_on_every_frame=reps_ok,
                     answers_without_score=int(g.risk.isna().sum()), frames_fewer_than_5=int((n_sc < 5).sum()),
                     frames_with_4_3_2_1_0=dist, forced_answers_used=int((forced & g.risk.notna()).sum())))
t = pd.DataFrame(rows)
pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 100)
print(t.to_string(index=False))
print(f'models {t.model.nunique()}: {list(t.model.unique())}; every model x condition with 5 repeats on every fit '
      f'frame: {bool(t.repeats_0_to_4_on_every_frame.all() and (t.frames == len(fit)).all())}')
_ = np
