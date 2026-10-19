"""How would Laureshyn et al. (2017) Delta-V0 / Extended Delta-V behave on the Waymo demo?
Delta-V (larger of the two road users) = R/(1+R) * |v1 - v2| (vector), R = heavier/lighter mass.
Extended Delta-V: both speeds first reduced to max(v0 - a*t, 0), t = time to the collision point.
usage: python -I delta_v_check.py"""
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.stdout.reconfigure(encoding='utf-8')
RES = r'D:/Ph.D/博士论文/benchmark/results/waymo_demo/frame_ssm.csv'
EXT = r'F:/benchmark/waymo perception/extracted/validation/'
f = pd.read_csv(RES)

for seg, g in f.groupby('segment'):
    t0 = datetime.fromtimestamp(g.timestamp_micros.iloc[0] / 1e6, tz=timezone.utc)
    print(f'{seg[:20]}: recorded {t0:%Y-%m-%d}, frames {len(g)}')

# ---- lead (rear-end, same direction: alpha = 0, so |v1 - v2| = closing speed), car-car R = 1 -> factor 0.5
lead = f[np.isfinite(f.TTC)].copy()
print(f'\nlead frames with a collision course (finite TTC): {len(lead)}; TTC min {lead.TTC.min():.3f} s; '
      f'ego speed in them {lead.v_ego.min():.3f}-{lead.v_ego.max():.3f} m/s')
lead['dv0'] = 0.5 * lead.closing
tau = kendalltau(lead.dv0, lead.closing, variant='b').statistic
print(f'Delta-V0 vs closing speed: Kendall tau-b = {tau:.4f}')
for a in (4.0, 6.7, 8.0):
    vf = np.clip(lead.v_ego - a * lead.TTC, 0, None)
    vl = np.clip(lead.v_lead - a * lead.TTC, 0, None)
    ext = 0.5 * np.abs(vf - vl)
    print(f'  a = {a} m/s2: Extended Delta-V > 0 in {(ext > 0).sum()} of {len(lead)} frames '
          f'(needs ego speed > a*TTC; max ego speed - a*TTC = {(lead.v_ego - a * lead.TTC).max():.3f} m/s)')

# ---- 2D-TTC partner (any direction): Extended Delta-V is non-zero only if someone is still moving at t = 2D-TTC
d2 = f[np.isfinite(f.TTC2D)].copy()
spd = {}
for seg in d2.segment.unique():
    o = pd.read_parquet(EXT + seg + '/objects.parquet', columns=['frame_idx', 'id', 'type', 'speed_x', 'speed_y'])
    o['v'] = np.hypot(o.speed_x, o.speed_y)
    spd.update({(seg, r.frame_idx, r.id): (r.v, r.type) for r in o.itertuples()})
d2['v_obj'] = [spd[(r.segment, r.frame_idx, r.TTC2D_obj)][0] for r in d2.itertuples()]
d2['obj_type'] = [spd[(r.segment, r.frame_idx, r.TTC2D_obj)][1] for r in d2.itertuples()]
print(f'\n2D-TTC frames: {len(d2)}; 2D-TTC min {d2.TTC2D.min():.3f} s; partner types {d2.obj_type.value_counts().to_dict()}')
for a in (4.0, 6.7, 8.0):
    moving = np.maximum(d2.ego_speed, d2.v_obj) > a * d2.TTC2D
    print(f'  a = {a} m/s2: someone still moving at the collision point in {moving.sum()} of {len(d2)} frames')
print(d2[['segment', 'frame_idx', 'TTC2D', 'ego_speed', 'v_obj', 'obj_type']].sort_values('TTC2D').head(8)
      .to_string(index=False))
