"""Is a TTC "value" just noise when the closing speed is tiny? (user, 2026-10-22; computed only)
On the frames where all 9 SSMs have a value (and the target frames among them), results_lead_final_excl:
  1. frames with closing speed (ego speed - lead speed) below 0.05, 0.1, 0.3 m/s;
  2. frames with TTC above 60, 100, 1,000 s;
  3. speed measurement error: ego smoothed (Savitzky-Golay) speed vs the ego speed Waymo reports with the FRONT
     image (all frames and frames away from the segment ends), and the labelled speed of vehicles that never move
     (as in code/experiments/exp_lowspeed_ttc2d.py: world centre within a 0.3 m range over >= 50 frames), all frames
     and away from the track ends (9 frames each side).

usage: python -I speed_noise.py <results root> <tables root>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
LEAD = ['TTC', 'THW', 'DRAC', 'MTTC', 'CI', 'PICUD', 'PSD']
ALL9 = LEAD + ['TTC2D', 'ACT']
HALF = 9


def has_value(d, s):
    v = d[s].to_numpy(float)
    val = (np.nan_to_num(v, nan=0.0) > 0) if s in ('DRAC', 'CI') else np.isfinite(v)
    if s in LEAD:
        val &= ~(d['reversing_ego'].astype(bool) | d['reversing_lead'].astype(bool)).to_numpy()
    return val


def q(x, ps=(50, 95, 99, 100)):
    return ' / '.join(f'{v:.3f}' for v in np.percentile(x, ps))


def main():
    res, tables = sys.argv[1:3]
    f = pd.concat([pd.read_csv(os.path.join(res, s, 'ssm', 'frame_ssm.csv'), dtype={'lead_map': str}).assign(split=s)
                   for s in ('validation', 'training')], ignore_index=True)
    ok = ~f['edge_frame'].astype(bool)
    all9 = ok & np.logical_and.reduce([has_value(f, s) for s in ALL9])
    tgt = all9 & (f['frame_idx'] % 5 == 0) & (f['frame_idx'] >= 20)
    for lab, m in (('all 9 SSMs with values', all9), ('  of them target frames', tgt)):
        x = f[m]
        print(f'{lab}: {len(x)} frames; closing speed < 0.05 / 0.1 / 0.3 m/s: '
              + ' / '.join(str(int((x.closing < t).sum())) for t in (0.05, 0.1, 0.3))
              + '; TTC > 60 / 100 / 1,000 s: ' + ' / '.join(str(int((x.TTC > t).sum())) for t in (60, 100, 1000))
              + f'; closing speed min {x.closing.min():.2e} m/s, median {x.closing.median():.3f} m/s')
    x = f[all9]
    for t in (0.05, 0.1, 0.3):
        y = x[x.closing < t]
        print(f'  closing < {t} m/s: TTC {y.TTC.min():.1f}-{y.TTC.max():.1f} s; lead slower than 0.5 m/s in '
              f'{int((y.v_lead < 0.5).sum())}; ego slower than 0.5 m/s in {int((y.v_ego < 0.5).sum())}')
    # 3a. ego speed: smoothed vs reported
    dif, dif_in, sgn = [], [], []
    st_all, st_in = [], []
    for split in ('validation', 'training'):
        g = f[f.split == split]
        for seg, h in g.groupby('segment'):
            fr = pd.read_parquet(os.path.join(tables, split, seg, 'frames.parquet'),
                                 columns=['frame_idx', 'ego_vx', 'ego_vy', 'world_from_vehicle']).sort_values('frame_idx')
            rep = np.hypot(fr['ego_vx'], fr['ego_vy']).to_numpy()
            sm = h.sort_values('frame_idx')['ego_speed'].to_numpy()
            d = sm - rep
            n = len(d)
            inner = np.zeros(n, bool)
            inner[HALF:n - HALF] = True
            dif.append(np.abs(d))
            dif_in.append(np.abs(d[inner]))
            sgn.append(d[inner])
            # 3b. static vehicles
            o = pd.read_parquet(os.path.join(tables, split, seg, 'objects.parquet'),
                                columns=['frame_idx', 'id', 'type', 'cx', 'cy', 'cz', 'speed_x', 'speed_y'])
            o = o[o['type'] == 'vehicle']
            if o.empty:
                continue
            pose = np.stack([np.asarray(p, float).reshape(4, 4) for p in fr['world_from_vehicle']])
            kk = o['frame_idx'].to_numpy()
            w = np.einsum('nij,nj->ni', pose[kk, :3, :3], o[['cx', 'cy', 'cz']].to_numpy(float)) + pose[kk, :3, 3]
            o = o.assign(wx=w[:, 0], wy=w[:, 1], spd=np.hypot(o['speed_x'], o['speed_y']))
            ext = o.groupby('id').agg(dx=('wx', lambda s: s.max() - s.min()), dy=('wy', lambda s: s.max() - s.min()),
                                      n=('wx', 'size'), lo=('frame_idx', 'min'), hi=('frame_idx', 'max'))
            ids = ext[(np.hypot(ext.dx, ext.dy) < 0.3) & (ext.n >= 50)]
            s = o[o['id'].isin(ids.index)]
            st_all.append(s['spd'].to_numpy())
            lo, hi = s['id'].map(ids['lo']), s['id'].map(ids['hi'])
            st_in.append(s.loc[(s.frame_idx - lo >= HALF) & (hi - s.frame_idx >= HALF), 'spd'].to_numpy())
    dif, dif_in, sgn = np.concatenate(dif), np.concatenate(dif_in), np.concatenate(sgn)
    print(f'\nego speed, smoothed minus Waymo-reported, |difference| percentiles 50 / 95 / 99 / max (m/s):')
    print(f'  all frames ({len(dif)}): {q(dif)}')
    print(f'  away from the segment ends ({len(dif_in)}): {q(dif_in)}; signed median {np.median(sgn):+.4f} m/s')
    st_all, st_in = np.concatenate(st_all), np.concatenate(st_in)
    print('\nlabelled speed of vehicles that never move (world centre within 0.3 m over >= 50 frames), '
          'percentiles 50 / 95 / 99 / max (m/s):')
    print(f'  all their boxes ({len(st_all)}): {q(st_all)}')
    print(f'  away from the track ends ({len(st_in)}): {q(st_in)}')


if __name__ == '__main__':
    main()
