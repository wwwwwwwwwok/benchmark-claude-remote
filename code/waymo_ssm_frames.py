"""Per-frame SSMs for Waymo segments extracted from v1.4.3, comparing three lead-vehicle pairing methods.

Usage:
    python -I waymo_ssm_frames.py <extracted_root> <out_dir> [--camera FRONT|all] [segment ...]

Pairing methods (wod_common.py)
    straight : straight corridor in the vehicle frame (|lateral| < 1.75 m)
    path     : corridor along the path the ego actually drives next
    map      : lane route from the HD map (falls back to `path` when the ego is not on a mapped lane)
Lead-vehicle SSMs are computed with the `map` pairing; 2D metrics use all objects.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wod_common import (Segment, LaneMap, lead_straight, lead_path, lead_map, wrap,  # noqa: E402
                        EGO_X_FRONT, EGO_X_REAR, EGO_HALF_W, LEAD_TYPES)

sys.stdout.reconfigure(encoding='utf-8')

# ---- parameters the user has to choose (demo values; see the workbook's hyperparameter discussion) ----
PARAMS = dict(
    tau=1.0,           # reaction time (s): DSS, PICUD, RCRI, RSS (rho)
    a_emerg=6.0,       # emergency braking deceleration (m/s^2): PICUD, RCRI, TTB
    mu=0.7,            # road friction coefficient: DSS
    t_safe=1.4,        # safety time gap (s): DST
    ttc_star=3.0,      # TTC threshold (s): TET, TIT
    rss_a_accel=3.8,   # RSS: max acceleration of the rear vehicle during the response time (m/s^2)
    rss_b_min=6.0,     # RSS: braking the rear vehicle applies after responding (m/s^2)
    rss_b_max=7.0,     # RSS: max braking of the front vehicle (m/s^2)
    ttc_cap=10.0,      # horizon of the 2D-TTC search (s)
)
G = 9.81
METHODS = ('straight', 'path', 'map')
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
METHOD_COLOR = {'map': BLUE, 'path': AQUA, 'straight': ORANGE}
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'

EGO_BOX = np.array([[EGO_X_FRONT, EGO_HALF_W], [EGO_X_FRONT, -EGO_HALF_W],
                    [EGO_X_REAR, -EGO_HALF_W], [EGO_X_REAR, EGO_HALF_W]])


# ------------------------------------------------------------------ 2D geometry
def rect(cx, cy, L, W, h):
    c, s = np.cos(h), np.sin(h)
    dx = np.array([1, 1, -1, -1]) * L / 2
    dy = np.array([1, -1, -1, 1]) * W / 2
    return np.stack([cx + dx * c - dy * s, cy + dx * s + dy * c], axis=-1)


def axes_of(q):
    e1, e2 = q[1] - q[0], q[3] - q[0]
    return [e1 / np.linalg.norm(e1), e2 / np.linalg.norm(e2)]


def ttc_2d(obj, rel_v, cap, dt=0.05):
    """First time the object box, moving at rel_v relative to the ego, overlaps the ego box."""
    t = np.arange(0.0, cap + 1e-9, dt)
    objs = obj[None] + t[:, None, None] * rel_v[None, None, :]
    hit = np.ones(len(t), bool)
    for ax in axes_of(EGO_BOX) + axes_of(obj):
        pe, po = EGO_BOX @ ax, objs @ ax
        hit &= (po.max(1) >= pe.min()) & (po.min(1) <= pe.max())
    i = np.flatnonzero(hit)
    return float(t[i[0]]) if len(i) else np.inf


def _pt_seg(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
    return np.linalg.norm(p - (a + t * ab))


def poly_dist(A, B):
    for ax in axes_of(A) + axes_of(B):
        pa, pb = A @ ax, B @ ax
        if pa.max() < pb.min() or pb.max() < pa.min():
            break
    else:
        return 0.0
    return min(min(_pt_seg(A[j], B[i], B[(i + 1) % 4]), _pt_seg(B[j], A[i], A[(i + 1) % 4]))
               for i in range(4) for j in range(4))


def act(obj, rel_v, h=0.1):
    d0 = poly_dist(EGO_BOX, obj)
    if d0 == 0:
        return 0.0, 0.0
    closing = (d0 - poly_dist(EGO_BOX, obj + h * rel_v[None])) / h
    return (d0 / closing if closing > 1e-6 else np.inf), d0


def mttc(gap, closing, dacc):
    if abs(dacc) < 1e-6:
        return gap / closing if closing > 0 else np.inf
    disc = closing ** 2 + 2 * dacc * gap
    if disc < 0:
        return np.inf
    r = [x for x in ((-closing + np.sqrt(disc)) / dacc, (-closing - np.sqrt(disc)) / dacc) if x > 0]
    return min(r) if r else np.inf


# ------------------------------------------------------------------ lead-vehicle SSMs
def lead_ssms(g, vf, vl, af, al):
    p = PARAMS
    closing = vf - vl
    out = dict(gap=g, v_ego_long=vf, v_lead_long=vl, closing=closing,
               TTC=g / closing if closing > 0 else np.inf,
               RTTC=closing / g if closing > 0 else 0.0,
               THW=g / vf if vf > 0.5 else np.inf,
               DRAC=closing ** 2 / (2 * g) if closing > 0 else 0.0)
    out['MTTC'] = m = mttc(g, closing, af - al)
    out['CI'] = (((vf + af * m) ** 2 - (vl + al * m) ** 2) / 2 / m) if np.isfinite(m) and m > 0 else 0.0
    stop_f = vf * p['tau'] + vf ** 2 / (2 * p['a_emerg'])
    out['PICUD'] = vl ** 2 / (2 * p['a_emerg']) + g - stop_f
    out['RCRI'] = int(out['PICUD'] <= 0)
    out['DSS'] = vl ** 2 / (2 * p['mu'] * G) + g - (vf * p['tau'] + vf ** 2 / (2 * p['mu'] * G))
    denom = g - vl * p['t_safe']
    out['DST'] = 0.0 if closing <= 0 else (closing ** 2 / (2 * denom) if denom > 0 else np.inf)
    out['TTB'] = out['TTC'] - closing / (2 * p['a_emerg']) if np.isfinite(out['TTC']) else np.inf
    rho, aa = p['tau'], p['rss_a_accel']
    dmin = vf * rho + 0.5 * aa * rho ** 2 + (vf + rho * aa) ** 2 / (2 * p['rss_b_min']) - vl ** 2 / (2 * p['rss_b_max'])
    out['RSS_dmin'] = max(dmin, 0.0)
    out['RSS_margin'] = g - out['RSS_dmin']
    return out


LEAD_COLS = ['gap', 'v_ego_long', 'v_lead_long', 'closing', 'TTC', 'RTTC', 'THW', 'DRAC', 'MTTC', 'CI',
             'PICUD', 'RCRI', 'DSS', 'DST', 'TTB', 'RSS_dmin', 'RSS_margin']


# ------------------------------------------------------------------ per segment
def process(seg, camera):
    lmap = LaneMap(seg.map)
    n = len(seg.frames)
    if lmap.ok:
        ego_lanes, ego_s = lmap.match_sequence(seg.ego_map[:, :2], seg.yaw)
    else:
        ego_lanes, ego_s = [None] * n, np.full(n, np.nan)
    rows, picks = [], []
    for k in range(n):
        o_all = seg.objects_at(k, camera=camera, types=['vehicle', 'pedestrian', 'cyclist'])
        o_lead = o_all[o_all['type'].isin(LEAD_TYPES)].reset_index(drop=True)
        R = seg.R[k]
        ego_v_veh = (R.T @ seg.ego_v[k])[:2]
        row = dict(segment=seg.name, frame_idx=k, timestamp_micros=int(seg.ts[k]), t_s=round(float(seg.t[k]), 2),
                   ego_speed=float(np.hypot(*seg.ego_v[k, :2])), ego_lane=ego_lanes[k], n_objects=len(o_all))
        res = {'straight': lead_straight(seg, k, o_lead), 'path': lead_path(seg, k, o_lead),
               'map': lead_map(seg, k, o_lead, lmap, ego_lanes, ego_s)}
        for m, r in res.items():
            row[f'lead_{m}'] = o_lead.loc[r['i'], 'id'] if r else None
            row[f'gap_{m}'] = r['gap'] if r else np.nan
            if r:
                c = r['v_ego'] - r['v_lead']
                row[f'TTC_{m}'] = r['gap'] / c if c > 0 else np.inf
            else:
                row[f'TTC_{m}'] = np.nan
        r = res['map']
        row['map_fallback'] = bool(r.get('fallback')) if r else (ego_lanes[k] is None)
        row.update(lead_ssms(r['gap'], r['v_ego'], r['v_lead'], r['a_ego'], r['a_lead']) if r
                   else {c: np.nan for c in LEAD_COLS})

        # 2D metrics over all objects (vehicle frame, relative velocity)
        row.update(TTC2D=np.inf, TTC2D_obj=None, ACT=np.inf, ACT_obj=None, min_dist=np.inf, min_dist_obj=None)
        if len(o_all):
            _, _, v_g, _ = seg.to_map(k, o_all)
            for (oid, cx, cy, L, W, h), vg in zip(o_all[['id', 'cx', 'cy', 'length', 'width', 'heading']].itertuples(index=False), v_g):
                if np.hypot(cx, cy) > 80:
                    continue
                q = rect(cx, cy, L, W, h)
                rel = R[:2, :2].T @ vg - ego_v_veh
                a_val, d0 = act(q, rel)
                if d0 < row['min_dist']:
                    row['min_dist'], row['min_dist_obj'] = d0, oid
                if a_val < row['ACT']:
                    row['ACT'], row['ACT_obj'] = a_val, oid
                t2 = ttc_2d(q, rel, PARAMS['ttc_cap'])
                if t2 < row['TTC2D']:
                    row['TTC2D'], row['TTC2D_obj'] = t2, oid
        rows.append(row)
        picks.append(res)
    return pd.DataFrame(rows), lmap, ego_lanes, ego_s


def summarize(g):
    dt = np.median(np.diff(g['t_s']))
    ttc = g['TTC'].fillna(np.inf)
    below = ttc < PARAMS['ttc_star']
    both = lambda a, b: (g[f'lead_{a}'].fillna('-') == g[f'lead_{b}'].fillna('-')).mean()
    return pd.Series({
        'frames': len(g), 'ego_speed_mean': g['ego_speed'].mean(),
        'ego_on_mapped_lane_share': g['ego_lane'].notna().mean(),
        **{f'frames_with_lead_{m}': int(g[f'lead_{m}'].notna().sum()) for m in METHODS},
        'same_lead_map_vs_path': both('map', 'path'), 'same_lead_map_vs_straight': both('map', 'straight'),
        'min_TTC': g['TTC'].min(), 'min_THW': g['THW'].min(), 'max_DRAC': g['DRAC'].max(),
        'min_PICUD': g['PICUD'].min(), 'min_DSS': g['DSS'].min(), 'min_RSS_margin': g['RSS_margin'].min(),
        'TET_s': below.sum() * dt, 'TIT_s2': ((PARAMS['ttc_star'] - ttc[below]) * dt).sum(),
        'min_TTC2D': g['TTC2D'].min(), 'min_ACT': g['ACT'].min(), 'min_dist': g['min_dist'].min(),
    })


# ------------------------------------------------------------------ figures
def style(ax):
    ax.set_facecolor(SURF)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID)


def plot_ttc_methods(df, seg_name, out):
    cap = PARAMS['ttc_cap']
    fig, ax = plt.subplots(figsize=(10, 3.6), facecolor=SURF)
    style(ax)
    for m in ('map', 'path', 'straight'):
        y = df[f'TTC_{m}'].replace(np.inf, cap).clip(upper=cap)
        ax.plot(df['t_s'], y, color=METHOD_COLOR[m], lw=2 if m == 'map' else 1.4, label=m)
    ax.set_ylim(0, cap * 1.05)
    ax.grid(color=GRID, lw=0.8)
    ax.set_xlabel('time in segment (s)', color=INK2)
    ax.set_ylabel('TTC to lead (s)', color=INK2)
    ax.legend(frameon=False, ncol=3, loc='lower left', fontsize=9)
    ax.set_title(f'{seg_name}: TTC to the lead vehicle by pairing method (capped at {cap:.0f} s; gaps = no lead)',
                 fontsize=9, color=INK2, loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'ttc_by_method_{seg_name}.png'), dpi=110, facecolor=SURF)
    plt.close(fig)


def plot_snapshot(seg, df, k, lmap, ego_lanes, ego_s, out, camera):
    """Bird's-eye view (map frame) next to the front image, with each method's lead outlined."""
    row = df.iloc[k]
    leads = {m: row[f'lead_{m}'] for m in METHODS}
    fig, (ax, axi) = plt.subplots(1, 2, figsize=(15, 6.2), facecolor=SURF, gridspec_kw={'width_ratios': [1, 1.45]})
    style(ax)
    for r in seg.map.itertuples():
        if r.kind == 'lane':
            ax.plot(r.xs, r.ys, color='#c3c2b7', lw=0.8, zorder=1)
        elif r.kind == 'road_edge':
            ax.plot(r.xs, r.ys, color=INK2, lw=1.3, zorder=2)
    if ego_lanes[k] is not None:
        future = [l for l in ego_lanes[k:] if l is not None]
        _, route = lmap.route(ego_lanes[k], ego_s[k], future)
        ax.plot(route[:, 0], route[:, 1], color=BLUE, lw=5, alpha=0.25, zorder=3, label='map lane route')
    ax.plot(seg.ego_map[k:, 0], seg.ego_map[k:, 1], color=AQUA, lw=1.6, ls='--', zorder=3, label='ego path (rest of segment)')
    o = seg.objects_at(k, camera=camera if camera != 'all' else None, types=['vehicle', 'pedestrian', 'cyclist'])
    if len(o):
        P, H, _, _ = seg.to_map(k, o)
        for (x, y), h, L, W, oid in zip(P, H, o['length'], o['width'], o['id']):
            ax.add_patch(MplPolygon(rect(x, y, L, W, h), closed=True, fc='#d9d8d2', ec=MUTED, lw=0.6, zorder=4))
            for j, m in enumerate(('map', 'path', 'straight')):
                if leads[m] == oid:
                    grow = 0.6 + 0.5 * j
                    ax.add_patch(MplPolygon(rect(x, y, L + grow, W + grow, h), closed=True, fill=False,
                                            ec=METHOD_COLOR[m], lw=2.2, zorder=5 + j))
    ex, ey = seg.ego_map[k, :2]
    ax.add_patch(MplPolygon(rect(ex + np.cos(seg.yaw[k]) * (EGO_X_FRONT + EGO_X_REAR) / 2,
                                 ey + np.sin(seg.yaw[k]) * (EGO_X_FRONT + EGO_X_REAR) / 2,
                                 EGO_X_FRONT - EGO_X_REAR, 2 * EGO_HALF_W, seg.yaw[k]),
                            closed=True, fc=BLUE, ec=SURF, lw=1, zorder=8))
    span = 45
    cx, cy = ex + np.cos(seg.yaw[k]) * span * 0.6, ey + np.sin(seg.yaw[k]) * span * 0.6
    ax.set_xlim(cx - span, cx + span); ax.set_ylim(cy - span, cy + span)
    ax.set_aspect('equal')
    for m in ('map', 'path', 'straight'):
        ax.plot([], [], color=METHOD_COLOR[m], lw=2.2, label=f'lead by {m}')
    ax.legend(frameon=False, fontsize=8, loc='lower left')
    ax.set_title(f'frame {k}  t = {row.t_s:.1f} s  ego {row.ego_speed:.1f} m/s', fontsize=9, color=INK2, loc='left')

    img_path = os.path.join(seg.dir, 'images', 'FRONT', f'{k:03d}_{int(row.timestamp_micros)}.jpg')
    if os.path.exists(img_path):
        axi.imshow(Image.open(img_path))
        pb = pd.read_parquet(os.path.join(seg.dir, 'projected_boxes.parquet'))
        pb = pb[(pb['frame_idx'] == k) & (pb['camera'] == 'FRONT')]
        for b in pb.itertuples():
            lid = b.id[:-len('_FRONT')] if isinstance(b.id, str) and b.id.endswith('_FRONT') else b.id
            hit = [m for m in ('map', 'path', 'straight') if leads[m] == lid]
            if not hit:
                continue
            for j, m in enumerate(hit):
                pad = 6 * j
                axi.add_patch(Rectangle((b.box_cx - b.box_w / 2 - pad, b.box_cy - b.box_h / 2 - pad),
                                        b.box_w + 2 * pad, b.box_h + 2 * pad, fill=False,
                                        ec=METHOD_COLOR[m], lw=2.4))
        axi.set_title('FRONT camera: outlined = lead chosen by each method', fontsize=9, color=INK2, loc='left')
    axi.axis('off')
    fig.suptitle(seg.name, fontsize=9, color=INK2, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'snapshot_{seg.name}_f{k:03d}.png'), dpi=100, facecolor=SURF)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('out')
    ap.add_argument('segments', nargs='*')
    ap.add_argument('--camera', default='FRONT', help="only objects most visible in this camera, or 'all'")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    camera = None if a.camera == 'all' else a.camera
    segs = a.segments or sorted(d for d in os.listdir(a.root) if os.path.isdir(os.path.join(a.root, d)))
    frames = []
    for name in segs:
        seg = Segment(a.root, name)
        df, lmap, ego_lanes, ego_s = process(seg, camera)
        frames.append(df)
        plot_ttc_methods(df, name, a.out)
        diff = df.index[(df['lead_map'].fillna('-') != df['lead_path'].fillna('-')) |
                        (df['lead_map'].fillna('-') != df['lead_straight'].fillna('-'))]
        show = [diff[len(diff) // 2]] if len(diff) else []
        if df['TTC'].notna().any():
            show.append(int(df['TTC'].replace(np.inf, np.nan).fillna(1e9).idxmin()))
        for k in dict.fromkeys(show):
            plot_snapshot(seg, df, int(k), lmap, ego_lanes, ego_s, a.out, a.camera)
        print(f'{name}: {len(df)} frames, ego on a mapped lane in {df["ego_lane"].notna().mean():.0%} of frames, '
              f'{len(diff)} frames where the methods pick different leads')
    fr = pd.concat(frames, ignore_index=True)
    fr.to_csv(os.path.join(a.out, 'frame_ssm.csv'), index=False)
    summ = fr.groupby('segment').apply(summarize, include_groups=False)
    summ.to_csv(os.path.join(a.out, 'scene_summary.csv'))
    pd.set_option('display.width', 200)
    print(summ.round(3).T.to_string())
    with open(os.path.join(a.out, 'params.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(f'{k} = {v}' for k, v in PARAMS.items()) + f'\ncamera = {a.camera}\n')


if __name__ == '__main__':
    main()
