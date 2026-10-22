"""Per-frame SSMs for Waymo segments extracted from v1.4.3, comparing three lead-vehicle pairing methods.

Usage:
    python -I waymo_ssm_frames.py <extracted_root> <out_dir> [--cameras FRONT,FRONT_LEFT,...|none] [segment ...]

Objects count only if they appear in at least one of the given camera images (the views the VLM sees);
default = all five Waymo cameras. 'none' keeps every labelled object, including ones no camera sees.

Pairing methods (wod_common.py)
    straight : straight corridor in the vehicle frame (|lateral| < 1.75 m)
    path     : corridor along the path the ego actually drives next
    map      : lane route from the HD map (falls back to `path` when the ego is not on a mapped lane)
`path` and `map` also require the lead's box to overlap the ego sideways, measured across the route
(wod_common.lateral_overlap, decided 2026-10-21 and 2026-10-22).
Lead-vehicle SSMs are computed with the `map` pairing; 2D metrics use all objects. Frames where the ego or the lead
moves backwards are flagged (reversing_ego, reversing_lead) and left out of the lead-vehicle SSMs in the analysis.

Lead-vehicle SSMs follow the traffic-safety convention (textbook TTC): speeds are the magnitudes of the
two velocities and the closing speed is ego speed - lead speed; accelerations are the rates of change of
those speeds (wod_common.speed_and_accel). 2D metrics use the full relative velocity vector.
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
import wod_common as wc  # noqa: E402
from wod_common import (Segment, LaneMap, lead_straight, lead_path, lead_map, wrap,  # noqa: E402
                        LEAD_TYPES)

sys.stdout.reconfigure(encoding='utf-8')

# ---- parameters the user has to choose (all decided 2026-10-15) ----
PARAMS = dict(
    tau=1.26,          # reaction time (s): PICUD. SHRP2 crash/near-crash brake reaction time,
                       # mean 1.255 s (NCHRP Research Report 1081, 2024)
    a_emerg=6.7,       # emergency braking deceleration (m/s^2): PICUD, PSD. SHRP2 per-event maximum
                       # deceleration, mean 22.0 ft/s^2 = 6.7 m/s^2 (NCHRP Research Report 1081, 2024)
    thw_v_min=0.1,     # THW = inf when the ego is slower than this (m/s): stopped, d / v would only be noise
)
TTC_PLOT_MAX = 10.0    # figures only: TTC above this (or inf) is drawn at this value
METHODS = ('straight', 'path', 'map')
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
METHOD_COLOR = {'map': BLUE, 'path': AQUA, 'straight': ORANGE}
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'

EGO_BOX = None


def use_ego(front, rear, half_w):
    """Set the ego geometry for the lead gap and the 2D metrics.
    footprint: the real car, read per segment from the bumper/side lidar mounting positions
               (front +4.070, rear -1.154, half width 1.025 m on the demo segments);
    camera:    the ego as a point at the FRONT camera, i.e. distances measured from where the image is taken.
    Called for every segment before its metrics are computed; there is no default."""
    global EGO_BOX
    wc.set_ego_geometry(front, rear, half_w)
    EGO_BOX = np.array([[front, half_w], [front, -half_w], [rear, -half_w], [rear, half_w]])


# ------------------------------------------------------------------ 2D geometry
def rect(cx, cy, L, W, h):
    c, s = np.cos(h), np.sin(h)
    dx = np.array([1, 1, -1, -1]) * L / 2
    dy = np.array([1, -1, -1, 1]) * W / 2
    return np.stack([cx + dx * c - dy * s, cy + dx * s + dy * c], axis=-1)


def axes_of(q):
    e1, e2 = q[1] - q[0], q[3] - q[0]
    return [e1 / np.linalg.norm(e1), e2 / np.linalg.norm(e2)]


def ray_time(p, u, poly):
    """Smallest s >= 0 for which p + s*u lies on an edge of the polygon; inf if the ray misses it."""
    best = np.inf
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        e, w = b - a, a - p
        den = e[0] * u[1] - e[1] * u[0]
        if abs(den) < 1e-12:
            continue
        s = (e[0] * w[1] - e[1] * w[0]) / den
        r = (u[0] * w[1] - u[1] * w[0]) / den
        if s >= 0 and 0 <= r <= 1:
            best = min(best, s)
    return best


def ttc_2d(obj, rel_v):
    """2D-TTC: first time the object box, moving at rel_v relative to the ego, touches the ego box. Exact, with no
    time horizon or step (decided 2026-10-16): with constant velocities the first contact is a corner of one box
    reaching an edge of the other, so the answer is the smallest corner-ray time over both boxes (the definition of
    Jiao's Two-Dimensional-Time-To-Collision). 0 if the boxes overlap now, inf if they never touch."""
    if poly_dist(EGO_BOX, obj) == 0:
        return 0.0
    if np.hypot(*rel_v) < 1e-12:
        return np.inf
    return min(min(ray_time(p, rel_v, EGO_BOX) for p in obj),      # object corners moving into the ego
               min(ray_time(q, -rel_v, obj) for q in EGO_BOX))     # ego corners, seen from the object


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


# m/s: a road user slower than this is not "moving" (no heading line is drawn from it in the ACT collision-course
# check). Set above the speed noise of static boxes in the labels (demo maxima: vehicles 0.324, pedestrians
# 0.206 m/s): pedestrians 0.3 m/s, since slow walkers move at 0.3-0.5 m/s; vehicles, cyclists and the ego 0.5 m/s
# (decided 2026-10-16)
V_MOVING = {'vehicle': 0.5, 'cyclist': 0.5, 'pedestrian': 0.3}
V_MOVING_EGO = 0.5


def ray_hits(p, u, poly):
    """Does the ray p + s*u (s >= 0) cross an edge of the polygon?"""
    return bool(np.isfinite(ray_time(p, u, poly)))


def on_collision_course(obj, obj_heading, obj_moving, ego_moving):
    """ACT paper's sufficient condition (Venthuruthiyil & Chunchu 2022, Sec. 3.1.3, Eq. 15): a line from the
    approaching road user's closest corner, along its heading, must pass through the other one's body.
    Checked from both sides (the ego heading is +x in the vehicle frame)."""
    d = np.linalg.norm(EGO_BOX[:, None, :] - obj[None, :, :], axis=-1)
    ie, io = np.unravel_index(d.argmin(), d.shape)           # closest corner pair (paper Eq. 7)
    if ego_moving and ray_hits(EGO_BOX[ie], np.array([1.0, 0.0]), obj):
        return True
    return bool(obj_moving and ray_hits(obj[io], np.array([np.cos(obj_heading), np.sin(obj_heading)]), EGO_BOX))


def closest_points(A, B):
    """Shortest distance between two disjoint convex polygons and its end points (on A, on B):
    the minimum over all corner-to-edge pairs."""
    best = (np.inf, None, None)
    for P, Q, p_on_a in ((A, B, True), (B, A, False)):
        for p in P:
            for a, b in zip(Q, np.roll(Q, -1, axis=0)):
                ab = b - a
                q = a + np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1) * ab
                d = np.linalg.norm(p - q)
                if d < best[0]:
                    best = (d, p, q) if p_on_a else (d, q, p)
    return best


def act(obj, rel_v, obj_heading, obj_moving, ego_moving):
    """Anticipated Collision Time (Venthuruthiyil & Chunchu 2022): shortest distance / closing-in rate; inf when
    the two are not closing in or not on a collision course. The closing-in rate is the relative velocity
    projected on the shortest-distance direction: the paper's Eq. 4 with the velocity term only (acceleration and
    yaw rate taken as 0, as in Jiao's SSMsOnPlane implementation), evaluated analytically (decided 2026-10-16)."""
    d0 = poly_dist(EGO_BOX, obj)
    if d0 == 0:
        return 0.0, 0.0
    d, pe, po = closest_points(EGO_BOX, obj)
    closing = -float(np.dot(rel_v, (po - pe) / d))      # > 0 when the object moves towards the ego
    if closing <= 1e-6 or not on_collision_course(obj, obj_heading, obj_moving, ego_moving):
        return np.inf, d0
    return d0 / closing, d0


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
    out = dict(gap=g, v_ego=vf, v_lead=vl, a_ego=af, a_lead=al, closing=closing,
               TTC=g / closing if closing > 0 else np.inf,
               THW=g / vf if vf > p['thw_v_min'] else np.inf,
               DRAC=closing ** 2 / (2 * g) if closing > 0 else 0.0)
    out['MTTC'] = m = mttc(g, closing, af - al)
    out['CI'] = (((vf + af * m) ** 2 - (vl + al * m) ** 2) / 2 / m) if np.isfinite(m) and m > 0 else 0.0
    stop_f = vf * p['tau'] + vf ** 2 / (2 * p['a_emerg'])
    out['PICUD'] = vl ** 2 / (2 * p['a_emerg']) + g - stop_f
    # PSD (Allen et al. 1978), car-following form: remaining distance to the lead's rear / stopping distance
    # v^2 / (2 a); < 1 means the ego cannot stop in time. a only scales PSD, so it does not change rankings
    out['PSD'] = g / (vf ** 2 / (2 * p['a_emerg'])) if vf > p['thw_v_min'] else np.inf
    return out


LEAD_COLS = ['gap', 'v_ego', 'v_lead', 'a_ego', 'a_lead', 'closing', 'TTC', 'THW', 'DRAC', 'MTTC', 'CI', 'PICUD',
             'PSD']


# ------------------------------------------------------------------ per segment
def process(seg, cameras):
    lmap = LaneMap(seg.map)
    n = len(seg.frames)
    if lmap.ok:
        ego_lanes, ego_s = lmap.match_sequence(seg.ego_map[:, :2], seg.yaw)
    else:
        ego_lanes, ego_s = [None] * n, np.full(n, np.nan)
    rows, picks = [], []
    # speeds and accelerations are Savitzky-Golay estimates (Waymo labels and our ego alike); within half a window
    # (9 frames = 0.9 s) of the start or end of a track they come from one edge fit and are inaccurate, so such frames
    # are flagged (edge_ego: ego near the segment start/end; edge_lead: lead near its track start/end) and left out
    # of the main analysis (decided 2026-10-16)
    half = wc.KIN_WINDOW // 2
    span = seg.objects.groupby('id')['frame_idx'].agg(['min', 'max'])
    for k in range(n):
        o_all = seg.objects_at(k, cameras=cameras, types=['vehicle', 'pedestrian', 'cyclist'])
        o_rng = o_all[wc.box_nearest_dist(o_all['cx'], o_all['cy'], o_all['length'], o_all['width'],
                                          o_all['heading']) <= wc.LABEL_RANGE]
        o_lead = o_rng[o_rng['type'].isin(LEAD_TYPES)].reset_index(drop=True)
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
        row['edge_ego'] = bool(k < half or k > n - 1 - half)
        lo, hi = span.loc[row['lead_map']] if r else (k, k)
        row['edge_lead'] = bool(r) and bool(k - lo < half or hi - k < half)
        row['edge_frame'] = row['edge_ego'] or row['edge_lead']
        row['map_fallback'] = bool(r.get('fallback')) if r else (ego_lanes[k] is None)
        # the lead-vehicle SSMs use speed magnitudes, so they are wrong while the ego or the lead reverses: flagged
        # and left out of those SSMs in the analysis, like edge frames (decided 2026-10-21)
        row['reversing_ego'] = wc.moving_backwards(seg.ego_v[k], seg.yaw[k])
        row['reversing_lead'] = bool(r['lead_reversing']) if r else False
        row.update(lead_ssms(r['gap'], r['v_ego'], r['v_lead'], r['a_ego'], r['a_lead']) if r
                   else {c: np.nan for c in LEAD_COLS})

        # 2D metrics over all objects (vehicle frame, relative velocity)
        row.update(TTC2D=np.inf, TTC2D_obj=None, ACT=np.inf, ACT_obj=None, min_dist=np.inf, min_dist_obj=None)
        if len(o_rng):
            _, _, v_g, _ = seg.to_map(k, o_rng)
            ego_moving = np.hypot(*ego_v_veh) >= V_MOVING_EGO
            cols = ['id', 'type', 'cx', 'cy', 'length', 'width', 'heading']
            for (oid, typ, cx, cy, L, W, h), vg in zip(o_rng[cols].itertuples(index=False), v_g):
                q = rect(cx, cy, L, W, h)
                v_obj = R[:2, :2].T @ vg
                rel = v_obj - ego_v_veh
                a_val, d0 = act(q, rel, h, np.hypot(*v_obj) >= V_MOVING[typ], ego_moving)
                if d0 < row['min_dist']:
                    row['min_dist'], row['min_dist_obj'] = d0, oid
                if a_val < row['ACT']:
                    row['ACT'], row['ACT_obj'] = a_val, oid
                t2 = ttc_2d(q, rel)
                if t2 < row['TTC2D']:
                    row['TTC2D'], row['TTC2D_obj'] = t2, oid
        rows.append(row)
        picks.append(res)
    return pd.DataFrame(rows), lmap, ego_lanes, ego_s


def summarize(g):
    both = lambda a, b: (g[f'lead_{a}'].fillna('-') == g[f'lead_{b}'].fillna('-')).mean()
    ok = g[~g['edge_frame']]          # metric extremes over the frames used in the analysis
    return pd.Series({
        'frames': len(g), 'edge_frames': int(g['edge_frame'].sum()), 'ego_speed_mean': g['ego_speed'].mean(),
        'ego_on_mapped_lane_share': g['ego_lane'].notna().mean(),
        **{f'frames_with_lead_{m}': int(g[f'lead_{m}'].notna().sum()) for m in METHODS},
        'same_lead_map_vs_path': both('map', 'path'), 'same_lead_map_vs_straight': both('map', 'straight'),
        'min_TTC': ok['TTC'].min(), 'min_THW': ok['THW'].min(), 'max_DRAC': ok['DRAC'].max(),
        'min_PICUD': ok['PICUD'].min(), 'min_PSD': ok['PSD'].min(),
        'min_TTC2D': ok['TTC2D'].min(), 'min_ACT': ok['ACT'].min(), 'min_dist': ok['min_dist'].min(),
    })


# ------------------------------------------------------------------ figures
def style(ax):
    ax.set_facecolor(SURF)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID)


def plot_ttc_methods(df, seg_name, out):
    cap = TTC_PLOT_MAX
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


def plot_snapshot(seg, df, k, lmap, ego_lanes, ego_s, out, cameras):
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
    o = seg.objects_at(k, cameras=cameras, types=['vehicle', 'pedestrian', 'cyclist'])
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
    f_, r_, w_ = wc.footprint_from_lidars(seg.context)       # always draw the real car outline
    ax.add_patch(MplPolygon(rect(ex + np.cos(seg.yaw[k]) * (f_ + r_) / 2,
                                 ey + np.sin(seg.yaw[k]) * (f_ + r_) / 2,
                                 f_ - r_, 2 * w_, seg.yaw[k]),
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
    ap.add_argument('--ego', choices=['footprint', 'camera'], default='footprint',
                    help='footprint = real car size; camera = ego as a point at the FRONT camera')
    ap.add_argument('--cameras', default='FRONT,FRONT_LEFT,FRONT_RIGHT,SIDE_LEFT,SIDE_RIGHT',
                    help="objects must appear in one of these camera images; 'none' = no camera filter")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cameras = None if a.cameras == 'none' else a.cameras.split(',')
    segs = a.segments or sorted(d for d in os.listdir(a.root) if os.path.isdir(os.path.join(a.root, d)))
    frames, ego_lines, label_lines = [], [], []
    for name in segs:
        seg = Segment(a.root, name)
        if a.ego == 'camera':
            cam_x, _ = wc.camera_position(seg.context, 'FRONT')
            use_ego(cam_x, cam_x - 0.02, 0.01)      # a 2 cm box standing in for a point
        else:
            use_ego(*wc.footprint_from_lidars(seg.context))   # bumper and side lidar positions
        ego_lines.append(f'ego {name} = front {wc.EGO_X_FRONT:.3f} m, rear {wc.EGO_X_REAR:.3f} m, '
                         f'half width {wc.EGO_HALF_W:.3f} m')
        # check that the labels still end at LABEL_RANGE (nearest point of the box) in this segment
        o = seg.objects[seg.objects['type'].isin(['vehicle', 'pedestrian', 'cyclist'])]
        edge = float(wc.box_nearest_dist(o['cx'], o['cy'], o['length'], o['width'], o['heading']).max()) \
            if len(o) else float('nan')
        label_lines.append(f'labels {name} = farthest box nearest point {edge:.3f} m')
        if edge > wc.LABEL_RANGE:
            print(f'WARNING {name}: labelled boxes reach {edge:.3f} m, beyond LABEL_RANGE = {wc.LABEL_RANGE} m; '
                  f'those boxes are left out')
        df, lmap, ego_lanes, ego_s = process(seg, cameras)
        frames.append(df)
        plot_ttc_methods(df, name, a.out)
        diff = df.index[(df['lead_map'].fillna('-') != df['lead_path'].fillna('-')) |
                        (df['lead_map'].fillna('-') != df['lead_straight'].fillna('-'))]
        show = [diff[len(diff) // 2]] if len(diff) else []
        if df['TTC'].notna().any():
            show.append(int(df['TTC'].replace(np.inf, np.nan).fillna(1e9).idxmin()))
        for k in dict.fromkeys(show):
            plot_snapshot(seg, df, int(k), lmap, ego_lanes, ego_s, a.out, cameras)
        print(f'{name}: {len(df)} frames, ego on a mapped lane in {df["ego_lane"].notna().mean():.0%} of frames, '
              f'{len(diff)} frames where the methods pick different leads')
    fr = pd.concat(frames, ignore_index=True)
    fr.to_csv(os.path.join(a.out, 'frame_ssm.csv'), index=False)
    summ = fr.groupby('segment').apply(summarize, include_groups=False)
    summ.to_csv(os.path.join(a.out, 'scene_summary.csv'))
    pd.set_option('display.width', 200)
    print(summ.round(3).T.to_string())
    with open(os.path.join(a.out, 'params.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(f'{k} = {v}' for k, v in PARAMS.items()) + f'\ncameras = {a.cameras}\n'
                f'label_range = {wc.LABEL_RANGE} m (nearest point of the box)\n'
                f'v_moving = {V_MOVING}, ego {V_MOVING_EGO} m/s\n'
                f'lead (map, path) = same lane (|d| < {wc.LANE_HALF} m), same direction, ahead, nearest, and overlapping '
                f'the ego across the route (lead: four corners; ego: box centre +- half width; no margin)\n'
                f'reversing = speed >= {wc.REVERSING_MIN_SPEED} m/s and travelling more than 90 deg from the heading\n'
                f'ego = {a.ego}\n' + '\n'.join(ego_lines) + '\n' + '\n'.join(label_lines) + '\n')


if __name__ == '__main__':
    main()
