"""Sanity checks on segments written by wod_v143_extract.py, plus a bird's-eye map plot per segment.

Usage: python -I check_v143_extract.py <extracted_root> <out_dir> [segment ...]

Checks
  1. map features: counts by kind, and that only the first frame carries them
  2. map_pose_offset size
  3. ego velocity reported with the FRONT image vs. velocity from pose differences
  4. frame of the labels' speed: angle between speed vector and box heading, read as global vs vehicle frame
  5. map alignment: distance from the ego to the nearest lane centre line, with and without the offset
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wod_common import Segment, LaneMap, wrap  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
ROOT, OUT = sys.argv[1], sys.argv[2]
SEGS = sys.argv[3:] or sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
os.makedirs(OUT, exist_ok=True)

INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'


def lane_distance(lmap, P, H):
    out = []
    for p, h in zip(P, H):
        c = lmap.candidates(p, h, max_d=10.0)
        out.append(min(abs(v[2]) for v in c.values()) if c else np.nan)
    return np.array(out)


rows = []
for seg_name in SEGS:
    seg = Segment(ROOT, seg_name)
    fr, m = seg.frames, seg.map
    kinds = m['kind'].value_counts().to_dict()
    map_frames = fr.index[fr['n_map_features'] > 0].tolist()

    off_norm = np.linalg.norm(seg.offset, axis=1)

    # 3. ego velocity: reported vs pose differences (global frame)
    pos = seg.pose[:, :3, 3]
    v_pose = np.gradient(pos, seg.t, axis=0)
    dv = np.linalg.norm(seg.ego_v[:, :2] - v_pose[:, :2], axis=1)

    # 4. speed frame
    o = seg.objects[(seg.objects['type'] == 'vehicle')]
    sp = np.hypot(o['speed_x'], o['speed_y'])
    o = o[sp > 2.0]
    ang_v = np.arctan2(o['speed_y'], o['speed_x']).to_numpy()
    yaw = seg.yaw[o['frame_idx'].to_numpy()]
    d_global = np.abs(wrap(ang_v - (o['heading'].to_numpy() + yaw)))
    d_vehicle = np.abs(wrap(ang_v - o['heading'].to_numpy()))

    # 5. map alignment
    lmap = LaneMap(m)
    if lmap.ok:
        d_with = lane_distance(lmap, seg.ego_map[:, :2], seg.yaw)
        d_without = lane_distance(lmap, pos[:, :2], seg.yaw)
    else:
        d_with = d_without = np.full(len(fr), np.nan)

    rows.append(dict(
        segment=seg_name, frames=len(fr), map_features=len(m), lanes=kinds.get('lane', 0),
        road_lines=kinds.get('road_line', 0), road_edges=kinds.get('road_edge', 0),
        crosswalks=kinds.get('crosswalk', 0), stop_signs=kinds.get('stop_sign', 0),
        frames_with_map=str(map_frames[:3]) + ('...' if len(map_frames) > 3 else ''),
        offset_max_m=off_norm.max(), offset_median_m=np.median(off_norm),
        ego_v_vs_pose_median_mps=np.median(dv), ego_speed_mean_mps=np.hypot(*seg.ego_v[:, :2].T).mean(),
        moving_vehicle_labels=len(o),
        speed_vs_heading_if_global_deg=np.degrees(np.median(d_global)) if len(o) else np.nan,
        speed_vs_heading_if_vehicle_deg=np.degrees(np.median(d_vehicle)) if len(o) else np.nan,
        ego_to_lane_with_offset_median_m=np.nanmedian(d_with),
        ego_to_lane_with_offset_p90_m=np.nanpercentile(d_with, 90) if np.isfinite(d_with).any() else np.nan,
        ego_to_lane_without_offset_median_m=np.nanmedian(d_without),
        ego_unmatched_frames=int(np.isnan(d_with).sum()),
    ))

    # ---- bird's-eye plot: map + ego path + objects at the middle frame ----
    k = len(fr) // 2
    fig, ax = plt.subplots(figsize=(9, 9), facecolor=SURF)
    ax.set_facecolor(SURF)
    for r in m.itertuples():
        xs, ys = np.asarray(r.xs), np.asarray(r.ys)
        if r.kind == 'lane':
            ax.plot(xs, ys, color='#c3c2b7', lw=0.8, zorder=1)
        elif r.kind == 'road_edge':
            ax.plot(xs, ys, color=INK2, lw=1.4, zorder=2)
        elif r.kind == 'road_line':
            ax.plot(xs, ys, color=MUTED, lw=0.6, ls=(0, (4, 3)), zorder=1)
        elif r.kind in ('crosswalk', 'speed_bump', 'driveway') and len(xs) >= 3:
            ax.add_patch(MplPolygon(np.column_stack([xs, ys]), closed=True, fc='#e1e0d9', ec='none', zorder=0))
    ax.plot(seg.ego_map[:, 0], seg.ego_map[:, 1], color=BLUE, lw=2.2, zorder=4, label='ego path (map frame)')
    oo = seg.objects_at(k, types=['vehicle', 'pedestrian', 'cyclist'])
    if len(oo):
        P, H, _, _ = seg.to_map(k, oo)
        for (x, y), h, L, W, t in zip(P, H, oo['length'], oo['width'], oo['type']):
            c, s = np.cos(h), np.sin(h)
            corners = np.array([[L / 2, W / 2], [L / 2, -W / 2], [-L / 2, -W / 2], [-L / 2, W / 2]])
            pts = corners @ np.array([[c, s], [-s, c]]) + [x, y]
            col = ORANGE if t == 'vehicle' else AQUA
            ax.add_patch(MplPolygon(pts, closed=True, fc=col, ec=SURF, lw=1, alpha=0.9, zorder=5))
    ex, ey = seg.ego_map[k, :2]
    ax.plot([ex], [ey], marker='o', ms=9, color=BLUE, mec=SURF, mew=2, zorder=6)
    R = 70
    ax.set_xlim(ex - R, ex + R); ax.set_ylim(ey - R, ey + R)
    ax.set_aspect('equal')
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_title(f'{seg_name}\nframe {k}: lanes (grey), road edges (dark), ego path (blue), '
                 f'vehicles (orange), pedestrians/cyclists (aqua)', fontsize=9, color=INK2, loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f'bev_{seg_name}.png'), dpi=110, facecolor=SURF)
    plt.close(fig)

res = pd.DataFrame(rows).set_index('segment').T
pd.set_option('display.width', 200)
print(res.to_string())
res.to_csv(os.path.join(OUT, 'extract_checks.csv'))
