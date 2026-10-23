"""Follow-up of acceptance frames R02, R07, R10 (user, 2026-10-22). For one vehicle at frame k:
  - its track: labelled frames, fastest speed over the whole track and away from its ends (9 frames each side);
  - which cameras see it (projected 3D boxes);
  - when the ego drives past it (first later frame with the vehicle wholly behind the ego's rear bumper), the nearest
    distance between the two boxes up to then, and the side gap while they are alongside (their lengths overlap);
  - its vehicle type from the lidar segmentation (check_lidar_points.py per-track table);
  - where it sits across the road: offset from the nearest lane centre line, that lane's width from the map boundary
    lines, distance from its box to the nearest road edge; and its offset from the ego's lane route.
Then a camera sheet, one panel per second from 1 s before k to 1 s after the ego passes it (if it does; else to k+3 s),
up to the segment end. A panel uses FRONT if the vehicle has a box there, otherwise the camera where its box is largest,
and says which. Claude does not open the sheet.

usage: python -I followup_sheets.py <extracted split root> <lidar vehicle_type_per_track.csv> <segment> <frame> <id> <out.jpg>
"""
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser('~/benchmark-claude-remote/code'))
import wod_common as wc  # noqa: E402
import waymo_ssm_frames as wsf  # noqa: E402
from front_sequence import font, PANEL_W  # noqa: E402
from lead_conditions import lane_width  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
HALF = wc.KIN_WINDOW // 2


def seg_dist(p, A, B):
    AB = B - A
    t = np.clip(((p - A) * AB).sum(1) / np.maximum((AB ** 2).sum(1), 1e-12), 0, 1)
    return np.hypot(*(p - (A + t[:, None] * AB)).T)


def main():
    root, types_csv, name, k, oid, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6]
    seg = wc.Segment(root, name)
    fp = wc.footprint_from_lidars(seg.context)
    wc.set_ego_geometry(*fp)
    wsf.use_ego(*fp)
    n = len(seg.frames)
    tr = seg.objects[seg.objects['id'] == oid].sort_values('frame_idx').reset_index(drop=True)
    sp = np.hypot(tr['speed_x'], tr['speed_y']).to_numpy()
    lo, hi = int(tr['frame_idx'].min()), int(tr['frame_idx'].max())
    inner = ((tr['frame_idx'] - lo >= HALF) & (hi - tr['frame_idx'] >= HALF)).to_numpy()
    b = tr[tr['frame_idx'] == k].iloc[0]
    print(f'{name} frame {k}, vehicle {oid} ({b["type"]}, {b.length:.2f} x {b.width:.2f} m): labelled in frames {lo}-{hi} '
          f'of 0-{n - 1}; speed now {np.hypot(b.speed_x, b.speed_y):.3f} m/s')
    i = int(np.argmax(sp))
    print(f'  fastest over the whole track {sp[i]:.3f} m/s (frame {int(tr.frame_idx[i])}); away from its ends '
          + (f'{sp[inner].max():.3f} m/s (frame {int(tr.frame_idx[np.flatnonzero(inner)[np.argmax(sp[inner])]])})'
             if inner.any() else 'no frames (track too short)')
          + f'; frames faster than 0.5 m/s: {", ".join(str(int(f)) for f in tr.frame_idx[sp > 0.5]) or "none"}')
    pb = pd.read_parquet(os.path.join(seg.dir, 'projected_boxes.parquet'))
    pb = pb[pb['id'].astype(str).str.startswith(oid + '_')]
    cams = pb.groupby('camera')['frame_idx'].agg(['size', 'min', 'max'])
    print('  cameras with its box (frames, first-last): ' + ', '.join(f'{c} {r["size"]} ({r["min"]}-{r["max"]})'
                                                                     for c, r in cams.iterrows())
          + f'; at frame {k}: {sorted(pb.loc[pb.frame_idx == k, "camera"])}')
    # passing and clearances
    later = tr[tr['frame_idx'] > k]
    behind = later[later['cx'] + later['length'] / 2 < wc.EGO_X_REAR]
    f_pass = int(behind['frame_idx'].iloc[0]) if len(behind) else None
    rows = []
    for r in tr[tr['frame_idx'] >= k].itertuples():
        if f_pass is not None and r.frame_idx > f_pass:
            break
        q = wsf.rect(r.cx, r.cy, r.length, r.width, r.heading)
        along = (r.cx - r.length / 2 < wc.EGO_X_FRONT) and (r.cx + r.length / 2 > wc.EGO_X_REAR)
        rows.append(dict(frame=r.frame_idx, dist=wsf.poly_dist(wsf.EGO_BOX, q), along=along, cy=r.cy,
                         side=abs(r.cy) - wc.EGO_HALF_W - r.width / 2))
    d = pd.DataFrame(rows)
    j = d['dist'].idxmin()
    al = d[d['along']]
    print(f'  the ego drives past it: ' + (f'frame {f_pass} (+{(f_pass - k) / 10:.1f} s)' if f_pass is not None else
                                           'not within the segment')
          + f'; nearest distance between the boxes {d.at[j, "dist"]:.3f} m at frame {int(d.at[j, "frame"])} '
          f'(+{(d.at[j, "frame"] - k) / 10:.1f} s); '
          + (f'alongside in frames {int(al.frame.min())}-{int(al.frame.max())}, smallest gap between the boxes then '
             f'{al.dist.min():.3f} m (it is on the {"right" if al.cy.median() < 0 else "left"})' if len(al) else
             'never alongside'))
    t = pd.read_csv(types_csv)
    t = t[(t['segment'] == name) & (t['id'] == oid)]
    print('  lidar type: ' + (f'{t.seg_type.iloc[0]} (agreement {t.agreement.iloc[0]:.2f}, {t.labelled_frames.iloc[0]} '
                              f'labelled frames)' if len(t) else 'none (no lidar segmentation points inside it)'))
    # across the road at frame k
    P, hdg, _, _ = seg.to_map(k, tr[tr['frame_idx'] == k])
    lmap = wc.LaneMap(seg.map)
    best = min(((lane, wc.project(v['xy'], v['s'], P)[1][0]) for lane, v in lmap.lanes.items()), key=lambda x: abs(x[1]))
    w = lane_width(seg, lmap, [best[0]], P)
    m = seg.map[seg.map['kind'] == 'road_edge']
    corners = wc.box_corners(P[:, 0], P[:, 1], b.length, b.width, hdg)[0]
    edge = min(seg_dist(c, np.column_stack([r.xs, r.ys])[:-1], np.column_stack([r.xs, r.ys])[1:]).min()
               for r in m.itertuples() if len(r.xs) >= 2 for c in corners) if len(m) else np.nan
    el, es = lmap.match_sequence(seg.ego_map[:, :2], seg.yaw)
    route_d = np.nan
    if el[k] is not None:
        xy = lmap.route(el[k], es[k], [l for l in el[k:] if l is not None])[1]
        route_d = wc.project(xy, wc.cumlen(xy), P)[1][0]
    print(f'  across the road at frame {k}: nearest lane {int(best[0])}, centre offset {best[1]:.3f} m (+ = left of the '
          f'lane direction); that lane {w.get("width", float("nan")):.3f} m wide (left boundary '
          f'{w["left_boundaries"]}, right boundary {w["right_boundaries"]}); nearest road edge {edge:.3f} m from its box;'
          f' offset from the ego lane route {route_d:.3f} m')
    # sheet
    last = min(f_pass + 10, n - 1) if f_pass is not None else min(k + 30, n - 1)
    frames = [f for f in range(k - 10, last + 1, 10) if f >= 0]
    if frames[-1] != last:
        frames.append(last)
    ft, fs = font(26), font(20)
    panels = []
    for f in frames:
        here = pb[pb['frame_idx'] == f]
        if (here['camera'] == 'FRONT').any():
            cam = 'FRONT'
        elif len(here):
            cam = here.assign(a=here.box_w * here.box_h).sort_values('a').camera.iloc[-1]
        else:
            cam = 'FRONT'
        img = Image.open(os.path.join(seg.dir, 'images', cam, f'{f:03d}_{int(seg.ts[f])}.jpg')).convert('RGB')
        bx = here[here['camera'] == cam]
        if len(bx):
            r = bx.iloc[0]
            ImageDraw.Draw(img).rectangle([r.box_cx - r.box_w / 2, r.box_cy - r.box_h / 2, r.box_cx + r.box_w / 2,
                                           r.box_cy + r.box_h / 2], outline=(220, 30, 30), width=6)
        o = tr[tr['frame_idx'] == f]
        pos = (f'x {o.cx.iloc[0]:.1f} m, y {o.cy.iloc[0]:.1f} m, {np.hypot(o.speed_x.iloc[0], o.speed_y.iloc[0]):.2f} m/s'
               if len(o) else 'not labelled')
        cap = (f'{(f - k) / 10:+.1f} s (frame {f})  {cam}  ego {np.hypot(*seg.ego_v[f, :2]):.1f} m/s  vehicle {pos}'
               + ('' if len(bx) else '  [not visible in any camera]'))
        img = img.resize((PANEL_W, int(img.height * PANEL_W / img.width)))
        canvas = Image.new('RGB', (PANEL_W, 640 + 40), (252, 252, 251))
        canvas.paste(img, (0, 40))
        ImageDraw.Draw(canvas).text((8, 8), cap, fill=(30, 30, 30), font=fs)
        panels.append(canvas)
    cols, h = 2, max(p.height for p in panels)
    rows_n = (len(panels) + 1) // 2
    sheet = Image.new('RGB', (PANEL_W * cols, h * rows_n + 60), (252, 252, 251))
    ImageDraw.Draw(sheet).text((10, 12), f'{name} frame {k}: red box = vehicle {oid}; one panel per second, camera named '
                                         f'in each panel', fill=(30, 30, 30), font=ft)
    for i, p in enumerate(panels):
        sheet.paste(p, ((i % cols) * PANEL_W, 60 + (i // cols) * h))
    sheet.save(out, quality=88)
    print(f'  sheet: {out} (frames {frames})')


if __name__ == '__main__':
    main()
