"""Acceptance pictures for the lateral-overlap lead rule (user, 2026-10-21, step 5).
Frames (seed 20261021, one frame per segment as far as possible):
  A: 10 random frames whose old map lead had TTC < 3 s (not an edge frame in the old run) and is no longer the lead
     under the new rule (removed, or replaced by another vehicle);
  B: new-rule frames with TTC < 3 s (not edge, ego and lead not reversing): the 5 smallest TTC + 5 random.
For every frame two pictures:
  bev_only/     bird's-eye view only (map, lane route, ego path, objects, the old lead dashed red, the new leads by
                method) with the lateral intervals of the ego and of the old lead across the route;
  with_camera/  the same next to the FRONT camera image with the leads outlined, like the snapshot figures of
                waymo_ssm_frames.py. Not opened by Claude (the Waymo licence question is open).
Also writes frames.csv (the chosen frames and their numbers).

usage: python -I acceptance_snapshots.py <old results root> <new results root> <extracted root of the workspace> <out>
       (results roots hold <split>/ssm/frame_ssm.csv; the extracted root holds <split>/<segment>/)
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from PIL import Image

sys.path.insert(0, os.path.expanduser('~/benchmark-claude-remote/code'))
import wod_common as wc  # noqa: E402
import waymo_ssm_frames as wsf  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
SEED = 20261021
CAMERAS = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
OLD = '#d62728'


def load(root, split):
    return pd.read_csv(os.path.join(root, split, 'ssm', 'frame_ssm.csv'),
                       dtype={'lead_map': str, 'lead_path': str, 'lead_straight': str})


def pick_random(cand, n, rng, used):
    """n random frames, one per segment, from segments not in `used` first."""
    segs = [s for s in cand['segment'].unique() if s not in used]
    rng.shuffle(segs)
    rows = []
    for s in segs[:n]:
        g = cand[cand['segment'] == s]
        rows.append(g.iloc[rng.integers(len(g))])
    if len(rows) < n:                                         # not enough segments: allow a second frame per segment
        rest = cand.drop(index=[r.name for r in rows])
        rows += [rest.loc[i] for i in rng.choice(rest.index, n - len(rows), replace=False)]
    return pd.DataFrame(rows)


def choose(old_root, new_root):
    parts = []
    for split in ('validation', 'training'):
        o, n = load(old_root, split), load(new_root, split)
        if not o[['segment', 'frame_idx']].equals(n[['segment', 'frame_idx']]):
            sys.exit(f'{split}: old and new frames differ')
        d = n[['segment', 'frame_idx', 'lead_map', 'lead_path', 'lead_straight', 'TTC', 'v_lead', 'ego_speed', 'gap',
               'edge_frame', 'reversing_ego', 'reversing_lead']].copy()
        d['old_lead'], d['old_TTC'], d['old_v_lead'], d['old_gap'], d['old_edge'] = \
            o['lead_map'], o['TTC'], o['v_lead'], o['gap'], o['edge_frame']
        d['split'] = split
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    rng = np.random.default_rng(SEED)
    a = d[~d['old_edge'] & (d['old_TTC'] < 3) & (d['lead_map'].fillna('') != d['old_lead'])]
    A = pick_random(a, 10, rng, set())
    A['group'] = 'A_removed'
    b = d[~d['edge_frame'] & ~d['reversing_ego'] & ~d['reversing_lead'] & (d['TTC'] < 3)].sort_values('TTC')
    small, used = [], set()
    for _, r in b.iterrows():
        if r['segment'] not in used:
            small.append(r)
            used.add(r['segment'])
        if len(small) == 5:
            break
    B1 = pd.DataFrame(small)
    B1['group'] = 'B_smallest'
    B2 = pick_random(b.drop(index=B1.index), 5, rng, used)
    B2['group'] = 'B_random'
    print(f'pool A (old TTC < 3 s, lead no longer the lead): {len(a)} frames in {a["segment"].nunique()} segments; '
          f'pool B (new TTC < 3 s): {len(b)} frames in {b["segment"].nunique()} segments')
    return pd.concat([A, B1, B2], ignore_index=True)


def route_of(seg, k, lmap, ego_lanes, ego_s):
    if ego_lanes[k] is not None:
        future = [l for l in ego_lanes[k:] if l is not None]
        return lmap.route(ego_lanes[k], ego_s[k], future)[1], 'lane route'
    return wc.ego_future_path(seg, k), 'ego path (no mapped lane)'


def lateral(seg, k, xy, o_row):
    ext = wc.extend_back(xy, 10.0)
    s_ext = wc.cumlen(ext)
    _, de, _ = wc.project(ext, s_ext, wc.ego_corners_map(seg, k))
    P, hdg, _, _ = seg.to_map(k, o_row)
    _, dc, _ = wc.project(ext, s_ext, wc.box_corners(P[:, 0], P[:, 1], o_row['length'], o_row['width'], hdg)[0])
    return (de.min(), de.max()), (dc.min(), dc.max())


def draw(seg, k, r, lmap, ego_lanes, ego_s, path_bev, path_cam):
    xy, rname = route_of(seg, k, lmap, ego_lanes, ego_s)
    o = seg.objects_at(k, cameras=CAMERAS, types=['vehicle', 'pedestrian', 'cyclist'])
    new = {m: r[f'lead_{m}'] if isinstance(r[f'lead_{m}'], str) else None for m in ('map', 'path', 'straight')}
    old = r['old_lead'] if isinstance(r['old_lead'], str) else None
    info = []
    for lab, oid in (('old lead', old if old != new['map'] else None), ('new map lead', new['map'])):
        orow = seg.objects[(seg.objects['frame_idx'] == k) & (seg.objects['id'] == oid)] if oid else []
        if len(orow):
            (e0, e1), (a0, a1) = lateral(seg, k, xy, orow)
            ov = min(e1, a1) - max(e0, a0)
            info.append(f'{lab} [{a0:.2f}, {a1:.2f}] m, ' + (f'overlap {ov:.2f} m' if ov > 0 else f'apart by {-ov:.2f} m'))
    if info:
        _, _, ph = wc.project(xy, wc.cumlen(xy), seg.ego_map[k:k + 1, :2])
        info = (f'across the route: ego [{e0:.2f}, {e1:.2f}] m (heading {np.degrees(abs(wc.wrap(seg.yaw[k] - ph[0]))):.1f} '
                f'deg off the route); ' + '; '.join(info))
    else:
        info = ''
    title = (f'{r["group"]}  {r["split"]} {seg.name}  frame {k}  t = {seg.t[k]:.1f} s  ego {r["ego_speed"]:.2f} m/s\n'
             f'old lead {old[:6] if old else "-"}: TTC {r["old_TTC"]:.3f} s, gap {r["old_gap"]:.2f} m, speed '
             f'{r["old_v_lead"]:.2f} m/s  |  new map lead {new["map"][:6] if new["map"] else "-"}: TTC {r["TTC"]:.3f} s'
             f'  |  path {new["path"][:6] if new["path"] else "-"}, straight {new["straight"][:6] if new["straight"] else "-"}'
             + (f'\n{info}' if info else ''))
    for with_cam, path in ((False, path_bev), (True, path_cam)):
        if with_cam:
            fig, (ax, axi) = plt.subplots(1, 2, figsize=(15, 6.6), facecolor=wsf.SURF,
                                          gridspec_kw={'width_ratios': [1, 1.45]})
        else:
            fig, ax = plt.subplots(figsize=(8, 8.3), facecolor=wsf.SURF)
        wsf.style(ax)
        for m_ in seg.map.itertuples():
            if m_.kind == 'lane':
                ax.plot(m_.xs, m_.ys, color='#c3c2b7', lw=0.8, zorder=1)
            elif m_.kind == 'road_edge':
                ax.plot(m_.xs, m_.ys, color=wsf.INK2, lw=1.3, zorder=2)
        ax.plot(xy[:, 0], xy[:, 1], color=wsf.BLUE, lw=5, alpha=0.25, zorder=3, label=rname)
        ax.plot(seg.ego_map[k:, 0], seg.ego_map[k:, 1], color=wsf.AQUA, lw=1.4, ls='--', zorder=3,
                label='ego path (rest of segment)')
        if len(o):
            P, H, _, _ = seg.to_map(k, o)
            for (x, y), h, L, W, oid in zip(P, H, o['length'], o['width'], o['id']):
                ax.add_patch(MplPolygon(wsf.rect(x, y, L, W, h), closed=True, fc='#d9d8d2', ec=wsf.MUTED, lw=0.6,
                                        zorder=4))
                for j, m in enumerate(('map', 'path', 'straight')):
                    if new[m] == oid:
                        g = 0.6 + 0.5 * j
                        ax.add_patch(MplPolygon(wsf.rect(x, y, L + g, W + g, h), closed=True, fill=False,
                                                ec=wsf.METHOD_COLOR[m], lw=2.2, zorder=5 + j))
                if old == oid and old != new['map']:
                    ax.add_patch(MplPolygon(wsf.rect(x, y, L + 2.4, W + 2.4, h), closed=True, fill=False, ec=OLD,
                                            lw=2.0, ls='--', zorder=9))
        ax.add_patch(MplPolygon(wc.ego_corners_map(seg, k), closed=True, fc=wsf.BLUE, ec=wsf.SURF, lw=1, zorder=8))
        ex, ey = seg.ego_map[k, :2]
        span = 35
        cx, cy = ex + np.cos(seg.yaw[k]) * span * 0.55, ey + np.sin(seg.yaw[k]) * span * 0.55
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.set_aspect('equal')
        for m in ('map', 'path', 'straight'):
            ax.plot([], [], color=wsf.METHOD_COLOR[m], lw=2.2, label=f'new lead by {m}')
        ax.plot([], [], color=OLD, lw=2.0, ls='--', label='old map lead (no longer the lead)')
        ax.legend(frameon=False, fontsize=7, loc='lower left')
        if with_cam:
            img = os.path.join(seg.dir, 'images', 'FRONT', f'{k:03d}_{int(seg.ts[k])}.jpg')
            if os.path.exists(img):
                axi.imshow(Image.open(img))
                pb = pd.read_parquet(os.path.join(seg.dir, 'projected_boxes.parquet'))
                pb = pb[(pb['frame_idx'] == k) & (pb['camera'] == 'FRONT')]
                for b in pb.itertuples():
                    lid = b.id[:-len('_FRONT')] if isinstance(b.id, str) and b.id.endswith('_FRONT') else b.id
                    hits = [m for m in ('map', 'path', 'straight') if new[m] == lid]
                    for j, m in enumerate(hits):
                        p = 6 * j
                        axi.add_patch(Rectangle((b.box_cx - b.box_w / 2 - p, b.box_cy - b.box_h / 2 - p),
                                                b.box_w + 2 * p, b.box_h + 2 * p, fill=False,
                                                ec=wsf.METHOD_COLOR[m], lw=2.4))
                    if old == lid and old != new['map']:
                        p = 6 * len(hits) + 4
                        axi.add_patch(Rectangle((b.box_cx - b.box_w / 2 - p, b.box_cy - b.box_h / 2 - p),
                                                b.box_w + 2 * p, b.box_h + 2 * p, fill=False, ec=OLD, lw=2.4, ls='--'))
                axi.set_title('FRONT camera: dashed red = old map lead, solid = new lead by method', fontsize=9,
                              color=wsf.INK2, loc='left')
            axi.axis('off')
        fig.suptitle(title, fontsize=8, color=wsf.INK2, x=0.01, ha='left')
        fig.tight_layout()
        fig.savefig(path, dpi=100, facecolor=wsf.SURF)
        plt.close(fig)


def main():
    old_root, new_root, ext_root, out = sys.argv[1:5]
    for sub in ('bev_only', 'with_camera'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    sel = choose(old_root, new_root)
    sel.insert(0, 'name', [f'{g[0]}{i + 1:02d}_{s[:8]}_f{k:03d}' for i, (g, s, k) in
                           enumerate(zip(sel['group'], sel['segment'], sel['frame_idx']))])
    sel.to_csv(os.path.join(out, 'frames.csv'), index=False)
    for (split, name), g in sel.groupby(['split', 'segment'], sort=False):
        seg = wc.Segment(os.path.join(ext_root, split), name)
        wc.set_ego_geometry(*wc.footprint_from_lidars(seg.context))
        wsf.use_ego(*wc.footprint_from_lidars(seg.context))
        lmap = wc.LaneMap(seg.map)
        ego_lanes, ego_s = lmap.match_sequence(seg.ego_map[:, :2], seg.yaw) if lmap.ok else ([None] * len(seg.frames), None)
        for _, r in g.iterrows():
            k = int(r['frame_idx'])
            draw(seg, k, r, lmap, ego_lanes, ego_s, os.path.join(out, 'bev_only', r['name'] + '.png'),
                 os.path.join(out, 'with_camera', r['name'] + '.png'))
            print(f'{r["name"]}: old TTC {r["old_TTC"]:.3f} -> new {r["TTC"]:.3f}')
    pd.set_option('display.width', 250)
    print(sel[['name', 'group', 'split', 'segment', 'frame_idx', 'old_lead', 'old_TTC', 'old_v_lead', 'lead_map', 'TTC',
               'v_lead', 'lead_path']].to_string(index=False, float_format=lambda x: f'{x:.3f}'))


if __name__ == '__main__':
    main()
