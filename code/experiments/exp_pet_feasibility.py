"""Quick PET feasibility check on the demo (not the final implementation).
Cell-based 'same position' PET (FHWA SSAM): rasterise the ego and every labelled road user (world frame, 0.5 m cells,
frame times), record first/last occupancy per cell; PET per common cell = later first arrival - earlier last occupancy.
Keep encounters whose heading difference at the minimum-PET cell is >= 30 deg (crossing, not following).
Frames assigned under option A: from the first frame both are camera-visible until the later one arrives.
usage: python -I exp_pet_feasibility.py <repo_code_dir> <extracted_root>"""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
code, root = sys.argv[1:3]
sys.path.insert(0, code)
import wod_common as wc  # noqa: E402

CELL = 0.5
CAMS = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
SEGS = ['13184115878756336167_1354_000_1374_000', '14486517341017504003_3406_349_3426_349',
        '16229547658178627464_380_000_400_000']


def cells_of(cx, cy, L, W, h):
    """Grid cells whose centre lies inside the rectangle (world frame)."""
    c, s = np.cos(h), np.sin(h)
    r = 0.5 * (abs(L * c) + abs(W * s)), 0.5 * (abs(L * s) + abs(W * c))
    xs = np.arange(np.floor((cx - r[0]) / CELL), np.ceil((cx + r[0]) / CELL) + 1)
    ys = np.arange(np.floor((cy - r[1]) / CELL), np.ceil((cy + r[1]) / CELL) + 1)
    X, Y = np.meshgrid(xs, ys)
    px, py = (X + 0.5) * CELL - cx, (Y + 0.5) * CELL - cy
    inside = (np.abs(px * c + py * s) <= L / 2) & (np.abs(-px * s + py * c) <= W / 2)
    return list(zip(X[inside].astype(int), Y[inside].astype(int)))


def occupancy(track):
    """track: rows (t, cx, cy, L, W, heading) -> {cell: [t_first, t_last, heading_first]}"""
    occ = {}
    for t, cx, cy, L, W, h in track:
        for cell in cells_of(cx, cy, L, W, h):
            if cell in occ:
                occ[cell][1] = t
            else:
                occ[cell] = [t, t, h]
    return occ


rows, frames_with = [], 0
total_frames = 0
for name in SEGS:
    seg = wc.Segment(root, name)
    fr, rear, hw = wc.footprint_from_lidars(seg.context)
    n = len(seg.frames)
    total_frames += n
    t = seg.t
    yaw = seg.yaw
    pos = seg.pose[:, :2, 3]
    # ego footprint centre in the world: midpoint of the box along the heading
    off = (fr + rear) / 2
    ego_track = [(t[k], pos[k, 0] + off * np.cos(yaw[k]), pos[k, 1] + off * np.sin(yaw[k]), fr - rear, 2 * hw, yaw[k])
                 for k in range(n)]
    ego_occ = occupancy(ego_track)
    o = seg.objects[seg.objects['type'].isin(['vehicle', 'pedestrian', 'cyclist'])]
    k = o['frame_idx'].to_numpy()
    xyw = np.einsum('nij,nj->ni', seg.R[k], o[['cx', 'cy', 'cz']].to_numpy(float)) + seg.pose[k, :3, 3]
    o = o.assign(wx=xyw[:, 0], wy=xyw[:, 1], wh=wc.wrap(o['heading'].to_numpy(float) + yaw[k]), t=t[k])
    vis = {}
    for kk in range(n):
        for oid in seg.objects_at(kk, cameras=CAMS)['id']:
            vis.setdefault(oid, []).append(kk)
    assigned = np.full(n, np.inf)
    for oid, g in o.groupby('id'):
        if oid not in vis:
            continue
        occ = occupancy(g[['t', 'wx', 'wy', 'length', 'width', 'wh']].to_numpy())
        best = None
        for cell in occ.keys() & ego_occ.keys():
            e0, e1, eh = ego_occ[cell]
            o0, o1, oh = occ[cell]
            if e1 < o0:
                pet, later_arrival, dh = o0 - e1, o0, abs(wc.wrap(oh - eh))
            elif o1 < e0:
                pet, later_arrival, dh = e0 - o1, e0, abs(wc.wrap(oh - eh))
            else:
                pet, later_arrival, dh = 0.0, max(e0, o0), abs(wc.wrap(oh - eh))
            if best is None or pet < best[0]:
                best = (pet, later_arrival, dh)
        if best is None:
            continue
        pet, arr, dh = best
        crossing = np.degrees(dh) >= 30 and np.degrees(dh) <= 180
        first_vis = min(vis[oid])
        k_arr = int(np.searchsorted(t, arr))
        rows.append(dict(seg=name[:6], id=oid, type=g['type'].iloc[0], pet=pet, dh=np.degrees(dh), crossing=crossing,
                         frames_A=max(0, k_arr - first_vis)))
        if crossing:
            sl = slice(first_vis, k_arr)
            assigned[sl] = np.minimum(assigned[sl], pet)
    frames_with += int(np.isfinite(assigned).sum())
    print(f'{name[:6]}: frames with an option-A PET value: {int(np.isfinite(assigned).sum())} of {n}')
d = pd.DataFrame(rows)
print(f'ego-object pairs sharing at least one cell: {len(d)}; crossing (heading diff >= 30 deg): {d.crossing.sum()}')
print(d[d.crossing].sort_values('pet')[['seg', 'type', 'pet', 'dh', 'frames_A']].round(2).to_string(index=False))
print(f'same-direction pairs (< 30 deg): {(~d.crossing).sum()}, their PET median {d[~d.crossing].pet.median():.2f} s')
print(f'total frames with a PET value: {frames_with} of {total_frames}')
