"""Shared loaders, geometry and lead-vehicle pairing for the Waymo SSM pipeline.

All inputs come from the folders written by wod_v143_extract.py.

Coordinate frames
    vehicle frame : x forward, y left, origin = Waymo vehicle-frame origin (pose reference point)
    global frame  : world_from_vehicle @ vehicle-frame point
    map frame     : global + map_pose_offset of that frame (aligns the frame with the map features)
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ---- ego footprint in the vehicle frame. Checked against the lidar mounting positions in the calibration:
# front-bumper lidar x = +4.07 m, rear-bumper lidar x = -1.15 m, side lidars y = +-1.02 m,
# so the origin is at the rear axle and the car is about 5.2 m x 2.0 m ----
EGO_X_FRONT, EGO_X_REAR, EGO_HALF_W = 4.0, -1.1, 1.0
LANE_HALF = 1.75                     # same-lane test: |lateral offset| < half of a 3.5 m lane
LEAD_MAX_DHEADING = np.deg2rad(30)   # lead must travel roughly in the ego direction
LOOKAHEAD = 150.0                    # how far ahead (m) a lead vehicle is searched
LEAD_TYPES = ('vehicle', 'cyclist')
SPEED_FRAME = 'global'               # frame of the labels' speed/accel (checked by check_v143_extract.py)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


FOOTPRINT = (EGO_X_FRONT, EGO_X_REAR, EGO_HALF_W)


def set_ego_geometry(front, rear, half_w):
    """Change the ego footprint used by the lead-vehicle gap (front) and by 2D metrics (whole box)."""
    global EGO_X_FRONT, EGO_X_REAR, EGO_HALF_W
    EGO_X_FRONT, EGO_X_REAR, EGO_HALF_W = front, rear, half_w


def camera_position(context, camera='FRONT'):
    """(x, y) of a camera in the vehicle frame, from the calibration in context.json."""
    for c in context['camera_calibrations']:
        if c['name'] == camera:
            E = np.asarray(c['extrinsic'], float).reshape(4, 4)
            return float(E[0, 3]), float(E[1, 3])
    raise KeyError(camera)


# ------------------------------------------------------------------ loading
class Segment:
    def __init__(self, root, seg):
        self.dir = os.path.join(root, seg)
        self.name = seg
        with open(os.path.join(self.dir, 'context.json'), encoding='utf-8') as f:
            self.context = json.load(f)
        fr = pd.read_parquet(os.path.join(self.dir, 'frames.parquet')).sort_values('frame_idx')
        self.frames = fr.reset_index(drop=True)
        self.ts = fr['timestamp_micros'].to_numpy(np.int64)
        self.t = (self.ts - self.ts[0]) / 1e6
        self.pose = np.stack([np.asarray(p, float).reshape(4, 4) for p in fr['world_from_vehicle']])
        self.R = self.pose[:, :3, :3]
        self.yaw = np.arctan2(self.R[:, 1, 0], self.R[:, 0, 0])
        off = fr[['map_pose_offset_x', 'map_pose_offset_y', 'map_pose_offset_z']].to_numpy(float)
        self.offset = np.nan_to_num(off)
        self.ego_map = self.pose[:, :3, 3] + self.offset            # ego origin in the map frame
        v = fr[['ego_vx', 'ego_vy', 'ego_vz']].to_numpy(float)       # global frame
        self.ego_v = v
        dt = np.median(np.diff(self.t))
        win = min(11, len(v) - (1 - len(v) % 2))
        self.ego_a = savgol_filter(v, win, 2, deriv=1, delta=dt, axis=0) if win >= 5 else np.gradient(v, self.t, axis=0)
        self.objects = pd.read_parquet(os.path.join(self.dir, 'objects.parquet'))
        self.map = pd.read_parquet(os.path.join(self.dir, 'map.parquet'))
        pb = pd.read_parquet(os.path.join(self.dir, 'projected_boxes.parquet'), columns=['frame_idx', 'camera', 'id'])
        pb['laser_id'] = [i[:-(len(c) + 1)] if isinstance(i, str) and i.endswith('_' + c) else i
                          for i, c in zip(pb['id'], pb['camera'])]
        self._visible = pb[['frame_idx', 'camera', 'laser_id']]

    def objects_at(self, k, cameras=None, types=None):
        """Objects of frame k. `cameras`: keep only objects that appear (have a projected 3D box) in at
        least one of these camera images; None keeps every labelled object, including ones no camera sees."""
        o = self.objects[self.objects['frame_idx'] == k]
        if cameras:
            v = self._visible
            ids = set(v.loc[(v['frame_idx'] == k) & v['camera'].isin(cameras), 'laser_id'])
            o = o[o['id'].isin(ids)]
        if types:
            o = o[o['type'].isin(types)]
        return o

    def to_map(self, k, o):
        """Object centres, headings, velocities and accelerations in the map frame (2D)."""
        R, t = self.R[k], self.pose[k, :3, 3]
        c = o[['cx', 'cy', 'cz']].to_numpy(float)
        xy = (c @ R.T + t + self.offset[k])[:, :2]
        hdg = wrap(o['heading'].to_numpy(float) + self.yaw[k])
        v = o[['speed_x', 'speed_y', 'speed_z']].to_numpy(float)
        a = o[['accel_x', 'accel_y', 'accel_z']].to_numpy(float)
        if SPEED_FRAME == 'vehicle':
            v, a = v @ R.T, a @ R.T
        return xy, hdg, v[:, :2], a[:, :2]


# ------------------------------------------------------------------ polyline helpers
def cumlen(xy):
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]


def project(xy, s_cum, P):
    """Project points P (M,2) onto polyline xy (N,2): arc length s, signed lateral offset d (left +),
    and the polyline heading at the projection."""
    A, AB = xy[:-1], np.diff(xy, axis=0)
    L2 = np.maximum((AB ** 2).sum(1), 1e-12)
    AP = P[:, None, :] - A[None]
    t = np.clip((AP * AB[None]).sum(-1) / L2[None], 0, 1)
    Q = A[None] + t[..., None] * AB[None]
    dist = np.linalg.norm(P[:, None, :] - Q, axis=-1)
    j = dist.argmin(1)
    r = np.arange(len(P))
    s = s_cum[j] + t[r, j] * np.sqrt(L2[j])
    cross = AB[j, 0] * (P[:, 1] - A[j, 1]) - AB[j, 1] * (P[:, 0] - A[j, 0])
    return s, np.sign(cross) * dist[r, j], np.arctan2(AB[j, 1], AB[j, 0])


def heading_at(xy, s_cum, s):
    j = np.clip(np.searchsorted(s_cum, s) - 1, 0, len(xy) - 2)
    d = xy[j + 1] - xy[j]
    return np.arctan2(d[..., 1], d[..., 0])


def extend_straight(xy, length, baseline=10.0):
    """Append a straight piece along the end direction, measured over the last `baseline` metres
    (a single short segment gives a noisy direction that drifts sideways far ahead)."""
    back = np.hypot(*(xy - xy[-1]).T)
    far = np.flatnonzero(back >= baseline)
    d = xy[-1] - (xy[far[-1]] if len(far) else xy[0])
    n = np.linalg.norm(d)
    if n < 1e-6:
        return xy
    return np.vstack([xy, xy[-1] + d / n * length])


# ------------------------------------------------------------------ lane map
class LaneMap:
    def __init__(self, map_df):
        self.lanes = {}
        A, B, lane_of, seg_idx = [], [], [], []
        for r in map_df[map_df['kind'] == 'lane'].itertuples():
            xy = np.column_stack([r.xs, r.ys]).astype(float)
            if len(xy) < 2:
                continue
            self.lanes[r.feature_id] = dict(xy=xy, s=cumlen(xy), exits=list(r.exit_lanes),
                                            entries=list(r.entry_lanes), interp=bool(r.interpolating))
            A.append(xy[:-1]); B.append(xy[1:])
            lane_of.append(np.full(len(xy) - 1, r.feature_id)); seg_idx.append(np.arange(len(xy) - 1))
        self.ok = bool(self.lanes)
        if self.ok:
            self.A, self.B = np.vstack(A), np.vstack(B)
            self.lane_of, self.seg_idx = np.concatenate(lane_of), np.concatenate(seg_idx)
            self.AB = self.B - self.A
            self.L2 = np.maximum((self.AB ** 2).sum(1), 1e-12)
            self.hdg = np.arctan2(self.AB[:, 1], self.AB[:, 0])

    def candidates(self, p, heading, max_d=2.5, max_dh=np.deg2rad(45)):
        """Best match per lane near point p with a compatible heading: {lane: (cost, s, d)}."""
        t = np.clip(((p - self.A) * self.AB).sum(1) / self.L2, 0, 1)
        Q = self.A + t[:, None] * self.AB
        dist = np.hypot(*(p - Q).T)
        dh = np.abs(wrap(self.hdg - heading))
        ok = np.flatnonzero((dist < max_d) & (dh < max_dh))
        out = {}
        for i in ok:
            lane = self.lane_of[i]
            cost = dist[i] + 2.0 * dh[i]
            if lane not in out or cost < out[lane][0]:
                s = self.lanes[lane]['s'][self.seg_idx[i]] + t[i] * np.sqrt(self.L2[i])
                cross = self.AB[i, 0] * (p[1] - self.A[i, 1]) - self.AB[i, 1] * (p[0] - self.A[i, 0])
                out[lane] = (cost, s, np.sign(cross) * dist[i])
        return out

    def match_sequence(self, P, H, overlap_d=1.0):
        """Map-match a trajectory, preferring to stay on the same lane or move to one of its exits.

        A backward pass then fixes lane splits: where two lanes start on top of each other (e.g. a
        left-turn pocket branching off), the forward pass can pick the branch the ego never takes.
        If the lane of frame k+1 is not a legal successor of frame k's lane but also fits frame k
        (lateral offset < overlap_d), frame k is relabelled to it. Real lane changes are untouched,
        because the target lane is about a lane width away before the change."""
        lanes, ss, cands, prev = [], [], [], None
        for p, h in zip(P, H):
            c = self.candidates(p, h)
            cands.append(c)
            if not c:
                lanes.append(None); ss.append(np.nan); prev = None
                continue
            def cost(item):
                lane, (cst, _, _) = item
                bonus = 0.0
                if prev is not None and (lane == prev or lane in self.lanes[prev]['exits']):
                    bonus = 1.0
                return cst - bonus
            lane, (_, s, _) = min(c.items(), key=cost)
            lanes.append(lane); ss.append(s); prev = lane
        for k in range(len(lanes) - 2, -1, -1):
            a, b = lanes[k], lanes[k + 1]
            if a is None or b is None or a == b or b in self.lanes[a]['exits']:
                continue
            if b in cands[k] and abs(cands[k][b][2]) < overlap_d:
                lanes[k], ss[k] = b, cands[k][b][1]
        return lanes, np.array(ss)

    def route(self, lane0, s0, future_lanes, lookahead=LOOKAHEAD):
        """Polyline of the ego lane from its start, continued through exit lanes (preferring the ones
        the ego actually drives later), long enough to reach `lookahead` metres past s0."""
        chain, cur = [lane0], lane0
        total = self.lanes[lane0]['s'][-1] - s0
        while total < lookahead:
            exits = [e for e in self.lanes[cur]['exits'] if e in self.lanes and e not in chain]
            if not exits:
                break
            nxt = next((l for l in future_lanes if l in exits), None)
            if nxt is None:
                end = self.lanes[cur]['xy']
                h_end = np.arctan2(*(end[-1] - end[-2])[::-1])
                nxt = min(exits, key=lambda e: abs(wrap(np.arctan2(*(self.lanes[e]['xy'][1] - self.lanes[e]['xy'][0])[::-1]) - h_end)))
            chain.append(nxt)
            total += self.lanes[nxt]['s'][-1]
            cur = nxt
        xy = np.vstack([self.lanes[chain[0]]['xy']] + [self.lanes[l]['xy'][1:] for l in chain[1:]])
        if total < lookahead:
            xy = extend_straight(xy, lookahead - total + 1)
        return chain, xy


# ------------------------------------------------------------------ lead-vehicle pairing
def _pick_lead(s_rel, d, dh, o, gap_ok_min=0.0):
    L = o['length'].to_numpy(float)
    gap = s_rel - L / 2 - EGO_X_FRONT
    ok = (s_rel > 0) & (np.abs(d) < LANE_HALF) & (np.abs(dh) < LEAD_MAX_DHEADING) & (gap > gap_ok_min)
    if not ok.any():
        return None
    i = np.flatnonzero(ok)[np.argmin(s_rel[ok])]
    return i, gap[i]


def lead_straight(seg, k, o):
    """Straight corridor in the vehicle frame (the earlier demo)."""
    if o.empty:
        return None
    cx, cy = o['cx'].to_numpy(float), o['cy'].to_numpy(float)
    res = _pick_lead(cx, cy, wrap(o['heading'].to_numpy(float)), o)
    if res is None:
        return None
    i, gap = res
    R = seg.R[k]
    _, _, v, a = seg.to_map(k, o.iloc[[i]])
    return dict(i=i, gap=gap, v_lead=(R[:2, :2].T @ v[0])[0], a_lead=(R[:2, :2].T @ a[0])[0],
                v_ego=(R.T @ seg.ego_v[k])[0], a_ego=(R.T @ seg.ego_a[k])[0])


def _lead_on_polyline(seg, k, o, xy, s_ego):
    if o.empty:
        return None
    s_cum = cumlen(xy)
    P, hdg, v, a = seg.to_map(k, o)
    s, d, ph = project(xy, s_cum, P)
    res = _pick_lead(s - s_ego, d, wrap(hdg - ph), o)
    if res is None:
        return None
    i, gap = res
    te = heading_at(xy, s_cum, s_ego)
    tl = ph[i]
    ue, ul = np.array([np.cos(te), np.sin(te)]), np.array([np.cos(tl), np.sin(tl)])
    return dict(i=i, gap=gap, v_lead=v[i] @ ul, a_lead=a[i] @ ul,
                v_ego=seg.ego_v[k, :2] @ ue, a_ego=seg.ego_a[k, :2] @ ue)


def ego_future_path(seg, k, min_step=0.5):
    """Ego positions from frame k onward (map frame), thinned, extended straight to LOOKAHEAD."""
    pts = [seg.ego_map[k, :2]]
    for p in seg.ego_map[k + 1:, :2]:
        if np.linalg.norm(p - pts[-1]) >= min_step:
            pts.append(p)
    if len(pts) < 2:   # stationary until the end: use the current heading
        h = seg.yaw[k]
        pts.append(pts[0] + np.array([np.cos(h), np.sin(h)]))
    xy = np.array(pts)
    return extend_straight(xy, LOOKAHEAD)


def lead_path(seg, k, o):
    """Corridor along the path the ego actually drives next."""
    return _lead_on_polyline(seg, k, o, ego_future_path(seg, k), 0.0)


def lead_map(seg, k, o, lmap, ego_lanes, ego_s):
    """Lane route from the HD map; falls back to the ego path when the ego is not on a mapped lane."""
    lane0 = ego_lanes[k]
    if lane0 is None:
        r = lead_path(seg, k, o)
        if r is not None:
            r['fallback'] = True
        return r
    future = [l for l in ego_lanes[k:] if l is not None]
    chain, xy = lmap.route(lane0, ego_s[k], future)
    r = _lead_on_polyline(seg, k, o, xy, ego_s[k])
    if r is not None:
        r['fallback'] = False
    return r
