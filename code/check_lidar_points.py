"""Check the label boxes against the raw TOP-lidar points (Waymo Open Dataset Perception v1.4.3).

Two checks, both read the TFRecord files directly (plain protobuf, no TensorFlow):
1. Lead rear face: for every frame with a lead vehicle (frame_ssm.csv, column lead_map), the rear-most
   dense part of the lidar points on the lead is compared with the rear edge of its label box
   (centre - length/2). The difference is how much the gap would change if it were measured from the
   points instead of the box. A single nearest point is not used: sparse returns behind a car (spray,
   road surface) can sit up to 1 m behind it, so the rear is the first 10 cm slab that holds enough points.
2. Vehicle type: 3D label boxes only say "vehicle". About 30 frames per segment carry lidar semantic
   segmentation labels (car, truck, bus, other_vehicle, motorcycle, ...); the majority class of the
   labelled points inside a box gives its type, and the tracking id carries it to every frame.

Points are built from the first return of the TOP lidar range image with the per-pixel pose (ego motion
compensation), following the steps of the official convert_range_image_to_point_cloud.

Usage:
    python -I code/check_lidar_points.py <tfrecord_dir> <extracted_root> <frame_ssm.csv> <out_dir>
Writes lead_rear_points_vs_box.csv, vehicle_type_from_lidar_seg.csv (per box and frame) and
vehicle_type_per_track.csv to <out_dir>, and prints the summaries.
"""
import glob
import os
import struct
import sys
import time
import zlib

import numpy as np
import pandas as pd
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

sys.stdout.reconfigure(encoding='utf-8')

FD = descriptor_pb2.FieldDescriptorProto
D, F32, I32, I64, S, B = FD.TYPE_DOUBLE, FD.TYPE_FLOAT, FD.TYPE_INT32, FD.TYPE_INT64, FD.TYPE_STRING, FD.TYPE_BYTES
# subset of dataset.proto (field numbers checked against waymo-research/waymo-open-dataset master)
SCHEMA = {
    'Transform': [('transform', 1, D, 1)],
    'LaserCalibration': [('name', 1, I32, 0), ('beam_inclinations', 2, D, 1), ('beam_inclination_min', 3, D, 0),
                         ('beam_inclination_max', 4, D, 0), ('extrinsic', 5, '.m.Transform', 0)],
    'Context': [('name', 1, S, 0), ('laser_calibrations', 3, '.m.LaserCalibration', 1)],
    'RangeImage': [('range_image_compressed', 2, B, 0), ('range_image_pose_compressed', 4, B, 0),
                   ('segmentation_label_compressed', 6, B, 0)],
    'Laser': [('name', 1, I32, 0), ('ri_return1', 2, '.m.RangeImage', 0)],
    'Frame': [('context', 1, '.m.Context', 0), ('timestamp_micros', 2, I64, 0), ('pose', 3, '.m.Transform', 0),
              ('lasers', 5, '.m.Laser', 1)],
    'MatrixShape': [('dims', 1, I32, 1)],
    'MatrixFloat': [('data', 1, F32, 1), ('shape', 2, '.m.MatrixShape', 0)],
    'MatrixInt32': [('data', 1, I32, 1), ('shape', 2, '.m.MatrixShape', 0)],
}
# protos/segmentation.proto, Segmentation.Type (road users only)
SEG_TYPES = {1: 'car', 2: 'truck', 3: 'bus', 4: 'other_vehicle', 5: 'motorcyclist', 6: 'bicyclist',
             12: 'bicycle', 13: 'motorcycle'}
TOP = 1
REAR_MARGIN, SIDE_MARGIN, GROUND_CLEAR = 1.0, 0.3, 0.3   # m, search window around the lead box


def build_classes():
    fdp = descriptor_pb2.FileDescriptorProto(name='m.proto', package='m', syntax='proto2')
    for mname, fields in SCHEMA.items():
        msg = fdp.message_type.add(name=mname)
        for fname, num, ftype, rep in fields:
            f = msg.field.add(name=fname, number=num, label=FD.LABEL_REPEATED if rep else FD.LABEL_OPTIONAL)
            if isinstance(ftype, str):
                f.type, f.type_name = FD.TYPE_MESSAGE, ftype
            else:
                f.type = ftype
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fdp)
    return [message_factory.GetMessageClass(pool.FindMessageTypeByName('m.' + n))
            for n in ('Frame', 'MatrixFloat', 'MatrixInt32')]


Frame, MatrixFloat, MatrixInt32 = build_classes()


def records(path):
    with open(path, 'rb') as f:
        while True:
            head = f.read(12)
            if len(head) < 12:
                return
            n = struct.unpack('<Q', head[:8])[0]
            data = f.read(n)
            f.read(4)
            yield data


def decode(blob, M, dtype):
    m = M()
    m.ParseFromString(zlib.decompress(blob))
    return np.asarray(m.data, dtype).reshape(tuple(m.shape.dims))


def rot_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.stack([np.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], -1),
                     np.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], -1),
                     np.stack([-sp, cp * sr, cp * cr], -1)], -2)


def top_points(frame, cal):
    """TOP lidar first return -> (points (H,W,3) in the frame's vehicle frame, valid (H,W), seg (H,W,2) or None)."""
    las = next(l for l in frame.lasers if l.name == TOP)
    ri = decode(las.ri_return1.range_image_compressed, MatrixFloat, np.float32)
    pix_pose = decode(las.ri_return1.range_image_pose_compressed, MatrixFloat, np.float32).astype(float)
    H, W = ri.shape[:2]
    incl = np.array(cal.beam_inclinations, float)
    if len(incl) == 0:
        incl = (np.arange(H) + 0.5) / H * (cal.beam_inclination_max - cal.beam_inclination_min) + cal.beam_inclination_min
    incl = incl[::-1]                                   # row 0 is the top beam
    E = np.array(cal.extrinsic.transform, float).reshape(4, 4)
    az = ((np.arange(W, 0, -1) - 0.5) / W * 2 - 1) * np.pi - np.arctan2(E[1, 0], E[0, 0])
    r = ri[..., 0].astype(float)
    ci, si = np.cos(incl)[:, None], np.sin(incl)[:, None]
    p_sensor = np.stack([np.cos(az)[None] * ci * r, np.sin(az)[None] * ci * r, si * r], -1)
    p_vehicle_at_capture = p_sensor @ E[:3, :3].T + E[:3, 3]
    p_world = np.einsum('hwij,hwj->hwi', rot_rpy(*pix_pose[..., :3].transpose(2, 0, 1)),
                        p_vehicle_at_capture) + pix_pose[..., 3:6]
    P = np.array(frame.pose.transform, float).reshape(4, 4)   # world_from_vehicle at the frame time
    pts = (p_world - P[:3, 3]) @ P[:3, :3]
    blob = las.ri_return1.segmentation_label_compressed
    seg = decode(blob, MatrixInt32, np.int32) if len(blob) else None
    return pts, r > 0, seg


def box_local(pts, b):
    """Points in the box frame: u forward along the box heading, v to the left, w up (from the centre)."""
    d = pts - np.array([b.cx, b.cy, b.cz])
    c, s = np.cos(b.heading), np.sin(b.heading)
    return d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c, d[:, 2]


def dense_rear(u, slab=0.10):
    """Rear-most u where a 10 cm slab holds enough points (skips sparse spray and road returns)."""
    us = np.sort(u)
    need = max(5, min(15, 0.1 * len(us)))
    count = np.searchsorted(us, us + slab, side='right') - np.arange(len(us))
    ok = np.flatnonzero(count >= need)
    return us[ok[0]] if len(ok) else np.nan


def check_segment(path, seg_name, objs, leads):
    rear_rows, type_rows, cal = [], [], None
    by_ts = {ts: g for ts, g in objs.groupby('timestamp_micros')}
    for k, rec in enumerate(records(path)):
        frame = Frame()
        frame.ParseFromString(rec)
        if cal is None:
            cal = next(c for c in frame.context.laser_calibrations if c.name == TOP)
        ts = frame.timestamp_micros
        need_lead = ts in leads.index
        has_seg = any(l.name == TOP and len(l.ri_return1.segmentation_label_compressed) for l in frame.lasers)
        if not (need_lead or has_seg) or ts not in by_ts:
            continue
        pts, valid, seg = top_points(frame, cal)
        P = pts[valid]
        boxes = by_ts[ts]
        if need_lead:
            lead_id = leads.loc[ts, 'lead_map']
            hit = boxes[boxes['id'] == lead_id]
            if len(hit):
                b = hit.iloc[0]
                u, v, w = box_local(P, b)
                m = ((np.abs(u) <= b.length / 2 + REAR_MARGIN) & (np.abs(v) <= b.width / 2 + SIDE_MARGIN)
                     & (w >= -b.height / 2 + GROUND_CLEAR) & (w <= b.height / 2 + SIDE_MARGIN))
                if m.sum() >= 10:
                    uu, vv = u[m], v[m]
                    rear = dense_rear(uu)
                    rear_rows.append(dict(
                        segment=seg_name, frame_idx=k, lead_id=lead_id, n_points=int(m.sum()),
                        length=b.length, width=b.width, gap_box=leads.loc[ts, 'gap_map'],
                        rear_nearest_minus_box=uu.min() + b.length / 2,     # > 0: points start ahead of the box rear
                        rear_dense_minus_box=rear + b.length / 2,          # = gap from points - gap from box
                        visible_length=np.percentile(uu, 98) - rear,
                        visible_width=np.percentile(vv, 98) - np.percentile(vv, 2)))
        if seg is not None:
            sems = seg[..., 1][valid]
            for b in boxes.itertuples():
                u, v, w = box_local(P, b)
                m = ((np.abs(u) <= b.length / 2) & (np.abs(v) <= b.width / 2)
                     & (w <= b.height / 2) & (w >= -b.height / 2 + 0.1))
                lab = sems[m]
                lab = lab[np.isin(lab, list(SEG_TYPES))]
                if len(lab) >= 5:
                    vals, cnt = np.unique(lab, return_counts=True)
                    type_rows.append(dict(segment=seg_name, frame_idx=k, id=b.id, length=b.length, width=b.width,
                                          height=b.height, seg_type=SEG_TYPES[int(vals[cnt.argmax()])],
                                          share=cnt.max() / cnt.sum(), n_points=len(lab)))
    return rear_rows, type_rows


def main():
    tf_dir, ex_root, ssm_csv, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    fs = pd.read_csv(ssm_csv)
    rear_rows, type_rows, all_vehicles = [], [], []
    for path in sorted(glob.glob(os.path.join(tf_dir, '*.tfrecord'))):
        seg_name = next((s for s in fs.segment.unique() if s in os.path.basename(path)), None)
        if seg_name is None:
            continue
        t0 = time.time()
        objs = pd.read_parquet(os.path.join(ex_root, seg_name, 'objects.parquet'))
        objs = objs[objs['type'] == 'vehicle']
        all_vehicles.append(objs.assign(segment=seg_name)[['segment', 'id']])
        leads = fs[(fs.segment == seg_name) & fs.lead_map.notna()].set_index('timestamp_micros')
        r, t = check_segment(path, seg_name, objs, leads)
        rear_rows += r
        type_rows += t
        print(f'{seg_name}: {time.time() - t0:.0f} s', flush=True)

    pd.set_option('display.width', 220)
    rr = pd.DataFrame(rear_rows)
    rr.to_csv(os.path.join(out, 'lead_rear_points_vs_box.csv'), index=False)
    if len(rr):
        print('\n== Lead rear face: gap measured from the points minus gap from the box (m)')
        print(rr.groupby(['segment', 'lead_id']).agg(
            frames=('n_points', 'size'), points_med=('n_points', 'median'),
            gap_min=('gap_box', 'min'), gap_max=('gap_box', 'max'), length=('length', 'first'),
            nearest_med=('rear_nearest_minus_box', 'median'), dense_med=('rear_dense_minus_box', 'median'),
            dense_p05=('rear_dense_minus_box', lambda x: x.quantile(.05)),
            dense_p95=('rear_dense_minus_box', lambda x: x.quantile(.95)),
            visible_length_med=('visible_length', 'median')).round(3).to_string())
        for c in ('rear_nearest_minus_box', 'rear_dense_minus_box'):
            q = rr[c].abs().quantile([.5, .95, 1.0]).to_numpy()
            print(f'|{c}| over {len(rr)} lead frames: median {q[0]:.3f}, 95th pct {q[1]:.3f}, max {q[2]:.3f}')

    tt = pd.DataFrame(type_rows)
    tt.to_csv(os.path.join(out, 'vehicle_type_from_lidar_seg.csv'), index=False)
    if len(tt):
        per_track = tt.groupby(['segment', 'id']).agg(
            seg_type=('seg_type', lambda x: x.value_counts().index[0]),
            agreement=('seg_type', lambda x: x.value_counts().iloc[0] / len(x)),
            labelled_frames=('seg_type', 'size'), length=('length', 'first'), width=('width', 'first'),
            height=('height', 'first'))
        per_track.to_csv(os.path.join(out, 'vehicle_type_per_track.csv'))
        veh = pd.concat(all_vehicles)
        typed = veh.set_index(['segment', 'id']).index.isin(per_track.index)
        n_tracks = veh.drop_duplicates().shape[0]
        print(f'\n== Vehicle type from lidar segmentation: {len(per_track)} of {n_tracks} vehicle tracks '
              f'({len(per_track) / n_tracks:.0%}), {typed.mean():.0%} of vehicle boxes')
        print(per_track.groupby('seg_type').agg(
            tracks=('length', 'size'), length_med=('length', 'median'), length_min=('length', 'min'),
            length_max=('length', 'max'), height_med=('height', 'median'),
            agreement_med=('agreement', 'median')).round(2).to_string())
        lead_ids = set(fs.lead_map.dropna())
        pt = per_track.reset_index()
        print('\nLead vehicles:')
        print(pt[pt['id'].isin(lead_ids)].round(2).to_string(index=False))


if __name__ == '__main__':
    main()
