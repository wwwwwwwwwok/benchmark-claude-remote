"""Extract what the SSM pipeline needs from Waymo Open Dataset Perception v1.4.3 TFRecord files.

Works on Windows with plain `protobuf` (no TensorFlow, no official waymo package): the message
definitions below are a subset of the official protos (dataset.proto, label.proto, protos/map.proto,
protos/vector.proto, checked against the waymo-research/waymo-open-dataset master branch).
Fields not listed are skipped by the parser.

Usage:
    python -I wod_v143_extract.py <tfrecord_dir or one .tfrecord file> <out_root> [--cameras FRONT,FRONT_LEFT,...]

Writes one folder per segment, <out_root>/<segment>/:
    context.json            segment name, stats (time of day, location, weather), camera and lidar calibrations
                            (lidar mounting positions also give the ego footprint: front/rear bumper, sides)
    frames.parquet          per frame: timestamp, ego pose (world_from_vehicle, 4x4 row-major),
                            map_pose_offset, ego velocity reported with the FRONT image
    objects.parquet         3D lidar labels (vehicle frame) with speed, acceleration, visibility
    projected_boxes.parquet 3D labels projected into each camera image (pixel boxes)
    camera_boxes.parquet    2D boxes labelled directly on camera images
    map.parquet             map features (lanes, road lines, road edges, stop signs, crosswalks, ...)
    no_label_zones.parquet  only if the segment has any
    images/<CAMERA>/<frame>_<timestamp>.jpg
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
import pandas as pd
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

sys.stdout.reconfigure(encoding='utf-8')

PKG = 'wodmin'
FD = descriptor_pb2.FieldDescriptorProto
SCALAR = {'double': FD.TYPE_DOUBLE, 'float': FD.TYPE_FLOAT, 'int64': FD.TYPE_INT64, 'int32': FD.TYPE_INT32,
          'bool': FD.TYPE_BOOL, 'string': FD.TYPE_STRING, 'bytes': FD.TYPE_BYTES}

# message -> [(field, number, type, repeated)]; enums are read as int32 (same wire format)
SCHEMA = {
    'Vector3d': [('x', 1, 'double', 0), ('y', 2, 'double', 0), ('z', 3, 'double', 0)],
    'Transform': [('transform', 1, 'double', 1)],
    'Velocity': [('v_x', 1, 'float', 0), ('v_y', 2, 'float', 0), ('v_z', 3, 'float', 0),
                 ('w_x', 4, 'double', 0), ('w_y', 5, 'double', 0), ('w_z', 6, 'double', 0)],
    'CameraCalibration': [('name', 1, 'int32', 0), ('intrinsic', 2, 'double', 1), ('extrinsic', 3, 'Transform', 0),
                          ('width', 4, 'int32', 0), ('height', 5, 'int32', 0),
                          ('rolling_shutter_direction', 6, 'int32', 0)],
    'ObjectCount': [('type', 1, 'int32', 0), ('count', 2, 'int32', 0)],
    'Stats': [('laser_object_counts', 1, 'ObjectCount', 1), ('time_of_day', 2, 'string', 0),
              ('location', 3, 'string', 0), ('weather', 4, 'string', 0),
              ('camera_object_counts', 5, 'ObjectCount', 1)],
    'LaserCalibration': [('name', 1, 'int32', 0), ('extrinsic', 5, 'Transform', 0)],
    'Context': [('name', 1, 'string', 0), ('camera_calibrations', 2, 'CameraCalibration', 1),
                ('laser_calibrations', 3, 'LaserCalibration', 1), ('stats', 4, 'Stats', 0)],
    'CameraImage': [('name', 1, 'int32', 0), ('image', 2, 'bytes', 0), ('pose', 3, 'Transform', 0),
                    ('velocity', 4, 'Velocity', 0), ('pose_timestamp', 5, 'double', 0), ('shutter', 6, 'double', 0),
                    ('camera_trigger_time', 7, 'double', 0), ('camera_readout_done_time', 8, 'double', 0)],
    'Box': [('center_x', 1, 'double', 0), ('center_y', 2, 'double', 0), ('center_z', 3, 'double', 0),
            ('width', 4, 'double', 0), ('length', 5, 'double', 0), ('height', 6, 'double', 0),
            ('heading', 7, 'double', 0)],
    'Metadata': [('speed_x', 1, 'double', 0), ('speed_y', 2, 'double', 0), ('accel_x', 3, 'double', 0),
                 ('accel_y', 4, 'double', 0), ('speed_z', 5, 'double', 0), ('accel_z', 6, 'double', 0)],
    'Association': [('laser_object_id', 1, 'string', 0)],
    'Label': [('box', 1, 'Box', 0), ('metadata', 2, 'Metadata', 0), ('type', 3, 'int32', 0), ('id', 4, 'string', 0),
              ('detection_difficulty_level', 5, 'int32', 0), ('tracking_difficulty_level', 6, 'int32', 0),
              ('num_lidar_points_in_box', 7, 'int32', 0), ('association', 10, 'Association', 0),
              ('most_visible_camera_name', 11, 'string', 0), ('camera_synced_box', 12, 'Box', 0),
              ('num_top_lidar_points_in_box', 13, 'int32', 0)],
    'CameraLabels': [('name', 1, 'int32', 0), ('labels', 2, 'Label', 1)],
    'Polygon2dProto': [('x', 1, 'double', 1), ('y', 2, 'double', 1), ('id', 3, 'string', 0)],
    'MapPoint': [('x', 1, 'double', 0), ('y', 2, 'double', 0), ('z', 3, 'double', 0)],
    'BoundarySegment': [('lane_start_index', 1, 'int32', 0), ('lane_end_index', 2, 'int32', 0),
                        ('boundary_feature_id', 3, 'int64', 0), ('boundary_type', 4, 'int32', 0)],
    'LaneNeighbor': [('feature_id', 1, 'int64', 0), ('self_start_index', 2, 'int32', 0),
                     ('self_end_index', 3, 'int32', 0), ('neighbor_start_index', 4, 'int32', 0),
                     ('neighbor_end_index', 5, 'int32', 0), ('boundaries', 6, 'BoundarySegment', 1)],
    'LaneCenter': [('speed_limit_mph', 1, 'double', 0), ('type', 2, 'int32', 0), ('interpolating', 3, 'bool', 0),
                   ('polyline', 8, 'MapPoint', 1), ('entry_lanes', 9, 'int64', 1), ('exit_lanes', 10, 'int64', 1),
                   ('left_neighbors', 11, 'LaneNeighbor', 1), ('right_neighbors', 12, 'LaneNeighbor', 1),
                   ('left_boundaries', 13, 'BoundarySegment', 1), ('right_boundaries', 14, 'BoundarySegment', 1)],
    'RoadEdge': [('type', 1, 'int32', 0), ('polyline', 2, 'MapPoint', 1)],
    'RoadLine': [('type', 1, 'int32', 0), ('polyline', 2, 'MapPoint', 1)],
    'StopSign': [('lane', 1, 'int64', 1), ('position', 2, 'MapPoint', 0)],
    'Crosswalk': [('polygon', 1, 'MapPoint', 1)],
    'SpeedBump': [('polygon', 1, 'MapPoint', 1)],
    'Driveway': [('polygon', 1, 'MapPoint', 1)],
    'MapFeature': [('id', 1, 'int64', 0), ('lane', 3, 'LaneCenter', 0), ('road_line', 4, 'RoadLine', 0),
                   ('road_edge', 5, 'RoadEdge', 0), ('stop_sign', 7, 'StopSign', 0), ('crosswalk', 8, 'Crosswalk', 0),
                   ('speed_bump', 9, 'SpeedBump', 0), ('driveway', 10, 'Driveway', 0)],
    'Frame': [('context', 1, 'Context', 0), ('timestamp_micros', 2, 'int64', 0), ('pose', 3, 'Transform', 0),
              ('images', 4, 'CameraImage', 1), ('laser_labels', 6, 'Label', 1),
              ('no_label_zones', 7, 'Polygon2dProto', 1), ('camera_labels', 8, 'CameraLabels', 1),
              ('projected_lidar_labels', 9, 'CameraLabels', 1), ('map_features', 10, 'MapFeature', 1),
              ('map_pose_offset', 11, 'Vector3d', 0)],
}

CAMERAS = {1: 'FRONT', 2: 'FRONT_LEFT', 3: 'FRONT_RIGHT', 4: 'SIDE_LEFT', 5: 'SIDE_RIGHT'}
LASERS = {1: 'TOP', 2: 'FRONT', 3: 'SIDE_LEFT', 4: 'SIDE_RIGHT', 5: 'REAR'}
LABEL_TYPES = {0: 'unknown', 1: 'vehicle', 2: 'pedestrian', 3: 'sign', 4: 'cyclist'}
MAP_KINDS = ('lane', 'road_line', 'road_edge', 'stop_sign', 'crosswalk', 'speed_bump', 'driveway')


def build_frame_class():
    fdp = descriptor_pb2.FileDescriptorProto(name='wodmin.proto', package=PKG, syntax='proto2')
    for mname, fields in SCHEMA.items():
        m = fdp.message_type.add(name=mname)
        for fname, num, ftype, rep in fields:
            f = m.field.add(name=fname, number=num, label=FD.LABEL_REPEATED if rep else FD.LABEL_OPTIONAL)
            if ftype in SCALAR:
                f.type = SCALAR[ftype]
            else:
                f.type, f.type_name = FD.TYPE_MESSAGE, f'.{PKG}.{ftype}'
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fdp)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName(f'{PKG}.Frame'))


def tfrecords(path):
    """Yield the payload of each record (TFRecord: uint64 length, uint32 crc, data, uint32 crc)."""
    with open(path, 'rb') as f:
        while True:
            head = f.read(12)
            if not head:
                return
            if len(head) < 12:
                raise IOError(f'{path}: truncated record header')
            n = struct.unpack('<Q', head[:8])[0]
            data = f.read(n)
            if len(data) < n or len(f.read(4)) < 4:
                raise IOError(f'{path}: truncated record')
            yield data


def box_cols(b, prefix=''):
    return {prefix + 'cx': b.center_x, prefix + 'cy': b.center_y, prefix + 'cz': b.center_z,
            prefix + 'length': b.length, prefix + 'width': b.width, prefix + 'height': b.height,
            prefix + 'heading': b.heading}


def map_rows(features):
    rows = []
    for mf in features:
        kind = next((k for k in MAP_KINDS if mf.HasField(k)), None)
        r = dict(feature_id=mf.id, kind=kind, subtype=-1, speed_limit_mph=np.nan, interpolating=False,
                 entry_lanes=[], exit_lanes=[], left_neighbors=[], right_neighbors=[],
                 left_boundaries=[], right_boundaries=[], stop_sign_lanes=[])
        pts = []
        if kind == 'lane':
            ln = mf.lane
            pts = ln.polyline
            r.update(subtype=ln.type, interpolating=ln.interpolating,
                     speed_limit_mph=ln.speed_limit_mph if ln.HasField('speed_limit_mph') else np.nan,
                     entry_lanes=list(ln.entry_lanes), exit_lanes=list(ln.exit_lanes),
                     left_neighbors=[n.feature_id for n in ln.left_neighbors],
                     right_neighbors=[n.feature_id for n in ln.right_neighbors],
                     left_boundaries=[b.boundary_feature_id for b in ln.left_boundaries],
                     right_boundaries=[b.boundary_feature_id for b in ln.right_boundaries])
        elif kind in ('road_line', 'road_edge'):
            g = getattr(mf, kind)
            pts, r['subtype'] = g.polyline, g.type
        elif kind == 'stop_sign':
            pts, r['stop_sign_lanes'] = [mf.stop_sign.position], list(mf.stop_sign.lane)
        elif kind is not None:
            pts = getattr(mf, kind).polygon
        r['xs'] = [p.x for p in pts]
        r['ys'] = [p.y for p in pts]
        r['zs'] = [p.z for p in pts]
        rows.append(r)
    return rows


def extract(path, out_root, cameras, Frame):
    frames, objects, projected, camboxes, zones = [], [], [], [], []
    ctx_info, mrows, seg = None, None, None
    for k, rec in enumerate(tfrecords(path)):
        fr = Frame()
        fr.ParseFromString(rec)
        ts = fr.timestamp_micros
        if seg is None:
            seg = fr.context.name
            seg_dir = os.path.join(out_root, seg)
            os.makedirs(seg_dir, exist_ok=True)          # also when no camera images are written
            for cam in cameras:
                os.makedirs(os.path.join(seg_dir, 'images', cam), exist_ok=True)
            st = fr.context.stats
            ctx_info = dict(
                segment=seg, source_file=os.path.basename(path),
                time_of_day=st.time_of_day, location=st.location, weather=st.weather,
                laser_object_counts={LABEL_TYPES.get(c.type, c.type): c.count for c in st.laser_object_counts},
                camera_object_counts={LABEL_TYPES.get(c.type, c.type): c.count for c in st.camera_object_counts},
                camera_calibrations=[dict(name=CAMERAS.get(c.name, c.name), intrinsic=list(c.intrinsic),
                                          extrinsic=list(c.extrinsic.transform), width=c.width, height=c.height,
                                          rolling_shutter_direction=c.rolling_shutter_direction)
                                     for c in fr.context.camera_calibrations],
                laser_calibrations=[dict(name=LASERS.get(c.name, c.name), extrinsic=list(c.extrinsic.transform))
                                    for c in fr.context.laser_calibrations])
        if len(fr.map_features):
            if mrows is not None:
                print(f'  note: frame {k} also carries map features; keeping the first set')
            else:
                mrows = map_rows(fr.map_features)

        front = next((im for im in fr.images if im.name == 1), None)
        off = fr.map_pose_offset
        frames.append(dict(
            frame_idx=k, timestamp_micros=ts, world_from_vehicle=list(fr.pose.transform),
            map_pose_offset_x=off.x if fr.HasField('map_pose_offset') else np.nan,
            map_pose_offset_y=off.y if fr.HasField('map_pose_offset') else np.nan,
            map_pose_offset_z=off.z if fr.HasField('map_pose_offset') else np.nan,
            ego_vx=front.velocity.v_x if front else np.nan, ego_vy=front.velocity.v_y if front else np.nan,
            ego_vz=front.velocity.v_z if front else np.nan, ego_wz=front.velocity.w_z if front else np.nan,
            front_pose_timestamp=front.pose_timestamp if front else np.nan,
            n_laser_labels=len(fr.laser_labels), n_map_features=len(fr.map_features)))

        for im in fr.images:
            cam = CAMERAS.get(im.name)
            if cam in cameras:
                with open(os.path.join(seg_dir, 'images', cam, f'{k:03d}_{ts}.jpg'), 'wb') as f:
                    f.write(im.image)

        for lab in fr.laser_labels:
            md = lab.metadata if lab.HasField('metadata') else None
            row = dict(frame_idx=k, timestamp_micros=ts, id=lab.id, type=LABEL_TYPES.get(lab.type, lab.type),
                       **box_cols(lab.box),
                       speed_x=md.speed_x if md else np.nan, speed_y=md.speed_y if md else np.nan,
                       speed_z=md.speed_z if md else np.nan, accel_x=md.accel_x if md else np.nan,
                       accel_y=md.accel_y if md else np.nan, accel_z=md.accel_z if md else np.nan,
                       num_lidar_points=lab.num_lidar_points_in_box,
                       num_top_lidar_points=lab.num_top_lidar_points_in_box,
                       detection_difficulty=lab.detection_difficulty_level,
                       tracking_difficulty=lab.tracking_difficulty_level,
                       most_visible_camera=lab.most_visible_camera_name)
            if lab.HasField('camera_synced_box'):
                row.update(box_cols(lab.camera_synced_box, 'synced_'))
            objects.append(row)

        for src, sink in ((fr.projected_lidar_labels, projected), (fr.camera_labels, camboxes)):
            for cl in src:
                for lab in cl.labels:
                    b = lab.box
                    sink.append(dict(frame_idx=k, timestamp_micros=ts, camera=CAMERAS.get(cl.name, cl.name),
                                     id=lab.id, type=LABEL_TYPES.get(lab.type, lab.type),
                                     box_cx=b.center_x, box_cy=b.center_y, box_w=b.length, box_h=b.width,
                                     laser_object_id=lab.association.laser_object_id
                                     if lab.HasField('association') else None))

        for z in fr.no_label_zones:
            zones.append(dict(frame_idx=k, timestamp_micros=ts, zone_id=z.id, xs=list(z.x), ys=list(z.y)))

    if seg is None:
        raise IOError(f'{path}: no frames')
    ctx_info['n_frames'] = len(frames)
    with open(os.path.join(seg_dir, 'context.json'), 'w', encoding='utf-8') as f:
        json.dump(ctx_info, f, ensure_ascii=False, indent=1)
    pd.DataFrame(frames).to_parquet(os.path.join(seg_dir, 'frames.parquet'), index=False)
    pd.DataFrame(objects).to_parquet(os.path.join(seg_dir, 'objects.parquet'), index=False)
    pd.DataFrame(projected).to_parquet(os.path.join(seg_dir, 'projected_boxes.parquet'), index=False)
    pd.DataFrame(camboxes).to_parquet(os.path.join(seg_dir, 'camera_boxes.parquet'), index=False)
    pd.DataFrame(mrows or []).to_parquet(os.path.join(seg_dir, 'map.parquet'), index=False)
    if zones:
        pd.DataFrame(zones).to_parquet(os.path.join(seg_dir, 'no_label_zones.parquet'), index=False)
    kinds = pd.Series([r['kind'] for r in (mrows or [])]).value_counts().to_dict()
    print(f'{seg}: {len(frames)} frames, {len(objects)} 3D labels, {len(projected)} projected boxes, '
          f'{len(camboxes)} camera boxes, map features {kinds}, no-label zones {len(zones)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tfrecord_dir', help='folder whose .tfrecord files are all extracted, or one .tfrecord file')
    ap.add_argument('out_root')
    ap.add_argument('--cameras', default=','.join(CAMERAS.values()),
                    help='cameras whose JPEGs are written (comma separated); others are skipped')
    a = ap.parse_args()
    cameras = [c for c in a.cameras.split(',') if c]
    Frame = build_frame_class()
    if os.path.isdir(a.tfrecord_dir):
        files = sorted(os.path.join(a.tfrecord_dir, f) for f in os.listdir(a.tfrecord_dir) if f.endswith('.tfrecord'))
    else:
        files = [a.tfrecord_dir] if a.tfrecord_dir.endswith('.tfrecord') and os.path.isfile(a.tfrecord_dir) else []
    if not files:
        sys.exit(f'no .tfrecord files in {a.tfrecord_dir}')
    for path in files:
        extract(path, a.out_root, cameras, Frame)


if __name__ == '__main__':
    main()
