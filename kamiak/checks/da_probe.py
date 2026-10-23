"""What the first frame of a Waymo domain adaptation segment holds (2026-10-22, user's question): parsed with the same
message subset as code/wod_v143_extract.py. Reports: segment name and stats (location, time of day, weather); cameras
(images, calibrations); lidar calibrations (mounting positions, the ego footprint they give); 3D labels by type and
whether they carry speed / acceleration; camera 2D labels and projected 3D labels per camera; map features by kind,
lanes with entry / exit lanes, road edges; map_pose_offset.

usage: python -I da_probe.py <file with the first record> [...]
"""
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.expanduser('~/benchmark-claude-remote/code'))
from wod_v143_extract import CAMERAS, LABEL_TYPES, LASERS, MAP_KINDS, build_frame_class, tfrecords  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')


def main():
    Frame = build_frame_class()
    for path in sys.argv[1:]:
        rec = next(tfrecords(path))
        fr = Frame()
        fr.ParseFromString(rec)
        c, st = fr.context, fr.context.stats
        print(f'\n===== {os.path.basename(path)}: record {len(rec):,} bytes')
        print(f'segment {c.name}; location {st.location!r}, time of day {st.time_of_day!r}, weather {st.weather!r}; '
              f'timestamp {fr.timestamp_micros}')
        print(f'object counts in the stats: laser {[(LABEL_TYPES.get(x.type, x.type), x.count) for x in st.laser_object_counts]}, '
              f'camera {[(LABEL_TYPES.get(x.type, x.type), x.count) for x in st.camera_object_counts]}')
        print(f'camera images: {[CAMERAS.get(i.name, i.name) for i in fr.images]} (sizes '
              f'{[len(i.image) for i in fr.images]} bytes); camera calibrations: '
              f'{[CAMERAS.get(x.name, x.name) for x in c.camera_calibrations]}')
        pos = {LASERS.get(x.name, x.name): np.asarray(x.extrinsic.transform, float).reshape(4, 4)[:3, 3]
               for x in c.laser_calibrations}
        print('lidar calibrations: ' + ', '.join(f'{k} ({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}) m' for k, v in pos.items()))
        if all(k in pos for k in ('FRONT', 'REAR', 'SIDE_LEFT', 'SIDE_RIGHT')):
            hw = (abs(pos['SIDE_LEFT'][1]) + abs(pos['SIDE_RIGHT'][1])) / 2
            print(f'  ego footprint from them: front {pos["FRONT"][0]:.3f} m, rear {pos["REAR"][0]:.3f} m, half width '
                  f'{hw:.3f} m')
        L = fr.laser_labels
        types = Counter(LABEL_TYPES.get(x.type, x.type) for x in L)
        md = [x for x in L if x.HasField('metadata')]
        print(f'3D labels: {len(L)} {dict(types)}; with metadata {len(md)}')
        if md:
            sp = np.array([[m.metadata.speed_x, m.metadata.speed_y, m.metadata.accel_x, m.metadata.accel_y] for m in md])
            print(f'  speed_x / speed_y / accel_x / accel_y non-zero in {[(int((sp[:, i] != 0).sum())) for i in range(4)]} '
                  f'of {len(md)} labels; |speed| max {np.hypot(sp[:, 0], sp[:, 1]).max():.3f} m/s')
            print(f'  ids unique {len({x.id for x in L})}; num_lidar_points_in_box > 0 in '
                  f'{sum(x.num_lidar_points_in_box > 0 for x in L)}; camera_synced_box in '
                  f'{sum(x.HasField("camera_synced_box") for x in L)}')
        for name, src in (('camera 2D labels', fr.camera_labels), ('projected 3D labels', fr.projected_lidar_labels)):
            per = {CAMERAS.get(cl.name, cl.name): len(cl.labels) for cl in src}
            print(f'{name}: {sum(per.values())} {per}')
        mf = fr.map_features
        kinds = Counter(next((k for k in MAP_KINDS if f.HasField(k)), None) for f in mf)
        lanes = [f.lane for f in mf if f.HasField('lane')]
        print(f'map features: {len(mf)} {dict(kinds)}')
        if lanes:
            print(f'  lanes with entry lanes {sum(len(l.entry_lanes) > 0 for l in lanes)}, with exit lanes '
                  f'{sum(len(l.exit_lanes) > 0 for l in lanes)}, with neighbours '
                  f'{sum(len(l.left_neighbors) + len(l.right_neighbors) > 0 for l in lanes)}, with boundaries '
                  f'{sum(len(l.left_boundaries) + len(l.right_boundaries) > 0 for l in lanes)} of {len(lanes)}; '
                  f'polyline points per lane median {int(np.median([len(l.polyline) for l in lanes]))}')
        print(f'map_pose_offset present: {fr.HasField("map_pose_offset")} '
              f'({fr.map_pose_offset.x:.3f}, {fr.map_pose_offset.y:.3f}, {fr.map_pose_offset.z:.3f})')


if __name__ == '__main__':
    main()
