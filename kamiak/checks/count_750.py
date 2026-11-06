"""Read-only (user, 2026-11-06, item 2): the comparison list's frames that are not in the main list (vlm_frames/
comparison_lead7_targets.csv, in_main False): how many come from segments outside the main list's 241; of those
segments the split, Waymo's time of day, location and weather (context.json stats) and the user's weather label
(labels/scene_labels.csv, if the segment has one); and whether their VLM inputs are ready: the frame in
vlm_inputs/grid/targets.csv and separate/targets.csv with every listed file there, else the five cameras' source images
of the five moments (t - 2.0 ... t, every 0.5 s) in the segment's tables folder (images/<CAMERA>/<frame>_<timestamp>.jpg,
looked for by the frame part of the name).
Nothing is opened but the csv / json files; images are only checked for existence.
usage: python -I count_750.py <repository root> <vlm_inputs folder> <segments folder (segments_1873)>"""
import json
import os
import sys

import pandas as pd

ROOT, V, SEGS = sys.argv[1:4]
CAMS = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
cmp_ = pd.read_csv(os.path.join(ROOT, 'vlm_frames', 'comparison_lead7_targets.csv'), comment='#')
main = pd.read_csv(os.path.join(ROOT, 'vlm_frames', 'main_all9_targets.csv'), comment='#')
msegs = set(main.segment)
extra = cmp_[~cmp_.in_main.astype(bool)].copy()
extra['main_segment'] = extra.segment.isin(msegs)
print(f'comparison list {len(cmp_)} frames in {cmp_.segment.nunique()} segments; main list {len(main)} frames in '
      f'{len(msegs)} segments; not in the main list {len(extra)} frames')
for flag, name in ((True, 'in the main list\'s segments'), (False, 'outside the main list\'s 241 segments')):
    q = extra[extra.main_segment == flag]
    print(f'  {name}: {len(q)} frames in {q.segment.nunique()} segments (split: '
          f'{q.drop_duplicates("segment").split.value_counts().to_dict()}; frames per split '
          f'{q.split.value_counts().to_dict()})')
sl = pd.read_csv(os.path.join(ROOT, 'labels', 'scene_labels.csv'), dtype=str, keep_default_na=False).set_index('segment')
out = extra[~extra.main_segment]
rows = []
for s, g in out.groupby('segment'):
    st = json.load(open(os.path.join(SEGS, s, 'context.json'), encoding='utf-8'))   # the stats are top-level keys
    rows.append(dict(segment=s, frames=len(g), split=g.split.iloc[0], time_of_day=st.get('time_of_day'),
                     location=st.get('location'), weather_waymo=st.get('weather'),
                     weather_user=sl.loc[s, 'weather'] if s in sl.index else '(not labelled)'))
t = pd.DataFrame(rows)
for col in ('time_of_day', 'location', 'weather_waymo', 'weather_user'):
    print(f'  outside segments by {col}: segments {t[col].value_counts().to_dict()}; frames '
          f'{t.groupby(col).frames.sum().to_dict()}')
gt = pd.read_csv(os.path.join(V, 'grid', 'targets.csv'))
sp = pd.read_csv(os.path.join(V, 'separate', 'targets.csv'))
gk = {(a, int(b)): imgs for a, b, imgs in zip(gt.segment, gt.frame_idx, gt.images)}
sk = {(a, int(b)): (x, y) for a, b, x, y in zip(sp.segment, sp.frame_idx, sp.images_by_time, sp.images_by_camera)}
ready, missing_files = {True: 0, False: 0}, 0
src_ok = src_bad = 0
for r in extra.itertuples():
    k = (r.segment, int(r.frame_idx))
    if k in gk and k in sk:
        files = gk[k].split(';') + sk[k][0].split(';') + sk[k][1].split(';')
        miss = [f for f in files if not os.path.exists(os.path.join(V, 'grid' if 'step_' in f else 'separate', f))]
        missing_files += len(miss)
        ready[r.main_segment] += int(not miss)
    elif not r.main_segment:
        need = [int(r.frame_idx) - d for d in (20, 15, 10, 5, 0)]
        base = os.path.join(SEGS, r.segment, 'images')
        have = os.path.isdir(base)
        for c in CAMS:
            names = set(n[:4] for n in os.listdir(os.path.join(base, c))) if have and \
                os.path.isdir(os.path.join(base, c)) else set()
            have = have and all(f'{f:03d}_' in names for f in need)
        src_ok += int(have)
        src_bad += int(not have)
n_main_seg = int(extra.main_segment.sum())
print(f'inputs ready (in grid/targets.csv and separate/targets.csv, every file there): frames in the main segments '
      f'{ready[True]} of {n_main_seg}; frames outside {ready[False]} of {len(out)}; files listed but missing {missing_files}')
print(f'frames outside without inputs: the five cameras\' source images of the five moments in the tables folder: '
      f'{src_ok} frames have them all, {src_bad} do not')
img_counts = {s: (sum(len(os.listdir(os.path.join(SEGS, s, 'images', c))) for c in CAMS
                      if os.path.isdir(os.path.join(SEGS, s, 'images', c))))
              for s in out.segment.unique()}
print(f'source images in the tables folders of the {len(img_counts)} outside segments: total '
      f'{sum(img_counts.values())}; segments with none {sum(1 for v in img_counts.values() if v == 0)}')
t.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'count_750_outside_segments.csv'), index=False)
