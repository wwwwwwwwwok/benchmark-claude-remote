"""Local time of the main result's Dawn/Dusk segments (user, 2026-11-12; a small check: only each segment's frames.parquet
is read, no image). The segments of the 1,084 frames (analysis/after_qwen3.5-4b/main_1084/ssm_target_frames.csv) whose
time_of_day is Dawn/Dusk (labels/scene_labels.csv); for each, the timestamp_micros of the segment's first frame (the
smallest frame_idx in frames.parquet), turned into UTC exactly (microseconds after 1970-01-01 UTC) and into local time
by city: location_sf America/Los_Angeles, location_phx America/Phoenix, location_other both UTC and
America/Los_Angeles (zoneinfo, daylight saving by the date); before 12:00 local = morning, otherwise evening. Frames =
the segment's frames among the 1,084. Stops before writing anything when a year is not 2016-2021 (then the timestamps
are not real times) and prints the raw values.
usage: python -I dawn_dusk_local_time.py <waymo_v143 folder> <scene_labels.csv> <out csv>
"""
import os
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

L, SL, OUT = sys.argv[1:4]
frames = pd.read_csv(os.path.join(L, 'analysis', 'after_qwen3.5-4b', 'main_1084', 'ssm_target_frames.csv'))
cnt = frames['segment'].value_counts()
lab = pd.read_csv(SL, dtype=str, keep_default_na=False).set_index('segment')
segs = sorted(s for s in cnt.index if lab.loc[s, 'time_of_day'].strip() == 'Dawn/Dusk')
print(f'main result: {len(frames)} frames in {len(cnt)} segments; time_of_day Dawn/Dusk: {len(segs)} segments, '
      f'{int(cnt[segs].sum())} frames')
if len(segs) != 20:
    sys.exit(f'{len(segs)} Dawn/Dusk segments, not 20: stop')
TZ = {'location_sf': ['America/Los_Angeles'], 'location_phx': ['America/Phoenix'],
      'location_other': ['UTC', 'America/Los_Angeles']}
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
rows, raw, bad = [], [], []
for s in segs:
    fr = pd.read_parquet(os.path.join(L, 'vlm_inputs', 'segments', s, 'frames.parquet'),
                         columns=['frame_idx', 'timestamp_micros']).sort_values('frame_idx')
    k0, ts = int(fr['frame_idx'].iloc[0]), int(fr['timestamp_micros'].iloc[0])
    utc = EPOCH + timedelta(microseconds=ts)
    loc = lab.loc[s, 'location'].strip()
    raw.append((s, loc, k0, ts, utc.isoformat()))
    if not 2016 <= utc.year <= 2021:
        bad.append(s)
    for tz in TZ[loc]:
        lt = utc.astimezone(ZoneInfo(tz))
        rows.append(dict(segment=s, location=loc, frames=int(cnt[s]), first_frame_idx=k0, timestamp_micros=ts,
                         utc=utc.strftime('%Y-%m-%d %H:%M:%S.%f'), time_zone=tz,
                         local=lt.strftime('%Y-%m-%d %H:%M:%S.%f %Z (UTC%z)'),
                         part='morning' if lt.hour < 12 else 'evening'))
if bad:
    print(f'years outside 2016-2021 for {len(bad)} segments: the timestamps are not real times; stop. Raw values:')
    for s, loc, k0, ts, u in raw:
        print(f'  {s} {loc} first frame {k0}: timestamp_micros {ts} -> {u}')
    sys.exit(1)
t = pd.DataFrame(rows)
t.to_csv(OUT, index=False)
print(f'written {OUT} (line = row + 1)')
for i, d in enumerate(rows):
    print(f"  line {i + 2}: {d['segment']} {d['location']:14s} frames {d['frames']:2d}, first frame {d['first_frame_idx']}, "
          f"timestamp_micros {d['timestamp_micros']}; UTC {d['utc']}; {d['time_zone']}: {d['local']} -> {d['part']}")
for other_tz in ('America/Los_Angeles', 'UTC'):
    use = t[(t.location != 'location_other') | (t.time_zone == other_tz)]
    print(f'summary (location_other taken in {other_tz}):')
    for part in ('morning', 'evening'):
        q = use[use.part == part]
        print(f'  {part}: {q.segment.nunique()} segments, {int(q.frames.sum())} frames '
              f'({", ".join(f"{loc} {int((q.location == loc).sum())}" for loc in TZ)})')
