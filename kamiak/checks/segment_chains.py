"""Segments cut from the same drive (user, 2026-11-12; describe only, no analysis is changed; only each segment's
frames.parquet is read, no image). For the main result's segments (238) and the sensitivity's (241)
(analysis/after_qwen3.5-4b/<frames>/ssm_target_frames.csv): each segment's first and last frame (smallest and largest
frame_idx) and their timestamp_micros; the segments sorted by their start; a segment whose start is less than T after
the latest end of the chain so far (or before it: overlapping) joins that chain, otherwise it starts a new one. This
sweep gives the same chains as linking every two segments that overlap or lie less than T apart and taking the connected
groups. T = 1 s and 60 s. Reported per T: the number of chains and their sizes; the segments in chains of two or more
and their frames (the analysis's frames), as counts and shares of all; every chain of two or more with its segments,
their first and last times (UTC) and the gaps between them.
usage: python -I segment_chains.py <waymo_v143 folder> <scene_labels.csv> <out dir>
"""
import collections
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

L, SL, O = sys.argv[1:4]
os.makedirs(O, exist_ok=True)
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
lab = pd.read_csv(SL, dtype=str, keep_default_na=False).set_index('segment')


def utc(ts):
    return (EPOCH + timedelta(microseconds=int(ts))).strftime('%Y-%m-%d %H:%M:%S.%f')


for sub in ('main_1084', 'sensitivity_1123'):
    frames = pd.read_csv(os.path.join(L, 'analysis', 'after_qwen3.5-4b', sub, 'ssm_target_frames.csv'))
    cnt = frames['segment'].value_counts()
    segs = sorted(cnt.index)
    rows, not_mono = [], []
    for s in segs:
        fr = pd.read_parquet(os.path.join(L, 'vlm_inputs', 'segments', s, 'frames.parquet'),
                             columns=['frame_idx', 'timestamp_micros']).sort_values('frame_idx')
        ts = fr['timestamp_micros'].astype('int64').to_numpy()
        if (ts[1:] <= ts[:-1]).any():
            not_mono.append(s)
        rows.append(dict(segment=s, location=lab.loc[s, 'location'].strip(), frames=int(cnt[s]),
                         first_frame_idx=int(fr['frame_idx'].iloc[0]), last_frame_idx=int(fr['frame_idx'].iloc[-1]),
                         first_ts=int(ts[0]), last_ts=int(ts[-1]), first_utc=utc(ts[0]), last_utc=utc(ts[-1])))
    t = pd.DataFrame(rows).sort_values(['first_ts', 'segment']).reset_index(drop=True)
    n_seg, n_fr = len(t), int(t.frames.sum())
    print('=' * 110)
    print(f'{sub}: {n_seg} segments, {n_fr} frames; segments whose timestamps do not rise with frame_idx: '
          f'{not_mono if not_mono else "none"}; segment lengths (last - first) '
          f'{(t.last_ts - t.first_ts).min() / 1e6:.3f}-{(t.last_ts - t.first_ts).max() / 1e6:.3f} s')
    for T in (1, 60):
        chain, end, k = [], None, -1
        for r_ in t.itertuples():
            if end is None or r_.first_ts - end >= T * 1_000_000:
                k += 1
                end = r_.last_ts
            else:
                end = max(end, r_.last_ts)
            chain.append(k)
        t[f'chain_{T}s'] = chain
        size = t.groupby(f'chain_{T}s').segment.transform('size')
        t[f'chain_size_{T}s'] = size
        multi = t[size >= 2]
        dist = collections.Counter(t.groupby(f'chain_{T}s').size())
        print(f'\nT = {T} s: {k + 1} chains; chains by number of segments {dict(sorted(dist.items()))}')
        print(f'  segments in chains of 2 or more: {len(multi)} of {n_seg} ({len(multi) / n_seg!r}); their frames '
              f'{int(multi.frames.sum())} of {n_fr} ({int(multi.frames.sum()) / n_fr!r})')
        for j, (cid, g) in enumerate(multi.groupby(f'chain_{T}s', sort=True), 1):
            print(f'  chain {j} ({len(g)} segments, {int(g.frames.sum())} frames): {g.first_utc.iloc[0]} to '
                  f'{utc(g.last_ts.max())} UTC')
            prev_end = None
            for r_ in g.itertuples():
                gap = '' if prev_end is None else f'; gap after the previous {(r_.first_ts - prev_end) / 1e6:.6f} s'
                print(f'    {r_.segment} {r_.location:14s} frames {r_.frames:2d}: {r_.first_utc} to {r_.last_utc}{gap}')
                prev_end = r_.last_ts if prev_end is None else max(prev_end, r_.last_ts)
    out = os.path.join(O, f'segment_chains_{sub}.csv')
    t.to_csv(out, index=False)
    print(f'\nwritten {out} (one row per segment, sorted by start)')
