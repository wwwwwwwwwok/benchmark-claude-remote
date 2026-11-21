"""Supplement "Further examples": replace frames hit by the lead fallback (user, 2026-11-20, after 6a4e1a3; exploratory).
Qwen3.6-27B thinking, sep_camera, main 1,084 frames. Reads tables and answer files only; opens no image.
Exclusion rule (user): a frame is excluded if it uses the fallback pairing (ego not matched to a map lane, so lead_map =
lead_path: frame_ssm map_fallback True) AND its lead never exceeds 0.5 m/s in the slow-lead rule's window (the frames of
the lead's track at least KIN_WINDOW // 2 = 9 frames from both track ends, as wod_common.moving_tracks; a track with no
such frame counts as never exceeding).
1. Per quadrant, all 1,084 frames ranked by Euclidean distance to the corner in the plane (score percentile, median of the
   9 SSM percentiles), percentile = (average rank - 1) / (1084 - 1) x 100, as in 20f9f43 (stable sort, so ties keep the
   table order, as np.argmin did in 20f9f43).
2. Walk each list from the nearest frame, take the first frame not excluded; every skipped frame is reported.
3. The frames CSV row and the 5 repeats for the chosen frame of each quadrant, with the same code as
   supp_further_examples_20261120.py (rows of unchanged quadrants are checked to be the same text as the 20f9f43 CSVs);
   nearest_to_corner_check = nearest among the frames not excluded.
usage: python -I supp_further_examples_fallback_20261120.py <out dir (the 20f9f43 folder)>"""
import io
import os
import sys
import types

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = '/home/shuyi.wang/benchmark-claude-remote'
sys.path.insert(0, f'{REPO}/code')
import analyze_vlm_ssm as av
import wod_common as wc

L = '/data/lab/cejli/sy/waymo_v143'
A = f'{L}/analysis/after_gemma-4-31b-it/main_1084'
OUT = sys.argv[1]
P = 'exploratory_further_examples_fallback_excl_'
NEW = [f'{OUT}/{P}{x}.csv' for x in ('frames', 'repeats', 'walk', 'flags_all')]
OLD_FRAMES, OLD_REPEATS = f'{OUT}/exploratory_further_examples_frames.csv', f'{OUT}/exploratory_further_examples_repeats.csv'
CURRENT = {'Q1': ('4967385055468388261_720_000_740_000', 180), 'Q2': ('11588853832866011756_2184_462_2204_462', 185),
           'Q3': ('16951470340360921766_2840_000_2860_000', 180), 'Q4': ('3270384983482134275_3220_000_3240_000', 160)}
QNAME = {'Q1': 'both high', 'Q2': 'both low', 'Q3': 'score high, SSM low', 'Q4': 'SSM high, score low'}
CORNER = {'Q1': (100, 100), 'Q2': (0, 0), 'Q3': (100, 0), 'Q4': (0, 100)}
SSMS = ['TTC', 'THW', 'DRAC', 'MTTC', 'CI', 'PICUD', 'PSD', 'TTC2D', 'ACT']
NAME = {'TTC2D': '2D-TTC'}
UNIT = {'TTC': 's', 'THW': 's', 'DRAC': 'm/s^2', 'MTTC': 's', 'CI': 'm^2/s^3', 'PICUD': 'm', 'PSD': '1 (ratio)',
        'TTC2D': 's', 'ACT': 's'}
failed = []


def check(ok, msg):
    print(('ok    ' if ok else 'FAIL  ') + msg)
    if not ok:
        failed.append(msg)


for f in NEW:
    if os.path.exists(f):
        sys.exit(f'{f} exists: stop')

fs = pd.read_csv(f'{A}/vlm_frame_scores.csv', float_precision='round_trip')
fs = fs[(fs.model == 'qwen3.6-27b') & (fs.condition == 'sep_camera')]
td = pd.read_csv(f'{A}/vlm_requests_tidy.csv', low_memory=False)
td = td[(td.model == 'qwen3.6-27b') & (td.condition == 'sep_camera')]
ssm = pd.read_csv(f'{L}/results_lead_final_excl/frame_ssm_validation_training.csv', float_precision='round_trip',
                  low_memory=False)
truth = pd.read_csv(f'{L}/quantities/truth_main.csv', float_precision='round_trip')
scene = pd.read_csv(f'{REPO}/labels/scene_labels.csv')
tgt = pd.read_csv(f'{L}/vlm_inputs/separate/targets_main.csv')
stf = pd.read_csv(f'{A}/ssm_target_frames.csv', float_precision='round_trip')
check(len(fs) == 1084 and len(stf) == 1084, f'1,084 frames in vlm_frame_scores and ssm_target_frames: {len(fs)}, {len(stf)}')

# ---------------------------------------------------------------- plane (as 20f9f43)
pl = stf[['segment', 'frame_idx'] + [f'{s}_risk' for s in SSMS]].merge(
    fs[['segment', 'frame_idx', 'score_mean']], on=['segment', 'frame_idx'], validate='one_to_one')
check(len(pl) == 1084, f'score and SSM tables matched on 1,084 frames: {len(pl)}')
pct = lambda x: (rankdata(x, method='average') - 1) / (len(x) - 1) * 100
pl['score_pct'] = pct(pl.score_mean.to_numpy(float))
for s in SSMS:
    pl[f'{s}_pct'] = pct(pl[f'{s}_risk'].to_numpy(float))
pl['ssm_pct_median'] = np.median(pl[[f'{s}_pct' for s in SSMS]].to_numpy(float), axis=1)

# ---------------------------------------------------------------- exclusion flags for all 1,084 frames
fl = pl[['segment', 'frame_idx']].merge(ssm[['segment', 'frame_idx', 'ego_lane', 'map_fallback', 'lead_map', 'lead_path',
                                             'v_lead', 'gap', 'gap_map', 'gap_path']],
                                        on=['segment', 'frame_idx'], validate='one_to_one', how='left')
check(fl.lead_map.notna().all(), 'every main frame has a lead (lead_map)')
fb = fl.map_fallback.astype(str).map({'True': True, 'False': False})
check(fb.notna().all(), 'map_fallback is True or False in every main frame')
fl['map_fallback'] = fb.astype(bool)
check(((fl.map_fallback) == fl.ego_lane.isna()).all(),
      f'map_fallback True exactly where ego_lane is empty ({int(fl.map_fallback.sum())} frames)')
check((fl.loc[fl.map_fallback, 'lead_map'] == fl.loc[fl.map_fallback, 'lead_path']).all() and
      (fl.loc[fl.map_fallback, 'gap_map'] == fl.loc[fl.map_fallback, 'gap_path']).all(),
      'in fallback frames lead_map = lead_path and gap_map = gap_path')
half = wc.KIN_WINDOW // 2
win = {}
for seg in sorted(fl.loc[fl.map_fallback, 'segment'].unique()):
    split = [d for d in ('training', 'validation') if os.path.isdir(f'{L}/tables/{d}/{seg}')]
    assert len(split) == 1, (seg, split)
    ob = pd.read_parquet(f'{L}/tables/{split[0]}/{seg}/objects.parquet',
                         columns=['frame_idx', 'id', 'type', 'speed_x', 'speed_y'])
    moving = wc.moving_tracks(types.SimpleNamespace(objects=ob))     # the rule's own function, on this segment
    span = ob.groupby('id')['frame_idx'].agg(['min', 'max'])
    for r in fl[(fl.segment == seg) & fl.map_fallback].itertuples():
        t = ob[ob['id'] == r.lead_map].sort_values('frame_idx')
        lo, hi = span.loc[r.lead_map, 'min'], span.loc[r.lead_map, 'max']
        sp = np.hypot(t.speed_x.to_numpy(float), t.speed_y.to_numpy(float))
        k = t.frame_idx.to_numpy(int)
        inner = (k - lo >= half) & (hi - k >= half)
        mx = float(sp[inner].max()) if inner.any() else float('nan')
        at = float(sp[k == r.frame_idx][0])
        never = not (inner.any() and mx > wc.LEAD_SLOW_SPEED)
        check(never == (r.lead_map not in moving), f'{seg} {r.frame_idx}: window max agrees with wod_common.moving_tracks')
        check(abs(at - float(r.v_lead)) <= 1e-13 * max(1.0, abs(float(r.v_lead))),
              f'{seg} {r.frame_idx}: label speed at t {at!r} = frame_ssm v_lead {float(r.v_lead)!r}')
        win[(seg, r.frame_idx)] = dict(lead_track_first=int(lo), lead_track_last=int(hi),
                                       window_first=int(k[inner].min()) if inner.any() else '',
                                       window_last=int(k[inner].max()) if inner.any() else '',
                                       window_frames=int(inner.sum()), lead_max_speed_window=mx,
                                       lead_type=t['type'].iloc[0])
fl['lead_speed_t'] = fl.v_lead.astype(float)
for c in ('lead_track_first', 'lead_track_last', 'window_first', 'window_last', 'window_frames',
          'lead_max_speed_window', 'lead_type'):
    fl[c] = [win.get((s, f), {}).get(c, '') for s, f in zip(fl.segment, fl.frame_idx)]
fl['excluded'] = [bool(fb_) and not (w != '' and w > wc.LEAD_SLOW_SPEED) for fb_, w in
                  zip(fl.map_fallback, fl.lead_max_speed_window)]
excl = {(s, int(f)) for s, f, e in zip(fl.segment, fl.frame_idx, fl.excluded) if e}
print(f'fallback frames among the 1,084: {int(fl.map_fallback.sum())}; excluded by the rule: {len(excl)}')
check(CURRENT['Q4'] in excl, 'the old Q4 frame is excluded by the rule')

# ---------------------------------------------------------------- walk each list
walk, chosen = [], {}
for q in ('Q1', 'Q2', 'Q3', 'Q4'):
    cx, cy = CORNER[q]
    d = np.hypot(pl.score_pct.to_numpy(float) - cx, pl.ssm_pct_median.to_numpy(float) - cy)
    order = np.argsort(d, kind='stable')
    for rank, i in enumerate(order, 1):
        key = (pl.segment.iloc[i], int(pl.frame_idx.iloc[i]))
        e = key in excl
        if not e:
            chosen[q] = (key, rank, float(d[i]))
            tie = int((d == d[i]).sum())
            check(tie == 1, f'{q}: no other frame at the same distance as the chosen one ({tie - 1} others)')
        walk.append(dict(quadrant=q, rank=rank, segment=key[0], frame_idx=key[1], distance=float(d[i]),
                         status='selected' if not e else 'skipped (excluded)'))
        if not e:
            break
    cur = CURRENT[q]
    ci = int(np.flatnonzero((pl.segment == cur[0]).to_numpy() & (pl.frame_idx == cur[1]).to_numpy())[0])
    if cur != chosen[q][0] and cur not in {(w['segment'], w['frame_idx']) for w in walk if w['quadrant'] == q}:
        walk.append(dict(quadrant=q, rank=int(np.flatnonzero(order == ci)[0]) + 1, segment=cur[0], frame_idx=cur[1],
                         distance=float(d[ci]), status='current frame of 20f9f43'))
    for w in walk:
        if w['quadrant'] == q and (w['segment'], w['frame_idx']) == cur and w['status'] != 'current frame of 20f9f43':
            w['status'] += '; current frame of 20f9f43'
    print(f'{q}: chosen {chosen[q][0]} (rank {chosen[q][1]}, distance {chosen[q][2]!r}); 20f9f43 frame {cur}')
wk = pd.DataFrame(walk).merge(fl.drop(columns=['v_lead', 'gap_map', 'gap_path']), on=['segment', 'frame_idx'], how='left')
wk = wk.merge(pl[['segment', 'frame_idx', 'score_pct', 'ssm_pct_median', 'score_mean']], on=['segment', 'frame_idx'], how='left')

# ---------------------------------------------------------------- frames and repeats (code of supp_further_examples_20261120.py)
want = {chosen[q][0] for q in chosen} | set(CURRENT.values())
got = {}
for stem in ('main_qwen3.6-27b', 'main_qwen3.6-27b_forced_answers'):
    with av.open_jsonl(av.jsonl_file(f'{L}/vlm_scores/qwen3.6-27b', stem)) as f:
        for raw in f:
            d_ = av.load_light(raw)
            k = (d_['segment'], int(d_['frame_idx']))
            if d_['condition'] == 'sep_camera' and k in want:
                got.setdefault(k + (int(d_['repeat']),), []).append((stem, d_.get('finish_reason'), d_.get('parsed')))


def build(q, seg, fr, allowed):
    """frames row and repeat rows for one frame; `allowed`: frames eligible for the nearest-corner check."""
    rr = []
    for r in range(5):
        cand = [c for c in got.get((seg, fr, r), []) if c[2]]
        check(len(cand) == 1, f'{q} {seg} {fr} repeat {r}: exactly one parsed answer ({len(cand)})')
        stem, fin, p = cand[0]
        t = td[(td.segment == seg) & (td.frame_idx == fr) & (td['repeat'] == r)]
        check(len(t) == 1 and float(t.risk.iloc[0]) == float(p['risk']) and bool(t.forced.iloc[0]) == (stem != 'main_qwen3.6-27b'),
              f'{q} {seg} {fr} repeat {r}: score {p["risk"]!r} and forced flag equal to vlm_requests_tidy.csv')
        rr.append(dict(quadrant=q, segment=seg, frame_idx=fr, repeat=r, score=p['risk'],
                       main_hazard=p.get('main_hazard'), reason=p.get('reason'),
                       hazard_type=p.get('hazard_type'), main_hazard_camera=p.get('main_hazard_camera'),
                       forced_answer=stem != 'main_qwen3.6-27b', source_file=stem, finish_reason=fin))
    m = fs[(fs.segment == seg) & (fs.frame_idx == fr)]
    smean = float(m.score_mean.iloc[0])
    own = float(np.mean([float(x['score']) for x in rr]))
    check(len(m) == 1 and int(m.n_answers.iloc[0]) == 5 and smean == own,
          f'{q} {seg} {fr}: 5 answers; frame mean {smean!r} = mean of the 5 scores {own!r}')
    rep = int(np.argmin([abs(float(x['score']) - smean) for x in rr]))      # lowest index on ties
    for x in rr:
        x['abs_diff_from_frame_mean'] = abs(float(x['score']) - smean)
        x['representative'] = x['repeat'] == rep
    s = ssm[(ssm.segment == seg) & (ssm.frame_idx == fr)]
    check(len(s) == 1, f'{q} {seg} {fr}: one row in frame_ssm')
    s = s.iloc[0]
    row = dict(quadrant=q, quadrant_name=QNAME[q], segment=seg, frame_idx=fr, t_s=s.t_s,
               timestamp_micros=int(s.timestamp_micros), score_mean=smean, representative_repeat=rep,
               representative_score=rr[rep]['score'])
    flags = []
    for c in SSMS:
        v = float(s[c])
        row[f'{NAME.get(c, c)} [{UNIT[c]}]'] = v
        if not np.isfinite(v):
            flags.append(f'{NAME.get(c, c)}={v} ({"not closing" if v == np.inf else "undefined"})')
    st = stf[(stf.segment == seg) & (stf.frame_idx == fr)].iloc[0]
    rel = max(abs(float(st[c]) - float(s[c])) / abs(float(s[c])) for c in SSMS)
    check(rel <= 1e-13, f'{q} {seg} {fr}: 9 SSM values in ssm_target_frames equal frame_ssm up to the written digits ({rel!r})')
    check(all(bool(st[f'{c}_use']) for c in SSMS), f'{q} {seg} {fr}: all 9 SSMs used in the analysis')
    row['ssm_undefined_or_capped'] = '; '.join(flags) if flags else 'none (all 9 finite; no value is capped in frame_ssm)'
    row['closing_speed [m/s]'] = float(s.closing)
    row['TTC2D_object'] = s.TTC2D_obj
    row['ACT_object'] = s.ACT_obj
    tr = truth[(truth.segment == seg) & (truth.frame_idx == fr)]
    check(len(tr) == 1 and bool(tr.has_lead.iloc[0]), f'{q} {seg} {fr}: one row in truth_main.csv, has a lead')
    tr = tr.iloc[0]
    rel = max(abs(float(tr[c]) - float(s[c])) / abs(float(s[c])) for c in ('gap', 'v_ego', 'v_lead', 'a_ego', 'a_lead'))
    check(rel <= 1e-13, f'{q} {seg} {fr}: truth_main.csv equals frame_ssm up to the written digits ({rel!r})')
    check(float(s.gap) == float(s.gap_map), f'{q} {seg} {fr}: gap = gap_map')
    row.update({'lead_gap_bumper_to_rear_along_lane [m]': float(s.gap),
                'lead_distance_from_front_camera [m]': float(tr.distance), 'front_camera_offset [m]': float(tr.camera_offset),
                'ego_speed [m/s]': float(s.v_ego), 'lead_speed [m/s]': float(s.v_lead),
                'ego_acceleration [m/s^2]': float(s.a_ego), 'lead_acceleration [m/s^2]': float(s.a_lead)})
    split = [d for d in ('training', 'validation') if os.path.isdir(f'{L}/tables/{d}/{seg}')]
    check(len(split) == 1, f'{q} {seg}: one split {split}')
    ob = pd.read_parquet(f'{L}/tables/{split[0]}/{seg}/objects.parquet')
    ob = ob[ob.frame_idx == fr]
    for name, oid in (('lead', s.lead_map), ('TTC2D_object', s.TTC2D_obj), ('ACT_object', s.ACT_obj)):
        o = ob[ob['id'] == oid]
        check(len(o) == 1, f'{q} {seg} {fr}: {name} {oid!r} found once in objects.parquet')
        row[f'{name}_type'] = o['type'].iloc[0]
        if name == 'lead':
            row['lead_id'] = oid
            row['lead_box_length [m]'] = float(o['length'].iloc[0])
            row['lead_box_width [m]'] = float(o['width'].iloc[0])
    sc = scene[scene.segment == seg]
    check(len(sc) == 1, f'{q} {seg}: one row in scene_labels.csv')
    sc = sc.iloc[0]
    row.update(weather_user_label=sc.weather, weather_waymo=sc.weather_waymo, time_of_day_waymo=sc.time_of_day,
               location_waymo=sc.location)
    g = tgt[(tgt.segment == seg) & (tgt.frame_idx == fr)]
    rel_ = [p_ for p_ in g.images_by_camera.iloc[0].split(';') if os.path.basename(p_) == f'FRONT_{fr}.jpg']
    check(len(g) == 1 and len(rel_) == 1, f'{q} {seg} {fr}: FRONT_{fr}.jpg is one of the 25 sep_camera inputs')
    pin = f'{L}/vlm_inputs/separate/{rel_[0]}'
    porig = f'{L}/tables/{split[0]}/{seg}/images/FRONT/{fr:03d}_{int(s.timestamp_micros)}.jpg'
    check(os.path.isfile(pin) and os.path.isfile(porig), f'{q} {seg} {fr}: both FRONT image paths exist (not opened)')
    row['front_image_model_input_sep_camera_now'] = pin
    row['front_image_original_full_resolution'] = porig
    pp = pl[(pl.segment == seg) & (pl.frame_idx == fr)].iloc[0]
    row['score_percentile'] = float(pp.score_pct)
    row['ssm_percentile_median'] = float(pp.ssm_pct_median)
    for c in SSMS:
        row[f'{NAME.get(c, c)}_percentile'] = float(pp[f'{c}_pct'])
    cx, cy = CORNER[q]
    sub = pl[allowed]
    dd = np.hypot(sub.score_pct - cx, sub.ssm_pct_median - cy).to_numpy()
    near = sub.iloc[int(np.argmin(dd))]
    row['nearest_to_corner_check'] = (near.segment == seg and int(near.frame_idx) == fr)
    return row, rr


def to_text(rows):
    b = io.StringIO()
    pd.DataFrame(rows).to_csv(b, index=False, float_format=None)
    return b.getvalue()


everything = np.ones(len(pl), bool)
not_excl = np.array([(s, int(f)) not in excl for s, f in zip(pl.segment, pl.frame_idx)])
# reproduction check: the 20f9f43 rows, rebuilt with this code and no exclusion, must be the same text
old_rows, old_reps = [], []
for q in ('Q1', 'Q2', 'Q3', 'Q4'):
    r_, rr_ = build(q, *CURRENT[q], everything)
    old_rows.append(r_)
    old_reps += rr_
check(to_text(old_rows) == open(OLD_FRAMES, encoding='utf-8').read(), '20f9f43 frames CSV rebuilt by this code: same text')
check(to_text(old_reps) == open(OLD_REPEATS, encoding='utf-8').read(), '20f9f43 repeats CSV rebuilt by this code: same text')

rows, reps = [], []
for q in ('Q1', 'Q2', 'Q3', 'Q4'):
    r_, rr_ = build(q, *chosen[q][0], not_excl)
    check(r_['nearest_to_corner_check'] is True or r_['nearest_to_corner_check'] == True,
          f'{q}: chosen frame is the nearest among frames not excluded')
    rows.append(r_)
    reps += rr_
    if chosen[q][0] == CURRENT[q]:
        old_line = to_text([old_rows[int(q[1]) - 1]]).splitlines()[1]
        check(to_text([r_]).splitlines()[1] == old_line, f'{q}: unchanged frame, row text equal to 20f9f43')
os.makedirs(OUT, exist_ok=True)
pd.DataFrame(rows).to_csv(NEW[0], index=False, float_format=None)
pd.DataFrame(reps).to_csv(NEW[1], index=False, float_format=None)
wk.to_csv(NEW[2], index=False, float_format=None)
fl.merge(pl[['segment', 'frame_idx', 'score_pct', 'ssm_pct_median']], on=['segment', 'frame_idx']).drop(
    columns=['gap_map', 'gap_path', 'v_lead']).to_csv(NEW[3], index=False, float_format=None)
print(wk.to_string(max_colwidth=60))
print('wrote', *NEW, sep='\n  ')
print(f'checks not met: {len(failed)}', *failed, sep='\n  ')
