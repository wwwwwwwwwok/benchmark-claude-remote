"""Supplement "Further examples", Q4 only: continue the walk of 65c8960 (user, 2026-11-20; exploratory). Qwen3.6-27B
thinking, sep_camera, main 1,084 frames. Reads tables and answer files only; opens no image.
Rule: the fallback exclusion of 65c8960 (exploratory_further_examples_fallback_excl_flags_all.csv, column excluded) still
applies; in addition the user skipped rank 3 (2336233899565126347_1180_000_1200_000 f185: lead parked at the curb, judged
by the user from the FRONT image). This script lists the next 6 frames after rank 3 that are not excluded (frames skipped
by the fallback exclusion on the way are listed too), with the lead re-paired by the repository's own functions to get
its lateral offset from the reference line (ego lane centreline route, or the ego future path when the ego has no map
lane), and writes for each the frames-CSV row and the 5 repeats with the code of supp_further_examples_fallback_20261120.py
(build() copied verbatim except one fix: FRONT input names are zero-padded, see the comment in build(); checked by
rebuilding the 65c8960 Q4 row and repeats, which must be the same text).
nearest_to_corner_check here = nearest among frames not excluded by the fallback rule and not skipped by the user so far,
so only the first of the 6 can be True.
usage: python -I supp_q4_walk_continue_20261120.py <out dir (the 20f9f43 folder)>"""
import io
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = '/home/shuyi.wang/benchmark-claude-remote'
sys.path.insert(0, f'{REPO}/code')
import analyze_vlm_ssm as av
import waymo_ssm_frames as wsf
import wod_common as wc

L = '/data/lab/cejli/sy/waymo_v143'
A = f'{L}/analysis/after_gemma-4-31b-it/main_1084'
OUT = sys.argv[1]
P = 'exploratory_further_examples_q4_continue_'
NEW = [f'{OUT}/{P}{x}.csv' for x in ('candidates', 'frames', 'repeats')]
FLAGS = f'{OUT}/exploratory_further_examples_fallback_excl_flags_all.csv'
WALK1 = f'{OUT}/exploratory_further_examples_fallback_excl_walk.csv'
F65, R65 = f'{OUT}/exploratory_further_examples_fallback_excl_frames.csv', f'{OUT}/exploratory_further_examples_fallback_excl_repeats.csv'
USER_SKIPPED = {('2336233899565126347_1180_000_1200_000', 185): 'lead parked at the curb (user, FRONT image)'}
N_WANT = 6
QNAME = {'Q1': 'both high', 'Q2': 'both low', 'Q3': 'score high, SSM low', 'Q4': 'SSM high, score low'}
CORNER = {'Q1': (100, 100), 'Q2': (0, 0), 'Q3': (100, 0), 'Q4': (0, 100)}
SSMS = ['TTC', 'THW', 'DRAC', 'MTTC', 'CI', 'PICUD', 'PSD', 'TTC2D', 'ACT']
NAME = {'TTC2D': '2D-TTC'}
UNIT = {'TTC': 's', 'THW': 's', 'DRAC': 'm/s^2', 'MTTC': 's', 'CI': 'm^2/s^3', 'PICUD': 'm', 'PSD': '1 (ratio)',
        'TTC2D': 's', 'ACT': 's'}
CAMS = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
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
check(len(fs) == 1084 and len(stf) == 1084, f'1,084 frames: {len(fs)}, {len(stf)}')
pl = stf[['segment', 'frame_idx'] + [f'{s}_risk' for s in SSMS]].merge(
    fs[['segment', 'frame_idx', 'score_mean']], on=['segment', 'frame_idx'], validate='one_to_one')
pct = lambda x: (rankdata(x, method='average') - 1) / (len(x) - 1) * 100
pl['score_pct'] = pct(pl.score_mean.to_numpy(float))
for s in SSMS:
    pl[f'{s}_pct'] = pct(pl[f'{s}_risk'].to_numpy(float))
pl['ssm_pct_median'] = np.median(pl[[f'{s}_pct' for s in SSMS]].to_numpy(float), axis=1)

flags = pd.read_csv(FLAGS, float_precision='round_trip')
check(len(flags) == 1084 and int(flags.excluded.sum()) == 8, f'65c8960 flags: {len(flags)} frames, {int(flags.excluded.sum())} excluded')
excl = {(s, int(f)) for s, f, e in zip(flags.segment, flags.frame_idx, flags.excluded) if e}
chk = pl.merge(flags[['segment', 'frame_idx', 'score_pct', 'ssm_pct_median']], on=['segment', 'frame_idx'], suffixes=('', '_65'))
check(len(chk) == 1084 and (chk.score_pct == chk.score_pct_65).all() and (chk.ssm_pct_median == chk.ssm_pct_median_65).all(),
      'plane recomputed = the 65c8960 flags table, every frame')
cx, cy = CORNER['Q4']
dist = np.hypot(pl.score_pct.to_numpy(float) - cx, pl.ssm_pct_median.to_numpy(float) - cy)
order = np.argsort(dist, kind='stable')
w1 = pd.read_csv(WALK1, float_precision='round_trip')
w1 = w1[w1.quadrant == 'Q4']
check([(pl.segment.iloc[i], int(pl.frame_idx.iloc[i]), float(dist[i])) for i in order[:3]] ==
      [(s, int(f), float(d)) for s, f, d in zip(w1.segment, w1.frame_idx, w1.distance)],
      'ranks 1-3 and distances = the 65c8960 Q4 walk')


# ---------------------------------------------------------------- build() and to_text(): copied verbatim from supp_further_examples_fallback_20261120.py
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
    # the one change to the copied build(): input names are zero-padded to 3 digits (FRONT_020.jpg); the copied code
    # matched FRONT_{fr}.jpg, which is the same for every frame >= 100 (all frames of 20f9f43 and 65c8960)
    rel_ = [p_ for p_ in g.images_by_camera.iloc[0].split(';') if os.path.basename(p_) == f'FRONT_{fr:03d}.jpg']
    check(len(g) == 1 and len(rel_) == 1, f'{q} {seg} {fr}: FRONT_{fr:03d}.jpg is one of the 25 sep_camera inputs')
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




# ---------------------------------------------------------------- continue the walk after rank 3
cand, earlier_segs = [], {}
for rank, i in enumerate(order[:3], 1):
    earlier_segs.setdefault(pl.segment.iloc[i], []).append(f'rank {rank} f{int(pl.frame_idx.iloc[i])}')
rank = 3
skipped_now = set(USER_SKIPPED) | excl
for i in order[3:]:
    rank += 1
    key = (pl.segment.iloc[i], int(pl.frame_idx.iloc[i]))
    same = '; '.join(earlier_segs.get(key[0], []))
    cand.append(dict(rank=rank, segment=key[0], frame_idx=key[1], distance=float(dist[i]),
                     score_percentile=float(pl.score_pct.iloc[i]), ssm_percentile_median=float(pl.ssm_pct_median.iloc[i]),
                     status='skipped (fallback exclusion of 65c8960)' if key in excl else 'candidate',
                     same_segment_as_earlier=same if same else 'no'))
    earlier_segs.setdefault(key[0], []).append(f'rank {rank} f{key[1]}')
    if sum(c['status'] == 'candidate' for c in cand) == N_WANT:
        break
check(sum(c['status'] == 'candidate' for c in cand) == N_WANT, f'{N_WANT} candidates found')

# lead re-paired at each candidate frame
segcache = {}
for c in cand:
    s = ssm[(ssm.segment == c['segment']) & (ssm.frame_idx == c['frame_idx'])].iloc[0]
    c.update(lead_id=s.lead_map, map_fallback=str(s.map_fallback), ego_lane='' if pd.isna(s.ego_lane) else str(int(float(s.ego_lane))),
             **{'lead_speed_t [m/s]': float(s.v_lead), 'ego_speed [m/s]': float(s.v_ego), 'gap [m]': float(s.gap)})
    if c['status'] != 'candidate':
        continue
    name, k = c['segment'], c['frame_idx']
    if name not in segcache:
        split = [d for d in ('training', 'validation') if os.path.isdir(f'{L}/tables/{d}/{name}')]
        assert len(split) == 1, (name, split)
        sg = wc.Segment(f'{L}/tables/{split[0]}', name)
        lm = wc.LaneMap(sg.map)
        n = len(sg.frames)
        el, es = lm.match_sequence(sg.ego_map[:, :2], sg.yaw) if lm.ok else ([None] * n, np.full(n, np.nan))
        segcache[name] = (sg, lm, el, es)
    sg, lm, el, es = segcache[name]
    wsf.use_ego(*wc.footprint_from_lidars(sg.context))          # this segment's footprint (global in wod_common)
    o_all = sg.objects_at(k, cameras=CAMS, types=['vehicle', 'pedestrian', 'cyclist'])
    o_rng = o_all[wc.box_nearest_dist(o_all['cx'], o_all['cy'], o_all['length'], o_all['width'], o_all['heading'])
                  <= wc.LABEL_RANGE]
    o_lead = o_rng[o_rng['type'].isin(wc.LEAD_TYPES)].reset_index(drop=True)
    r = wc.lead_map(sg, k, o_lead, lm, el, es)
    check(r is not None and o_lead.loc[r['i'], 'id'] == s.lead_map and
          abs(r['gap'] - float(s.gap)) <= 1e-13 * abs(float(s.gap)) and bool(r['fallback']) == (str(s.map_fallback) == 'True'),
          f'rank {c["rank"]} {name} f{k}: rebuilt pairing = frame_ssm (lead {s.lead_map}, gap, fallback {s.map_fallback})')
    lane0 = el[k]
    if lane0 is None:
        xy, s_ego, ref = wc.ego_future_path(sg, k), 0.0, 'ego future path (no ego lane matched)'
    else:
        _, xy = lm.route(lane0, es[k], [l for l in el[k:] if l is not None])
        s_ego, ref = float(es[k]), f'ego lane centreline route (map lane {lane0})'
    Pm, hdg, _, _ = sg.to_map(k, o_lead)
    _, d, ph = wc.project(xy, wc.cumlen(xy), Pm)
    i = int(r['i'])
    tr = sg.objects[sg.objects['id'] == s.lead_map]
    sp = np.hypot(tr.speed_x.to_numpy(float), tr.speed_y.to_numpy(float))
    kk = tr.frame_idx.to_numpy(int)
    c.update({'lead_lateral_offset [m]': float(d[i]), 'reference_line': ref,
              'lead_heading_minus_reference [deg]': float(np.degrees(wc.wrap(hdg[i] - ph[i]))),
              'lead_max_speed_whole_track [m/s]': float(sp.max()), 'lead_max_speed_frame': int(kk[sp.argmax()]),
              'lead_track_first': int(kk.min()), 'lead_track_last': int(kk.max()),
              'lead_frames_above_0.5_m/s': int((sp > 0.5).sum()), 'lead_type': tr['type'].iloc[0]})
    check(abs(sp[kk == k][0] - float(s.v_lead)) <= 1e-13 * max(1.0, float(s.v_lead)),
          f'rank {c["rank"]}: lead label speed at t = frame_ssm v_lead')

# ---------------------------------------------------------------- frames rows and repeats
picked = [(c['segment'], c['frame_idx']) for c in cand if c['status'] == 'candidate']
want = set(picked) | set(USER_SKIPPED)
got = {}
for stem in ('main_qwen3.6-27b', 'main_qwen3.6-27b_forced_answers'):
    with av.open_jsonl(av.jsonl_file(f'{L}/vlm_scores/qwen3.6-27b', stem)) as f:
        for raw in f:
            d_ = av.load_light(raw)
            k = (d_['segment'], int(d_['frame_idx']))
            if d_['condition'] == 'sep_camera' and k in want:
                got.setdefault(k + (int(d_['repeat']),), []).append((stem, d_.get('finish_reason'), d_.get('parsed')))

not_excl = np.array([(s, int(f)) not in excl for s, f in zip(pl.segment, pl.frame_idx)])
# reproduction: the 65c8960 Q4 row and repeats (rank 3, nearest among frames not excluded by the fallback rule)
r3, rr3 = build('Q4', *list(USER_SKIPPED)[0], not_excl)
check(to_text([r3]).splitlines()[1] == open(F65, encoding='utf-8').read().splitlines()[4], '65c8960 Q4 frames row rebuilt: same text')
check(to_text(rr3).splitlines()[1:] == open(R65, encoding='utf-8').read().splitlines()[16:21], '65c8960 Q4 repeats rebuilt: same text')
allowed = np.array([(s, int(f)) not in skipped_now for s, f in zip(pl.segment, pl.frame_idx)])
rows, reps = [], []
for j, key in enumerate(picked):
    r_, rr_ = build('Q4', *key, allowed)
    check(r_['nearest_to_corner_check'] == (j == 0), f'candidate {j + 1}: nearest_to_corner_check {r_["nearest_to_corner_check"]}')
    rows.append(r_)
    reps += rr_
    cc = next(c for c in cand if (c['segment'], c['frame_idx']) == key)
    cc['score_mean'] = r_['score_mean']
    cc['representative_repeat'] = r_['representative_repeat']
    check(cc['ego_speed [m/s]'] == r_['ego_speed [m/s]'] and cc['gap [m]'] == r_['lead_gap_bumper_to_rear_along_lane [m]'],
          f'candidate {j + 1}: ego speed and gap equal in both tables')
pd.DataFrame(cand).to_csv(NEW[0], index=False, float_format=None)
pd.DataFrame(rows).to_csv(NEW[1], index=False, float_format=None)
pd.DataFrame(reps).to_csv(NEW[2], index=False, float_format=None)
print(pd.DataFrame(cand).to_string(max_colwidth=50))
print('wrote', *NEW, sep='\n  ')
print(f'checks not met: {len(failed)}', *failed, sep='\n  ')
