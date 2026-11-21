"""Supplement "Further examples", 4 agreement quadrants (user, 2026-11-20; exploratory). Qwen3.6-27B thinking, sep_camera,
main 1,084-frame set. Reads tables and answer files only; opens no image (image paths are only looked up and checked to exist).
Same sources as the 2026-11-16/17 lookups (qualitative_example_lookup.py, supp_20261117_items8_9.py):
  answers   vlm_scores/qwen3.6-27b/main_qwen3.6-27b(.jsonl.gz) + main_qwen3.6-27b_forced_answers (parsed JSON, verbatim)
  scores    analysis/after_gemma-4-31b-it/main_1084/vlm_frame_scores.csv, vlm_requests_tidy.csv
  SSMs      results_lead_final_excl/frame_ssm_validation_training.csv (raw values, read with round_trip precision)
  truth     quantities/truth_main.csv; lead object type from tables/<split>/<segment>/objects.parquet
  labels    labels/scene_labels.csv (weather = the user's label; time_of_day = Waymo)
Writes <out>/exploratory_further_examples_frames.csv (one row per frame) and _repeats.csv (20 rows).
usage: python -I supp_further_examples_20261120.py <out dir (must not exist)>"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = '/home/shuyi.wang/benchmark-claude-remote'
sys.path.insert(0, f'{REPO}/code')
import analyze_vlm_ssm as av

L = '/data/lab/cejli/sy/waymo_v143'
A = f'{L}/analysis/after_gemma-4-31b-it/main_1084'
OUT = sys.argv[1]
FRAMES = [('Q1', 'both high', '4967385055468388261_720_000_740_000', 180, 64.0),
          ('Q2', 'both low', '11588853832866011756_2184_462_2204_462', 185, 8.0),
          ('Q3', 'score high, SSM low', '16951470340360921766_2840_000_2860_000', 180, 62.0),
          ('Q4', 'SSM high, score low', '3270384983482134275_3220_000_3240_000', 160, 11.0)]
SSMS = ['TTC', 'THW', 'DRAC', 'MTTC', 'CI', 'PICUD', 'PSD', 'TTC2D', 'ACT']
NAME = {'TTC2D': '2D-TTC'}
UNIT = {'TTC': 's', 'THW': 's', 'DRAC': 'm/s^2', 'MTTC': 's', 'CI': 'm^2/s^3', 'PICUD': 'm', 'PSD': '1 (ratio)',
        'TTC2D': 's', 'ACT': 's'}
failed = []


def check(ok, msg):
    print(('ok    ' if ok else 'FAIL  ') + msg)
    if not ok:
        failed.append(msg)


if os.path.exists(OUT):
    sys.exit(f'{OUT} exists: stop')
want = {(s, f) for _, _, s, f, _ in FRAMES}

# ---- answers, verbatim from the parsed JSON (forced answer used where the main run's reply has none)
got = {}
for stem in ('main_qwen3.6-27b', 'main_qwen3.6-27b_forced_answers'):
    with av.open_jsonl(av.jsonl_file(f'{L}/vlm_scores/qwen3.6-27b', stem)) as f:
        for raw in f:
            d = av.load_light(raw)
            k = (d['segment'], int(d['frame_idx']))
            if d['condition'] == 'sep_camera' and k in want:
                got.setdefault(k + (int(d['repeat']),), []).append((stem, d.get('finish_reason'), d.get('parsed')))
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
check(len(fs) == 1084 and len(stf) == 1084, f'1,084 frames in vlm_frame_scores (qwen3.6-27b, sep_camera): {len(fs)}; '
      f'in ssm_target_frames: {len(stf)}')

# ---- percentile coordinates as the user defined them (check of the frame choice only)
pl = stf[['segment', 'frame_idx'] + [f'{s}_risk' for s in SSMS]].merge(
    fs[['segment', 'frame_idx', 'score_mean']], on=['segment', 'frame_idx'], validate='one_to_one')
check(len(pl) == 1084, f'score and SSM tables matched on 1,084 frames: {len(pl)}')
pct = lambda x: (rankdata(x, method='average') - 1) / (len(x) - 1) * 100
pl['score_pct'] = pct(pl.score_mean.to_numpy(float))
for s in SSMS:
    pl[f'{s}_pct'] = pct(pl[f'{s}_risk'].to_numpy(float))
pl['ssm_pct_median'] = np.median(pl[[f'{s}_pct' for s in SSMS]].to_numpy(float), axis=1)
CORNER = {'Q1': (100, 100), 'Q2': (0, 0), 'Q3': (100, 0), 'Q4': (0, 100)}

rows, reps = [], []
for q, qname, seg, fr, mean_expected in FRAMES:
    print(f'===== {q} ({qname}): {seg} frame {fr}')
    rr = []
    for r in range(5):
        cand = [c for c in got.get((seg, fr, r), []) if c[2]]
        check(len(cand) == 1, f'{q} repeat {r}: exactly one parsed answer ({len(cand)}; '
              f'{[(c[0], c[1]) for c in got.get((seg, fr, r), [])]})')
        stem, fin, p = cand[0]
        t = td[(td.segment == seg) & (td.frame_idx == fr) & (td['repeat'] == r)]
        check(len(t) == 1 and float(t.risk.iloc[0]) == float(p['risk']) and bool(t.forced.iloc[0]) == (stem != 'main_qwen3.6-27b'),
              f'{q} repeat {r}: score {p["risk"]!r} and forced flag equal to vlm_requests_tidy.csv')
        rr.append(dict(quadrant=q, segment=seg, frame_idx=fr, repeat=r, score=p['risk'],
                       main_hazard=p.get('main_hazard'), reason=p.get('reason'),
                       hazard_type=p.get('hazard_type'), main_hazard_camera=p.get('main_hazard_camera'),
                       forced_answer=stem != 'main_qwen3.6-27b', source_file=stem, finish_reason=fin))
    m = fs[(fs.segment == seg) & (fs.frame_idx == fr)]
    smean = float(m.score_mean.iloc[0])
    own = float(np.mean([float(x['score']) for x in rr]))
    check(len(m) == 1 and int(m.n_answers.iloc[0]) == 5, f'{q}: in the main 1,084 frames, 5 answers')
    check(smean == own == mean_expected, f'{q}: frame mean {smean!r} (vlm_frame_scores) = mean of the 5 scores {own!r} '
          f'= expected {mean_expected!r}')
    dist = [abs(float(x['score']) - smean) for x in rr]
    rep = int(np.argmin(dist))                          # argmin returns the lowest index on ties
    for x in rr:
        x['abs_diff_from_frame_mean'] = abs(float(x['score']) - smean)
        x['representative'] = x['repeat'] == rep
    reps += rr

    s = ssm[(ssm.segment == seg) & (ssm.frame_idx == fr)]
    check(len(s) == 1, f'{q}: one row in frame_ssm')
    s = s.iloc[0]
    row = dict(quadrant=q, quadrant_name=qname, segment=seg, frame_idx=fr, t_s=s.t_s,
               timestamp_micros=int(s.timestamp_micros), score_mean=smean, representative_repeat=rep,
               representative_score=rr[rep]['score'])
    flags = []
    for c in SSMS:
        v = float(s[c])
        row[f'{NAME.get(c, c)} [{UNIT[c]}]'] = v
        if not np.isfinite(v):
            flags.append(f'{NAME.get(c, c)}={v} ({"not closing" if v == np.inf else "undefined"})')
    st = stf[(stf.segment == seg) & (stf.frame_idx == fr)].iloc[0]
    # ssm_target_frames.csv and truth_main.csv were written with 16 significant digits, so the last digit can differ
    # from frame_ssm (the source of every value written here); tolerance: relative difference <= 1e-13
    rel = max(abs(float(st[c]) - float(s[c])) / abs(float(s[c])) for c in SSMS)
    check(rel <= 1e-13, f'{q}: 9 SSM values in ssm_target_frames equal frame_ssm up to the written digits '
          f'(max relative difference {rel!r})')
    check(all(bool(st[f'{c}_use']) for c in SSMS), f'{q}: all 9 SSMs used in the analysis (_use true)')
    row['ssm_undefined_or_capped'] = '; '.join(flags) if flags else 'none (all 9 finite; no value is capped in frame_ssm)'
    row['closing_speed [m/s]'] = float(s.closing)
    row['TTC2D_object'] = s.TTC2D_obj
    row['ACT_object'] = s.ACT_obj

    tr = truth[(truth.segment == seg) & (truth.frame_idx == fr)]
    check(len(tr) == 1 and bool(tr.has_lead.iloc[0]), f'{q}: one row in truth_main.csv, has a lead')
    tr = tr.iloc[0]
    rel = max(abs(float(tr[c]) - float(s[c])) / abs(float(s[c])) for c in ('gap', 'v_ego', 'v_lead', 'a_ego', 'a_lead'))
    check(rel <= 1e-13, f'{q}: gap, speeds and accelerations in truth_main.csv equal frame_ssm up to the written digits '
          f'(max relative difference {rel!r})')
    check(float(s.gap) == float(s.gap_map), f'{q}: gap = gap_map (the lead used is lead_map)')
    row.update({'lead_gap_bumper_to_rear_along_lane [m]': float(s.gap),
                'lead_distance_from_front_camera [m]': float(tr.distance), 'front_camera_offset [m]': float(tr.camera_offset),
                'ego_speed [m/s]': float(s.v_ego), 'lead_speed [m/s]': float(s.v_lead),
                'ego_acceleration [m/s^2]': float(s.a_ego), 'lead_acceleration [m/s^2]': float(s.a_lead)})
    split = [d for d in ('training', 'validation') if os.path.isdir(f'{L}/tables/{d}/{seg}')]
    check(len(split) == 1, f'{q}: segment in exactly one split: {split}')
    ob = pd.read_parquet(f'{L}/tables/{split[0]}/{seg}/objects.parquet')
    ob = ob[ob.frame_idx == fr]
    for name, oid in (('lead', s.lead_map), ('TTC2D_object', s.TTC2D_obj), ('ACT_object', s.ACT_obj)):
        o = ob[ob['id'] == oid]
        check(len(o) == 1, f'{q}: {name} {oid!r} found once in objects.parquet at frame {fr}')
        row[f'{name}_type'] = o['type'].iloc[0]
        if name == 'lead':
            row['lead_id'] = oid
            row['lead_box_length [m]'] = float(o['length'].iloc[0])
            row['lead_box_width [m]'] = float(o['width'].iloc[0])
    print(f'{q}: objects.parquet columns: {list(ob.columns)}')

    sc = scene[scene.segment == seg]
    check(len(sc) == 1, f'{q}: one row in scene_labels.csv')
    sc = sc.iloc[0]
    row.update(weather_user_label=sc.weather, weather_waymo=sc.weather_waymo, time_of_day_waymo=sc.time_of_day,
               location_waymo=sc.location)

    g = tgt[(tgt.segment == seg) & (tgt.frame_idx == fr)]
    check(len(g) == 1, f'{q}: one row in vlm_inputs/separate/targets_main.csv')
    rel = [p for p in g.images_by_camera.iloc[0].split(';') if os.path.basename(p) == f'FRONT_{fr}.jpg']
    check(len(rel) == 1, f'{q}: FRONT_{fr}.jpg is one of the 25 sep_camera inputs: {rel}')
    pin = f'{L}/vlm_inputs/separate/{rel[0]}'
    porig = f'{L}/tables/{split[0]}/{seg}/images/FRONT/{fr:03d}_{int(s.timestamp_micros)}.jpg'
    check(os.path.isfile(pin) and os.path.isfile(porig), f'{q}: both FRONT image paths exist (not opened)')
    row['front_image_model_input_sep_camera_now'] = pin
    row['front_image_original_full_resolution'] = porig

    pp = pl[(pl.segment == seg) & (pl.frame_idx == fr)].iloc[0]
    row['score_percentile'] = float(pp.score_pct)
    row['ssm_percentile_median'] = float(pp.ssm_pct_median)
    for c in SSMS:
        row[f'{NAME.get(c, c)}_percentile'] = float(pp[f'{c}_pct'])
    cx, cy = CORNER[q]
    d = np.hypot(pl.score_pct - cx, pl.ssm_pct_median - cy)
    near = pl.iloc[int(np.argmin(d.to_numpy()))]
    row['nearest_to_corner_check'] = (near.segment == seg and int(near.frame_idx) == fr)
    print(f'{q}: score percentile {pp.score_pct!r}, median SSM percentile {pp.ssm_pct_median!r}; nearest frame to '
          f'corner {CORNER[q]}: {near.segment} {int(near.frame_idx)} (distance {float(d.min())!r}; this frame '
          f'{float(d[(pl.segment == seg) & (pl.frame_idx == fr)].iloc[0])!r})')
    rows.append(row)

os.makedirs(OUT)
pd.DataFrame(rows).to_csv(f'{OUT}/exploratory_further_examples_frames.csv', index=False, float_format=None)
pd.DataFrame(reps).to_csv(f'{OUT}/exploratory_further_examples_repeats.csv', index=False, float_format=None)
back = pd.read_csv(f'{OUT}/exploratory_further_examples_frames.csv', float_precision='round_trip')
check(all(back[f'{NAME.get(c, c)} [{UNIT[c]}]'].tolist() == [r[f'{NAME.get(c, c)} [{UNIT[c]}]'] for r in rows] for c in SSMS),
      'SSM values read back from the CSV equal the source floats (full precision)')
print(f'wrote {OUT}: {sorted(os.listdir(OUT))}')
print(f'checks not met: {len(failed)}', *failed, sep='\n  ')
