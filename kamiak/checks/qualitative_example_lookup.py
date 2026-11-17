"""Qualitative example (user, 2026-11-16): items 6 and 95 of the 200-item validation sample. Reads tables only; the 25
sep_camera input images are copied byte for byte (shutil.copyfile, never decoded) and checked against sha256_separate.txt."""
import gzip, hashlib, json, os, shutil, sys
import pandas as pd
sys.path.insert(0, '/home/shuyi.wang/benchmark-claude-remote/code')
import analyze_vlm_ssm as av
L = '/data/lab/cejli/sy/waymo_v143'
A = f'{L}/analysis/after_gemma-4-31b-it/main_1084'
key = pd.read_csv(f'{L}/analysis/reason_27b/main_1084_llm_labels/exploratory_llm_validation_key.csv')
sample = pd.read_csv(f'{L}/to_office/reason_labels/reason_labels_sample.csv')
scene = pd.read_csv('/home/shuyi.wang/benchmark-claude-remote/labels/scene_labels.csv')
fs = pd.read_csv(f'{A}/vlm_frame_scores.csv', float_precision='round_trip')
td = pd.read_csv(f'{A}/vlm_requests_tidy.csv', low_memory=False)
ssm = pd.read_csv(f'{L}/results_lead_final_excl/frame_ssm_validation_training.csv', float_precision='round_trip', low_memory=False)
tgt = pd.read_csv(f'{L}/vlm_inputs/separate/targets_main.csv')
sha = dict(reversed(l.split(None, 1)) for l in open(f'{L}/vlm_inputs/sha256_separate.txt').read().splitlines())
sha = {k.strip(): v for k, v in sha.items()}
items = {6: 'item006', 95: 'item095'}
want = {}
for it in items:
    r = key[key.item == it].iloc[0]
    want[(r.segment, int(r.frame_idx), int(r['repeat']))] = it
found = {}
for stem in ('main_qwen3.6-27b', 'main_qwen3.6-27b_forced_answers'):
    with av.open_jsonl(av.jsonl_file(f'{L}/vlm_scores/qwen3.6-27b', stem)) as f:
        for raw in f:
            d = av.load_light(raw)
            k = (d['segment'], int(d['frame_idx']), int(d['repeat']))
            if d['condition'] == 'sep_camera' and k in want and d.get('parsed'):
                found[k] = (stem, d['parsed'], d.get('finish_reason'))
TIMES = ['minus2.0', 'minus1.5', 'minus1.0', 'minus0.5', 'now']
for k, it in want.items():
    seg, fr, rep = k
    print(f'===== item {it}: segment {seg}, frame_idx {fr}, repeat {rep}')
    print('sample text (reason) as given to the annotators:', repr(sample[sample.item == it].reason.iloc[0]))
    sc = scene[scene.segment == seg].iloc[0]
    print(f'time_of_day {sc.time_of_day}; weather (user label) {sc.weather}; weather_waymo {sc.weather_waymo}; location {sc.location}')
    stem, p, fin = found[k]
    print(f'parsed answer (from {stem}, finish_reason {fin}):')
    for f_ in ('risk', 'main_hazard_camera', 'main_hazard', 'hazard_type', 'reason'):
        print(f'   {f_}: {p.get(f_)!r}')
    print('   all keys:', json.dumps(p, ensure_ascii=False))
    t = td[(td.model == 'qwen3.6-27b') & (td.condition == 'sep_camera') & (td.segment == seg) & (td.frame_idx == fr)].sort_values('repeat')
    print('5 answers in tidy (repeat, risk, forced):', list(zip(t['repeat'], t.risk, t.forced)))
    m = fs[(fs.model == 'qwen3.6-27b') & (fs.condition == 'sep_camera') & (fs.segment == seg) & (fs.frame_idx == fr)]
    print('frame score_mean', repr(float(m.score_mean.iloc[0])), '(mean of tidy risks', t.risk.mean(), ')')
    s = ssm[(ssm.segment == seg) & (ssm.frame_idx == fr)].iloc[0]
    cols = ['TTC', 'THW', 'DRAC', 'MTTC', 'CI', 'PICUD', 'PSD', 'TTC2D', 'ACT', 'gap', 'v_ego', 'a_ego', 'v_lead', 'a_lead', 'closing',
            'lead_map', 'lead_path', 'map_fallback', 'TTC2D_obj', 'ACT_obj', 'min_dist', 'min_dist_obj', 'ego_speed', 't_s']
    for c in cols:
        print(f'   {c}: {s[c]!r}')
    split = [d for d in ('training', 'validation') if os.path.isdir(f'{L}/tables/{d}/{seg}')][0]
    ob = pd.read_parquet(f'{L}/tables/{split}/{seg}/objects.parquet')
    if 'frame_idx' in ob.columns:
        ob = ob[ob.frame_idx == fr]
    idc = [c for c in ob.columns if c in ('id', 'object_id', 'track_id')][0]
    lead = s.lead_map if not (isinstance(s.map_fallback, (bool,)) and s.map_fallback) and str(s.map_fallback) not in ('True', '1') else s.lead_path
    for name, oid in (('lead used', lead), ('TTC2D_obj', s.TTC2D_obj), ('ACT_obj', s.ACT_obj), ('min_dist_obj', s.min_dist_obj)):
        o = ob[ob[idc] == oid]
        typ = o['type'].iloc[0] if len(o) and 'type' in o.columns else 'not found'
        extra = {c: o[c].iloc[0] for c in o.columns if c in ('length', 'width', 'speed')} if len(o) else {}
        print(f'   {name}: id {oid!r}, type {typ!r} {extra}')
    row = tgt[(tgt.segment == seg) & (tgt.frame_idx == fr)].iloc[0]
    paths = row.images_by_camera.split(';')
    out = f'{L}/to_office/qualitative_example/{items[it]}'
    if os.path.exists(out):
        sys.exit(f'{out} exists: stop')
    os.makedirs(out)
    tot = 0
    for j, rel in enumerate(paths):
        cam = os.path.basename(rel).rsplit('_', 1)[0]
        name = f'{cam}_{TIMES[j % 5]}.jpg'
        src = f'{L}/vlm_inputs/separate/{rel}'
        shutil.copyfile(src, f'{out}/{name}')
        h = hashlib.sha256(open(f'{out}/{name}', 'rb').read()).hexdigest()
        assert h == sha[rel], (rel, 'sha256 mismatch')
        tot += os.path.getsize(f'{out}/{name}')
        print(f'   {name} <- {rel} ({os.path.getsize(src)} bytes, sha256 ok)')
    print(f'   folder {out}: {len(os.listdir(out))} files, {tot} bytes')
