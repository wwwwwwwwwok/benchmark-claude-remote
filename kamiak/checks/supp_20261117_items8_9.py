"""Supplement facts (user, 2026-11-17): item 8 (one frame's five answers; its 25 sep_camera inputs copied by bytes, never
decoded, checked against sha256_separate.txt) and item 9 (9-setting heatmap; the drop option first checked to leave the
11-setting output of 2026-11-15 byte-identical)."""
import filecmp, hashlib, os, shutil, sys
import pandas as pd
sys.path.insert(0, '/home/shuyi.wang/benchmark-claude-remote/code')
import analyze_vlm_ssm as av
import supplement_20261114 as sup
import sec47_package as s47
L = '/data/lab/cejli/sy/waymo_v143'
P = os.path.dirname(os.path.abspath(__file__))
# ---- item 8
SEG, FR = '17912777897400903477_2047_500_2067_500', 75
A = f'{L}/analysis/after_gemma-4-31b-it/main_1084'
fs = pd.read_csv(f'{A}/vlm_frame_scores.csv', float_precision='round_trip')
m = fs[(fs.model == 'qwen3.6-27b') & (fs.condition == 'sep_camera') & (fs.segment == SEG) & (fs.frame_idx == FR)]
print(f'item 8: {SEG} frame {FR}; in the main 1,084 frames: {len(m) == 1}; score_mean {m.score_mean.iloc[0] if len(m) else None!r}')
got = {}
for stem in ('main_qwen3.6-27b', 'main_qwen3.6-27b_forced_answers'):
    with av.open_jsonl(av.jsonl_file(f'{L}/vlm_scores/qwen3.6-27b', stem)) as f:
        for raw in f:
            d = av.load_light(raw)
            if d['condition'] == 'sep_camera' and d['segment'] == SEG and int(d['frame_idx']) == FR:
                if d.get('parsed') or stem == 'main_qwen3.6-27b':
                    if d.get('parsed') or int(d['repeat']) not in got:
                        got[int(d['repeat'])] = (stem, d.get('finish_reason'), d.get('parsed'))
for r in sorted(got):
    stem, fin, p = got[r]
    p = p or {}
    print(f'  repeat {r} ({stem}, finish {fin}): risk {p.get("risk")!r}; camera {p.get("main_hazard_camera")!r}; '
          f'main_hazard {p.get("main_hazard")!r}; hazard_type {p.get("hazard_type")!r}\n     reason: {p.get("reason")!r}')
sc = pd.read_csv('/home/shuyi.wang/benchmark-claude-remote/labels/scene_labels.csv')
s = sc[sc.segment == SEG].iloc[0]
print(f'  time_of_day {s.time_of_day}, location {s.location}, weather (user) {s.weather}')
tgt = pd.read_csv(f'{L}/vlm_inputs/separate/targets_main.csv')
row = tgt[(tgt.segment == SEG) & (tgt.frame_idx == FR)].iloc[0]
sha = {}
for l in open(f'{L}/vlm_inputs/sha256_separate.txt').read().splitlines():
    h, n = l.split(None, 1)
    sha[n.strip()] = h
TIMES = ['minus2.0', 'minus1.5', 'minus1.0', 'minus0.5', 'now']
out = f'{L}/to_office/qualitative_example/{SEG}_f{FR:03d}'
if os.path.exists(out):
    sys.exit(f'{out} exists: stop')
os.makedirs(out)
tot = 0
for j, rel in enumerate(row.images_by_camera.split(';')):
    name = f'{os.path.basename(rel).rsplit("_", 1)[0]}_{TIMES[j % 5]}.jpg'
    shutil.copyfile(f'{L}/vlm_inputs/separate/{rel}', f'{out}/{name}')
    assert hashlib.sha256(open(f'{out}/{name}', 'rb').read()).hexdigest() == sha[rel], rel
    tot += os.path.getsize(f'{out}/{name}')
print(f'  copied 25 images (sha256 all match sha256_separate.txt) to {out}: {len(os.listdir(out))} files, {tot} bytes')
# ---- item 9
chk = f'{P}/heatmap_11_check'
os.makedirs(chk, exist_ok=True)
sup.heatmap(L, chk, folder=s47.FOLDER, order=s47.ORDER)
ref = f'{L}/analysis/sec47_20261115'
for f_ in ('exploratory_supp_heatmap_models_formats.csv', 'exploratory_supp_heatmap_models_formats.pdf',
           'exploratory_supp_heatmap_models_formats.png'):
    sup.check(filecmp.cmp(f'{chk}/{f_}', f'{ref}/{f_}', shallow=False), f'11-setting heatmap rerun with the edited program: {f_} byte-identical to {ref}')
O9 = f'{L}/to_office/analysis/heatmap_9settings_20261117'
if os.path.exists(O9):
    sys.exit(f'{O9} exists: stop')
os.makedirs(O9)
ORDER9 = ['qwen3.6-27b', 'qwen3-vl-32b-thinking', 'qwen3-vl-32b', 'gemma-4-31b-it', 'gemma-4-31b-it-nothink',
          'internvl3.5-38b-thinking', 'internvl3.5-38b', 'internvl3.5-8b-thinking', 'internvl3.5-8b']
p_ag, models = sup.heatmap(L, O9, folder=s47.FOLDER, order=ORDER9, drop=['qwen-drive-1.0-4b', 'qwen3.5-4b'])
d9 = pd.read_csv(f'{O9}/exploratory_supp_heatmap_models_formats.csv', dtype=str)
d11 = pd.read_csv(f'{ref}/exploratory_supp_heatmap_models_formats.csv', dtype=str)
same = d11[d11.model.isin(ORDER9)].sort_values(['condition', 'model', 'ssm']).reset_index(drop=True)
sup.check(d9.sort_values(['condition', 'model', 'ssm']).reset_index(drop=True).equals(same) and len(d9) == 9 * 5 * 9,
          f'9-setting CSV: 405 rows, each the same text as the 11-setting CSV row')
print(f'item 9: source {p_ag}; settings {models}; SSMs {sup.SSMS}; output {O9}: {sorted(os.listdir(O9))}')
print(f'checks not met: {len(sup.failed)}', *sup.failed, sep='\n  ')
