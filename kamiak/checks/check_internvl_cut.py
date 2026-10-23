import json, sys
sys.path.insert(0, '/home/shuyi.wang/benchmark-claude-remote/code')
from run_vlm import parse
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/data/lab/cejli/sy/models/InternVL3_5-38B')
for line in open('/home/shuyi.wang/benchmark-claude-remote/results/vlm_scores/internvl3.5-38b-thinking/pilot_internvl3.5-38b-thinking.jsonl'):
    d = json.loads(line)
    if d['finish_reason'] != 'length':
        continue
    r = d['reasoning'] or ''; t = d['text'] or ''
    th = r.strip(); n_th = len(tok.encode(th, add_special_tokens=False))
    total = d['usage']['prompt_tokens'] + n_th + 30 + 1024
    print(f"{d['segment'][:6]} f{d['frame_idx']:3d} {d['condition']:10s} r{d['repeat']} parsed={d['parsed'] is not None!s:5s} gen {d['usage']['completion_tokens']:6,} "
          f"thinking stripped {n_th:6,} tok | answer part {len(t.strip())} chars | JSON inside thinking: {parse(th) is not None} | "
          f"prompt+thinking+stop+1024 = {total:,} {'OK' if total <= 40960 else 'TOO LONG'}")
