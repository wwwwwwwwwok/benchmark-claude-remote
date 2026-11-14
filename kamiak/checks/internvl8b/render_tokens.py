"""InternVL3.5-8B (user, 2026-11-13), before its GPU runs, CPU only: the prompt tokens of 5 requests (the first main frame,
repeat 0, one per condition; the requests Gemma's check used, original JPEG inputs) as the official code builds them,
for thinking off (the model's chat_template.jinja) and on (kamiak/internvl_thinking_chat_template.jinja), against the
prompt tokens InternVL3.5-38B's main runs recorded for the same requests.
Official: the template rendered with transformers apply_chat_template (system message + user parts, images as
{"type": "image"}, add_generation_prompt), each '<image>' replaced as modeling_internvl_chat.py does
('<img>' + '<IMG_CONTEXT>' x num_image_token x tiles + '</img>'), tiles from the README's dynamic_preprocess (max_num 12,
image_size 448, thumbnail), num_image_token = (448 // 14)^2 x 0.5^2 from config.json; tokens = the tokenizer's ids
(no special tokens added). Exit 1 when a count differs from the 38B's.
usage: python -I render_tokens.py <8B model folder> <requests_original.json> <38B main jsonl.gz> <38B-thinking main jsonl.gz>
       <thinking template>
"""
import gzip
import json
import sys

from PIL import Image
from transformers import AutoTokenizer


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):   # README, as written
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def tiles(path, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    w, h = Image.open(path).size
    target_ratios = sorted(set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1)
                               for j in range(1, n + 1) if min_num <= i * j <= max_num), key=lambda x: x[0] * x[1])
    r = find_closest_aspect_ratio(w / h, target_ratios, w, h, image_size)
    blocks = r[0] * r[1]
    return blocks + (1 if use_thumbnail and blocks != 1 else 0), (w, h), r


def recorded(path, keys):
    out = {}
    with gzip.open(path, 'rt') as f:
        for line in f:
            d = json.loads(line)
            k = (d['segment'], int(d['frame_idx']), d['condition'], int(d['repeat']))
            if k in keys:
                out[k] = (d.get('usage') or {}).get('prompt_tokens')
    return out


def main():
    mdir, req_f, g38, g38t, think_t = sys.argv[1:6]
    cfg = json.load(open(f'{mdir}/config.json'))
    vc = cfg['vision_config']
    nit = int((vc['image_size'] // vc['patch_size']) ** 2 * cfg['downsample_ratio'] ** 2)
    tok = AutoTokenizer.from_pretrained(mdir, trust_remote_code=True)
    tmpl = {'off': open(f'{mdir}/chat_template.jinja').read(), 'on': open(think_t).read()}
    reqs = json.load(open(req_f))
    keys = {(r['segment'], int(r['frame_idx']), r['condition'], int(r['repeat'])) for r in reqs}
    rec = {'off': recorded(g38, keys), 'on': recorded(g38t, keys)}
    print(f'num_image_token {nit}; tokenizer {type(tok).__name__}; {len(reqs)} requests')
    bad = 0
    for r in reqs:
        k = (r['segment'], int(r['frame_idx']), r['condition'], int(r['repeat']))
        parts, n_t, sizes = [], [], set()
        for typ, val in r['parts']:
            if typ == 'text':
                parts.append({'type': 'text', 'text': val})
            else:
                parts.append({'type': 'image'})
                t, wh, ratio = tiles(val)
                n_t.append(t)
                sizes.add((wh, ratio, t))
        msgs = [{'role': 'system', 'content': r['system']}, {'role': 'user', 'content': parts}]
        for mode in ('off', 'on'):
            text = tok.apply_chat_template(msgs, chat_template=tmpl[mode], add_generation_prompt=True, tokenize=False)
            pieces = text.split('<image>')
            assert len(pieces) == len(n_t) + 1
            full = pieces[0] + ''.join('<img>' + '<IMG_CONTEXT>' * nit * t + '</img>' + p
                                       for t, p in zip(n_t, pieces[1:]))
            n = len(tok(full, add_special_tokens=False)['input_ids'])
            n0 = len(tok(text, add_special_tokens=False)['input_ids'])
            want = rec[mode].get(k)
            ok = n == want
            bad += not ok
            print(f'{r["condition"]:<10} thinking {mode:<3}: official {n} (before expansion {n0}; {len(n_t)} images, '
                  f'tiles {sorted(set(n_t))}, sizes {sorted(sizes)}), 38B recorded {want}: {"same" if ok else "DIFFERENT"}')
            if r['condition'] == 'ctrl_sep' and mode == 'on':
                print('  rendered (ctrl_sep, thinking on), images collapsed:\n' + text.replace('\n', '\\n\n'))
    print(f'differences: {bad}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
