"""Gemma 4 31B-it without thinking (user, 2026-11-13): CPU checks before its GPU runs. Nothing is sent to a model; no
image is shown. vllm_cu129 environment (transformers 5.19.0, vLLM 0.32.0), from the repository root.
1. The 5 requests of render_check.py (one per input; the comparison PNGs and the original JPEGs of the same frame)
   rendered with the official tokenizer and processor as vLLM renders them, but with enable_thinking False (the
   server default --default-chat-template-kwargs '{"enable_thinking": false}'): tokenizer.apply_chat_template(
   add_generation_prompt=True, enable_thinking=False), then the processor expands each <|image|> at max_soft_tokens
   560. Per input: prompt tokens before / after the expansion, visual tokens against the table; PNG = JPEG ids.
2. Against the same requests with enable_thinking True (the thinking runs): the two rendered texts differ in exactly
   the template's two places, '<|think|>\\n' at the top of the system turn (thinking only) and the empty thinking
   channel '<|channel>thought\\n<channel|>' right after '<|turn>model\\n' (no thinking only); the thinking render's
   expanded ids = official_prompts.json of render_check.py (3,020 / 13,895 / 13,891 / 818 / 2,956 tokens).
3. vLLM 0.32's gemma4 reasoning parser (vllm.parser.gemma4.Gemma4Parser) built as the server builds it from its
   default kwargs (chat_template_kwargs {'enable_thinking': False}), on the non-streaming path the chat completion
   handler takes for run_vlm.py's requests (_single_pass_parse: reset to the configured start state, feed the reply
   text and token ids, finish): (a) the answer texts of the thinking pilot (first 200 lines; the text the model wrote
   after its thinking): reasoning must be None and content the text; (b) reported, not checks: a reply that starts
   with 'thought\\n' and has no <channel|>, a reply with its own '<|channel>thought ... <channel|>' block, a reply
   with the word "thought" inside; and the start state the parser would take from each rendered prompt
   (adjust_initial_state_from_prompt; must be CONTENT, the prompt ending with <channel|>).
Writes official_prompts_nothink.json (condition -> prompt_ids, prompt_ids_unexpanded) and rendered_prompts_nothink.txt.
usage: python -I render_nothink.py <model folder> <requests json> <requests_original json> <official_prompts.json>
       <thinking pilot jsonl> <out folder>
"""
import json
import os
import sys

from PIL import Image
from transformers import AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemma4_common import MAX_SOFT_TOKENS, TABLE, image_runs, openai_messages, shorten, vllm_conversation  # noqa: E402

model, req_json, orig_json, think_json, pilot, out = sys.argv[1:7]
os.makedirs(out, exist_ok=True)
bad = []
proc = AutoProcessor.from_pretrained(model)
tok = proc.tokenizer
THINK_TOP = '<|think|>\n'
EMPTY_CHANNEL = '<|channel>thought\n<channel|>'


def render(r, thinking):
    conv = vllm_conversation(openai_messages([tuple(p) for p in r['parts']], r['system']))
    text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    ids = tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=True, enable_thinking=thinking,
                                  return_dict=False)
    imgs = [Image.open(v).convert('RGB') for k, v in r['parts'] if k == 'image']
    enc = proc(text=[text], images=[imgs], add_special_tokens=False, return_tensors='pt',
               images_kwargs={'max_soft_tokens': MAX_SOFT_TOKENS})
    return text, ids, enc['input_ids'][0].tolist(), imgs


official_think = json.load(open(think_json))
print('=== 1, 2. the 5 requests, enable_thinking False against True (official tokenizer and processor, 560)')
results, rendered = {}, []
for label, path in (('PNG inputs', req_json), ('original JPEGs', orig_json)):
    for r in json.load(open(path)):
        c = r['condition']
        text_n, ids_n, exp_n, imgs = render(r, False)
        text_t, ids_t, exp_t, _ = render(r, True)
        runs = image_runs(exp_n)
        n_img = sum(n + 2 for n in runs)
        head = '<|turn>system\n'
        want_t = text_n.replace(head, head + THINK_TOP, 1)
        want_t = want_t[:-len(EMPTY_CHANNEL)] if want_t.endswith('<|turn>model\n' + EMPTY_CHANNEL) else None
        two_places = (want_t == text_t and text_n.count(head) == 1 and THINK_TOP not in text_n
                      and text_n.endswith('<|turn>model\n' + EMPTY_CHANNEL) and not text_t.endswith(EMPTY_CHANNEL))
        think_same = exp_t == official_think[c]['prompt_ids']
        d_tokens = len(exp_n) - len(exp_t)
        ok = n_img == TABLE[c] and len(runs) == len(imgs) and two_places and think_same and ids_n[0] == 2 \
            and ids_n[1] != 2
        print(f'{label:14s} {c:10s}: without thinking {len(ids_n)} tokens before and {len(exp_n)} after the expansion '
              f'(with thinking {len(exp_t)}; difference {d_tokens:+d}); visual tokens {n_img} (table {TABLE[c]}); the '
              f'texts differ only in the two places: {two_places}; the thinking render = official_prompts.json: '
              f'{think_same} -> {"ok" if ok else "NOT MET"}')
        if not ok:
            bad.append(f'{label} {c}')
        if label == 'PNG inputs':
            results[c] = dict(prompt_ids_unexpanded=ids_n, prompt_ids=exp_n, image_tokens=runs, text_prompt=text_n)
            rendered.append(f'##### {c} ({len(exp_n)} tokens)\n{shorten(tok.decode(exp_n, skip_special_tokens=False))}')
        else:
            same = exp_n == results[c]['prompt_ids']
            print(f'{"":14s} {"":10s}  expanded ids equal to the PNG inputs\': {same}')
            if not same:
                bad.append(f'original JPEGs {c}: ids differ from the PNG inputs')
diff_ids = {c: (THINK_TOP, tok.encode(THINK_TOP, add_special_tokens=False), EMPTY_CHANNEL,
                tok.encode(EMPTY_CHANNEL, add_special_tokens=False)) for c in list(results)[:1]}
for c, (a_, ia, b_, ib) in diff_ids.items():
    print(f'   tokens of {a_!r}: {ia} ({len(ia)}); of {b_!r}: {ib} ({len(ib)}); so without thinking {len(ib) - len(ia):+d}')
with open(os.path.join(out, 'official_prompts_nothink.json'), 'w') as f:
    json.dump(results, f)
with open(os.path.join(out, 'rendered_prompts_nothink.txt'), 'w') as f:
    f.write('\n\n'.join(rendered) + '\n')
print('prompt tokens without thinking (official): ' + ', '.join(f'{c} {len(v["prompt_ids"])}' for c, v in results.items()))
print('--- end of the grid prompt as rendered without thinking: ' + repr(results['grid']['text_prompt'][-200:]))

print('\n=== 3. vLLM 0.32\'s gemma4 reasoning parser with enable_thinking False (the non-streaming path)')
try:
    from vllm.parser.engine.parser_engine_config import ParserState
    from vllm.parser.gemma4 import Gemma4Parser
    parser = Gemma4Parser(tok, None, chat_template_kwargs={'enable_thinking': False})
except Exception as e:                                   # reported; the GPU small test checks the real server
    parser = None
    print(f'   could not build the parser on the CPU: {type(e).__name__}: {e}')
    bad.append('3 parser not built')
if parser is not None:
    print(f'   configured start state {parser.parser_engine_config.initial_state}; wait_for_reasoning '
          f'{parser.parser_engine_config.wait_for_reasoning}')

    def parse(text):
        ids = tok.encode(text, add_special_tokens=False)
        reasoning, content, _ = parser._single_pass_parse(text, ids)
        return reasoning, content

    texts = []
    with open(pilot, encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            if d.get('text'):
                texts.append(d['text'])
            if len(texts) == 200:
                break
    n_ok = 0
    first_bad = None
    for t in texts:
        rs, ct = parse(t)
        good = rs is None and ct is not None and ct.strip() == t.strip()
        n_ok += good
        if not good and first_bad is None:
            first_bad = (t[:200], rs if rs is None else rs[:200], ct if ct is None else ct[:200])
    print(f'   (a) {len(texts)} answer texts of the thinking pilot: reasoning None and content = the text in {n_ok}'
          + (f'; first one that is not: {first_bad}' if first_bad else ''))
    if n_ok != len(texts) or not texts:
        bad.append('3a real answers')
    a0 = texts[0] if texts else '{"risk": 10}'
    for name, t in (('starts with "thought\\n", no <channel|>', 'thought\n' + a0),
                    ('its own <|channel>thought ... <channel|> block', '<|channel>thought\nLooking at the cars.<channel|>'
                     + a0),
                    ('the word "thought" inside', a0.replace('{', '{"note": "I thought so", ', 1))):
        rs, ct = parse(t)
        print(f'   (b) {name}: reasoning {None if rs is None else repr(rs[:80])}; content '
              f'{None if ct is None else repr(ct[:80])}')
    for c, v in results.items():
        parser._reset()
        parser.adjust_initial_state_from_prompt(v['prompt_ids'])
        st = parser._engine.state
        print(f'   start state from the rendered {c} prompt: {st}')
        if st != ParserState.CONTENT:
            bad.append(f'3 start state {c}')

print(f'\nchecks not met: {len(bad)}')
for b in bad:
    print('  ', b)
sys.exit(1 if bad else 0)
