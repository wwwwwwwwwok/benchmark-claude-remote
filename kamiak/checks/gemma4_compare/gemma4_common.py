"""Shared pieces of the Gemma 4 31B-it checks (user, 2026-11-12).

vllm_conversation(): the conversation vLLM 0.32 hands to this model's chat template. vLLM detects the template's
content format with its own _detect_content_format (render_check.py prints what it finds; this template loops over
message content, so 'openai'); in that format every message's content reaches the template as a list of parts, the
system prompt included: text parts {'type': 'text', 'text': ...}, image parts {'type': 'image'}; an assistant
message's `reasoning` is copied to `reasoning` and `reasoning_content` (vllm/renderers/chat_utils.py,
_parse_chat_message_content_part and _parse_chat_message_content). The official side gets the same conversation, so
both sides render the same text; the GPU check then compares the prompt token ids vLLM really used.
"""
import re

MAX_SOFT_TOKENS = 560                    # user, 2026-11-12
IMAGE_TOKEN, BOI, EOI = 258880, 255999, 258882      # config.json image_token_id, boi_token_id, eoi_token_id
EOS = [1, 106, 50]                       # generation_config.json eos_token_id
TABLE = {'grid': 2670, 'sep_time': 13470, 'sep_camera': 13470, 'ctrl_grid': 534, 'ctrl_sep': 2694}  # hand-computed


def vllm_conversation(messages):
    conv = []
    for m in messages:
        content = m.get('content')
        if content is None:
            content = []
        elif isinstance(content, str):
            content = [{'type': 'text', 'text': content}]
        parts = []
        for p in content:
            if p['type'] == 'text':
                parts.append({'type': 'text', 'text': p['text']})
            elif p['type'] in ('image_url', 'image'):
                parts.append({'type': 'image'})
            else:
                raise ValueError(f'unexpected content part {p["type"]}')
        c = {'role': m['role'], 'content': parts}
        if m['role'] == 'assistant' and m.get('reasoning') is not None:
            c['reasoning'] = c['reasoning_content'] = m['reasoning']
        conv.append(c)
    return conv


def openai_messages(parts, system):
    """Like code/run_vlm.py's to_messages, but without reading the images (only their places matter here)."""
    content = [{'type': 'text', 'text': v} if k == 'text' else {'type': 'image'} for k, v in parts]
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]


def image_runs(ids):
    """Per picture in the expanded prompt: the number of image tokens between <|image> and <image|>."""
    runs, i = [], 0
    while i < len(ids):
        if ids[i] == BOI:
            j = i + 1
            while j < len(ids) and ids[j] == IMAGE_TOKEN:
                j += 1
            assert j < len(ids) and ids[j] == EOI, ('picture not closed', i, j)
            runs.append(j - i - 1)
            i = j + 1
        else:
            assert ids[i] != IMAGE_TOKEN, ('image token outside a picture', i)
            i += 1
    return runs


def shorten(text, image_text='<|image|>'):
    """The rendered prompt with each run of two or more image tokens written as <|image|>xN."""
    return re.sub(f'(?:{re.escape(image_text)}){{2,}}',
                  lambda m: f'{image_text}x{len(m.group(0)) // len(image_text)}', text)
