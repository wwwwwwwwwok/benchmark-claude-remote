"""Do the two 4B models get the same sampling and the same stop tokens from vLLM? (user, 2026-10-27: apart from the
3 values we send, the default sampling (repetition_penalty, min_p, ...) and the stop tokens must be the same; Qwen-Drive
has a generation_config.json, Qwen3.5-4B does not.) Done with vLLM's own code, as its server and engine do it:
try_get_generation_config(...).to_diff_dict() gives the server's default sampling values (the keys
ModelConfig.get_diff_sampling_param looks at) and the extra eos ids; our request (temperature 0.7, top_p 0.8, top_k
20, max_tokens 1024, top_logprobs 5, as run_vlm.py sends it) is filled from those defaults where it leaves a value
unset, then SamplingParams.update_from_generation_config(diff, tokenizer eos) adds the stop tokens. Prints every
sampling field of the final parameters for each model and whether the two are equal.
usage: python -I check_defaults.py <model folder A> <model folder B>
"""
import sys

from vllm import SamplingParams
from vllm.tokenizers.registry import get_tokenizer
from vllm.transformers_utils.config import try_get_generation_config

KEYS = ['repetition_penalty', 'presence_penalty', 'frequency_penalty', 'temperature', 'top_k', 'top_p', 'min_p',
        'max_new_tokens']
SENT = dict(temperature=0.7, top_p=0.8, top_k=20, max_tokens=1024, logprobs=5)
FIELDS = ['temperature', 'top_p', 'top_k', 'min_p', 'repetition_penalty', 'presence_penalty', 'frequency_penalty',
          'max_tokens', 'min_tokens', 'stop', 'stop_token_ids', 'ignore_eos', 'logprobs', 'seed', 'bad_words',
          'skip_special_tokens', 'spaces_between_special_tokens', 'include_stop_str_in_output']
res = {}
for path in sys.argv[1:3]:
    gc = try_get_generation_config(path, trust_remote_code=False)
    diff = gc.to_diff_dict() if gc is not None else {}
    defaults = {k: diff[k] for k in KEYS if diff.get(k) is not None}
    tok = get_tokenizer(path)
    kw = {('max_tokens' if k == 'max_new_tokens' else k): v for k, v in defaults.items()}
    kw.update(SENT)                       # what the request sets wins over the defaults
    sp = SamplingParams(**kw)
    sp.update_from_generation_config(diff, tok.eos_token_id)
    final = {f: getattr(sp, f) for f in FIELDS}
    final['all_stop_token_ids'] = sorted(sp._all_stop_token_ids)
    res[path] = final
    print(f'{path}:\n  generation config (non-default values): {diff}\n  server default sampling: {defaults}\n'
          f'  tokenizer eos {tok.eos_token_id} ({tok.eos_token!r}); final sampling: {final}')
a, b = res.values()
diffs = {f: (a[f], b[f]) for f in a if a[f] != b[f]}
print(f'\nfields that differ between the two: {diffs if diffs else "none"}')
print(f'SAME SAMPLING AND STOP TOKENS: {not diffs}')
