"""Driver check (user, 2026-11-12): camas has NVIDIA driver 615.71.10 since its 11-12 maintenance; every earlier run had
535.274.02. Qwen3.6-27B without thinking answered the 25 greedy requests of 10-29 (first pilot frame x 5 conditions x
5 quantities, temperature 0, presence_penalty 1.5, one worker: kamiak/main_run/quantities_27b_nothink.sbatch step 4)
twice on the new driver (driver_greedy.sbatch). Compared request by request: the reply text character for character
and the number read (`parsed`), between the two new runs and between each new run and the two old runs on 535.274.02
(pp1.5 and pp1.5_again of job 31488638; the two old ones against each other counted again), plus the prompt tokens and
the settings recorded on every line. The user's reading: all 25 numbers the same -> accept the new driver.
usage: python -I driver_compare.py <new run 1> <new run 2> <old pp1.5> <old pp1.5_again>
"""
import json
import sys

paths = dict(zip(['new1', 'new2', 'old1', 'old2'], sys.argv[1:5]))
KEY = ('segment', 'frame_idx', 'condition', 'quantity', 'repeat')
SET = ('model', 'temperature', 'top_p', 'top_k', 'presence_penalty', 'max_tokens')
runs = {}
for name, p in paths.items():
    rows = [json.loads(line) for line in open(p, encoding='utf-8')]
    runs[name] = {tuple(d[k] for k in KEY): d for d in rows}
    sets = sorted({tuple(d.get(k) for k in SET) for d in rows})
    print(f'{name}: {p}\n   {len(rows)} lines, {len(runs[name])} distinct requests; settings {SET}: {sets}; finish '
          f'{sorted({d.get("finish_reason") for d in rows})}')
keys = sorted(runs['old1'])
ok_keys = all(sorted(r) == keys for r in runs.values()) and len(keys) == 25
print(f'the same 25 requests in all four runs: {ok_keys}')
same_prompt = sum(len({runs[n][k]['usage']['prompt_tokens'] for n in runs}) == 1 for k in keys)
print(f'prompt tokens the same in all four runs: {same_prompt} of {len(keys)}')


def compare(a, b):
    text = sum(runs[a][k]['text'] == runs[b][k]['text'] for k in keys)
    num = sum(runs[a][k]['parsed'] == runs[b][k]['parsed'] for k in keys)
    return text, num


print('\npairs (of 25): reply text the same character for character / number read the same')
for a, b, label in (('new1', 'new2', 'new driver, run 1 vs run 2'),
                    ('new1', 'old1', 'new run 1 vs old pp1.5'), ('new1', 'old2', 'new run 1 vs old pp1.5_again'),
                    ('new2', 'old1', 'new run 2 vs old pp1.5'), ('new2', 'old2', 'new run 2 vs old pp1.5_again'),
                    ('old1', 'old2', 'old driver, pp1.5 vs pp1.5_again (10-29, recounted)')):
    t, nm = compare(a, b)
    print(f'  {label:52s}: text {t:2d}, number {nm:2d}')

print('\nper request (number read: old pp1.5 / old pp1.5_again / new 1 / new 2; text the same as old pp1.5: new 1, new 2)')
diff_num = 0
for k in keys:
    v = [runs[n][k]['parsed'] for n in ('old1', 'old2', 'new1', 'new2')]
    same_t = [runs[n][k]['text'] == runs['old1'][k]['text'] for n in ('new1', 'new2')]
    first = next((i for i, (x, y) in enumerate(zip(runs['new1'][k]['text'], runs['old1'][k]['text'])) if x != y), None)
    flag = '' if len(set(map(str, v))) == 1 else '   NUMBER DIFFERS'
    diff_num += bool(flag)
    print(f"  {k[2]:10s} {k[3]:10s}: {v[0]} / {v[1]} / {v[2]} / {v[3]}; text same: {same_t[0]}, {same_t[1]}"
          + ('' if same_t[0] else f' (new 1 parts from old at character {first})') + flag)
all_same = ok_keys and diff_num == 0
print(f'\nrequests whose number differs anywhere among the four runs: {diff_num}')
print(f'ALL 25 NUMBERS THE SAME (new and old driver): {all_same}')
sys.exit(0 if all_same else 1)
