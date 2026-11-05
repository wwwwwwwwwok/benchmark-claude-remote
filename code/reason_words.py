"""EXPLORATORY, added after the trees, the ceiling and the resampled trees were seen (user, 2026-11-05; CLAUDE.md
"找模型自己的规则", the entry of 2026-11-05): which words Qwen3.6-27B writes in its answers on the frames it scores high and
on those it scores low. Only counts are written; no reply's text is written or printed.

Replies: the risk run's <model dir>/main_<model>.jsonl(.gz) and its _forced_answers file, read as analyze_vlm_ssm.py does
(read_model's rules: a forced answer replaces a reply without a readable answer); of each reply's final JSON (`parsed`)
the two fields main_hazard and reason, not the thinking. Checked against the risk analysis's vlm_requests_tidy.csv:
every request of these frames is there once, the risk score and main_hazard the same.
Words: lower case; only the letters a-z kept (anything else, hyphens and digits too, splits words); scikit-learn's
ENGLISH_STOP_WORDS taken out; no stemming. A frame has a word when one of its replies (main_hazard or reason) has it.
Per condition: the frames' risk score (score_mean) ranked among them (scipy rankdata, ties the mean rank) and cut at the
tertiles of the rank (np.quantile 1/3, 2/3: low <= q1 < middle <= q2 < high, as analyze_rules.tertile); for every word in
at least MIN_FRAMES frames: the frames with it in each tier and the share of the tier; difference high - low. The 25
words with the largest and the 25 with the smallest difference (ties: alphabetical), per condition; the words in every
condition's 25.
Outputs (out_dir/<subfolder>): exploratory_words.csv (every word in MIN_FRAMES frames or more), exploratory_words_top.csv,
exploratory_words_common.csv, exploratory_words_params.txt.
usage: python -I reason_words.py <model dir (vlm_scores/qwen3.6-27b)> <risk analysis folder (after_qwen3.5-4b)>
       <out_dir> [subfolder ...]   (default main_1084 sensitivity_1123)
"""
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_vlm_ssm as av  # noqa: E402

MODEL = 'qwen3.6-27b'
FIELDS = ('main_hazard', 'reason')
MIN_FRAMES = 30
TOP = 25
WORD = re.compile(r'[a-z]+')


def words_of(text):
    """The words of one text: lower case, runs of a-z, stop words out (None: no words; anything else not a string is
    taken as its str())."""
    if text is None:
        return set()
    if not isinstance(text, str):
        text = str(text)
    return {w for w in WORD.findall(text.lower()) if w not in ENGLISH_STOP_WORDS}


def frame_words_of(replies):
    """The words of one frame: replies = [(risk, main_hazard, reason), ...]; the union over the replies with a
    readable answer (risk not NaN) of the words of main_hazard and reason."""
    out = set()
    for r_, mh, rs in replies:
        if not np.isnan(r_):
            out |= words_of(mh) | words_of(rs)
    return out


def same_text(x, y):
    """A reply's field (None, a string, or something else) against the tidy table's cell (read with only '' as NaN)."""
    if x is None or (isinstance(x, str) and x == ''):
        return pd.isna(y)
    return (x if isinstance(x, str) else str(x)) == y


def tertile(x):
    q1, q2 = np.quantile(x, [1 / 3, 2 / 3])
    return np.where(x <= q1, 1, np.where(x <= q2, 2, 3)), (float(q1), float(q2))


def word_table(frame_words, score):
    """frame_words: list of sets (one per frame); score: the frames' risk scores. Returns the tiers, their cuts and the
    table of every word in MIN_FRAMES frames or more."""
    rk = rankdata(score) / len(score)
    tier, cuts = tertile(rk)
    n_t = {t: int((tier == t).sum()) for t in (1, 2, 3)}
    count = {}
    for ws, t in zip(frame_words, tier):
        for w in ws:
            c = count.setdefault(w, [0, 0, 0])
            c[t - 1] += 1
    rows = []
    for w, (lo, mid, hi) in count.items():
        tot = lo + mid + hi
        if tot >= MIN_FRAMES:
            rows.append(dict(word=w, frames=tot, frames_high=hi, frames_middle=mid, frames_low=lo,
                             share_high=hi / n_t[3], share_low=lo / n_t[1], difference=hi / n_t[3] - lo / n_t[1]))
    t = pd.DataFrame(rows, columns=['word', 'frames', 'frames_high', 'frames_middle', 'frames_low', 'share_high',
                                    'share_low', 'difference'])
    return tier, cuts, n_t, t


def top_lists(t):
    hi = t.sort_values(['difference', 'word'], ascending=[False, True]).head(TOP)
    lo = t.sort_values(['difference', 'word'], ascending=[True, True]).head(TOP)
    return hi, lo


def read_replies(model_dir, keys):
    """{request key: (risk, main_hazard, reason)} for the requests in keys, the forced answer where the reply has none."""
    main = av.jsonl_file(model_dir, f'main_{MODEL}')
    forced = av.jsonl_file(model_dir, f'main_{MODEL}_forced_answers')
    out, n_lines = {}, 0
    with av.open_jsonl(main) as f:
        for raw in f:
            n_lines += 1
            d = av.load_light(raw)
            k = (d['segment'], int(d['frame_idx']), d['condition'], int(d['repeat']))
            if k not in keys:
                continue
            if k in out:
                raise ValueError(f'{main}: duplicate request {k}')
            p = d.get('parsed')
            out[k] = (float(p['risk']), p.get('main_hazard'), p.get('reason')) if p else (np.nan, None, None)
    n_forced = 0
    if forced:
        with av.open_jsonl(forced) as f:
            for raw in f:
                d = av.load_light(raw)
                k = (d['segment'], int(d['frame_idx']), d['condition'], int(d['repeat']))
                if k not in keys:
                    continue
                if not np.isnan(out[k][0]):
                    raise ValueError(f'{forced}: forced answer for {k}, but the original reply has one')
                p = d.get('parsed')
                if p:
                    out[k] = (float(p['risk']), p.get('main_hazard'), p.get('reason'))
                    n_forced += 1
    return out, dict(main=main, forced=forced, main_lines=n_lines, forced_used=n_forced)


def main():
    model_dir, risk_root, out_root = sys.argv[1:4]
    subs = sys.argv[4:] or ['main_1084', 'sensitivity_1123']
    t0 = time.time()
    tidy = {}
    for s in subs:
        if os.path.exists(os.path.join(out_root, s, 'exploratory_words_params.txt')):
            sys.exit(f'{out_root}/{s} has exploratory_words_params.txt already: give a new folder')
        rq = pd.read_csv(os.path.join(risk_root, s, 'vlm_requests_tidy.csv'), low_memory=False,
                         usecols=['model', 'condition', 'segment', 'frame_idx', 'repeat', 'risk', 'main_hazard'],
                         keep_default_na=False, na_values=[''], dtype={'main_hazard': str})
        tidy[s] = rq[rq.model == MODEL].reset_index(drop=True)
    keys = set()
    for s in subs:
        keys |= set(zip(tidy[s].segment, tidy[s].frame_idx.astype(int), tidy[s].condition, tidy[s]['repeat'].astype(int)))
    rep, info = read_replies(model_dir, keys)
    print(f'read {info["main_lines"]} lines of {info["main"]}; {len(rep)} requests of these frames; forced answers used '
          f'{info["forced_used"]} ({time.time() - t0:.0f} s)', flush=True)
    failed = []
    for s in subs:
        rq = tidy[s]
        out = os.path.join(out_root, s)
        os.makedirs(out, exist_ok=True)
        ks = list(zip(rq.segment, rq.frame_idx.astype(int), rq.condition, rq['repeat'].astype(int)))
        missing = sum(1 for k in ks if k not in rep)
        r_here = np.array([rep[k][0] if k in rep else np.nan for k in ks])
        r_tidy = rq.risk.to_numpy(float)
        bad_risk = int((~((r_here == r_tidy) | (np.isnan(r_here) & np.isnan(r_tidy)))).sum())
        mh_here = [rep[k][1] if k in rep else None for k in ks]
        bad_mh = int(sum(1 for x, y in zip(mh_here, rq.main_hazard) if not same_text(x, y)))
        checks = [f'requests of {MODEL} in vlm_requests_tidy.csv: {len(ks)}; not among the replies read {missing}; '
                  f'risk score not the same {bad_risk}; main_hazard not the same {bad_mh}']
        if missing or bad_risk or bad_mh:
            failed.append(f'{s}: replies against vlm_requests_tidy.csv')
        fsc = pd.read_csv(os.path.join(risk_root, s, 'vlm_frame_scores.csv'), float_precision='round_trip')
        st = pd.read_csv(os.path.join(risk_root, s, 'ssm_target_frames.csv'))[['segment', 'frame_idx']] \
            .sort_values(['segment', 'frame_idx']).reset_index(drop=True)
        mi = pd.MultiIndex.from_frame(st)
        allw, tops, info_rows = [], {}, []
        for c in av.CONDITIONS:
            q = rq[rq.condition == c]
            by_frame = {}
            for sg, fi, rp in zip(q.segment, q.frame_idx.astype(int), q['repeat'].astype(int)):
                by_frame.setdefault((sg, fi), []).append(rep[(sg, fi, c, rp)])
            fw = {k_: frame_words_of(v) for k_, v in by_frame.items()}
            sc = fsc[(fsc.model == MODEL) & (fsc.condition == c)].set_index(['segment', 'frame_idx'])['score_mean'] \
                .reindex(mi).to_numpy(float)
            frame_words = [fw.get((sg, int(fi)), set()) for sg, fi in zip(st.segment, st.frame_idx)]
            tier, cuts, n_t, t = word_table(frame_words, sc)
            hi, lo = top_lists(t)
            tops[c] = (set(hi.word), set(lo.word))
            t = t.assign(condition=c, rank_high=t.word.map({w: i + 1 for i, w in enumerate(hi.word)}),
                         rank_low=t.word.map({w: i + 1 for i, w in enumerate(lo.word)}))
            allw.append(t)
            info_rows.append(dict(condition=c, frames=len(st), frames_low=n_t[1], frames_middle=n_t[2],
                                  frames_high=n_t[3], rank_cut_1=cuts[0], rank_cut_2=cuts[1],
                                  score_low_max=float(sc[tier == 1].max()), score_middle_min=float(sc[tier == 2].min()),
                                  score_middle_max=float(sc[tier == 2].max()), score_high_min=float(sc[tier == 3].min()),
                                  frames_without_words=sum(1 for w in frame_words if not w),
                                  distinct_words=len(set().union(*frame_words)), words_in_30_frames=len(t)))
            print(f'{s} {c}: tiers {n_t}, words in {MIN_FRAMES}+ frames {len(t)} ({time.time() - t0:.0f} s)', flush=True)
        words = pd.concat(allw, ignore_index=True)
        words.to_csv(os.path.join(out, 'exploratory_words.csv'), index=False)
        top = words[words.rank_high.notna() | words.rank_low.notna()]
        top.to_csv(os.path.join(out, 'exploratory_words_top.csv'), index=False)
        pd.DataFrame(info_rows).to_csv(os.path.join(out, 'exploratory_words_tiers.csv'), index=False)
        common = []
        for side, i in (('high > low (top 25)', 0), ('low > high (top 25)', 1)):
            ws = set.intersection(*[tops[c][i] for c in av.CONDITIONS])
            for w in sorted(ws):
                common.append(dict(side=side, word=w))
        pd.DataFrame(common, columns=['side', 'word']).to_csv(os.path.join(out, 'exploratory_words_common.csv'),
                                                              index=False)
        with open(os.path.join(out, 'exploratory_words_params.txt'), 'w', encoding='utf-8') as f:
            f.write(f'EXPLORATORY, added after the trees, the ceiling and the resampled trees were seen (user, '
                    f'2026-11-05): reason_words.py run {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'replies: {info["main"]} ({info["main_lines"]} lines) and {info["forced"]} (forced answers used '
                    f'for these frames, all runs read: {info["forced_used"]}); fields {FIELDS} of the final JSON '
                    '(parsed), not the thinking\n')
            f.write(f'risk analysis = {os.path.abspath(os.path.join(risk_root, s))}\n')
            f.write(f'words: lower case, runs of a-z ({WORD.pattern}), scikit-learn ENGLISH_STOP_WORDS '
                    f'({len(ENGLISH_STOP_WORDS)} words) out, no stemming; a frame has a word when one of its replies '
                    f'has it; words in at least {MIN_FRAMES} frames; tiers at the tertiles of the rank of score_mean\n')
            for c_ in checks:
                f.write(f'check: {c_}\n')
            f.write(f'checks not met: {[x for x in failed if x.startswith(s)] or "none"}\n')
    print(f'done in {time.time() - t0:.0f} s; checks not met: {failed if failed else "none"}')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
