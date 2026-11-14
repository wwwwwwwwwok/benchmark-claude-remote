"""EXPLORATORY (user, 2026-11-13; the plan is in CLAUDE.md "理由分析" -> "第三轮", recorded and committed before this
program was written; it is not changed after the results): Qwen3.6-27B's answers in preset categories. A: its own
hazard_type and main_hazard_camera; B: keyword classes written before the clusters of rounds 1-2 were seen (one conflict
rule added for stop sign); C: the thinking text. Section 6 ("post-hoc variant", added after the clusters were seen) is
computed and reported apart. sep_camera, the main result's 1,084 frames (238 segments), 5 answers each (5,420, 26 of them
forced answers). CPU only; no image is opened; only the text and fields of the replies are read.
Data: every answer from vlm_scores/qwen3.6-27b/main_qwen3.6-27b.jsonl.gz (field `parsed`: risk, hazard_type,
main_hazard_camera, main_hazard, reason; field `reasoning`: the thinking) and, where the reply has no readable answer,
the forced answer's `parsed` from main_qwen3.6-27b_forced_answers.jsonl.gz (analyze_vlm_ssm.read_model's rule). A forced
answer's thinking is the reply's own `reasoning`, the thinking cut at max_tokens (the forced record keeps the same text:
checked). Frame scores: score_mean of the risk analysis (after_gemma-4-31b-it/main_1084/vlm_frame_scores.csv); each
answer's own score: its parsed risk (= the risk analysis's vlm_requests_tidy.csv, checked). hazard_type and camera:
strip and lower case only; not an allowed value = invalid (each written form listed, not merged); a missing or null field
= missing. SSM risk percentiles: round 2's exploratory_reason_frames.csv (checked against a recomputation); cues: round
2's files and columns, and D's main crossing version new_cues_min30.csv for "a crosswalk / a lane crossing within 100 m
ahead along the lane" (the cue's own range).
Keyword rules: lower case; whole tokens (runs of a-z0-9; hyphens and dashes as spaces, so U-turn = U turn); every word of
a keyword gets its plural (-s / -es / consonant-y -> -ies; irregular child -> children, person -> people) and its verb
forms -s, -ed, -ing (e-dropping, consonant-y -> -ied, consonant-vowel-consonant: doubled and undoubled), made from its
base (an inflected keyword is mapped to its base by LEMMA and keeps its own plural too); prepositions, adverbs and modal
verbs, and the letter u of U-turn (NO_FORMS: neither nouns nor verbs) are not changed; a phrase = the product of its
words' forms; every form is written to exploratory_keywords_forms.csv and params.txt. Conflicts: a traffic word inside traffic light / traffic signal
is not class 7; a signal word inside turn signal / blinker is not class 11; no standalone close; stopped is not class 5;
a stop word inside stop sign is not class 5 nor 20. A word may count in several classes otherwise. Negation: no / not /
without / none / never among the 3 tokens before the keyword; per answer and class: mentioned if any mention is not
negated, otherwise negated if any. main_hazard and reason are matched apart and combined. Class 16: a number followed by
m, meter(s), foot / feet, second(s), mph or km/h (singular and plural, as the rule above). Motion talk: classes 4, 5 or 6
mentioned. A frame mentions a class when 3 or more of its 5 answers do (main), 1 or more (sensitivity). Thinking: the
same classes and rules; estimates: a number followed by a distance (m, meter(s), metre(s), ft, feet), time (s, sec,
second(s)) or speed unit (mph, km/h, kph, m/s); a frame "estimated" when 3+ (main) / 1+ (sensitivity) of its thinkings
have one. Post-hoc variant: an approach word followed within 4 tokens by intersection, junction, crosswalk, crossing,
light, signal, sign, turn, corner or curve (or their plurals) is not class 6; the extra forms rainy, foggy (12),
suddenly, potentially, unexpectedly (15), deceleration (5), acceleration (6), signalized (11); only the class score
differences (frames, main) and the motion-talk tau-b of the two groups and their difference (main and sensitivity).
Within-group tau-b = S / sqrt(nx ny) over the frame pairs inside the group (as 4.3); two-group differences = the named
group minus the rest. Intervals: the main analysis's 2,000 resamples of whole segments (default_rng(20261021), drawn
segments' sha256 48bc1f41...; every frame / answer of a drawn segment, as often as drawn), 2.5 / 97.5 % percentiles (numpy
linear) over the resamples where the value is defined, the others counted as missing; a group of fewer than 50 frames:
tau-b without an interval (a difference too when either group is under 50); a class or type with fewer than 10 frames
(answers, at the answer level): score difference without an interval. Per section: how many intervals are read against
0 (or 50), how many leave it out, and that count x 0.05.
Checks (exit 1 when one is not met): 5,420 answers, 1,084 frames, 238 segments, 5 per frame; the answers' risk = the
risk analysis's tidy risk; frame means = score_mean; every frame has every SSM; the 9 tau-b over all pairs =
agreement.csv lines 1001-1025 within 1e-12; hazard_type and camera values add up to 5,420; the percentiles = round 2's
frame table; the forced answers' thinking = the reply's. The run script compares a second run byte for byte.
usage: python -I reason_keywords.py <waymo_v143 folder> <round-2 out_dir> <out_dir> [--workers N]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import collections  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import multiprocessing as mp  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_rules as ar  # noqa: E402
import analyze_vlm_ssm as av  # noqa: E402
import reason_clusters as rc  # noqa: E402

MODEL, COND = 'qwen3.6-27b', 'sep_camera'
SEED, N_BOOT, BOOT_SEED = 20261113, 2000, 20261021
HASH = '48bc1f413f9d5b99da57ff4edde7bbb4eb378c5a9343c44ccad4e0165466604c'
AG_FIRST, AG_STEP, TOL = 1001, 3, 1e-12
MIN_TAU, MIN_DIFF, N_SAMPLE = 50, 10, 20
HAZ_ALLOWED = ['rear-end', 'lane change or side', 'crossing or intersection', 'pedestrian or cyclist', 'oncoming',
               'other', 'none']
CAM_ALLOWED = ['front-left', 'front', 'front-right', 'left-side', 'right-side', 'none']
CATS = [(1, 'pedestrian/cyclist', ['pedestrian', 'person', 'people', 'child', 'cyclist', 'bicycle', 'bike', 'scooter']),
        (2, 'crosswalk/intersection', ['crosswalk', 'crossing', 'intersection', 'junction']),
        (3, 'lead vehicle/following', ['lead vehicle', 'vehicle ahead', 'car ahead', 'in front', 'following',
                                       'tailgating']),
        (4, 'distance', ['distance', 'gap', 'headway', 'closely', 'too close', 'close behind', 'close proximity']),
        (5, 'braking/slowing', ['brake', 'braking', 'brake lights', 'slowing', 'decelerating', 'stopping']),
        (6, 'speed/approaching', ['speed', 'fast', 'slow', 'approaching', 'closing in', 'accelerating']),
        (7, 'traffic density', ['traffic', 'congestion', 'congested', 'dense', 'queue', 'multiple vehicles',
                                'many vehicles']),
        (8, 'lane change/merging/side', ['lane change', 'merging', 'cut in', 'adjacent lane', 'beside', 'alongside']),
        (9, 'turning', ['turn', 'turning', 'left turn', 'right turn', 'U-turn', 'turn signal', 'blinker']),
        (10, 'parked vehicles', ['parked', 'parking']),
        (11, 'traffic lights/signs', ['traffic light', 'signal', 'red light', 'green light', 'stop sign', 'yield']),
        (12, 'weather/lighting', ['rain', 'wet', 'night', 'dark', 'glare', 'visibility', 'fog']),
        (13, 'large vehicles', ['truck', 'bus', 'van', 'trailer']),
        (14, 'oncoming', ['oncoming', 'opposite direction']),
        (15, 'hypothetical', ['could', 'might', 'may', 'possibly', 'potential', 'unpredictable', 'sudden', 'unexpected']),
        (16, 'numbers with units', []),
        (17, 'motorcycle', ['motorcycle', 'motorbike', 'motorcyclist']),
        (18, 'construction/obstacle', ['construction', 'cones', 'barrier', 'road work', 'debris', 'obstacle']),
        (19, 'occlusion/view', ['occluded', 'blocked view', 'blind spot', 'limited visibility', 'obstructed']),
        (20, 'stopped/waiting', ['stopped', 'stationary', 'waiting', 'idling', 'queued'])]
CAT_IDS = [c for c, _, _ in CATS]
CAT_NAME = {c: n for c, n, _ in CATS}
MOTION = (4, 5, 6)
LEMMA = {'braking': 'brake', 'slowing': 'slow', 'decelerating': 'decelerate', 'stopping': 'stop', 'following': 'follow',
         'tailgating': 'tailgate', 'merging': 'merge', 'turning': 'turn', 'accelerating': 'accelerate',
         'congested': 'congest', 'parked': 'park', 'parking': 'park', 'occluded': 'occlude', 'obstructed': 'obstruct',
         'blocked': 'block', 'stopped': 'stop', 'waiting': 'wait', 'idling': 'idle', 'queued': 'queue',
         'approaching': 'approach', 'closing': 'close', 'crossing': 'cross', 'cones': 'cone', 'lights': 'light',
         'vehicles': 'vehicle', 'people': 'person', 'limited': 'limit'}
IRREG = {'child': {'children'}, 'person': {'people'}}
NO_FORMS = {'u', 'in', 'ahead', 'too', 'closely', 'behind', 'beside', 'alongside', 'could', 'might', 'may', 'possibly'}
NEG = {'no', 'not', 'without', 'none', 'never'}
VARIANT_EXTRA = {12: ['rainy', 'foggy'], 15: ['suddenly', 'potentially', 'unexpectedly'], 5: ['deceleration'],
                 6: ['acceleration'], 11: ['signalized']}
APPROACH = {'approach', 'approaches', 'approached', 'approaching'}
LOCATION_BASE = ['intersection', 'junction', 'crosswalk', 'crossing', 'light', 'signal', 'sign', 'turn', 'corner', 'curve']
VOW = set('aeiou')
CAT16_RE = re.compile(r'(?<![a-z0-9.])\d+(?:\.\d+)?\s*(km/h|mph|meters|meter|feet|foot|seconds|second|m)(?![a-z0-9/])')
EST_RE = re.compile(r'(?<![a-z0-9.])\d+(?:\.\d+)?\s*(m/s|km/h|kph|mph|metres|metre|meters|meter|feet|ft|seconds|second|'
                    r'sec|s|m)(?![a-z0-9/])')
EST_GROUP = {'m': 'distance', 'meter': 'distance', 'meters': 'distance', 'metre': 'distance', 'metres': 'distance',
             'ft': 'distance', 'feet': 'distance', 's': 'time', 'sec': 'time', 'second': 'time', 'seconds': 'time',
             'mph': 'speed', 'km/h': 'speed', 'kph': 'speed', 'm/s': 'speed'}
EST_KINDS = ['distance', 'time', 'speed']
TOK = re.compile(r'[a-z0-9]+')
DASHES = str.maketrans({'-': ' ', '‐': ' ', '‑': ' ', '‒': ' ', '–': ' ', '—': ' '})
STAT_NAME = {2: 'mentioned', 1: 'negated', 0: 'none'}
G = {}                                                            # data for the worker processes (fork)


# ---------------------------------------------------------------------------------------------------- word forms
def plural(w):
    out = set(IRREG.get(w, ()))
    if w.endswith(('s', 'x', 'z', 'ch', 'sh')):
        out.add(w + 'es')
    elif len(w) > 1 and w.endswith('y') and w[-2] not in VOW:
        out.add(w[:-1] + 'ies')
    else:
        out.add(w + 's')
    return out


def cvc(w):
    return len(w) >= 3 and w[-1] not in VOW and w[-1] not in 'wxy' and w[-2] in VOW and w[-3] not in VOW


def verb_forms(w):
    out = set(plural(w)) - set(IRREG.get(w, ()))                 # third person = the regular -s / -es / -ies
    if w.endswith('e'):
        out |= {w + 'd', w[:-1] + 'ing'}
        if w.endswith(('ee', 'ye', 'oe', 'ue')):
            out.add(w + 'ing')
    elif len(w) > 1 and w.endswith('y') and w[-2] not in VOW:
        out |= {w[:-1] + 'ied', w + 'ing'}
    elif cvc(w):
        out |= {w + w[-1] + 'ed', w + 'ed', w + w[-1] + 'ing', w + 'ing'}
    else:
        out |= {w + 'ed', w + 'ing'}
    return out


def word_forms(word):
    if word in NO_FORMS:
        return [word]
    base = LEMMA.get(word, word)
    return sorted({word, base} | plural(base) | verb_forms(base) | plural(word))


def phrase_forms(kw):
    words = TOK.findall(kw.lower().translate(DASHES))
    return {tuple(p) for p in itertools.product(*[word_forms(w) for w in words])}


def build_index(phrases):
    idx = collections.defaultdict(list)
    for t in phrases:
        idx[t[0]].append(t)
    return idx


def find(idx, w):
    out = []
    for i, tok in enumerate(w):
        for t in idx.get(tok, ()):
            if tuple(w[i:i + len(t)]) == t:
                out.append((i, i + len(t)))
    return out


def tokens(text):
    low = text.lower().translate(DASHES)
    return low, [(m.group(0), m.start(), m.end()) for m in TOK.finditer(low)]


class Matcher:
    """The keyword classes with all forms; match(text) -> {class: 2 mentioned / 1 negated / 0 none} and the matched
    character spans (class, start, end, negated) in text.lower()."""

    def __init__(self, variant=False):
        self.variant = variant
        self.forms = []                                           # (class, keyword, form) for the forms file
        index = collections.defaultdict(list)
        for c, _, kws in CATS:
            for kw in kws:
                fs = phrase_forms(kw)
                if c == 5:
                    fs.discard(('stopped',))                      # conflict: stopped is only class 20
                for f in sorted(fs):
                    self.forms.append((c, kw, ' '.join(f)))
                    index[f[0]].append((f, c))
            if c == 16:
                self.forms.append((16, '(a number followed by a unit)', 'regex ' + CAT16_RE.pattern))
            if variant:
                for x in VARIANT_EXTRA.get(c, ()):
                    self.forms.append((c, f'(post-hoc variant) {x}', x))
                    index[x].append(((x,), c))
        self.index = index
        self.traffic_sig = build_index(phrase_forms('traffic light') | phrase_forms('traffic signal'))
        self.turn_sig = build_index(phrase_forms('turn signal') | phrase_forms('blinker'))
        self.stop_sign = build_index(phrase_forms('stop sign'))
        self.location = {w for b in LOCATION_BASE for w in {b} | plural(b)}

    def match(self, text):
        status = dict.fromkeys(CAT_IDS, 0)
        spans = []
        if not isinstance(text, str) or not text:
            return status, spans
        low, toks = tokens(text)
        w = [t for t, _, _ in toks]
        found = []
        for i, tok in enumerate(w):
            for t, c in self.index.get(tok, ()):
                if tuple(w[i:i + len(t)]) == t:
                    found.append((c, i, i + len(t)))
        ts, tn, sp = find(self.traffic_sig, w), find(self.turn_sig, w), find(self.stop_sign, w)

        def inside(i, j, sps):
            return any(a <= i and j <= b for a, b in sps)

        keep = []
        for c, i, j in found:
            if c == 7 and inside(i, j, ts):
                continue
            if c == 11 and inside(i, j, tn):
                continue
            if c in (5, 20) and inside(i, j, sp):
                continue
            if self.variant and c == 6 and w[i] in APPROACH and \
                    any(w[k] in self.location for k in range(j, min(j + 4, len(w)))):
                continue
            keep.append((c, i, toks[i][1], toks[j - 1][2]))
        start_tok = {s: k for k, (_, s, _) in enumerate(toks)}
        for m in CAT16_RE.finditer(low):
            k = start_tok.get(m.start())
            if k is not None:
                keep.append((16, k, m.start(), m.end()))
        for c, i, a_, b_ in keep:
            neg = any(w[k] in NEG for k in range(max(0, i - 3), i))
            status[c] = max(status[c], 1 if neg else 2)
            spans.append((c, a_, b_, neg))
        return status, spans


def estimates(text):
    """Thinking estimates: {kind: count}, the unit spellings, the match spans in text.lower()."""
    out = dict.fromkeys(EST_KINDS, 0)
    units, spans = [], []
    if not isinstance(text, str):
        return out, units, spans
    for m in EST_RE.finditer(text.lower()):
        u = m.group(1)
        out[EST_GROUP[u]] += 1
        units.append((EST_GROUP[u], u))
        spans.append((m.start(), m.end()))
    return out, units, spans


# ---------------------------------------------------------------------------------------------------- statistics
def tau_of(S, nx, ny):
    return float(S / np.sqrt(nx * ny)) if nx * ny > 0 else np.nan


def fleiss(mat):
    if len(mat) == 0:
        return np.nan
    n = mat.sum(1)[0]
    p_j = mat.sum(0) / (len(mat) * n)
    P_i = (mat * (mat - 1)).sum(1) / (n * (n - 1))
    pe = (p_j ** 2).sum()
    return float((P_i.mean() - pe) / (1 - pe)) if pe < 1 else np.nan


def gmeans(code, w, K):
    cnt = np.bincount(code, minlength=K)
    s = np.bincount(code, weights=w, minlength=K)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def mdiff(x, m):
    return float(x[m].mean() - x[~m].mean()) if m.any() and (~m).any() else np.nan


def med(x, m):
    return float(np.median(x[m])) if m.any() else np.nan


def stats(idx):
    g = G
    ia = g['rows5'][idx].ravel()
    sc, ar_ = g['score'][idx], g['ans_risk'][ia]
    pc = {s: g['pct'][s][idx] for s in av.SSMS}
    res = {}
    ov, ova = float(sc.mean()), float(ar_.mean())
    # A: hazard types and cameras, frames and answers
    for key, code_f, code_a, K, KA in (('hz', g['hz_f'], g['hz_a'], g['KH'], g['KA_hz']),
                                       ('cam', g['cam_f'], g['cam_a'], g['KC'], g['KA_cam'])):
        cf = code_f[idx]
        mf = gmeans(cf, sc, K)
        ma = gmeans(code_a[ia], ar_, KA)
        for k in range(K):
            res[(key, 'f_mean', k)] = mf[k]
            res[(key, 'f_dall', k)] = mf[k] - ov
        for k in range(KA):
            res[(key, 'a_dall', k)] = ma[k] - ova
        if key == 'hz':
            for k in range(K):
                res[(key, 'f_drear', k)] = mf[k] - mf[g['REAR_F']]
                for s in av.SSMS:
                    res[('hz', 'pct', k, s)] = med(pc[s], cf == k)
            for k in range(KA):
                res[(key, 'a_drear', k)] = ma[k] - ma[g['REAR_A']]
        mat = g['kappa_mat_' + key][idx]
        res[('kappa', key)] = fleiss(mat[g['kappa_ok_' + key][idx]])
    # B: class score differences (frames main / sensitivity, answers), post-hoc variant (frames main)
    cnt, cntv, st = g['cnt'][idx], g['cnt_v'][idx], g['ans_status'][ia]
    for j, c in enumerate(CAT_IDS):
        res[('B', c, 'main')] = mdiff(sc, cnt[:, j] >= 3)
        res[('B', c, 'sens')] = mdiff(sc, cnt[:, j] >= 1)
        res[('Ba', c)] = mdiff(ar_, st[:, j] == 2)
        res[('V', c)] = mdiff(sc, cntv[:, j] >= 3)
    # two-group splits (motion talk, thinking estimates): means and SSM percentile medians
    for name, arr in g['desc_two'].items():
        m = arr[idx]
        for sd_, mm in ((1, m), (0, ~m)):
            res[('two', name, 'mean', sd_)] = float(sc[mm].mean()) if mm.any() else np.nan
            for s in av.SSMS:
                res[('two', name, 'pct', sd_, s)] = med(pc[s], mm)
    # tau-b over the frame pairs inside each group
    p, q = np.triu_indices(len(idx), 1)
    sx = np.sign(sc[p] - sc[q])
    sy = {s: np.sign(g['risk'][s][idx][p] - g['risk'][s][idx][q]) for s in av.SSMS}
    for name, lab_arr, K in g['groupings']:
        lab = lab_arr[idx]
        code = np.where(lab[p] == lab[q], lab[p], K).astype(np.int64)
        nx = np.bincount(code, weights=np.abs(sx), minlength=K + 1)
        for s in av.SSMS:
            S = np.bincount(code, weights=sx * sy[s], minlength=K + 1)
            ny = np.bincount(code, weights=np.abs(sy[s]), minlength=K + 1)
            for k in range(K):
                res[('tau', name, k, s)] = tau_of(S[k], nx[k], ny[k])
            if name == 'all':
                res[('tau', name, 'all', s)] = tau_of(S.sum(), nx.sum(), ny.sum())
            if K == 2:
                res[('tau', name, 'diff', s)] = res[('tau', name, 1, s)] - res[('tau', name, 0, s)]
    return res


def one(b):
    idx = np.concatenate([G['seg_frames'][s] for s in G['draws'][b]])
    rr = stats(idx)
    return np.array([rr.get(k, np.nan) for k in G['keys']], float)


def ci(v):
    ok = ~np.isnan(v)
    if not ok.any():
        return np.nan, np.nan, 0, int((~ok).sum())
    lo, hi = np.percentile(v[ok], [2.5, 97.5])
    return float(lo), float(hi), int(ok.sum()), int((~ok).sum())


def side(lo, hi, ref=0.0):
    if lo is None or np.isnan(lo):
        return 'no interval'
    return f'above {ref:g}' if lo > ref else (f'below {ref:g}' if hi < ref else f'contains {ref:g}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('waymo')
    ap.add_argument('round2')
    ap.add_argument('out')
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    L, R2, O = a.waymo, a.round2, a.out
    A, MD, FSSM = rc.paths(L)
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.exists(os.path.join(O, 'exploratory_keywords_params.txt')):
        sys.exit(f'{O}/exploratory_keywords_params.txt exists already: give a new folder')
    os.makedirs(O, exist_ok=True)
    t0 = time.time()
    failed, rows_ck = [], []

    def check(ok, what):
        print(f'  check {"ok" if ok else "NOT MET"}: {what}', flush=True)
        rows_ck.append(dict(check=what, ok=bool(ok)))
        if not ok:
            failed.append(what)

    with open(os.path.abspath(__file__), 'rb') as f:
        prog_sha = hashlib.sha256(f.read()).hexdigest()
    print(f'reason analysis round 3 (preset categories), {MODEL} {COND} (user, 2026-11-13; EXPLORATORY); run '
          f'{time.strftime("%Y-%m-%d %H:%M:%S")}; program sha256 {prog_sha}\n\n0. inputs', flush=True)

    # ------------------------------------------------------------------------------------------------- replies
    td = pd.read_csv(os.path.join(A, 'vlm_requests_tidy.csv'), low_memory=False, keep_default_na=False, na_values=[''],
                     usecols=['model', 'condition', 'segment', 'frame_idx', 'repeat', 'risk', 'forced'])
    td = td[(td.model == MODEL) & (td.condition == COND)].sort_values(['segment', 'frame_idx', 'repeat']) \
        .reset_index(drop=True)
    td['frame_idx'], td['repeat'] = td.frame_idx.astype(int), td['repeat'].astype(int)
    td['forced'] = td.forced.astype(str).str.lower().eq('true')
    keys = {(s, k, r): i for i, (s, k, r) in enumerate(zip(td.segment, td.frame_idx, td['repeat']))}
    main_f = av.jsonl_file(MD, f'main_{MODEL}')
    forced_f = av.jsonl_file(MD, f'main_{MODEL}_forced_answers')
    rec = [None] * len(td)
    with av.open_jsonl(main_f) as f:
        for raw in f:
            d = av.load_light(raw)
            if d['condition'] != COND:
                continue
            i = keys.get((d['segment'], int(d['frame_idx']), int(d['repeat'])))
            if i is None:
                continue
            if rec[i] is not None:
                sys.exit(f'duplicate request {d["segment"]} {d["frame_idx"]} {d["repeat"]}')
            rec[i] = dict(parsed=d.get('parsed'), thinking=d.get('reasoning'), forced=False)
    forced_think_same = n_forced_f = 0
    with av.open_jsonl(forced_f) as f:
        for raw in f:
            d = av.load_light(raw)
            if d['condition'] != COND:
                continue
            i = keys.get((d['segment'], int(d['frame_idx']), int(d['repeat'])))
            if i is None:
                continue
            if rec[i]['parsed']:
                sys.exit(f'forced answer for a reply that has one: {d["segment"]} {d["frame_idx"]}')
            if d.get('parsed'):
                n_forced_f += 1
                forced_think_same += d.get('reasoning') == rec[i]['thinking']
                rec[i]['parsed'], rec[i]['forced'] = d['parsed'], True
    ok_all = all(r is not None and r['parsed'] for r in rec)
    risk_p = np.array([float(r['parsed']['risk']) for r in rec]) if ok_all else np.zeros(len(rec))
    check(ok_all and np.array_equal(risk_p, td.risk.to_numpy(float))
          and [r['forced'] for r in rec] == td.forced.tolist(),
          f'{len(rec)} answers read from {main_f} (parsed, reasoning) and {forced_f} (parsed); every one has an answer; '
          f'the risk = the tidy table\'s; forced as the tidy table ({int(td.forced.sum())})')
    check(n_forced_f == int(td.forced.sum()) and forced_think_same == n_forced_f,
          f'the {n_forced_f} forced answers\' thinking = the reply\'s own reasoning (the thinking cut at max_tokens): '
          f'{forced_think_same} of {n_forced_f} the same text')
    if not ok_all:
        sys.exit('stopped: not every answer was read')
    n_think = sum(isinstance(r['thinking'], str) and len(r['thinking']) > 0 for r in rec)
    print(f'  answers with a thinking text: {n_think} of {len(rec)}', flush=True)
    frames = pd.read_csv(os.path.join(A, 'ssm_target_frames.csv'))[['segment', 'frame_idx']] \
        .sort_values(['segment', 'frame_idx']).reset_index(drop=True)
    n = len(frames)
    mi = pd.MultiIndex.from_frame(frames)
    fpos = pd.Series(np.arange(n), index=mi).reindex(pd.MultiIndex.from_arrays([td.segment, td.frame_idx])).to_numpy()
    reps = collections.defaultdict(dict)
    for i, (fp, rp) in enumerate(zip(fpos, td['repeat'])):
        if not np.isnan(fp):
            reps[int(fp)][int(rp)] = i
    check(len(td) == 5420 and n == 1084 and frames.segment.nunique() == 238 and not np.isnan(fpos.astype(float)).any()
          and all(sorted(reps[f_]) == [0, 1, 2, 3, 4] for f_ in range(n)),
          f'{len(td)} answers, {n} frames, {frames.segment.nunique()} segments, 5 answers per frame (repeats 0-4)')
    rows5 = np.array([[reps[f_].get(r, 0) for r in range(5)] for f_ in range(n)])
    p_fs = os.path.join(A, 'vlm_frame_scores.csv')
    fs = pd.read_csv(p_fs, float_precision='round_trip')
    score = fs[(fs.model == MODEL) & (fs.condition == COND)].set_index(['segment', 'frame_idx'])['score_mean'] \
        .reindex(mi).to_numpy(float)
    fm = td.groupby(['segment', 'frame_idx'], sort=True)['risk'].mean().reindex(mi).to_numpy(float)
    check(not np.isnan(score).any() and np.array_equal(fm, score),
          f'frame means of the 5 answers = {p_fs} score_mean frame by frame')

    # ------------------------------------------------------------------------------------------------- A fields
    def norm_field(p, key, allowed):
        if key not in p or p[key] is None:
            return 'missing', None
        v = p[key]
        if not isinstance(v, str):
            return 'invalid', repr(v)
        s = v.strip().lower()
        return (s, None) if s in allowed else ('invalid', v)

    hz, cam, hz_w, cam_w = [], [], collections.Counter(), collections.Counter()
    for r in rec:
        v, wtn = norm_field(r['parsed'], 'hazard_type', HAZ_ALLOWED)
        hz.append(v)
        if wtn is not None:
            hz_w[wtn] += 1
        v, wtn = norm_field(r['parsed'], 'main_hazard_camera', CAM_ALLOWED)
        cam.append(v)
        if wtn is not None:
            cam_w[wtn] += 1
    hz, cam = np.array(hz, dtype=object), np.array(cam, dtype=object)
    HZ_VALS = HAZ_ALLOWED + ['invalid', 'missing']
    CAM_VALS = CAM_ALLOWED + ['invalid', 'missing']
    check(sum(int((hz == v).sum()) for v in HZ_VALS) == len(td) and sum(int((cam == v).sum()) for v in CAM_VALS) == len(td),
          f'hazard_type values add up to {len(td)} (invalid {int((hz == "invalid").sum())}, missing '
          f'{int((hz == "missing").sum())}); camera values add up to {len(td)} (invalid {int((cam == "invalid").sum())}, '
          f'missing {int((cam == "missing").sum())})')

    def mode5(vals):
        c = collections.Counter(vals)
        top = max(c.values())
        w = sorted(k for k, v in c.items() if v == top)
        return (w[0] if len(w) == 1 else 'mixed'), top

    hz_mode = [mode5(list(hz[rows5[f_]])) for f_ in range(n)]
    cam_mode = [mode5(list(cam[rows5[f_]])) for f_ in range(n)]
    HZ_GROUPS = [v for v in HZ_VALS + ['mixed'] if any(m == v for m, _ in hz_mode)]
    CAM_GROUPS = [v for v in CAM_VALS + ['mixed'] if any(m == v for m, _ in cam_mode)]
    hz_f = np.array([HZ_GROUPS.index(m) for m, _ in hz_mode])
    cam_f = np.array([CAM_GROUPS.index(m) for m, _ in cam_mode])
    hz_a = np.array([HZ_VALS.index(v) for v in hz])
    cam_a = np.array([CAM_VALS.index(v) for v in cam])

    def kappa_inputs(vals, allowed):
        mat = np.array([[int((vals[rows5[f_]] == v).sum()) for v in allowed] for f_ in range(n)])
        return mat, mat.sum(1) == 5

    km_hz, ok_hz = kappa_inputs(hz, HAZ_ALLOWED)
    km_cam, ok_cam = kappa_inputs(cam, CAM_ALLOWED)

    # ------------------------------------------------------------------------------------------------- B, C matching
    print('\n1. keyword matching', flush=True)
    mt, mv = Matcher(), Matcher(variant=True)
    st = np.zeros((len(td), 20), int)                              # 2 mentioned, 1 negated, 0 none
    stv = np.zeros((len(td), 20), int)
    st_th = np.zeros((len(td), 20), int)
    spans_ans = []
    est = np.zeros((len(td), 3), int)
    est_units = collections.Counter()
    est_spans = []
    for i, r in enumerate(rec):
        p = r['parsed']
        sp_i = []
        for fld in ('main_hazard', 'reason'):
            s_, sp = mt.match(p.get(fld))
            sv, _ = mv.match(p.get(fld))
            for j, c in enumerate(CAT_IDS):
                st[i, j] = max(st[i, j], s_[c])
                stv[i, j] = max(stv[i, j], sv[c])
            sp_i.append((fld, sp))
        spans_ans.append(sp_i)
        s_, _ = mt.match(r['thinking'])
        for j, c in enumerate(CAT_IDS):
            st_th[i, j] = s_[c]
        e_, units, sps = estimates(r['thinking'])
        est[i] = [e_[k] for k in EST_KINDS]
        est_units.update(units)
        est_spans += [(i, a_, b_) for a_, b_ in sps]
    jm = [CAT_IDS.index(c) for c in MOTION]
    motion = (st[:, jm] == 2).any(1)
    motion_v = (stv[:, jm] == 2).any(1)
    has_est = est.sum(1) > 0
    cnt = np.stack([(st[rows5, j] == 2).sum(1) for j in range(20)], axis=1)
    cnt_v = np.stack([(stv[rows5, j] == 2).sum(1) for j in range(20)], axis=1)
    motion_cnt, motion_cnt_v, est_cnt = motion[rows5].sum(1), motion_v[rows5].sum(1), has_est[rows5].sum(1)
    print(f'  done ({time.time() - t0:.0f} s): forms {len(mt.forms)} (post-hoc variant {len(mv.forms)})', flush=True)

    # ------------------------------------------------------------------------------------------------- SSMs, truth, cues
    ssm_df = av.ssm_table(FSSM, frames)
    risk = {s: ssm_df[s + '_risk'].to_numpy(float) for s in av.SSMS}
    check(all(ssm_df[s + '_conflict'].all() and ssm_df[s + '_use'].all() for s in av.SSMS)
          and not any(np.isnan(risk[s]).any() for s in av.SSMS), 'every frame has a value of every SSM')
    pct_new = {s: (rankdata(risk[s], method='average') - 1) / (n - 1) * 100 for s in av.SSMS}
    p_r2 = os.path.join(R2, 'exploratory_reason_frames.csv')
    r2 = pd.read_csv(p_r2, float_precision='round_trip').set_index(['segment', 'frame_idx']).reindex(mi)
    pct = {s: r2[f'{s}_risk_percentile'].to_numpy(float) for s in av.SSMS}
    check(all(np.array_equal(pct[s], pct_new[s]) for s in av.SSMS),
          f'the 9 SSM risk percentiles of {p_r2} (used) = a recomputation, frame by frame')
    truth = pd.read_csv(os.path.join(L, 'quantities', 'truth_main.csv'), float_precision='round_trip')
    tr = frames.merge(truth, on=['segment', 'frame_idx'], how='left', validate='one_to_one')
    rawq = {'true_distance': tr['distance'].to_numpy(float),
            'true_closing_speed': (tr['v_ego'] - tr['v_lead']).to_numpy(float),
            'ego_speed': tr['v_ego'].to_numpy(float), 'ego_accel': tr['a_ego'].to_numpy(float),
            'lead_accel': tr['a_lead'].to_numpy(float)}
    labels_csv = os.path.join(REPO, 'labels', 'scene_labels.csv')
    sl = pd.read_csv(labels_csv, dtype=str, keep_default_na=False).set_index('segment')
    weather = sl.loc[frames.segment, 'weather'].str.strip().to_numpy()
    tod = sl.loc[frames.segment, 'time_of_day'].str.strip().to_numpy()
    NE = 'analysis/rule_explore/new_cues'
    CUE_DEF = [('n_pedestrian_20m >= 1', 'analysis/rule_explore/nearby_counts.csv', 'n_pedestrian_20m', 1, False),
               ('n_ped_cyc_on_road >= 1', f'{NE}/new_cues.csv', 'n_ped_cyc_on_road', 1, False),
               ('n_ped_cyc_in_crosswalk_50m >= 1', f'{NE}/new_cues_e.csv', 'n_ped_cyc_in_crosswalk_50m', 1, False),
               ('dist_crosswalk < 100 (a crosswalk within 100 m ahead along the lane; D, crossings of 30 deg or more)',
                f'{NE}/new_cues_min30.csv', 'dist_crosswalk', 100, True),
               ('dist_lane_crossing < 100 (a lane crossing of 30 deg or more within 100 m ahead along the lane; D)',
                f'{NE}/new_cues_min30.csv', 'dist_lane_crossing', 100, True),
               ('max_vehicle_length_30m >= 7.0', f'{NE}/new_cues_e2.csv', 'max_vehicle_length_30m', 7.0, False),
               ('n_vehicle_stationary_other_20m >= 1', f'{NE}/new_cues_e.csv', 'n_vehicle_stationary_other_20m', 1,
                False),
               ('n_oncoming_vehicle >= 1', f'{NE}/new_cues.csv', 'n_oncoming_vehicle', 1, False),
               ('n_side_vehicle >= 1', f'{NE}/new_cues_e.csv', 'n_side_vehicle', 1, False)]
    cues, cue_src = {}, {}
    for name, file_, col_, thr, less in CUE_DEF:
        p_ = os.path.join(L, file_)
        t_ = pd.read_csv(p_, float_precision='round_trip')
        v = frames.merge(t_[['segment', 'frame_idx', col_]], on=['segment', 'frame_idx'], how='left',
                         validate='one_to_one')[col_].to_numpy(float)
        cues[name] = np.where(np.isnan(v), np.nan, ((v < thr) if less else (v >= thr)).astype(float))
        cue_src[name] = f'{p_} column {col_}'
    cues['weather == rain or wet'] = np.isin(weather, ['rain', 'wet']).astype(float)
    cues['time_of_day == Night'] = (tod == 'Night').astype(float)
    cue_src['weather == rain or wet'] = f'{labels_csv} column weather'
    cue_src['time_of_day == Night'] = f'{labels_csv} column time_of_day'
    # class 12's rain / wet and night / dark words alone (main rules: forms, negation)
    sub_idx = {'rain/wet words': build_index(phrase_forms('rain') | phrase_forms('wet')),
               'night/dark words': build_index(phrase_forms('night') | phrase_forms('dark'))}

    def sub_status(text, idx_):
        if not isinstance(text, str):
            return 0
        _, toks = tokens(text)
        w = [t for t, _, _ in toks]
        s_ = 0
        for i, _ in find(idx_, w):
            s_ = max(s_, 1 if any(w[k] in NEG for k in range(max(0, i - 3), i)) else 2)
        return s_

    sub_cnt = {}
    for nm, idx_ in sub_idx.items():
        ss = np.array([max(sub_status(r['parsed'].get(f_), idx_) for f_ in ('main_hazard', 'reason')) for r in rec])
        sub_cnt[nm] = (ss[rows5] == 2).sum(1)

    # ------------------------------------------------------------------------------------------------- resamples
    seg_names = frames.segment.to_numpy()
    seg_list = np.array(sorted(set(seg_names)))
    nb_r, seed_r, frames_r, segs_r = ar.read_params_risk(os.path.join(A, 'params.txt'))
    check((nb_r, seed_r, frames_r) == (N_BOOT, BOOT_SEED, n) and list(seg_list) == segs_r,
          f'{A}/params.txt: {nb_r} resamples, seed {seed_r}, {frames_r} frames, the same {len(seg_list)} segments')
    rng = np.random.default_rng(BOOT_SEED)
    draws = [rng.choice(seg_list, len(seg_list)) for _ in range(N_BOOT)]
    hh = hashlib.sha256()
    for dr in draws:
        hh.update(np.searchsorted(seg_list, dr).astype(np.int32).tobytes())
    check(hh.hexdigest() == HASH, f"drawn segments' sha256 {hh.hexdigest()}")
    REAR_F = HZ_GROUPS.index('rear-end')
    desc_two = {'motion_main': motion_cnt >= 3, 'motion_sens': motion_cnt >= 1, 'thinking_est_main': est_cnt >= 3,
                'thinking_est_sens': est_cnt >= 1}
    tau_two = {**desc_two, 'variant_motion_main': motion_cnt_v >= 3, 'variant_motion_sens': motion_cnt_v >= 1,
               'rear_end_vs_rest': hz_f == REAR_F}
    groupings = [('all', np.zeros(n, int), 1), ('hz_type', hz_f, len(HZ_GROUPS))] + \
        [(nm, arr.astype(int), 2) for nm, arr in tau_two.items()]
    G.update(rows5=rows5, score=score, ans_risk=risk_p, hz_f=hz_f, hz_a=hz_a, cam_f=cam_f, cam_a=cam_a,
             KH=len(HZ_GROUPS), KC=len(CAM_GROUPS), KA_hz=len(HZ_VALS), KA_cam=len(CAM_VALS), REAR_F=REAR_F,
             REAR_A=HZ_VALS.index('rear-end'), pct=pct, risk=risk, kappa_mat_hz=km_hz, kappa_ok_hz=ok_hz,
             kappa_mat_cam=km_cam, kappa_ok_cam=ok_cam, cnt=cnt, cnt_v=cnt_v, ans_status=st, desc_two=desc_two,
             groupings=groupings, draws=draws, seg_frames={s: np.flatnonzero(seg_names == s) for s in seg_list})
    point = stats(np.arange(n))
    p_ag = os.path.join(A, 'agreement.csv')
    with open(p_ag, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        ag = {d['ssm']: (rd.line_num, d) for d in rd
              if d['model'] == MODEL and d['condition'] == COND and d['grouping'] == 'all'}
    lines = [ag[s][0] for s in av.SSMS]
    dmax = max(abs(point[('tau', 'all', 'all', s)] - float(ag[s][1]['tau_b_conflict'])) for s in av.SSMS)
    check(lines == list(range(AG_FIRST, AG_FIRST + 9 * AG_STEP, AG_STEP)) and dmax <= TOL,
          f'the 9 tau-b over all frame pairs (S / sqrt(nx ny)) = {p_ag} lines {lines}: largest |difference| {dmax!r}')
    keys_ = list(point)
    G['keys'] = keys_
    col = {k: i for i, k in enumerate(keys_)}
    print(f'\n2. resamples: {N_BOOT}, {a.workers} worker(s), {len(keys_)} numbers each', flush=True)
    tb = time.time()
    with mp.get_context('fork').Pool(max(a.workers, 1)) as pool:
        boot = np.stack(list(pool.imap(one, range(N_BOOT), chunksize=8)))
    print(f'  done in {time.time() - tb:.0f} s', flush=True)

    comps = []                                                    # (section, kind, what, side)

    def with_ci(k, section=None, kind='', what='', ref=0.0, allow=True):
        lo, hi, nv, miss = ci(boot[:, col[k]]) if allow else (np.nan, np.nan, 0, 0)
        sd = side(lo, hi, ref)
        if section and allow and not np.isnan(lo):
            comps.append((section, kind, what, sd))
        return dict(value=point[k], lo=lo, hi=hi, nboot=nv, missing=miss, side=sd)

    def raw_stats(m, which):
        out = {}
        for q_ in which:
            vv = rawq[q_][m]
            vv = vv[~np.isnan(vv)]
            if len(vv):
                q1, md, q3 = np.percentile(vv, [25, 50, 75])
                out[q_] = (float(md), float(q1), float(q3), float(q3 - q1), len(vv))
            else:
                out[q_] = (np.nan,) * 4 + (0,)
        return out

    RAW3 = ['true_distance', 'true_closing_speed', 'ego_speed']
    RAW5 = RAW3 + ['ego_accel', 'lead_accel']
    SD, SP, ST, STD = 'score difference', 'SSM percentile median vs 50', 'tau-b', 'tau-b difference'

    # ------------------------------------------------------------------------------------------------- A tables
    rows_dist = []
    for field, vals, allowed in (('hazard_type', hz, HZ_VALS), ('main_hazard_camera', cam, CAM_VALS)):
        for v in allowed:
            k_ = int((vals == v).sum())
            rows_dist.append(dict(field=field, value=v, answers=k_, share=k_ / len(vals)))
    rows_inv = [dict(field='hazard_type', written=w_, answers=c_) for w_, c_ in sorted(hz_w.items())] + \
        [dict(field='main_hazard_camera', written=w_, answers=c_) for w_, c_ in sorted(cam_w.items())]
    ct = pd.crosstab(pd.Categorical(hz, categories=HZ_VALS), pd.Categorical(cam, categories=CAM_VALS),
                     rownames=['hazard_type'], colnames=['main_hazard_camera'], dropna=False)
    rows_stab = []
    for field, modes, kok, f_ in (('hazard_type', hz_mode, ok_hz, 'hz'), ('main_hazard_camera', cam_mode, ok_cam, 'cam')):
        tops = collections.Counter(t for _, t in modes)
        kk = with_ci(('kappa', f_))
        rows_stab.append(dict(field=field, frames=n, top_value_5of5=tops.get(5, 0), top_value_4of5=tops.get(4, 0),
                              top_value_3of5=tops.get(3, 0), top_value_2of5_or_fewer=sum(v for t, v in tops.items() if t <= 2),
                              fleiss_kappa=kk['value'], kappa_frames_all_5_valid=int(kok.sum()), kappa_lo=kk['lo'],
                              kappa_hi=kk['hi'], kappa_nboot=kk['nboot'], kappa_missing=kk['missing']))
    rows_hg, rows_ha, rows_cg, rows_hp, rows_hr, rows_ht = [], [], [], [], [], []
    for k, gname in enumerate(HZ_GROUPS):
        m = hz_f == k
        segc = pd.Series(seg_names[m]).value_counts()
        mm = with_ci(('hz', 'f_mean', k))
        dall = with_ci(('hz', 'f_dall', k), 'A', SD, f'frame type {gname} - all frames', allow=m.sum() >= MIN_DIFF)
        drear = with_ci(('hz', 'f_drear', k), 'A', SD, f'frame type {gname} - rear-end',
                        allow=m.sum() >= MIN_DIFF and k != REAR_F)
        rows_hg.append(dict(group=gname, frames=int(m.sum()), segments=len(segc),
                            largest_segment_frames=int(segc.iloc[0]) if len(segc) else 0, mean=mm['value'],
                            mean_lo=mm['lo'], mean_hi=mm['hi'], mean_missing=mm['missing'],
                            median=float(np.median(score[m])), sd_ddof0=float(score[m].std(ddof=0)),
                            minus_all=dall['value'], minus_all_lo=dall['lo'], minus_all_hi=dall['hi'],
                            minus_all_nboot=dall['nboot'], minus_all_missing=dall['missing'], minus_all_side=dall['side'],
                            minus_rear=drear['value'] if k != REAR_F else np.nan, minus_rear_lo=drear['lo'],
                            minus_rear_hi=drear['hi'], minus_rear_nboot=drear['nboot'],
                            minus_rear_missing=drear['missing'], minus_rear_side=drear['side'] if k != REAR_F else ''))
        for s in av.SSMS:
            pp = with_ci(('hz', 'pct', k, s), 'A', SP, f'frame type {gname} {s}', 50.0)
            rows_hp.append(dict(group=gname, ssm=s, frames=int(m.sum()), median_percentile=pp['value'], lo=pp['lo'],
                                hi=pp['hi'], nboot=pp['nboot'], missing=pp['missing'], side=pp['side']))
        for q_, (md, q1, q3, iqr, nv) in raw_stats(m, RAW3).items():
            rows_hr.append(dict(group=gname, quantity=q_, frames_with_value=nv, median=md, q1=q1, q3=q3, iqr=iqr))
        for s in av.SSMS:
            tt = with_ci(('tau', 'hz_type', k, s), 'A', ST, f'within frame type {gname} {s}', allow=m.sum() >= MIN_TAU)
            rows_ht.append(dict(grouping='frame type', group=gname, ssm=s, frames=int(m.sum()), tau_b=tt['value'],
                                lo=tt['lo'], hi=tt['hi'], nboot=tt['nboot'], missing=tt['missing'], side=tt['side'],
                                fewer_than_50=bool(m.sum() < MIN_TAU)))
    mrear = hz_f == REAR_F
    for s in av.SSMS:
        # the rear-end group's own tau-b is already in the frame-type rows (counted there); the rest is counted here
        for sd_, nm_, mm_ in ((1, 'rear-end', mrear), (0, 'all other frames', ~mrear)):
            okg = mm_.sum() >= MIN_TAU
            tt = with_ci(('tau', 'rear_end_vs_rest', sd_, s), 'A' if sd_ == 0 else None, ST, f'{nm_} {s}', allow=okg)
            rows_ht.append(dict(grouping='rear-end vs the rest', group=nm_, ssm=s, frames=int(mm_.sum()), tau_b=tt['value'],
                                lo=tt['lo'], hi=tt['hi'], nboot=tt['nboot'], missing=tt['missing'], side=tt['side'],
                                fewer_than_50=bool(not okg)))
        tt = with_ci(('tau', 'rear_end_vs_rest', 'diff', s), 'A', STD, f'rear-end - rest {s}',
                     allow=min(mrear.sum(), (~mrear).sum()) >= MIN_TAU)
        rows_ht.append(dict(grouping='rear-end vs the rest', group='rear-end - rest', ssm=s, frames=n, tau_b=tt['value'],
                            lo=tt['lo'], hi=tt['hi'], nboot=tt['nboot'], missing=tt['missing'], side=tt['side'],
                            fewer_than_50=False))
    for k, v in enumerate(HZ_VALS):
        m = hz_a == k
        if not m.any():
            continue
        dall = with_ci(('hz', 'a_dall', k), 'A', SD + ' (answers)', f'answer type {v} - all answers',
                       allow=m.sum() >= MIN_DIFF)
        drear = with_ci(('hz', 'a_drear', k), 'A', SD + ' (answers)', f'answer type {v} - rear-end',
                        allow=m.sum() >= MIN_DIFF and v != 'rear-end')
        rows_ha.append(dict(type=v, answers=int(m.sum()), mean=float(risk_p[m].mean()), minus_all=dall['value'],
                            minus_all_lo=dall['lo'], minus_all_hi=dall['hi'], minus_all_nboot=dall['nboot'],
                            minus_all_missing=dall['missing'], minus_all_side=dall['side'],
                            minus_rear=drear['value'] if v != 'rear-end' else np.nan, minus_rear_lo=drear['lo'],
                            minus_rear_hi=drear['hi'], minus_rear_nboot=drear['nboot'], minus_rear_missing=drear['missing'],
                            minus_rear_side=drear['side'] if v != 'rear-end' else ''))
    for k, gname in enumerate(CAM_GROUPS):
        m = cam_f == k
        dall = with_ci(('cam', 'f_dall', k), 'A', SD, f'frame camera {gname} - all frames', allow=m.sum() >= MIN_DIFF)
        rows_cg.append(dict(level='frame (most frequent camera)', camera=gname, n=int(m.sum()),
                            mean=float(score[m].mean()), minus_all=dall['value'], lo=dall['lo'], hi=dall['hi'],
                            nboot=dall['nboot'], missing=dall['missing'], side=dall['side']))
    for k, v in enumerate(CAM_VALS):
        m = cam_a == k
        if not m.any():
            continue
        dall = with_ci(('cam', 'a_dall', k), 'A', SD + ' (answers)', f'answer camera {v} - all answers',
                       allow=m.sum() >= MIN_DIFF)
        rows_cg.append(dict(level='answer', camera=v, n=int(m.sum()), mean=float(risk_p[m].mean()), minus_all=dall['value'],
                            lo=dall['lo'], hi=dall['hi'], nboot=dall['nboot'], missing=dall['missing'], side=dall['side']))
    rear_a = hz == 'rear-end'
    rows_rc = [dict(camera=v, rear_end_answers=int((cam[rear_a] == v).sum()), all_rear_end_answers=int(rear_a.sum()),
                    share=float((cam[rear_a] == v).sum() / rear_a.sum())) for v in CAM_VALS]

    # ------------------------------------------------------------------------------------------------- B tables
    rows_bc, rows_bs, rows_bt, rows_bq, rows_vs = [], [], [], [], []
    for j, c in enumerate(CAT_IDS):
        ment, negd = st[:, j] == 2, st[:, j] == 1
        fm_, fs_ = cnt[:, j] >= 3, cnt[:, j] >= 1
        rows_bc.append(dict(category=c, name=CAT_NAME[c], answers_mentioned=int(ment.sum()), share=float(ment.mean()),
                            answers_negated_only=int(negd.sum()), frames_main=int(fm_.sum()),
                            segments_main=int(len(set(seg_names[fm_]))), frames_sens=int(fs_.sum()),
                            segments_sens=int(len(set(seg_names[fs_])))))
        for thr_name, mask in (('main', fm_), ('sens', fs_)):
            dd = with_ci(('B', c, thr_name), 'B', SD, f'class {c} frames ({thr_name})', allow=mask.sum() >= MIN_DIFF)
            rows_bs.append(dict(category=c, name=CAT_NAME[c], level=f'frame ({thr_name})', n_mentioned=int(mask.sum()),
                                n_not=int((~mask).sum()), mean_mentioned=float(score[mask].mean()) if mask.any() else np.nan,
                                mean_not=float(score[~mask].mean()) if (~mask).any() else np.nan, diff=dd['value'],
                                lo=dd['lo'], hi=dd['hi'], nboot=dd['nboot'], missing=dd['missing'], side=dd['side']))
            for grp, gm in (('mentioned', mask), ('not', ~mask)):
                for q_, (md, q1, q3, iqr, nv) in raw_stats(gm, RAW5).items():
                    rows_bt.append(dict(category=c, name=CAT_NAME[c], level=thr_name, group=grp, frames=int(gm.sum()),
                                        quantity=q_, frames_with_value=nv, median=md, q1=q1, q3=q3, iqr=iqr))
        dd = with_ci(('Ba', c), 'B', SD + ' (answers)', f'class {c} answers', allow=ment.sum() >= MIN_DIFF)
        rows_bs.append(dict(category=c, name=CAT_NAME[c], level='answer', n_mentioned=int(ment.sum()),
                            n_not=int((~ment).sum()), mean_mentioned=float(risk_p[ment].mean()) if ment.any() else np.nan,
                            mean_not=float(risk_p[~ment].mean()) if (~ment).any() else np.nan, diff=dd['value'],
                            lo=dd['lo'], hi=dd['hi'], nboot=dd['nboot'], missing=dd['missing'], side=dd['side']))
        rows_bs.append(dict(category=c, name=CAT_NAME[c], level='answer, negated only', n_mentioned=int(negd.sum()),
                            n_not=np.nan, mean_mentioned=float(risk_p[negd].mean()) if negd.any() else np.nan,
                            mean_not=np.nan, diff=np.nan, lo=np.nan, hi=np.nan, nboot=np.nan, missing=np.nan,
                            side='counts and mean only'))
        fmv = cnt_v[:, j] >= 3
        dd = with_ci(('V', c), 'variant', SD, f'class {c} frames (main), post-hoc variant', allow=fmv.sum() >= MIN_DIFF)
        mrow = [x for x in rows_bs if x['category'] == c and x['level'] == 'frame (main)'][0]
        rows_vs.append(dict(category=c, name=CAT_NAME[c], frames_mentioned_main_rules=mrow['n_mentioned'],
                            diff_main_rules=mrow['diff'], lo_main_rules=mrow['lo'], hi_main_rules=mrow['hi'],
                            side_main_rules=mrow['side'], frames_mentioned_variant=int(fmv.sum()),
                            diff_variant=dd['value'], lo_variant=dd['lo'], hi_variant=dd['hi'], nboot_variant=dd['nboot'],
                            missing_variant=dd['missing'], side_variant=dd['side']))
    CR, LC = CUE_DEF[3][0], CUE_DEF[4][0]
    PAIRS = [(1, None, 'n_pedestrian_20m >= 1'), (1, None, 'n_ped_cyc_on_road >= 1'),
             (1, None, 'n_ped_cyc_in_crosswalk_50m >= 1'), (2, None, 'n_ped_cyc_in_crosswalk_50m >= 1'), (2, None, CR),
             (2, None, LC), (13, None, 'max_vehicle_length_30m >= 7.0'), (10, None, 'n_vehicle_stationary_other_20m >= 1'),
             (14, None, 'n_oncoming_vehicle >= 1'), (8, None, 'n_side_vehicle >= 1'),
             (12, 'rain/wet words', 'weather == rain or wet'), (12, 'night/dark words', 'time_of_day == Night')]
    for c, sub, cue_name in PAIRS:
        for thr_name, thr in (('main', 3), ('sens', 1)):
            said = (sub_cnt[sub] if sub else cnt[:, CAT_IDS.index(c)]) >= thr
            cv = cues[cue_name]
            okv = ~np.isnan(cv)
            has_ = (cv == 1) & okv
            both = int((said & has_).sum())
            rows_bq.append(dict(category=c, words=sub or CAT_NAME[c], cue=cue_name, source=cue_src[cue_name],
                                level=thr_name, said_and_cue=both, said=int((said & okv).sum()),
                                share_cue_among_said=both / (said & okv).sum() if (said & okv).any() else np.nan,
                                cue_frames=int(has_.sum()),
                                share_said_among_cue=both / has_.sum() if has_.any() else np.nan,
                                frames_with_value=int(okv.sum()), share_cue_among_all=has_.sum() / okv.sum()))
    for c, why in ((9, 'no left-turn cue in D / E (D has only heading_change_deg, the heading change over the past 2 s; '
                       'the next-3-s heading change was not used, user 2026-11-06): none'),
                   (11, 'no cue (no traffic-signal state in v1.4.3): skipped')):
        rows_bq.append(dict(category=c, words=CAT_NAME[c], cue='none', source=why, level='', said_and_cue=np.nan,
                            said=np.nan, share_cue_among_said=np.nan, cue_frames=np.nan, share_said_among_cue=np.nan,
                            frames_with_value=0, share_cue_among_all=np.nan))

    # ------------------------------------------------------------------------------------------------- two-group tables
    rows_two, rows_two_pct, rows_two_raw, rows_two_tau = [], [], [], []
    TWO_INFO = {'motion_main': ('B', 'talks of motion (3+ of 5 answers)', 'does not'),
                'motion_sens': ('B', 'talks of motion (1+ of 5 answers)', 'does not'),
                'thinking_est_main': ('C', 'thinking has an estimate (3+ of 5)', 'does not'),
                'thinking_est_sens': ('C', 'thinking has an estimate (1+ of 5)', 'does not'),
                'variant_motion_main': ('variant', 'talks of motion, post-hoc variant (3+ of 5)', 'does not'),
                'variant_motion_sens': ('variant', 'talks of motion, post-hoc variant (1+ of 5)', 'does not')}
    for name, (sec, lab1, lab0) in TWO_INFO.items():
        m = tau_two[name]
        for sd_, lab, mm in ((1, lab1, m), (0, lab0, ~m)):
            if name in desc_two:
                me = with_ci(('two', name, 'mean', sd_))
                rows_two.append(dict(split=name, group=lab, frames=int(mm.sum()), segments=int(len(set(seg_names[mm]))),
                                     mean=me['value'], lo=me['lo'], hi=me['hi'], nboot=me['nboot'], missing=me['missing']))
                for s in av.SSMS:
                    pp = with_ci(('two', name, 'pct', sd_, s), sec, SP, f'{name} {lab} {s}', 50.0)
                    rows_two_pct.append(dict(split=name, group=lab, ssm=s, frames=int(mm.sum()),
                                             median_percentile=pp['value'], lo=pp['lo'], hi=pp['hi'], nboot=pp['nboot'],
                                             missing=pp['missing'], side=pp['side']))
                for q_, (md, q1, q3, iqr, nv) in raw_stats(mm, RAW3).items():
                    rows_two_raw.append(dict(split=name, group=lab, quantity=q_, frames_with_value=nv, median=md, q1=q1,
                                             q3=q3, iqr=iqr))
            else:
                rows_two.append(dict(split=name, group=lab, frames=int(mm.sum()), segments=int(len(set(seg_names[mm]))),
                                     mean=np.nan, lo=np.nan, hi=np.nan, nboot=np.nan, missing=np.nan))
            for s in av.SSMS:
                tt = with_ci(('tau', name, sd_, s), sec, ST, f'{name} {lab} {s}', allow=mm.sum() >= MIN_TAU)
                rows_two_tau.append(dict(split=name, group=lab, ssm=s, frames=int(mm.sum()), tau_b=tt['value'],
                                         lo=tt['lo'], hi=tt['hi'], nboot=tt['nboot'], missing=tt['missing'],
                                         side=tt['side'], fewer_than_50=bool(mm.sum() < MIN_TAU)))
        for s in av.SSMS:
            tt = with_ci(('tau', name, 'diff', s), sec, STD, f'{name} named - rest {s}',
                         allow=min(m.sum(), (~m).sum()) >= MIN_TAU)
            rows_two_tau.append(dict(split=name, group='named group - the rest', ssm=s, frames=n, tau_b=tt['value'],
                                     lo=tt['lo'], hi=tt['hi'], nboot=tt['nboot'], missing=tt['missing'], side=tt['side'],
                                     fewer_than_50=False))

    # ------------------------------------------------------------------------------------------------- C tables
    rows_te = [dict(kind=k, answers_with=int((est[:, i] > 0).sum()), answers=len(td), share=float((est[:, i] > 0).mean()))
               for i, k in enumerate(EST_KINDS)]
    rows_te.append(dict(kind='any of the three', answers_with=int(has_est.sum()), answers=len(td),
                        share=float(has_est.mean())))
    rows_te += [dict(kind=f'frames with an estimate in {t_}+ of 5 thinkings', answers_with=int((est_cnt >= t_).sum()),
                     answers=n, share=float((est_cnt >= t_).mean())) for t_ in (3, 1)]
    rows_tu = [dict(kind=k, unit=u, matches=c_) for (k, u), c_ in sorted(est_units.items())]
    rows_tc = [dict(category=c, name=CAT_NAME[c], thinking_answers_mentioned=int((st_th[:, j] == 2).sum()),
                    thinking_share=float((st_th[:, j] == 2).mean()),
                    thinking_answers_negated_only=int((st_th[:, j] == 1).sum()),
                    answer_text_answers_mentioned=int((st[:, j] == 2).sum()),
                    answer_text_share=float((st[:, j] == 2).mean())) for j, c in enumerate(CAT_IDS)]

    # ------------------------------------------------------------------------------------------------- sample (section 9)
    rs = np.random.default_rng(SEED)
    rows_sample = []

    def mark(text, spans_):
        merged = []
        for a_, b_ in sorted(spans_):
            if merged and a_ <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b_)
            else:
                merged.append([a_, b_])
        out, last = [], 0
        for a_, b_ in merged:
            out.append(text[last:a_] + '[[' + text[a_:b_] + ']]')
            last = b_
        return ''.join(out) + text[last:]

    def shown(t_):
        return t_ if len(t_.lower()) == len(t_) else t_.lower()      # spans are positions in text.lower()

    for j, c in enumerate(CAT_IDS):
        cand = np.flatnonzero(st[:, j] == 2)
        pick = np.sort(rs.choice(cand, size=min(N_SAMPLE, len(cand)), replace=False)) if len(cand) else []
        for i in pick:
            parts = []
            for fld, sp in spans_ans[i]:
                t_ = rec[i]['parsed'].get(fld)
                t_ = t_ if isinstance(t_, str) else ''
                parts.append(f'{fld}: ' + mark(shown(t_), [(a_, b_) for cc, a_, b_, neg in sp if cc == c and not neg]))
            rows_sample.append(dict(category=c, name=CAT_NAME[c], segment=td.segment[i], frame_idx=int(td.frame_idx[i]),
                                    repeat=int(td['repeat'][i]), text=' | '.join(parts), correct_1_or_0=''))
    if est_spans:
        pick = np.sort(rs.choice(len(est_spans), size=min(N_SAMPLE, len(est_spans)), replace=False))
        for k in pick:
            i, a_, b_ = est_spans[k]
            t_ = shown(rec[i]['thinking'])
            before, after = t_[:a_].split()[-15:], t_[b_:].split()[:15]
            rows_sample.append(dict(category='estimate', name='thinking estimate', segment=td.segment[i],
                                    frame_idx=int(td.frame_idx[i]), repeat=int(td['repeat'][i]),
                                    text=' '.join(before) + ' [[' + t_[a_:b_] + ']] ' + ' '.join(after),
                                    correct_1_or_0=''))

    # ------------------------------------------------------------------------------------------------- per answer, per frame
    ans = td[['segment', 'frame_idx', 'repeat', 'risk', 'forced']].copy()
    ans['hazard_type'], ans['main_hazard_camera'] = hz, cam
    for j, c in enumerate(CAT_IDS):
        ans[f'k{c:02d}'] = [STAT_NAME[v] for v in st[:, j]]
    ans['motion_talk'] = motion
    for j, c in enumerate(CAT_IDS):
        ans[f'thinking_k{c:02d}'] = st_th[:, j] == 2
    for i, k in enumerate(EST_KINDS):
        ans[f'thinking_{k}_estimate'] = est[:, i] > 0
    fr = frames.copy()
    fr['score_mean'] = score
    fr['hazard_type_most'] = [m for m, _ in hz_mode]
    fr['camera_most'] = [m for m, _ in cam_mode]
    for j, c in enumerate(CAT_IDS):
        fr[f'k{c:02d}_answers'] = cnt[:, j]
    fr['motion_talk_answers'], fr['thinking_estimate_answers'] = motion_cnt, est_cnt
    for s in av.SSMS:
        fr[f'{s}_risk_percentile'] = pct[s]

    # ------------------------------------------------------------------------------------------------- comparisons
    rows_comp = []
    for sec in ('A', 'B', 'C', 'variant'):
        lst = [x for x in comps if x[0] == sec]
        for kind in [None] + sorted({x[1] for x in lst}):
            sub = [x for x in lst if kind is None or x[1] == kind]
            n_out = sum(x[3].startswith(('above', 'below')) for x in sub)
            rows_comp.append(dict(section=sec, kind=kind or 'all', intervals=len(sub), leaving_out_0_or_50=n_out,
                                  by_chance_about=0.05 * len(sub)))
    rows_comp_list = [dict(section=s_, kind=k_, what=w_, side=d_) for s_, k_, w_, d_ in comps]

    # ------------------------------------------------------------------------------------------------- write
    def W(name, rows):
        pd.DataFrame(rows).to_csv(os.path.join(O, name), index=False)

    W('exploratory_hazard_distribution.csv', rows_dist)
    W('exploratory_hazard_invalid.csv', rows_inv if rows_inv else [dict(field='(none)', written='', answers=0)])
    ct.to_csv(os.path.join(O, 'exploratory_hazard_crosstab.csv'))
    W('exploratory_hazard_stability.csv', rows_stab)
    W('exploratory_hazard_frame_groups.csv', rows_hg)
    W('exploratory_hazard_answer_groups.csv', rows_ha)
    W('exploratory_hazard_camera_groups.csv', rows_cg)
    W('exploratory_hazard_rear_end_camera.csv', rows_rc)
    W('exploratory_hazard_ssm_percentile.csv', rows_hp)
    W('exploratory_hazard_raw.csv', rows_hr)
    W('exploratory_hazard_tau_b.csv', rows_ht)
    W('exploratory_keywords_forms.csv', [dict(category=c, keyword=k, form=f) for c, k, f in mt.forms])
    W('exploratory_keywords_forms_variant.csv', [dict(category=c, keyword=k, form=f) for c, k, f in mv.forms])
    W('exploratory_keywords_categories.csv', rows_bc)
    W('exploratory_keywords_scores.csv', rows_bs)
    W('exploratory_keywords_truth.csv', rows_bt)
    W('exploratory_keywords_cues.csv', rows_bq)
    W('exploratory_keywords_motion.csv', [x for x in rows_two if x['split'].startswith('motion')])
    W('exploratory_keywords_motion_ssm.csv', [x for x in rows_two_pct if x['split'].startswith('motion')])
    W('exploratory_keywords_motion_raw.csv', [x for x in rows_two_raw if x['split'].startswith('motion')])
    W('exploratory_keywords_motion_tau_b.csv', [x for x in rows_two_tau if x['split'].startswith('motion')])
    W('exploratory_keywords_variant_scores.csv', rows_vs)
    W('exploratory_keywords_variant_motion.csv', [x for x in rows_two if x['split'].startswith('variant')])
    W('exploratory_keywords_variant_motion_tau_b.csv', [x for x in rows_two_tau if x['split'].startswith('variant')])
    W('exploratory_keywords_comparisons.csv', rows_comp)
    W('exploratory_keywords_comparisons_list.csv', rows_comp_list)
    W('exploratory_keywords_sample.csv', rows_sample)
    W('exploratory_keywords_check.csv', rows_ck)
    ans.to_csv(os.path.join(O, 'exploratory_keywords_answers.csv'), index=False)
    fr.to_csv(os.path.join(O, 'exploratory_keywords_frames.csv'), index=False)
    W('exploratory_thinking_categories.csv', rows_tc)
    W('exploratory_thinking_estimates.csv', rows_te)
    W('exploratory_thinking_estimate_units.csv', rows_tu)
    W('exploratory_thinking_groups.csv', [x for x in rows_two if x['split'].startswith('thinking')])
    W('exploratory_thinking_ssm.csv', [x for x in rows_two_pct if x['split'].startswith('thinking')])
    W('exploratory_thinking_raw.csv', [x for x in rows_two_raw if x['split'].startswith('thinking')])
    W('exploratory_thinking_tau_b.csv', [x for x in rows_two_tau if x['split'].startswith('thinking')])
    with open(os.path.join(O, 'exploratory_keywords_params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'EXPLORATORY: reason analysis round 3 (preset categories), {MODEL} {COND}, main result 1,084 frames (user, '
                f'2026-11-13; plan in CLAUDE.md "理由分析" -> "第三轮")\nrun {time.strftime("%Y-%m-%d %H:%M:%S")}; out_dir '
                f'{os.path.abspath(O)}\nprogram {os.path.abspath(__file__)} sha256 {prog_sha}\n')
        f.write(f'replies: {main_f} (fields parsed.risk, parsed.hazard_type, parsed.main_hazard_camera, '
                f'parsed.main_hazard, parsed.reason, reasoning); forced answers: {forced_f} (parsed.*); the thinking of a '
                f'forced answer is the reply\'s own reasoning, the thinking cut at max_tokens, which the forced record '
                f'keeps unchanged ({forced_think_same} of {n_forced_f} the same text)\n')
        f.write(f'frame scores: {p_fs} score_mean; answer scores: parsed.risk; percentiles: {p_r2}; truth '
                f'{L}/quantities/truth_main.csv; cues: ' + '; '.join(f'{k}: {v}' for k, v in cue_src.items()) + '\n')
        f.write('keyword classes (as the plan):\n')
        for c, nm, kws in CATS:
            f.write(f'  {c} {nm}: {", ".join(kws) if kws else "regex " + CAT16_RE.pattern}\n')
        f.write(f'lemmas: {LEMMA}\nirregular plurals: { {k: sorted(v) for k, v in IRREG.items()} }\n'
                f'not changed (prepositions, adverbs, modal verbs: neither nouns nor verbs): {sorted(NO_FORMS)}\n')
        f.write('rules: lower case; tokens [a-z0-9]+ with hyphens / dashes as spaces; plural, -s, -ed, -ing forms (see the '
                'program\'s docstring), phrases as the product of their words\' forms; conflicts: traffic in traffic '
                'light / traffic signal not class 7; signal in turn signal / blinker not class 11; no standalone close; '
                'stopped not class 5; stop in stop sign not class 5 nor 20; negation: '
                f'{sorted(NEG)} among the 3 tokens before; motion talk = classes {MOTION}; frame mentions: 3+ of 5 '
                '(main), 1+ (sensitivity)\n')
        f.write(f'thinking estimates regex {EST_RE.pattern}; kinds {EST_GROUP}\n')
        f.write(f'post-hoc variant: approach words {sorted(APPROACH)} followed within 4 tokens by {sorted(mt.location)} not '
                f'class 6; extra forms {VARIANT_EXTRA}\n')
        f.write(f'hazard_type frame groups (most frequent of 5, ties = mixed): {HZ_GROUPS}; camera frame groups '
                f'{CAM_GROUPS}\n')
        f.write(f'intervals: {N_BOOT} resamples of whole segments, default_rng({BOOT_SEED}), drawn segments\' sha256 '
                f'{hh.hexdigest()}; percentiles 2.5 / 97.5 (numpy linear) over the resamples where the value is defined; '
                f'no tau-b interval under {MIN_TAU} frames; no score-difference interval under {MIN_DIFF} frames (answers); '
                f'Fleiss\' kappa and group-mean intervals are descriptive and not counted; sample default_rng({SEED}), '
                f'{N_SAMPLE} per class and {N_SAMPLE} estimates\n')
        f.write(f'every form of the main rules ({len(mt.forms)}):\n')
        for c, k, fm_ in mt.forms:
            f.write(f'  {c}\t{k}\t{fm_}\n')
        f.write('post-hoc variant: the same forms and ' + ', '.join(f'{x} ({c})' for c, xs in VARIANT_EXTRA.items()
                                                                   for x in xs) + '\n')

    # ------------------------------------------------------------------------------------------------- report
    print('\n' + '=' * 110)
    for r_ in rows_comp:
        if r_['kind'] == 'all':
            print(f"  {r_['section']}: {r_['intervals']} intervals, {r_['leaving_out_0_or_50']} leave out 0 (or 50), by "
                  f"chance about {r_['by_chance_about']:.1f}")
    print(f'\ndone in {time.time() - t0:.0f} s; checks not met: {len(failed)}')
    for w_ in failed:
        print('  ', w_)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
