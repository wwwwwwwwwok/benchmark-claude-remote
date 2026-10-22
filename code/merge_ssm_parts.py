"""Join what kamiak/waymo_ssm.sbatch writes for each block of segments into the files that one run over all the
segments would write:
    <parts>/<block>/ssm/     frame_ssm.csv, scene_summary.csv, params.txt, *.png   (code/waymo_ssm_frames.py)
    <parts>/<block>/checks/  extract_checks.csv, *.png                           (code/check_v143_extract.py)
 -> <out>/ssm/ and <out>/checks/, same file names; the PNGs are copied.
Tables are joined as text (read as strings), so every value stays exactly as the block wrote it.
Stops if a block lacks a file, the blocks used different parameters, a segment appears twice, or the joined tables
do not hold exactly <n_segments> segments in sorted order (the order of a single run).

usage: python -I merge_ssm_parts.py <parts> <out> <n_segments>
"""
import glob
import os
import shutil
import sys

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')


def read_text_table(path, index_col=None):
    return pd.read_csv(path, dtype=str, keep_default_na=False, index_col=index_col)


def per_segment(line):
    """params.txt lines written once per segment: 'ego <segment> = ...' and 'labels <segment> = ...'
    (the run-wide line 'ego = footprint' is not one of them)."""
    return line.startswith('labels ') or (line.startswith('ego ') and not line.startswith('ego = '))


def main():
    parts, out, n_seg = sys.argv[1], sys.argv[2], int(sys.argv[3])
    blocks = sorted(d for d in os.listdir(parts) if os.path.isdir(os.path.join(parts, d)))
    if not blocks:
        sys.exit(f'no blocks in {parts}')
    frames, summaries, checks = [], [], []
    head, ego_lines, label_lines = None, [], []
    for b in blocks:
        ssm, chk = os.path.join(parts, b, 'ssm'), os.path.join(parts, b, 'checks')
        need = [os.path.join(ssm, f) for f in ('frame_ssm.csv', 'scene_summary.csv', 'params.txt')] + \
               [os.path.join(chk, 'extract_checks.csv')]
        missing = [p for p in need if not os.path.exists(p)]
        if missing:
            sys.exit(f'block {b}: missing {missing}')
        frames.append(read_text_table(need[0]))
        summaries.append(read_text_table(need[1], index_col=0))
        checks.append(read_text_table(need[3], index_col=0))
        with open(need[2], encoding='utf-8') as f:
            lines = f.read().splitlines()
        h = [ln for ln in lines if not per_segment(ln)]
        if head is None:
            head = h
        elif h != head:
            sys.exit(f'block {b}: parameters differ from block {blocks[0]}:\n{h}\n{head}')
        ego_lines += [ln for ln in lines if ln.startswith('ego ') and per_segment(ln)]
        label_lines += [ln for ln in lines if ln.startswith('labels ')]

    fr = pd.concat(frames, ignore_index=True)
    summ = pd.concat(summaries)
    chk = pd.concat(checks, axis=1)
    if any(list(f.columns) != list(frames[0].columns) for f in frames) or \
            any(list(s.columns) != list(summaries[0].columns) for s in summaries) or \
            any(list(c.index) != list(checks[0].index) for c in checks):
        sys.exit('the blocks wrote different columns')
    segs = [s for f in frames for s in dict.fromkeys(f['segment'])]     # a segment in two blocks shows up twice
    for name, got in (('frame_ssm.csv', segs), ('scene_summary.csv', list(summ.index)),
                      ('extract_checks.csv', list(chk.columns)), ('params.txt ego', [ln.split()[1] for ln in ego_lines]),
                      ('params.txt labels', [ln.split()[1] for ln in label_lines])):
        if got != sorted(set(got)) or len(got) != n_seg or got != segs:
            sys.exit(f'{name}: {len(got)} segments, expected {n_seg} distinct ones in sorted order, the same in '
                     f'every file')

    os.makedirs(os.path.join(out, 'ssm'), exist_ok=True)
    os.makedirs(os.path.join(out, 'checks'), exist_ok=True)
    fr.to_csv(os.path.join(out, 'ssm', 'frame_ssm.csv'), index=False)
    summ.to_csv(os.path.join(out, 'ssm', 'scene_summary.csv'))
    chk.to_csv(os.path.join(out, 'checks', 'extract_checks.csv'))
    with open(os.path.join(out, 'ssm', 'params.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(head + ego_lines + label_lines) + '\n')
    n_png = 0
    for b in blocks:
        for sub in ('ssm', 'checks'):
            for p in glob.glob(os.path.join(parts, b, sub, '*.png')):
                shutil.copy2(p, os.path.join(out, sub, os.path.basename(p)))
                n_png += 1
    print(f'{len(blocks)} blocks -> {out}: {n_seg} segments, {len(fr)} frames, {n_png} figures')


if __name__ == '__main__':
    main()
