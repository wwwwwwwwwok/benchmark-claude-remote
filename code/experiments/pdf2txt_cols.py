"""Two-column PDF to text: rebuild lines from word boxes (for PDFs where get_text() puts every word on its own line).
usage: python -I pdf2txt_cols.py <pdf> <out.txt>"""
import sys

import pymupdf

src, dst = sys.argv[1], sys.argv[2]
doc = pymupdf.open(src)
with open(dst, 'w', encoding='utf-8') as f:
    for i, page in enumerate(doc):
        f.write(f"\n=== page {i + 1} ===\n")
        mid = page.rect.width / 2
        words = page.get_text('words')
        # a word belongs to the left column if it starts left of the middle and ends before ~middle+20
        cols = {0: [], 1: [], 2: []}   # 0 full-width (crosses the middle), 1 left, 2 right
        for w in words:
            x0, y0, x1, y1, txt = w[:5]
            if x0 < mid - 5 and x1 > mid + 5:
                cols[0].append(w)
            elif x0 < mid:
                cols[1].append(w)
            else:
                cols[2].append(w)
        for c in (1, 2, 0):
            lines = {}
            for w in cols[c]:
                key = round((w[1] + w[3]) / 2 / 2.5)   # group by vertical centre, 2.5 pt bins
                lines.setdefault(key, []).append(w)
            f.write(f"--- column {'full' if c == 0 else ('left' if c == 1 else 'right')} ---\n")
            for key in sorted(lines):
                ws = sorted(lines[key], key=lambda w: w[0])
                f.write(' '.join(w[4] for w in ws) + '\n')
print(dst, len(doc), 'pages')
