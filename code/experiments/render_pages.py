import sys, pymupdf
src, out, pages = sys.argv[1], sys.argv[2], [int(p) for p in sys.argv[3].split(',')]
doc = pymupdf.open(src)
for p in pages:
    pix = doc[p - 1].get_pixmap(dpi=130)
    pix.save(f'{out}/page{p}.png')
    print(f'{out}/page{p}.png', pix.width, pix.height)
