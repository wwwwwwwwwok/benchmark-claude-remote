"""FRONT camera images of one segment around a frame, stitched into one picture for the user (2026-10-22).
Frames: k-10 (1 s before), k, k+10, k+20, k+30 (up to the end of the segment); the given vehicle is outlined in red where
it has a projected box in the FRONT image; each panel is captioned with the time offset, the ego speed and the vehicle's
position in the ego frame. Claude does not open the output (Waymo licence question open).

usage: python -I front_sequence.py <extracted split root> <segment> <frame> <object id> <out.jpg>
       (or import make_sheet)
"""
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.expanduser('~/benchmark-claude-remote/code'))
import wod_common as wc  # noqa: E402

PANEL_W = 960


def font(size=26):
    try:
        return ImageFont.truetype('DejaVuSans-Bold.ttf', size)
    except OSError:
        return ImageFont.load_default()


def make_sheet(seg, k, oid, out, header=''):
    n = len(seg.frames)
    frames = [f for f in (k - 10, k, k + 10, k + 20, k + 30) if 0 <= f < n]
    pb = pd.read_parquet(os.path.join(seg.dir, 'projected_boxes.parquet'))
    pb = pb[(pb['camera'] == 'FRONT') & (pb['id'] == f'{oid}_FRONT')].set_index('frame_idx')
    ft = font()
    panels = []
    for f in frames:
        img = Image.open(os.path.join(seg.dir, 'images', 'FRONT', f'{f:03d}_{int(seg.ts[f])}.jpg')).convert('RGB')
        d = ImageDraw.Draw(img)
        if f in pb.index:
            b = pb.loc[f]
            d.rectangle([b.box_cx - b.box_w / 2, b.box_cy - b.box_h / 2, b.box_cx + b.box_w / 2, b.box_cy + b.box_h / 2],
                        outline=(220, 30, 30), width=6)
        o = seg.objects[(seg.objects['frame_idx'] == f) & (seg.objects['id'] == oid)]
        pos = (f'vehicle at x {o.cx.iloc[0]:.1f} m, y {o.cy.iloc[0]:.1f} m, '
               f'{np.hypot(o.speed_x.iloc[0], o.speed_y.iloc[0]):.2f} m/s' if len(o) else 'vehicle not labelled')
        cap = (f'{(f - k) / 10:+.1f} s (frame {f})   ego {np.hypot(*seg.ego_v[f, :2]):.1f} m/s   {pos}'
               + ('' if f in pb.index else '   [no box in FRONT]'))
        scale = PANEL_W / img.width
        img = img.resize((PANEL_W, int(img.height * scale)))
        canvas = Image.new('RGB', (PANEL_W, img.height + 44), (252, 252, 251))
        canvas.paste(img, (0, 44))
        ImageDraw.Draw(canvas).text((10, 8), cap, fill=(30, 30, 30), font=font(22))
        panels.append(canvas)
    cols = 2
    rows = (len(panels) + cols - 1) // cols
    h = max(p.height for p in panels)
    sheet = Image.new('RGB', (PANEL_W * cols, h * rows + 90), (252, 252, 251))
    dr = ImageDraw.Draw(sheet)
    dr.text((10, 10), f'{seg.name}  FRONT camera around frame {k}; red box = vehicle {oid}', fill=(30, 30, 30), font=ft)
    dr.text((10, 48), header, fill=(30, 30, 30), font=font(22))
    for i, p in enumerate(panels):
        sheet.paste(p, ((i % cols) * PANEL_W, 90 + (i // cols) * h))
    sheet.save(out, quality=90)
    return frames


def main():
    root, name, k, oid, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
    frames = make_sheet(wc.Segment(root, name), k, oid, out)
    print(f'{out}: frames {frames}')


if __name__ == '__main__':
    main()
