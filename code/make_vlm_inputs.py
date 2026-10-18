"""Build the VLM inputs (decided 2026-10-17): for every target frame t, the 5 time steps t-2.0, -1.5, -1.0, -0.5 and
0 s (2 frames per second, past only), each step as one labelled image with all five cameras:

    top row:    FRONT_LEFT | FRONT | FRONT_RIGHT
    bottom row: SIDE_LEFT  | (empty) | SIDE_RIGHT

A step image is shared by several targets (frame k is "now" for target k and "0.5 s ago" for target k+5), so it carries
no time label; the prompt states the order and spacing of the 5 images.

Target frames: every 0.5 s (frame index divisible by 5), at least 2 s into the segment, and not flagged edge_frame
in frame_ssm.csv (speeds and accelerations there are unreliable, so the analysis drops them).

Two layouts (to be compared in the pilot):
    --layout grid      (default) one 2 x 3 image per time step -> 5 images per target
    --layout separate  each camera as its own image (--cam_width px wide), captioned in the prompt -> 25 images per
                       target, ordered time step by time step, cameras front-left, front, front-right, left, right

Usage:
    python -I make_vlm_inputs.py <extracted_root> <frame_ssm.csv> <out_dir> [--layout grid|separate] [segment ...]

Output: the images under <out_dir>/<segment>/ (step_<frame>.jpg or <CAMERA>_<frame>.jpg), and <out_dir>/targets.csv
with one row per target: segment, frame_idx, t_s, ego_speed_kmh (Waymo's reported ego speed at t; metadata for the
analysis only, NOT given to the VLM), images (oldest first, ';'-separated)."""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wod_common as wc  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')

STEP = 5            # frames between time steps (0.5 s at 10 Hz)
N_STEPS = 5         # t-2.0 ... t
TOP = ['FRONT_LEFT', 'FRONT', 'FRONT_RIGHT']
BOTTOM = ['SIDE_LEFT', None, 'SIDE_RIGHT']
CAMERA_ORDER = ['FRONT_LEFT', 'FRONT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']   # 'separate' layout, left to right
FONT_FILES = ['C:/Windows/Fonts/arialbd.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf']


def font(size):
    for f in FONT_FILES:
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def label(draw, xy, text, fnt):
    x, y = xy
    l, t, r, b = draw.textbbox((x, y), text, font=fnt)
    draw.rectangle((l - 6, t - 4, r + 6, b + 4), fill=(0, 0, 0))
    draw.text((x, y), text, font=fnt, fill=(255, 255, 255))


def camera_files(seg_dir):
    files = {}
    for cam in TOP + [c for c in BOTTOM if c]:
        d = os.path.join(seg_dir, 'images', cam)
        files[cam] = {int(f.split('_')[0]): os.path.join(d, f) for f in os.listdir(d) if f.endswith('.jpg')}
    return files


def step_image(files, k, tile):
    """One time step: the five cameras of frame k on a 3 x 2 grid, each panel labelled with its camera."""
    ims = {cam: Image.open(files[cam][k]).convert('RGB') for cam in files}
    scaled = {cam: im.resize((tile, round(im.height * tile / im.width)), Image.LANCZOS) for cam, im in ims.items()}
    h_top = max(scaled[c].height for c in TOP)
    h_bot = max(scaled[c].height for c in BOTTOM if c)
    canvas = Image.new('RGB', (3 * tile, h_top + h_bot), (40, 40, 40))
    draw = ImageDraw.Draw(canvas)
    f_cam = font(max(14, tile // 28))
    for cams, y0 in ((TOP, 0), (BOTTOM, h_top)):
        for col, cam in enumerate(cams):
            if cam is None:
                continue
            canvas.paste(scaled[cam], (col * tile, y0))
            label(draw, (col * tile + 10, y0 + 8), cam.replace('_', ' ').lower(), f_cam)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('frame_ssm')
    ap.add_argument('out')
    ap.add_argument('segments', nargs='*')
    ap.add_argument('--layout', choices=['grid', 'separate'], default='grid',
                    help='grid: one labelled 2 x 3 image per time step (5 images per target); '
                         'separate: every camera as its own image (25 images per target, captioned in the prompt)')
    ap.add_argument('--tile', type=int, default=640, help='grid: width (px) of each camera panel')
    ap.add_argument('--cam_width', type=int, default=960, help='separate: width (px) of each camera image')
    a = ap.parse_args()
    fr = pd.read_csv(a.frame_ssm)
    segs = a.segments or sorted(fr['segment'].unique())
    rows = []
    for name in segs:
        seg = wc.Segment(a.root, name)
        files = camera_files(os.path.join(a.root, name))
        g = fr[fr['segment'] == name].set_index('frame_idx')
        targets = [k for k in g.index if k % STEP == 0 and k >= STEP * (N_STEPS - 1) and not g.loc[k, 'edge_frame']]
        need = sorted({k - STEP * j for k in targets for j in range(N_STEPS)})
        out_dir = os.path.join(a.out, name)
        os.makedirs(out_dir, exist_ok=True)
        for k in need:
            if a.layout == 'grid':
                path = os.path.join(out_dir, f'step_{k:03d}.jpg')
                if not os.path.exists(path):
                    step_image(files, k, a.tile).save(path, quality=90)
            else:
                for cam in CAMERA_ORDER:
                    path = os.path.join(out_dir, f'{cam}_{k:03d}.jpg')
                    if not os.path.exists(path):
                        im = Image.open(files[cam][k]).convert('RGB')
                        im.resize((a.cam_width, round(im.height * a.cam_width / im.width)), Image.LANCZOS).save(
                            path, quality=90)
        speed = np.hypot(seg.ego_v_reported[:, 0], seg.ego_v_reported[:, 1]) * 3.6
        for k in targets:
            steps = [k - STEP * j for j in range(N_STEPS - 1, -1, -1)]          # oldest first
            if a.layout == 'grid':
                images = [f'{name}/step_{s:03d}.jpg' for s in steps]
            else:                                                            # time step by time step, cameras in order
                images = [f'{name}/{cam}_{s:03d}.jpg' for s in steps for cam in CAMERA_ORDER]
            rows.append(dict(segment=name, frame_idx=k, t_s=round(float(seg.t[k]), 2),
                             ego_speed_kmh=round(float(speed[k]), 1), images=';'.join(images)))
        print(f'{name}: {len(targets)} targets, {len(need)} time steps')
    pd.DataFrame(rows).to_csv(os.path.join(a.out, 'targets.csv'), index=False)
    print(f'targets.csv: {len(rows)} rows')


if __name__ == '__main__':
    main()
