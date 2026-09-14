#!/usr/bin/env python3
# viz/make_g12_constellation.py -- every verified megaregion the pipeline has
# ever found, accreting onto the +-16,000-block search square in file order.
# Points zbuffer by score, so the final map shows the best find per pixel.
# Needs output_all.txt (or output.txt); skips gracefully if neither exists.

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

from style import (BG, TXT, DIM, HIT, ACCENT, WATER, WHITE,
                   font, blend, save_gif, with_rail)

R = 16000
SIZE = 720
W, H = 760, 836
HEAD = 70

def find_input():
    for p in ("output_all.txt", "output.txt"):
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            return p
    return None

def iter_points(path):
    """Yield (score, x, z) from output rows. Per-line format detection:
    a '.'/'e' in the first token means score-first, else seed-first."""
    with open(path, errors="replace") as fp:
        for line in fp:
            if line.startswith("#"):
                continue
            t = line.split(None, 4)   # only the first 4 tokens matter
            if len(t) < 4:
                continue
            try:
                if any(ch in t[0] for ch in ".eE"):
                    sc = float(t[0]); x = int(t[2]); z = int(t[3])
                else:
                    sc = float("nan"); x = int(t[1]); z = int(t[2])
            except ValueError:
                continue
            if sc <= -900:    # the fd() "not measured" sentinel
                sc = float("nan")
            yield sc, x, z

def score_color(s):
    if s != s:     return blend(BG, DIM, 0.8)      # unscored row
    if s < 0:      return blend(BG, WATER, 0.35)
    if s < 2:      return blend(BG, WATER, 0.45 + 0.15 * (s / 2))
    if s < 4:      return blend(WATER, HIT, (s - 2) / 2)
    if s < 6:      return blend(HIT, ACCENT, (s - 4) / 2)
    return blend(ACCENT, WHITE, min(1.0, (s - 6) / 4))

def cell(x, z):
    px = int((x + R) / (2 * R) * (SIZE - 1))
    py = int((z + R) / (2 * R) * (SIZE - 1))
    return max(0, min(SIZE - 1, px)), max(0, min(SIZE - 1, py))

path = find_input()
if path is None:
    print("g12: No output_all.txt / output.txt found -- skipping "
          "(this GIF needs a real corpus)")
    sys.exit(0)

print("g12: Reading %s (this can take a moment for large corpora)..." % path)
pts = list(iter_points(path))
print("g12: %d rows" % len(pts))

img = np.zeros((SIZE, SIZE, 3), np.uint8)
img[:] = BG
grid_c = blend(BG, DIM, 0.45)
for g in range(-R, R + 1, 4000):
    gx, gy = cell(g, g)
    img[:, gx] = grid_c
    img[gy, :] = grid_c
c0, r0 = cell(0, 0)
img[:, c0] = blend(BG, DIM, 0.9)
img[r0, :] = blend(BG, DIM, 0.9)

zbuf = np.full((SIZE, SIZE), -1e18)
best_sc = -1e18
best_px = None

def compose(l1, l2):
    fr = Image.new("RGB", (W, H), BG)
    fr.paste(Image.fromarray(img), (20, HEAD))
    dr = ImageDraw.Draw(fr)
    dr.rectangle([20, HEAD, 20 + SIZE - 1, HEAD + SIZE - 1], outline=DIM)
    dr.text((14, 10), l1, font=font(17), fill=TXT)
    dr.text((14, 40), l2, font=font(12), fill=DIM)
    dr.text((14, HEAD + SIZE + 12),
            "   Blue: 0–2  ·  Green: 2–4  ·  Gold: 4–6  ·  White-Gold: 6+  "
            "(each pixel represents one megaregion's core)",
            font=font(11), fill=DIM)
    return fr

frames, dur = [], []
TARGET_FRAMES = 110
chunk = max(1, len(pts) // TARGET_FRAMES)

for c0i in range(0, len(pts), chunk):
    for sc, x, z in pts[c0i:c0i + chunk]:
        px, py = cell(x, z)
        if sc > zbuf[py, px] or (sc != sc and zbuf[py, px] == -1e18):
            zbuf[py, px] = sc if sc == sc else -1e17
            img[py, px] = score_color(sc)
        if sc == sc and sc > best_sc:
            best_sc = sc
            best_px = (px, py)
    n = min(c0i + chunk, len(pts))
    frames.append(compose(
        "Every verified region in a sample run, in the order it was found:",
        "%s of %s rows; best score so far: %.2f"
        % (format(n, ","), format(len(pts), ","), best_sc)))
    dur.append(70)

# Pulse the best find.
if best_px is not None:
    bx, by = 20 + best_px[0], HEAD + best_px[1]
    for k in range(9):
        rr = 6 + k * 5
        fr = compose("A Constellation of Verified Regions from a Sample Run",
                     "Best score: %.2f" % best_sc)
        dr = ImageDraw.Draw(fr)
        col = blend(ACCENT, BG, k / 9.0)
        dr.ellipse([bx - rr, by - rr, bx + rr, by + rr], outline=col, width=2)
        if k >= 4:
            dr.ellipse([bx - 3, by - 3, bx + 3, by + 3], outline=WHITE, width=2)
        frames.append(fr)
        dur.append(100)

fin = compose("A Constellation of Verified Regions from a Sample Run",
              "Best score: %.2f, marked with a ring" % best_sc)
dr = ImageDraw.Draw(fin)
if best_px is not None:
    bx, by = 20 + best_px[0], HEAD + best_px[1]
    dr.ellipse([bx - 8, by - 8, bx + 8, by + 8], outline=ACCENT, width=2)
    dr.ellipse([bx - 2, by - 2, bx + 2, by + 2], outline=WHITE, width=2)
frames.append(fin)
dur.append(4200)

frames = [with_rail(f, 12) for f in frames]
save_gif(frames, dur, "g12_constellation.gif")