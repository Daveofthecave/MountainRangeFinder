#!/usr/bin/env python3
# viz/make_g01_octaves.py -- why the GPU can filter on two octaves: the same
# region's erosion field at increasing octave depth. The field is drawn as a
# translucent heatmap over the real terrain; a thin luminous line marks the
# -0.35 blob boundary. Everything is composited at field resolution, then
# resized once at the end (masks and images must share a grid).

import numpy as np
from PIL import Image, ImageDraw

from fields import climate_grid, height_grid
from style import (BG, TXT, DIM, HIT, ACCENT, WATER, WHITE,
                   font, save_gif, title, with_rail)

SEED = 6696478651374553046
CX, CZ = -14500, -15000          # roughly the magic blob's heart
EXT = 6000
PITCH = 64
N = 2 * EXT // PITCH + 1         # 188

LEVELS = [  # (label, nmax octaves, boundary color)
    ("Erosion: Octave 0A Only",                                  1, WATER),
    ("Erosion: Octaves 0A + 0B (what this pipeline uses)",       2, HIT),
    ("Erosion: Octaves 0A + 0B + 1A + 1B",                       4, ACCENT),
    ("Erosion: All Octaves (what Minecraft uses)",              -1, WHITE),
]
THRESH = -0.35

print("sampling fields (cached after first run)...")
h = height_grid(SEED, CX - EXT, CZ - EXT, N, N, PITCH)

def terrain(hh):
    # Four-stop land ramp (valley floor -> foothills -> high slopes -> snow)
    # with a gentle S-curve, so foothill/mid-ground bands actually separate.
    t = np.clip((hh - 55.0) / 265.0, 0, 1)
    t = t * t * (3 - 2 * t) * 0.55 + t * 0.45      # partial smoothstep
    c0 = np.array([58.0, 88.0, 66.0])    # valley floor: deep green
    c1 = np.array([108.0, 132.0, 88.0])  # foothills
    c2 = np.array([168.0, 168.0, 130.0]) # high slopes: warm grey-green
    c3 = np.array([246.0, 250.0, 252.0]) # snow
    b1 = t / 0.38
    b2 = (t - 0.38) / 0.34
    b3 = (t - 0.72) / 0.28
    land = np.empty(hh.shape + (3,), float)
    m = t < 0.38
    land[m] = c0 * (1 - b1[m, None]) + c1 * b1[m, None]
    m = (t >= 0.38) & (t < 0.72)
    land[m] = c1 * (1 - b2[m, None]) + c2 * b2[m, None]
    m = t >= 0.72
    land[m] = c2 * (1 - b3[m, None]) + c3 * b3[m, None]
    out = land.astype(np.uint8)
    out[hh <= 55] = (34, 66, 116)
    return out

base = terrain(h)                # N x N x 3, field resolution

def edge_of(m):
    inner = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    e = np.zeros_like(m)
    e[1:-1, 1:-1] = m[1:-1, 1:-1] & ~inner
    return e

def field_composite(ero, bcol):
    """Field heatmap over terrain + thin boundary line, at FIELD resolution."""
    lo, hi = np.percentile(ero, [3, 97])
    heat01 = np.clip((ero - lo) / (hi - lo), 0, 1)
    heat01 = np.clip((heat01 - 0.5) * 1.35 + 0.5, 0, 1)  # gentle contrast S-curve
    # Wide-luminance ramp, CV's convention (white = high): deep navy for low
    # erosion (mountains) to pale ice for high. The old ramp spanned only
    # ~100 gray levels on top of terrain, which is why it read as a wash.
    heat = (heat01[..., None] * np.array([170.0, 190.0, 215.0])
            + (1 - heat01[..., None]) * np.array([16.0, 22.0, 40.0]))
    comp = (base.astype(float) * 0.40 + heat * 0.75).clip(0, 255).astype(np.uint8)
    e = edge_of(ero <= THRESH)
    comp[e] = np.array(bcol, dtype=np.uint8)
    glow = np.zeros_like(e)
    glow[1:-1, 1:-1] = (e[:-2, 1:-1] | e[2:, 1:-1] | e[1:-1, :-2] | e[1:-1, 2:]) & ~e[1:-1, 1:-1]
    comp[glow] = (comp[glow].astype(float) * 0.5
                  + np.array(bcol, float) * 0.5).astype(np.uint8)
    return comp

def frame_img(ero, bcol):
    return Image.fromarray(field_composite(ero, bcol)).resize((720, 720), Image.NEAREST)

W, H = 760, 836

def compose(img, l1, l2, col):
    fr = Image.new("RGB", (W, H), BG)
    fr.paste(img, (20, 70))
    dr = ImageDraw.Draw(fr)
    dr.rectangle([20, 70, 739, 789], outline=DIM)
    title(dr, (14, 10), l1, max_w=W - 28)
    title(dr, (14, 40), l2, max_w=W - 28, size=13, fill=col)
    return fr

images = []
for name, nmax, col in LEVELS:
    ero = climate_grid(SEED, "erosion", nmax, (CX - EXT) // 4, (CZ - EXT) // 4,
                       N, N, PITCH // 4)
    images.append((name, col, frame_img(ero, col)))

frames, dur = [], []
intro = Image.new("RGB", (W, H), BG)
intro.paste(Image.fromarray(base).resize((720, 720), Image.NEAREST), (20, 70))
dri = ImageDraw.Draw(intro)
dri.rectangle([20, 70, 739, 789], outline=DIM)
title(dri, (14, 10), "Each climate field is a stack of octaves", max_w=W - 28)
title(dri, (14, 40), "Every octave layer adds detail; the pipeline filters on just the first two octaves",
      max_w=W - 28, size=13, fill=DIM)
frames.append(intro)
dur.append(3000)
# Start the ladder from the bare terrain backdrop, so level 1 fades in like
# every later level instead of popping.
prev = ("", Image.fromarray(base).resize((720, 720), Image.NEAREST))
for name, col, img in images:
    if prev is not None:
        for f in range(4):
            frames.append(compose(Image.blend(prev[1], img, (f + 1) / 4),
                                  "Octave Ladder: How Much Detail Is Enough?",
                                  name, col))
            dur.append(55)
    frames.append(compose(img,
                          "Octave Ladder: How Much Detail Is Enough?",
                          "%s (boundary: erosion ≤ %.2f)" % (name, THRESH), col))
    dur.append(2100)
    prev = (name, img)

# Finale: 0B boundary vs full boundary, composited at field resolution.
eroF = climate_grid(SEED, "erosion", -1, (CX - EXT) // 4, (CZ - EXT) // 4, N, N, PITCH // 4)
eroB = climate_grid(SEED, "erosion", 2, (CX - EXT) // 4, (CZ - EXT) // 4, N, N, PITCH // 4)
comp = field_composite(eroF, WHITE)
comp[edge_of(eroB <= THRESH)] = HIT
comp[edge_of(eroF <= THRESH)] = WHITE
fr = compose(Image.fromarray(comp).resize((720, 720), Image.NEAREST),
             "Green boundary: two octaves; White boundary: all octaves",
             "The truncated proxy rarely misplaces the boundary, and the CPU re-measures survivors in double precision", WHITE)
frames.append(fr)
dur.append(2600)

frames = [with_rail(f, 1) for f in frames]
save_gif(frames, dur, "g01_octaves.gif")
