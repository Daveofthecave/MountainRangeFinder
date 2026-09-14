#!/usr/bin/env python3
# viz/make_g02_hexgrid.py -- the anchor lattice over +-16,000 blocks, and why
# it's hexagonal (KernelCoverage, gpu.cu).

import math
from PIL import Image, ImageDraw

from geom import hex_anchors_blocks, square_lattice_edges, hex_lattice_edges
from fields import climate_grid
from style import (BG, TXT, DIM, HIT, MISS, ACCENT, WATER, WHITE,
                   font, blend, dashed_circle, xmark, save_gif, title, wrap_text,
                   with_rail)
import numpy as np

SEED = 6696478651374553046   # the magic seed's mountains star in the backdrop
R = 16000
PITCH = 320                  # backdrop field pitch (blocks)
N = 2 * R // PITCH + 1

ero = climate_grid(SEED, "erosion", 2, -R // 4, -R // 4, N, N, PITCH // 4)

MAP = 560              # a smaller map leaves the captions room to breathe
HEAD = 100
W, H = MAP, HEAD + MAP
S = MAP / (2 * R)            # px per block

def px(x, z):
    return ((x + R) * S, HEAD + (z + R) * S)

lo, hi = np.percentile(ero, [2, 98])
grey = np.clip((ero - lo) / (hi - lo), 0, 1)
base = (grey[..., None] * np.array([70.0, 90.0, 120.0]) * 0.55).astype(np.uint8)
base_img = Image.fromarray(base).resize((MAP, MAP), Image.NEAREST)

SQ = [(-R + 1600 * c, -R + 1600 * r) for r in range(21) for c in range(21)]
SQ_EDGES = square_lattice_edges()
HEX_RC = {(r, c): (x, z) for x, z, r, c in hex_anchors_blocks()}
HEX_EDGES = hex_lattice_edges()
# Pair each square anchor with its nearest still-free hex anchor, so the morph
# is a gentle local shuffle rather than a column shear. Square anchors left
# without a partner (the hex lattice has 42 fewer points) fade in place.
_free_hex = set(HEX_RC.values())
PAIRS = []
for _sq in [(-R + 1600 * c, -R + 1600 * r) for r in range(21) for c in range(21)]:
    _hbest = min(_free_hex,
                 key=lambda h2: (h2[0] - _sq[0]) ** 2 + (h2[1] - _sq[1]) ** 2,
                 default=None)
    if _hbest is not None and ((_hbest[0] - _sq[0]) ** 2
                               + (_hbest[1] - _sq[1]) ** 2) <= 2200 ** 2:
        _free_hex.discard(_hbest)
        PAIRS.append((_sq, _hbest))
    else:
        PAIRS.append((_sq, None))

SQ_COL = blend(BG, WATER, 0.75)
SQ_EDGE = blend(BG, WATER, 0.40)
HEX_COL = blend(BG, HIT, 0.85)
HEX_EDGE = blend(BG, HIT, 0.38)

def compose(dots, l1, l2, anns=None, edges=None, edge_col=DIM, edge_sets=None):
    fr = Image.new("RGB", (W, H), BG)
    fr.paste(base_img, (0, HEAD))
    dr = ImageDraw.Draw(fr)
    if edges:
        for (x1, z1, x2, z2) in edges:
            dr.line([px(x1, z1), px(x2, z2)], fill=edge_col)
    if edge_sets:
        for edge_list, col in edge_sets:
            for (x1, z1, x2, z2) in edge_list:
                dr.line([px(x1, z1), px(x2, z2)], fill=col)
    dr.rectangle([0, HEAD, W - 1, H - 1], outline=DIM)
    for (x, z, rad, col) in dots:
        if rad <= 0.2:
            continue
        cx, cy = px(x, z)
        dr.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=col)
    if anns:
        anns(dr)
    y = wrap_text(dr, (12, 8), l1, max_w=W - 24, size=16)
    wrap_text(dr, (12, max(34, y + 2)), l2, max_w=W - 24, size=12, fill=DIM)
    return fr

frames, dur = [], []

frames.append(compose([], "The Search Square: ±16,000 Blocks Around (0, 0)",
                      "Each anchor is a launch point for the whole pipeline"))
dur.append(1800)

sq_dots = [(x, z, 2.6, SQ_COL) for x, z in SQ]
for f in range(4):
    t = (f + 1) / 4
    frames.append(compose([(x, z, rad * t, col) for x, z, rad, col in sq_dots],
                          "Naive Approach: A Square Grid of Anchors",
                          "",
                          edges=SQ_EDGES, edge_col=blend(SQ_EDGE, BG, 1 - t)))
    dur.append(70)

def square_ann(dr):
    x0, z0 = px(0, 0)
    dashed_circle(dr, x0, z0, 800 * math.sqrt(2) * S, ACCENT)
    wx, wz = px(800, 800)
    xmark(dr, wx, wz, 7, MISS, 3)
    dr.line([x0, z0, wx, wz], fill=ACCENT, width=2)

frames.append(compose(sq_dots,
    "Naive Approach: A Square Grid of Anchors",
    "441 anchors, 1,600 blocks apart; a blob's center can sit 1,131 blocks "
    "from the nearest anchor (marked)",
    square_ann, edges=SQ_EDGES, edge_col=SQ_EDGE))
dur.append(3400)

MORPH = 12
for f in range(MORPH):
    t = (f + 1) / MORPH
    e = t * t * (3 - 2 * t)          # smoothstep easing
    dots = []
    for (sx, sz), dst in PAIRS:
        if dst is None:
            dots.append((sx, sz, 2.6 * (1 - e), SQ_COL))   # surplus anchors fade
        else:
            hx, hz = dst
            dots.append((sx + (hx - sx) * e, sz + (hz - sz) * e, 2.6,
                         blend(SQ_COL, HEX_COL, e)))
    # A true line morph isn't possible (840 square edges vs 1,118 hex edges),
    # so the lines crossfade alongside the dot morph: square out first,
    # hex in second.
    sq_fade = max(0.0, 1.0 - 2 * e)
    hex_fade = max(0.0, 2 * e - 1.0)
    frames.append(compose(dots, "Morphing to a Hexagonal Lattice...",
                          "Same area, but with better packing",
                          edge_sets=[(SQ_EDGES, blend(BG, SQ_EDGE, sq_fade)),
                                     (HEX_EDGES, blend(BG, HEX_EDGE, hex_fade))]))
    dur.append(60)

hex_dots = [(x, z, 2.6, HEX_COL) for x, z in HEX_RC.values()]

def hex_ann(dr):
    x0, z0 = px(0, 0)
    rad = 1920 / math.sqrt(3)        # covering radius of the hex lattice
    dashed_circle(dr, x0, z0, rad * S, ACCENT)
    pts = [(x0 + rad * math.cos(math.radians(90 + 60 * k)) * S,
            z0 - rad * math.sin(math.radians(90 + 60 * k)) * S) for k in range(6)]
    dr.line(pts + [pts[0]], fill=DIM, width=2)
    wx, wz = px(0, -rad)
    xmark(dr, wx, wz, 7, MISS, 3)
    dr.line([x0, z0, wx, wz], fill=ACCENT, width=2)

frames.append(compose(hex_dots,
    "Better Approach: A Hexagonal Lattice",
    "399 anchors, 1,920 blocks apart; the worst spot is 1,109 blocks out, "
    "with 42 fewer anchors to check per seed",
    hex_ann, edges=HEX_EDGES, edge_col=HEX_EDGE))
dur.append(3600)

frames.append(compose(hex_dots,
    "Each anchor runs the full verification sequence (with early exit):",
    "7-point erosion check, sunflower spiral check, climate window checks, flood fill",
    edges=HEX_EDGES, edge_col=HEX_EDGE))
dur.append(3000)

frames = [with_rail(f, 2 if i == 0 else 3) for i, f in enumerate(frames)]
save_gif(frames, dur, "g02_hexgrid.gif")
