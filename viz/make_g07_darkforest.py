#!/usr/bin/env python3
# viz/make_g07_darkforest.py -- CPU gate 1/3: the NOT dark forest veto
# (recipe steps 58-59; cpu.cpp). Two real r=1,365 discs on the magic seed,
# sampled in lockstep: the pipeline's anchor disc vs the darkest disc within
# +-3,600 blocks of it. Dark forest cells are tinted maroon so the threat
# reads at a glance.
#
# Verdict fractions are computed on the sampled disc grid; the ring-by-ring
# dots are the illustrative animation layer (the kernel's exact pitch is 64).

import math
import numpy as np
from PIL import Image, ImageDraw

from fields import biome_grid
from style import (BG, TXT, DIM, HIT, MISS, ACCENT, WHITE,
                   font, blend, save_gif, title, wrap_text, with_rail, gate_chips)

SEED = 6696478651374553046
import json as _json
try:
    with open("viz/out/demo_anchor.json") as fp:
        _a = _json.load(fp)
    AX, AZ = int(_a["x"]), int(_a["z"])
except Exception:
    AX, AZ = -16064, -15044        # the candidate anchor
DF_ID = 29                         # dark_forest (cubiomes biomes.h)
DF_R, GATE = 1365, 0.07            # recipe steps 58-59 (cpu.cpp: 1365, 64, 0.07)
PITCH = 32
FIELD = 5600                       # biome field half-extent (blocks)
N = 2 * FIELD // PITCH + 1         # 351
C0 = FIELD // PITCH                # anchor's cell index: 175
SCAN = 3600 // PITCH               # darkest-disc search window (cells)
MIN_SEP = 1600 // PITCH            # keep the second disc visually distinct

BIO_COLORS = {
    0: (40, 75, 130), 10: (60, 90, 140), 24: (30, 60, 110),
    44: (45, 90, 150), 45: (45, 90, 150), 46: (45, 90, 150),
    47: (35, 75, 130), 48: (35, 75, 130), 49: (35, 75, 130), 50: (35, 75, 130),
    7: (70, 120, 190), 11: (120, 150, 200),
    1: (140, 165, 85), 129: (170, 190, 90), 177: (150, 170, 95),
    185: (230, 160, 190), 4: (55, 110, 50), 132: (90, 140, 70),
    27: (85, 130, 65), 155: (110, 150, 80),
    5: (50, 100, 85), 30: (90, 120, 110), 32: (45, 90, 75), 160: (55, 100, 85),
    12: (215, 220, 225), 140: (180, 200, 225),
    178: (205, 210, 215), 179: (200, 208, 213),
    180: (190, 190, 200), 181: (225, 230, 240), 182: (150, 150, 155),
    2: (225, 210, 140), 35: (180, 160, 85), 36: (190, 170, 95), 163: (160, 130, 95),
    6: (80, 100, 65), 184: (70, 95, 75), 21: (70, 130, 60), 168: (85, 140, 70),
    37: (200, 115, 75), 165: (210, 125, 80), 38: (190, 105, 70),
    3: (110, 125, 110), 34: (120, 130, 110), 131: (115, 125, 105),
    16: (215, 205, 165), 26: (200, 195, 160), 25: (150, 150, 150),
}
DARK_TINT = (120, 45, 55)          # maroon: the threat color

print("sampling the biome field (351x351 = ~123k biomes; cached afterwards)...")
bio = biome_grid(SEED, AX - FIELD, AZ - FIELD, N, N, PITCH)
dark = bio == DF_ID

# --- exact disc fraction on the grid (this is what the verdict prints) --------
OFFS = [(di, dj)
        for di in range(-43, 44) for dj in range(-43, 44)
        if di * di + dj * dj <= (DF_R / PITCH) ** 2]

def disc_frac(cx, cy):
    tot = 0
    for di, dj in OFFS:
        tot += dark[cy + dj, cx + di]
    return tot / len(OFFS)

# --- darkest disc: shortlist via an integral image over square windows, then
# --- measure the true disc fraction at the argmax square ----------------------
side = (2 * DF_R) // PITCH
iimg = np.pad(dark.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)

def sq_sum(cx, cy):
    r = side // 2
    x0i, x1i, y0i, y1i = cx - r, cx + r + 1, cy - r, cy + r + 1
    return iimg[y1i, x1i] - iimg[y0i, x1i] - iimg[y1i, x0i] + iimg[y0i, x0i]

best = None
lo = max(44, C0 - SCAN)
hi = min(N - 44, C0 + SCAN + 1)
for cy in range(lo, hi):
    for cx in range(lo, hi):
        if (cx - C0) ** 2 + (cy - C0) ** 2 < MIN_SEP ** 2:
            continue
        v = sq_sum(cx, cy)
        if best is None or v > best[0]:
            best = (v, cx, cy)
_, BCX, BCY = best
frac_a = disc_frac(C0, C0)
frac_b = disc_frac(BCX, BCY)
print("anchor disc:  %.2f%% dark forest   (gate %.0f%%) -> %s"
      % (100 * frac_a, 100 * GATE, "PASS" if frac_a < GATE else "FAIL"))
print("darkest disc: %.2f%% dark forest  [center %+d, %+d rel. anchor]"
      % (100 * frac_b, (BCX - C0) * PITCH, (BCY - C0) * PITCH))

# --- panels -------------------------------------------------------------------
PANEL_PX = 320
SPAN_CELLS = 108                   # ~3,456 blocks across
DISC_PX = DF_R * PANEL_PX / (SPAN_CELLS * PITCH)
LX, RX, PY = 40, 430, 96
W, H = 810, 640

def panel_img(cx, cy):
    crop = bio[cy - SPAN_CELLS // 2: cy + SPAN_CELLS // 2,
               cx - SPAN_CELLS // 2: cx + SPAN_CELLS // 2]
    arr = np.zeros(crop.shape + (3,), np.uint8)
    for bid, col in BIO_COLORS.items():
        arr[crop == bid] = col
    arr[~np.isin(crop, list(BIO_COLORS))] = (70, 75, 80)
    arr[crop == DF_ID] = DARK_TINT
    return (Image.fromarray((arr * 0.70).astype(np.uint8))  # dimmed: dots must pop
            .resize((PANEL_PX, PANEL_PX), Image.NEAREST))

imgL = panel_img(C0, C0)
imgR = panel_img(BCX, BCY)

def compose(dots, hdr1, hdr2, cap_l, cap_r, chips=None):
    """dots: list of (ox_px, oy_px, is_dark_left, is_dark_right);
       caps: (line1, color, line2)."""
    fr = Image.new("RGB", (W, H), BG)
    fr.paste(imgL, (LX, PY))
    fr.paste(imgR, (RX, PY))
    dr = ImageDraw.Draw(fr)
    for px0 in (LX, RX):
        dr.rectangle([px0, PY, px0 + PANEL_PX - 1, PY + PANEL_PX - 1], outline=DIM)
        c = px0 + PANEL_PX // 2
        cy0 = PY + PANEL_PX // 2
        dr.ellipse([c - DISC_PX, cy0 - DISC_PX, c + DISC_PX, cy0 + DISC_PX],
                   outline=DIM)
    for (ox, oy, d_l, d_r) in dots:
        for px0, dk in ((LX, d_l), (RX, d_r)):
            px = px0 + PANEL_PX // 2 + ox
            py = PY + PANEL_PX // 2 + oy
            dr.ellipse([px - 2, py - 2, px + 2, py + 2],
                       fill=MISS if dk else blend(BG, HIT, 0.9))
    y = wrap_text(dr, (14, 10), hdr1, max_w=W - 28, size=16)
    wrap_text(dr, (14, max(36, y + 2)), hdr2, max_w=W - 28, size=12, fill=DIM)
    dr.text((LX, PY + PANEL_PX + 12), cap_l[0], font=font(14), fill=cap_l[1])
    dr.text((LX, PY + PANEL_PX + 34), cap_l[2], font=font(11), fill=DIM)
    dr.text((RX, PY + PANEL_PX + 12), cap_r[0], font=font(14), fill=cap_r[1])
    dr.text((RX, PY + PANEL_PX + 34), cap_r[2], font=font(11), fill=DIM)
    if chips:
        gate_chips(dr, 40, H - 40, chips)
    return fr

frames, dur = [], []
frames.append(compose([],
    "CPU Gate 1 of 3: The Dark Forest Veto",
    "The gate samples only the anchor's disc (left); the darkest nearby disc is shown for contrast only",
    ("The candidate's disc", TXT, "Radius 1,365 blocks around the anchor"),
    ("The darkest disc within ±3,600 blocks", TXT, "What the gate would eliminate")))
dur.append(3400)

# --- ring-by-ring sampling, both panels in lockstep ----------------------------
dots = []
cnt_l = cnt_r = shown = 0
for ring in range(1, 22):
    r = ring * 64
    k = max(2, int(round(2 * math.pi * r / 64)))
    for t in range(k):
        a = 2 * math.pi * t / k
        dx = int(round(r * math.cos(a)))
        dz = int(round(r * math.sin(a)))
        cdx, cdz = int(round(dx / PITCH)), int(round(dz / PITCH))
        d_l = bool(dark[C0 + cdz, C0 + cdx])
        d_r = bool(dark[BCY + cdz, BCX + cdx])
        shown += 1
        cnt_l += d_l
        cnt_r += d_r
        dots.append((int(dx * DISC_PX / DF_R), int(dz * DISC_PX / DF_R), d_l, d_r))
    fa = cnt_l / shown
    fb = cnt_r / shown
    frames.append(compose(dots,
        "Sampling both discs, ring by ring...",
        "Green dots = no dark forest; Red dots = dark forest found",
        ("Anchor disc: %.2f%% dark" % (100 * fa), HIT if fa < GATE else MISS,
         "The candidate's immediate surroundings"),
        ("Darkest disc: %.2f%% dark" % (100 * fb), MISS if fb >= GATE else HIT,
         "What the gate would eliminate"),
        chips=[1, 0, 0]))
    dur.append(200)

# --- verdicts -------------------------------------------------------------------
va = frac_a < GATE
vb = frac_b < GATE
cap_l = ("Anchor disc: %.2f%% dark: pass" % (100 * frac_a), HIT,
         "The candidate's immediate surroundings")
cap_r = ("Darkest disc: %.2f%% dark: %s"
         % (100 * frac_b, "still passes" if vb else "rejected"),
         HIT if vb else MISS,
         "What the gate would eliminate")
frames.append(compose(dots, "Verdicts", "", cap_l, cap_r, chips=[2, 0, 0]))
dur.append(3000)

msg = ("A few dark forest patches are fine; a thick blotch is not"
       if not vb else
       "This region stays clear of dark forest even at its worst")
frames.append(compose(dots, "A candidate anchored in the dark forest blotch would be dropped",
    msg, cap_l, cap_r, chips=[2, 0, 0]))
dur.append(3400)

frames.append(compose(dots, "Gate 1 Passed; the Erosion region re-measure is next",
                      "", cap_l, cap_r, chips=[2, 1, 0]))
dur.append(1600)

frames = [with_rail(f, 9, sub=0) for f in frames]
save_gif(frames, dur, "g07_darkforest.gif")
