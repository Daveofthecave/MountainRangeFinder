#!/usr/bin/env python3
# viz/make_g08_cpu_gates.py -- CPU gates 2-3 (cpu.cpp, in the code's real
# order): the blob re-measure (same flood fill as the GPU, but double
# precision and +-48,000-block reach) and the approx-height spiral.
#
# No text cards: the re-measure is shown as a zoom-out from the GPU's window,
# the gates' numbers are measured live (fields.blob_cells / blob_core_radius
# mirror probe.cpp), and the gate chips fill in the footer as we go.

import json, math
import numpy as np
from PIL import Image, ImageDraw

from fields import climate_grid, height_grid, blob_cells, blob_core_radius
from style import (BG, TXT, DIM, HIT, MISS, ACCENT, WHITE,
                   font, blend, save_gif, title, wrap_text, with_rail, gate_chips)

SEED = 6696478651374553046
try:
    with open("viz/out/demo_anchor.json") as fp:
        _a = json.load(fp)
    AX, AZ = int(_a["x"]), int(_a["z"])
except Exception:
    AX, AZ = -16064, -15044

HT_R, HT_STEP, HT_MIN, HT_MAX = 1600, 64, 230, 320   # cpu.cpp height window
MIN_AREA, MIN_CORE_CELLS = 18_000_000, 20            # common.h gates

print("measuring the blob (the same fill the verifier re-runs)...")
closed, (BX0, BZ0) = blob_cells(SEED, AX, AZ)
AREA = int(closed.sum()) * 4096
CORE_C = blob_core_radius(closed)
print("blob: %s blocks^2, core %d cells = %s blocks"
      % (format(AREA, ","), CORE_C, format(CORE_C * 64, ",")))

W, H = 760, 836
HEAD, FOOT = 62, 56
MAP = 700
MX, MY = 20, HEAD

# --- Act 1 backdrops: erosion field + blob mask at the GPU's +-8,000 view,
# --- then the same view zoomed out to +-16,000 (the search square) -----------
def field_img(ext, pitch):
    n = 2 * ext // pitch + 1
    ero = climate_grid(SEED, "erosion", 2, (BX0 - ext) // 4, (BZ0 - ext) // 4,
                       n, n, pitch // 4)
    lo, hi = np.percentile(ero, [2, 98])
    grey = np.clip((ero - lo) / (hi - lo), 0, 1)
    arr = (grey[..., None] * np.array([70.0, 90.0, 120.0]) * 0.45).astype(np.uint8)
    # blob cells tinted green; lattice coords -> this view's grid
    tint = np.array(HIT, float) * 0.35
    for j in range(closed.shape[0]):
        for i in range(closed.shape[1]):
            if not closed[j, i]:
                continue
            bx = BX0 + (i - 125) * 64
            bz = BZ0 + (j - 125) * 64
            gi = int(round((bx - (BX0 - ext)) / pitch))
            gj = int(round((bz - (BZ0 - ext)) / pitch))
            if 0 <= gi < n and 0 <= gj < n:
                arr[gj, gi] = (arr[gj, gi] * 0.65 + tint).astype(np.uint8)
    return Image.fromarray(arr).resize((MAP, MAP), Image.NEAREST)

img_near = field_img(8000, 64)
img_far = field_img(16000, 128)

def px(ext, bx, bz):
    return (MX + (bx - (BX0 - ext)) / (2 * ext) * MAP,
            MY + (bz - (BZ0 - ext)) / (2 * ext) * MAP)

def compose(img, l1, l2, chips, foot="", gpu_window_ext=None, view_ext=8000):
    fr = Image.new("RGB", (W, H), BG)
    fr.paste(img, (MX, MY))
    dr = ImageDraw.Draw(fr)
    dr.rectangle([MX, MY, MX + MAP - 1, MY + MAP - 1], outline=DIM)
    if gpu_window_ext is not None:
        x0, y0 = px(view_ext, BX0 - gpu_window_ext, BZ0 - gpu_window_ext)
        x1, y1 = px(view_ext, BX0 + gpu_window_ext, BZ0 + gpu_window_ext)
        dr.rectangle([x0, y0, x1, y1], outline=ACCENT, width=2)
        if gpu_window_ext < view_ext:
            dr.text((x0 + 4, y0 + 4), "the GPU's +-8,000 window",
                    font=font(11), fill=ACCENT)
    y = wrap_text(dr, (14, 10), l1, max_w=W - 28, size=16)
    wrap_text(dr, (14, max(36, y + 2)), l2, max_w=W - 28, size=12, fill=DIM)
    if foot:
        dr.text((MX + 4, MY + MAP + 8), foot, font=font(12), fill=TXT)
    gate_chips(dr, MX, H - 34, chips)
    return fr

frames, dur = [], []

frames.append(compose(img_near,
    "CPU Gate 2 of 3: Re-Measuring the Erosion Region",
    "This time with double precision; the CPU's reach is ±48,000 blocks",
    [2, 1, 0], gpu_window_ext=8000, view_ext=8000))
dur.append(2000)

ZF = 12
for f in range(ZF):
    t = (f + 1) / ZF
    e = t * t * (3 - 2 * t)
    frames.append(compose(Image.blend(img_near, img_far, e),
        "The GPU's window inside the low Erosion search area",
        "Whatever was clipped at the GPU window's edge is re-measured here",
        [2, 1, 0], gpu_window_ext=8000, view_ext=int(8000 + 8000 * e)))
    dur.append(65)

verdict_a = AREA >= MIN_AREA
verdict_c = CORE_C >= MIN_CORE_CELLS
frames.append(compose(img_far,
    "Area: %s blocks² (gate: %s): %s"
        % (format(AREA, ","), format(MIN_AREA, ","),
           "pass" if verdict_a else "fail"),
    "Core radius: %s blocks (gate: %s): %s"
        % (format(CORE_C * 64, ","), format(MIN_CORE_CELLS * 64, ","),
           "pass" if verdict_c else "fail"),
    [2, 2, 0], gpu_window_ext=8000, view_ext=16000,
    foot="(This low-Erosion region never touched the window edge)"))
dur.append(2800)

# --- Act 2: the height spiral (dimmed biome backdrop so the samples pop) ------
print("Sampling biome backdrop + height gate data (cached after first run)...")
EXT_B = HT_R + 320
BP = 24
BN = 2 * EXT_B // BP + 1
BIO_COLORS = {
    0: (40, 75, 130), 10: (60, 90, 140), 24: (30, 60, 110),
    44: (45, 90, 150), 45: (45, 90, 150), 46: (45, 90, 150),
    47: (35, 75, 130), 48: (35, 75, 130), 49: (35, 75, 130), 50: (35, 75, 130),
    7: (70, 120, 190), 11: (120, 150, 200),
    1: (140, 165, 85), 129: (170, 190, 90), 177: (150, 170, 95),
    185: (230, 160, 190), 4: (55, 110, 50), 132: (90, 140, 70),
    27: (85, 130, 65), 155: (110, 150, 80), 29: (25, 55, 28),
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
from fields import biome_grid
bio_bg = biome_grid(SEED, AX - EXT_B, AZ - EXT_B, BN, BN, BP)
bg_arr = np.zeros((BN, BN, 3), np.uint8)
for bid, col in BIO_COLORS.items():
    bg_arr[bio_bg == bid] = col
bg_arr[~np.isin(bio_bg, list(BIO_COLORS))] = (70, 75, 80)
bg_arr = (bg_arr * 0.55).astype(np.uint8)   # dimmed: the samples must pop
img_h = Image.fromarray(bg_arr).resize((MAP, MAP), Image.NEAREST)

def pxh(bx, bz):
    return (MX + (bx - (AX - EXT_B)) / (2 * EXT_B) * MAP,
            MY + (bz - (AZ - EXT_B)) / (2 * EXT_B) * MAP)

def hcolor(yv):
    if yv <= 62:
        return (70, 110, 170)
    t = min(1.0, max(0.0, (yv - 62) / 190.0))
    return tuple(int(round(a + (b - a) * t))
                 for a, b in zip((140, 160, 110), (250, 252, 255)))

HN = 2 * HT_R // HT_STEP + 1
hg = height_grid(SEED, AX - HT_R, AZ - HT_R, HN, HN, HT_STEP)

ring_order = [(0, 0)]
for r in range(1, HT_R // HT_STEP + 1):
    ring_order += ([(t, -r) for t in range(-r, r + 1)]
                   + [(t, r) for t in range(-r, r + 1)]
                   + [(-r, t) for t in range(-r + 1, r)]
                   + [(r, t) for t in range(-r + 1, r)])
# cpu.cpp clips the spiral to a disc (dx*dx + dz*dz > R^2 is skipped).
ring_order = [(di, dj) for di, dj in ring_order
              if di * di + dj * dj <= (HT_R // HT_STEP) ** 2]

hmax_seen = -1e9
hit_frame = None
hit_cell = None
per_frame = max(1, len(ring_order) // 46)
for k0 in range(0, len(ring_order), per_frame):
    fr = Image.new("RGB", (W, H), BG)
    fr.paste(img_h, (MX, MY))
    dr = ImageDraw.Draw(fr)
    dr.rectangle([MX, MY, MX + MAP - 1, MY + MAP - 1], outline=DIM)
    for (di, dj) in ring_order[:k0 + per_frame]:
        yv = hg[HT_R // HT_STEP + dj, HT_R // HT_STEP + di]
        if yv > hmax_seen:
            hmax_seen = yv
            hit_cell = (di, dj)          # current argmax
        xp, yp = pxh(AX + di * HT_STEP, AZ + dj * HT_STEP)
        dr.rectangle([xp - 4, yp - 4, xp + 4, yp + 4], fill=hcolor(yv))
    if hit_frame is None and hmax_seen >= HT_MIN:
        hit_frame = len(frames)
    title(dr, (14, 10), "CPU Gate 3 of 3: Approximate Surface Height",
          max_w=W - 28)
    title(dr, (14, 38), "Highest sample so far: %.0f (need one in [%d, %d])"
          % (hmax_seen, HT_MIN, HT_MAX), max_w=W - 28, size=12,
          fill=HIT if hmax_seen >= HT_MIN else DIM)
    if hit_cell is not None:
        hxp, hyp = pxh(AX + hit_cell[0] * HT_STEP, AZ + hit_cell[1] * HT_STEP)
        # Red: the dot ramp runs green->white, so red is the one hue no
        # sample can collide with, and it pops on both light and dark cells.
        dr.ellipse([hxp - 9, hyp - 9, hxp + 9, hyp + 9], outline=MISS, width=3)
        lbl = "y = %.0f" % hmax_seen
        lw = dr.textlength(lbl, font=font(13))
        lx = hxp + 12 if hxp + 12 + lw <= W - 6 else hxp - 14 - lw
        ly = min(max(hyp - 9, MY + 2), MY + MAP - 18)
        dr.text((lx + 1, ly + 1), lbl, font=font(13), fill=BG)   # dark shadow for contrast
        dr.text((lx, ly), lbl, font=font(13), fill=MISS)


    gate_chips(dr, MX, H - 34, [2, 2, 1])
    frames.append(fr)
    dur.append(60)

print("height gate: max %.0f (need %d..%d) -> %s"
      % (hmax_seen, HT_MIN, HT_MAX, "PASS" if HT_MIN <= hmax_seen <= HT_MAX else "?"))

verdict_h = HT_MIN <= hmax_seen <= HT_MAX
# Build the verdict frame on top of the last sweep frame, so the measured
# height dots linger instead of vanishing.
fr = frames[-1].copy()
dr = ImageDraw.Draw(fr)
dr.rectangle([0, 0, W, HEAD - 4], fill=BG)          # clear the title strip
title(dr, (14, 10), "All CPU Gates Passed", max_w=W - 28)
title(dr, (14, 38), "Next clip: data enrichment, pattern windows, and scoring",
      max_w=W - 28, size=12, fill=DIM)
dr.rectangle([0, H - 44, W, H], fill=BG)            # clear the chip strip
gate_chips(dr, MX, H - 34, [2, 2, 2 if verdict_h else 3])
frames.append(fr)
dur.append(3000)

frames = [with_rail(f, 9, sub=(1 if i <= ZF + 1 else 2)) for i, f in enumerate(frames)]
save_gif(frames, dur, "g08_cpu_gates.gif")
