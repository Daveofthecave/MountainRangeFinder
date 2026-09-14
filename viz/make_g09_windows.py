#!/usr/bin/env python3
# viz/make_g09_windows.py -- the coherent best-pattern windows (probe.cpp
# best_pattern_window / window_pattern_score) sweeping the magic blob.
# Uses real full-octave erosion, real approx heights, real biome census.
#
# Fidelity: window stats are taken over in-blob cells only (matching
# probe.cpp's prefix-sum counts) -- including the ocean fraction, which the
# previous version divided by the whole window.

import numpy as np
from PIL import Image, ImageDraw

from fields import climate_grid, height_grid, biome_grid
from style import (BG, TXT, DIM, HIT, MISS, ACCENT, WATER, WHITE,
                   font, blend, save_gif, title, with_rail)

SEED = 6696478651374553046
CX, CZ = -14500, -15000
EXT = 6400
PITCH = 128                       # the 128-block sublattice pitch of blob_enrich
N = 2 * EXT // PITCH + 1          # 101

# window_pattern_score reference constants (probe.cpp; keep in sync)
WREF_DH, WREF_HSTD, WREF_ERO = 36.0, 40.0, -0.90
WREF_HI, WREF_LO, OCN_SKIP = 0.10, 0.02, 0.15

OCEANS = {0, 10, 24, 44, 45, 46, 47, 48, 49, 50}
MTN = {178, 179, 180, 181, 182}   # grove..stony_peaks

def wscore(ero_m, h_sd, dh_m, mtn_f, hi_f, lo_f, ocn_f, cover):
    mtn_cap = min(mtn_f, 0.62)
    hi_z = max(-1.5, min(2.0, (hi_f - WREF_HI) / 0.10))
    lo_z = max(-1.5, min(2.0, (lo_f - WREF_LO) / 0.03))
    edge_pen = 1.2 * (0.85 - cover) / 0.15 if cover < 0.85 else 0.0
    return (1.00 * (dh_m - WREF_DH) / 6.0
            + 0.70 * (h_sd - WREF_HSTD) / 12.0
            + 0.40 * hi_z + 0.35 * lo_z
            + 0.35 * (WREF_ERO - ero_m) / 0.15
            + 0.30 * (mtn_cap - 0.35) / 0.15
            - 3.00 * ocn_f - edge_pen)

print("sampling fields (heights + biomes + 5 climate grids; cached after first run)...")
x0, z0 = CX - EXT, CZ - EXT
h     = height_grid(SEED, x0, z0, N, N, PITCH)
bio   = biome_grid(SEED, x0, z0, N, N, PITCH)
eroF  = climate_grid(SEED, "erosion", -1, x0 // 4, z0 // 4, N, N, PITCH // 4)
eroB  = climate_grid(SEED, "erosion", 2, x0 // 4, z0 // 4, N, N, PITCH // 4)
contB = climate_grid(SEED, "cont",    4, x0 // 4, z0 // 4, N, N, PITCH // 4)
tempB = climate_grid(SEED, "temp",    2, x0 // 4, z0 // 4, N, N, PITCH // 4)
blobm = (eroB <= -0.35) & (contB >= -0.12) & (tempB >= -0.45) & (tempB <= 0.20)

ocn = np.isin(bio, list(OCEANS))
mtn = np.isin(bio, list(MTN))

def window_field(R):
    """Pattern score (and ingredients) at every window center; NaN = skipped."""
    keys = ("sc", "ero", "hstd", "dh", "mtn", "hi", "lo", "ocn")
    F = {k: np.full((N, N), np.nan) for k in keys}
    for cy in range(R, N - R):
        for cx in range(R, N - R):
            sl = (slice(cy - R, cy + R + 1), slice(cx - R, cx + R + 1))
            m = blobm[sl]
            cnt = m.sum()
            cover = cnt / ((2 * R + 1) ** 2)
            if cover < 0.5:
                continue
            ocn_f = (ocn[sl] & m).sum() / cnt   # in-blob cells, like probe.cpp
            if ocn_f > OCN_SKIP:
                continue
            hw = np.where(m, h[sl], np.nan)     # stats run over blob cells only,
            ew = np.where(m, eroF[sl], np.nan)  # mirroring probe.cpp's prefix sums
            ero_m = np.nanmean(ew)
            h_sd = float(np.nanstd(hw))
            dhs = np.concatenate([np.abs(np.diff(hw, axis=0)).ravel(),
                                  np.abs(np.diff(hw, axis=1)).ravel()])
            dh_m = np.nanmean(dhs)
            if np.isnan(dh_m):
                continue
            mtn_f = (mtn[sl] & m).sum() / cnt
            hi_f = (hw >= 200).sum() / cnt          # NaN compares False: excluded
            lo_f = ((hw <= 63) & ~ocn[sl]).sum() / cnt
            sc = wscore(ero_m, h_sd, dh_m, mtn_f, hi_f, lo_f, ocn_f, cover)
            F["sc"][cy, cx] = sc
            F["ero"][cy, cx] = ero_m; F["hstd"][cy, cx] = h_sd
            F["dh"][cy, cx] = dh_m;   F["mtn"][cy, cx] = mtn_f
            F["hi"][cy, cx] = hi_f;   F["lo"][cy, cx] = lo_f
            F["ocn"][cy, cx] = ocn_f
    return F

print("scoring windows (896 / 1664 / 3200 blocks)...")
SC = {3: window_field(3), 6: window_field(6), 12: window_field(12)}
best = {}
for R, F in SC.items():
    sc = F["sc"]
    if np.any(~np.isnan(sc)):
        j, i = np.unravel_index(np.nanargmax(sc), sc.shape)
        best[R] = (i, j, sc[j, i])
        print("  %4d-block window: best score %.2f at (%d, %d)"
              % (R * 256 + 128, sc[j, i], x0 + i * PITCH, z0 + j * PITCH))
if 6 in best:
    print("recall headline reference: (-12992, -15040)")

# --- backdrop: terrain, translucent 0B-erosion tint, blob outline -------------
MAP = 640
SIDE = 260
W, H = MAP + SIDE + 40, 800

def height_ramp(hh):
    t = np.clip((hh - 62.0) / 190.0, 0, 1)
    out = np.zeros(hh.shape + (3,), np.uint8)
    land = (np.array([110.0, 130.0, 90.0]) * (1 - t[..., None])
            + np.array([240.0, 242.0, 248.0]) * t[..., None])
    out[:] = land.astype(np.uint8)
    out[ocn] = (38, 70, 120)
    return (out * 0.9).astype(np.uint8)

bm = blobm[1:-1, 1:-1] & ~(blobm[:-2, 1:-1] & blobm[2:, 1:-1]
                           & blobm[1:-1, :-2] & blobm[1:-1, 2:])
edge_img = np.zeros_like(blobm)
edge_img[1:-1, 1:-1] = bm

terr = height_ramp(h).astype(float)
gx = np.zeros_like(h); gz = np.zeros_like(h)
gx[1:-1, 1:-1] = h[1:-1, 2:] - h[1:-1, :-2]
gz[1:-1, 1:-1] = h[2:, 1:-1] - h[:-2, 1:-1]
terr = np.clip(terr + np.clip(gx + gz, -40, 40)[..., None] * 0.9, 0, 255)
base_arr = terr.astype(np.uint8)
base_arr[edge_img] = WHITE
base = Image.fromarray(base_arr).resize((MAP, MAP), Image.NEAREST)

def cell_px(i, j):
    return (10 + (i * MAP) // N, 70 + (j * MAP) // N)

def window_rect(dr, i, j, R, col, width=2):
    x0p, y0p = cell_px(i - R, j - R)
    x1p, y1p = cell_px(i + R + 1, j + R + 1)
    # clamp to the map so edge windows don't bleed into the border or panel
    x0p = max(11, x0p); y0p = max(71, y0p)
    x1p = min(10 + MAP - 2, x1p); y1p = min(70 + MAP - 2, y1p)
    dr.rectangle([x0p, y0p, x1p, y1p], outline=col, width=width)

# --- sweep of the 1664-block window -------------------------------------------
# Strictly left-to-right, carriage return at each row end: the same row-major
# order as probe.cpp's best_pattern_window (no serpentine).
R = 6
path = []
for j in range(R, N - R):
    for i in range(R, N - R):
        path.append((i, j))
TOT = len(path)

def stat_line(dr, x, y, label, val, fmt="%.2f"):
    dr.text((x, y), label, font=font(11), fill=DIM)
    s = "--" if val is None or (isinstance(val, float) and np.isnan(val)) else fmt % val
    dr.text((x + 128, y), s, font=font(11), fill=TXT)

hist = []          # (fraction_of_scan, score)
best_so_far = None
last_good = {}     # persistent ingredient values (no flicker on skipped windows)
frames, dur = [], []
k = 0
last_row = N - 2 * R - 1                        # 0-based index of the final sweep row
while k < TOT:
    row_i = path[k][1] - R                      # 0-based sweep row
    # Ease in/out: slow for the first/last couple of rows, full speed between.
    # (Illustrative pacing -- the sweep scans every window regardless.)
    d_edge = min(row_i, last_row - row_i)
    # Halve the step sizes to render more intermediate positions per row,
    # making the fast-forward sweep feel smoother without changing its speed.
    step = (4, 6, 10, 19)[min(d_edge, 3)]
    k = min(TOT, k + step)
    i, j = path[k - 1]
    sc = SC[R]["sc"][j, i]
    if not np.isnan(sc):
        hist.append(((k - 1) / (TOT - 1), sc))
        if best_so_far is None or sc > best_so_far[2]:
            best_so_far = (i, j, sc)
    fr = Image.new("RGB", (W, H), BG)
    fr.paste(base, (10, 70))
    dr = ImageDraw.Draw(fr)
    dr.rectangle([10, 70, 10 + MAP - 1, 70 + MAP - 1], outline=DIM)
    if best_so_far is not None:
        window_rect(dr, best_so_far[0], best_so_far[1], R, HIT, 2)
    window_rect(dr, i, j, R, WATER if not np.isnan(sc) else MISS, 2)
    title(dr, (14, 10), "Scanning for a special mountain pattern using a sliding 1,664-block window",
          max_w=W - 28)
    dr.text((14, 38), "The best window's center becomes the seed's reported coordinates",
            font=font(12), fill=DIM)
    px0 = MAP + 30
    # live ingredient list for the window under the cursor: this is the
    # pattern. Values persist through skipped (red) windows instead of
    # flickering to "--", so you can watch them evolve.
    dr.text((px0, 76), "Window Ingredients:", font=font(12), fill=ACCENT)
    F = SC[R]
    cur = {
        "ero":  F["ero"][j, i],
        "hstd": F["hstd"][j, i],
        "dh":   F["dh"][j, i],
        "mtn":  100 * F["mtn"][j, i],
        "hi":   100 * F["hi"][j, i],
        "lo":   100 * F["lo"][j, i],
        "sc":   sc,
    }
    for key in cur:
        if not (isinstance(cur[key], float) and np.isnan(cur[key])):
            last_good[key] = cur[key]
    stat_line(dr, px0, 96,  "Mean erosion",   last_good.get("ero"))
    stat_line(dr, px0, 112, "Height spread",  last_good.get("hstd"), "%.1f blocks")
    stat_line(dr, px0, 128, "Steepness", last_good.get("dh"),  "%.1f blocks")
    stat_line(dr, px0, 144, "Mountain cover", last_good.get("mtn"), "%.0f%%")
    stat_line(dr, px0, 160, "High ground",    last_good.get("hi"),  "%.0f%%")
    stat_line(dr, px0, 176, "Valley floor",   last_good.get("lo"),  "%.0f%%")
    stat_line(dr, px0, 196, "Window score",   last_good.get("sc"),  "%+.2f")
    # score history; x = true scan progress
    dr.text((px0, 232), "score over the scan:", font=font(12), fill=TXT)
    sy0, sy1 = 256, 376
    dr.rectangle([px0, sy0, px0 + 200, sy1], outline=DIM)
    if hist:
        lo_s = min(v for _, v in hist)
        hi_s = max(v for _, v in hist)
        span = max(1e-9, hi_s - lo_s)
        pts = [(px0 + 2 + p * 196, sy1 - 6 - (v - lo_s) / span * (sy1 - sy0 - 14))
               for p, v in hist]
        if len(pts) > 1:
            dr.line(pts, fill=ACCENT, width=2)
        if best_so_far is not None:
            dr.text((px0, sy1 + 8), "Best window so far: %+.2f" % best_so_far[2],
                    font=font(12), fill=HIT)
    dr.text((px0, 400), "Green: best window so far", font=font(11), fill=HIT)
    dr.text((px0, 418), "Blue: being scored right now", font=font(11), fill=WATER)
    dr.text((px0, 436), "Red: skipped (off-region or coastal)", font=font(11), fill=MISS)
    # time-warp indicator under the viewport
    if d_edge >= 2:
        note, ncol = ">>  Fast-forwarding the middle rows", ACCENT
    else:
        note, ncol = "<<  Slowing down near the edges", DIM
    tw = dr.textlength(note, font=font(12))
    dr.text((10 + (MAP - tw) / 2, 70 + MAP + 8), note, font=font(12), fill=ncol)
    frames.append(fr)
    # Scale duration down proportionally so the sweep speed remains the same
    dur.append(max(15, round(36 * step / 16)))

# --- finale: the three best windows nested + headline crosshair ---------------
fr = Image.new("RGB", (W, H), BG)
fr.paste(base, (10, 70))
dr = ImageDraw.Draw(fr)
dr.rectangle([10, 70, 10 + MAP - 1, 70 + MAP - 1], outline=DIM)
for Rw, col, tag in ((3, WATER, "896"), (6, HIT, "1664"), (12, ACCENT, "3200")):
    if Rw in best:
        bi, bj, bs = best[Rw]
        window_rect(dr, bi, bj, Rw, col, 3)
        x0p, y0p = cell_px(bi - Rw, bj - Rw)
        lx = min(max(x0p + 4, 14), 10 + MAP - 80)
        ly = min(max(y0p + 4, 74), 70 + MAP - 20)
        dr.text((lx, ly), "%s: %.2f" % (tag, bs), font=font(12), fill=col)
if 6 in best:
    bi, bj, _ = best[6]
    cxp, cyp = cell_px(bi, bj)
    dr.line([cxp - 10, cyp, cxp + 10, cyp], fill=WHITE, width=2)
    dr.line([cxp, cyp - 10, cxp, cyp + 10], fill=WHITE, width=2)
    dr.text((cxp + 12, cyp - 8), "best", font=font(12), fill=WHITE)
title(dr, (14, 10), "The Best Mountain-Pattern Window at Each Scale",
      max_w=W - 28)
dr.text((14, 38), "The 1,664-block pattern window's center becomes the reported coordinates",
        font=font(12), fill=DIM)
px0 = MAP + 30
dr.text((px0, 76), "Winning Window's Ingredients:", font=font(12), fill=ACCENT)
if 6 in best:
    bi, bj, _ = best[6]
    F = SC[6]
    stat_line(dr, px0, 96,  "Mean erosion",   F["ero"][bj, bi])
    stat_line(dr, px0, 112, "Height spread",  F["hstd"][bj, bi], "%.1f blocks")
    stat_line(dr, px0, 128, "Steepness", F["dh"][bj, bi],  "%.1f blocks")
    stat_line(dr, px0, 144, "Mountain cover", 100 * F["mtn"][bj, bi], "%.0f%%")
    stat_line(dr, px0, 160, "High ground",    100 * F["hi"][bj, bi], "%.0f%%")
    stat_line(dr, px0, 176, "Valley floor",   100 * F["lo"][bj, bi], "%.0f%%")
    stat_line(dr, px0, 196, "Window score",   F["sc"][bj, bi], "%+.2f")
dr.text((px0, 232), "Score over the scan:", font=font(12), fill=TXT)
sy0, sy1 = 256, 376
dr.rectangle([px0, sy0, px0 + 200, sy1], outline=DIM)
if hist:
    lo_s = min(v for _, v in hist)
    hi_s = max(v for _, v in hist)
    span = max(1e-9, hi_s - lo_s)
    pts = [(px0 + 2 + p * 196, sy1 - 6 - (v - lo_s) / span * (sy1 - sy0 - 14))
           for p, v in hist]
    if len(pts) > 1:
        dr.line(pts, fill=ACCENT, width=2)
if best_so_far is not None:
    dr.text((px0, sy1 + 8), "Best window: %+.2f" % best_so_far[2],
            font=font(12), fill=HIT)
dr.text((px0, 446), "Window sizes:", font=font(12), fill=TXT)
for q, (tag2, col2) in enumerate((("896", WATER), ("1,664", HIT), ("3,200", ACCENT))):
    dr.text((px0, 468 + 20 * q), tag2 + "-block window", font=font(11), fill=col2)
dr.text((px0, 468 + 20 * 3), "White cross: reported coordinates", font=font(11), fill=WHITE)
frames.append(fr)
dur.append(7000)

frames = [with_rail(f, 10) for f in frames]

# Palette lock: quantize every frame against one shared palette (built from
# the finale frame, which contains every key color) and save with
# optimize=False. This stops the green "best so far" box from shimmering when
# the blue current-window box appears -- that was per-frame GIF palette
# quantization reallocating the green's slot. dither=NONE additionally
# prevents dither pattern shimmer frame-to-frame.
import os
from style import TEMPO
# Palette lock, round 2: the finale's big gold 3200-box skewed the palette so
# hard that the scan's red "skipped" box mapped to yellow (the finale has no
# red for median-cut to defend). Build the palette from a stack of the last
# sweep frame and the finale, so every color family -- red, green, blue,
# gold, the rail -- gets its own slot.
last_sweep = frames[-2]          # final sweep frame: red/blue/green boxes
finale = frames[-1]              # best-windows finale: the gold 3200 box
stack = Image.new("RGB", (finale.width, finale.height * 2))
stack.paste(last_sweep, (0, 0))
stack.paste(finale, (0, finale.height))
pal = stack.quantize(colors=256, method=Image.MAXCOVERAGE, dither=Image.NONE)
frames = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
os.makedirs("viz/out", exist_ok=True)
frames[0].save("viz/out/g09_windows.gif", save_all=True, append_images=frames[1:],
               duration=[max(20, int(round(d * TEMPO))) for d in dur],
               loop=0, optimize=False)
print("wrote %s (%d frames, %.0f KiB)"
      % ("viz/out/g09_windows.gif", len(frames),
         os.path.getsize("viz/out/g09_windows.gif") / 1024))
