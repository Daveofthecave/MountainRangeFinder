#!/usr/bin/env python3
# viz/make_g06_floodfill.py -- the flagship: KernelBlob / blob_fill, run on the
# magic seed's real blob field.
#
# Acts:
#   1. the GPU's +-8,000-block window, then a zoom into the blob's bbox
#   2. wavefront BFS (amber frontier, white-hot leading edge); the GPU
#      prefilter's early-exit flash
#   3. 1-cell morphological closing (rivers can't split the range)
#   4. chamfer distance transform -> inscribed core radius (shell peel)
#   5. verdict card vs the real gates
#
# The fill here is a line-for-line port of blob_fill() in probe.cpp (world-
# aligned 64-block lattice, 4-neighbor BFS, 4-neighbor closing, chamfer core),
# over cubiomes' real climate fields -- so its totals match the pipeline's
# recall row for this seed (7,441 cells = 30,478,336 blocks^2, core 22)
# exactly. If they ever stop matching, the port (or the field cache) broke.

import math
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

from fields import climate_grid
from style import (BG, PANEL, TXT, DIM, HIT, MISS, ACCENT, WATER, WHITE,
                   font, blend, save_gif, title, wrap_text, with_rail)

# --- recipe constants (common.h / probe.cpp / gpu.cu KernelBlob) ------------
SEED = 6696478651374553046
# Continuity: use the anchor g03 chose. The fill is anchor-independent --
# blob_fill snaps to the world-aligned lattice and floods the whole connected
# region, so any interior anchor lands on the same blob (and the same
# 30,478,336 / 22 recall self-check below).
import json as _json
try:
    with open("viz/out/demo_anchor.json") as fp:
        _a = _json.load(fp)
    AX, AZ = int(_a["x"]), int(_a["z"])
except Exception:
    AX, AZ = -16064, -15044        # recall anchor, blocks (anchorX / anchorZ)


STEP = 64                      # blocks per lattice cell
HALF = 8000 // STEP            # 125  -> the GPU KernelBlob window (+-8,000)
LAT  = 2 * HALF + 1            # 251 cells per side

ERO_MAX  = -0.35               # the BLOB_* blob field (common.h)
CONT_MIN = -0.12
TEMP_MIN, TEMP_MAX = -0.45, 0.20

TARGET_CELLS   = 3867          # GPU prefilter early-exit target: main.cpp pushes
                               # 0.88 * min_blob_area / 4096 (= 3867 at the 18M default)
MIN_BLOB_AREA  = 18_000_000    # CPU area gate, blocks^2 (common.h)
MIN_CORE_CELLS = 20            # CPU inscribed-core gate, cells (common.h)

W, H = 760, 836
HEAD = 70
VIEW = 720
VIEW_X = (W - VIEW) // 2

# --- the real field ----------------------------------------------------------
# blob_fill() snaps the anchor down to the world-aligned lattice first.
BX0 = AX & ~63
BZ0 = AZ & ~63
x0q = (BX0 - 8000) // 4
z0q = (BZ0 - 8000) // 4

print("sampling the real climate field (3 x 63k quart points; "
      "cached in viz/cache/ after the first run)...")
ero  = climate_grid(SEED, "erosion", 2, x0q, z0q, LAT, LAT, STEP // 4)  # 0B
cont = climate_grid(SEED, "cont",    4, x0q, z0q, LAT, LAT, STEP // 4)  # 1B
temp = climate_grid(SEED, "temp",    2, x0q, z0q, LAT, LAT, STEP // 4)  # 0B
mask = (ero <= ERO_MAX) & (cont >= CONT_MIN) & (temp >= TEMP_MIN) & (temp <= TEMP_MAX)

# --- seed cell: the snapped anchor, nudging out 2 lattice rings if it fails --
def find_seed():
    c = HALF
    if mask[c, c]:
        return c, c
    for r in (1, 2):                       # probe.cpp's exact ring search
        for dj in range(-r, r + 1):
            for di in range(-r, r + 1):
                if mask[c + dj, c + di]:
                    return c + di, c + dj
    return None

seed = find_seed()
if seed is None:
    raise SystemExit("no passing cell near the anchor -- field cache stale?")

# --- wavefront BFS (visit order doubles as the animation's reveal order) -----
order = np.full((LAT, LAT), -1, np.int32)
cells = [seed]
order[seed[1], seed[0]] = 0
q = deque([seed])
while q:
    i, j = q.popleft()
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni, nj = i + di, j + dj
        if 0 <= ni < LAT and 0 <= nj < LAT and order[nj, ni] < 0 and mask[nj, ni]:
            order[nj, ni] = len(cells)
            cells.append((ni, nj))
            q.append((ni, nj))
N = len(cells)
print("blob fill: %d cells = %s blocks^2 before closing"
      % (N, format(N * 4096, ",")))

# --- 1-cell morphological closing (probe.cpp: dilate-any, then erode-all) ----
blob = np.zeros((LAT, LAT), bool)
for (i, j) in cells:
    blob[j, i] = True

dil = blob.copy()
dil[1:,  :] |= blob[:-1, :]                  # DILATE: any set 4-neighbor joins
dil[:-1, :] |= blob[1:, :]
dil[:, 1:] |= blob[:, :-1]
dil[:, :-1] |= blob[:, 1:]

closed = dil.copy()                          # ERODE: keep cells whose 4 dilated
closed[0, :] = closed[-1, :] = False         # neighbors are all present
closed[:, 0] = closed[:, -1] = False         # (out of lattice = unset)
closed[1:-1, 1:-1] &= (dil[:-2, 1:-1] & dil[2:, 1:-1]
                       & dil[1:-1, :-2] & dil[1:-1, 2:])
bridges = closed & ~blob
N_CLOSED = int(closed.sum())
AREA_CLOSED = N_CLOSED * 4096

# --- chamfer distance transform (two sweeps, as in probe.cpp) -----------------
SQ2 = math.sqrt(2.0)
dt = np.where(closed, 1e9, 0.0)
for j in range(LAT):                         # forward: W, N, NW, NE
    row = dt[j]
    for i in range(LAT):
        v = row[i]
        if v == 0.0:
            continue
        if i > 0:            v = min(v, row[i - 1] + 1.0)
        if j > 0:            v = min(v, dt[j - 1][i] + 1.0)
        if i > 0 and j > 0:  v = min(v, dt[j - 1][i - 1] + SQ2)
        if i < LAT - 1 and j > 0: v = min(v, dt[j - 1][i + 1] + SQ2)
        row[i] = v
for j in range(LAT - 1, -1, -1):             # backward: E, S, SE, SW
    row = dt[j]
    for i in range(LAT - 1, -1, -1):
        v = row[i]
        if v == 0.0:
            continue
        if i < LAT - 1:            v = min(v, row[i + 1] + 1.0)
        if j < LAT - 1:            v = min(v, dt[j + 1][i] + 1.0)
        if i < LAT - 1 and j < LAT - 1: v = min(v, dt[j + 1][i + 1] + SQ2)
        if i > 0 and j < LAT - 1:  v = min(v, dt[j + 1][i - 1] + SQ2)
        row[i] = v
CORE = int(round(float(dt.max())))
print("after closing: %d cells = %s blocks^2 | inscribed core: %d cells = %s blocks"
      % (N_CLOSED, format(AREA_CLOSED, ","), CORE, format(CORE * 64, ",")))
MATCH = (AREA_CLOSED == 30_478_336 and CORE == 22)
print("recall self-check vs the pipeline's recall row (30,478,336 / 22): %s"
      % ("MATCH -- bit-for-bit identical" if MATCH else "DIFFERS"))

# --- rendering ----------------------------------------------------------------
lo, hi = np.percentile(ero, [2, 98])
grey = np.clip((ero - lo) / (hi - lo), 0, 1)
base_arr = (grey[..., None] * np.array([70.0, 90.0, 120.0]) * 0.55).astype(np.uint8)

SETTLE = blend(BG, HIT, 0.45)
SHALLOW = blend(BG, HIT, 0.18)
vis = base_arr.copy()

# Zoom target: the blob's bbox + margin, squared up.
js, is_ = np.where(blob)
MARGIN = 6
cx = (is_.min() + is_.max() + 1) / 2.0
cy = (js.min() + js.max() + 1) / 2.0
side = max(is_.max() - is_.min() + 1, js.max() - js.min() + 1) + 2 * MARGIN
# Clamp the zoom target into the lattice: without this, a blob crowding one
# side of the +-8,000-block window makes the view slide off the field,
# producing a letterboxing effect.
side = min(side, float(LAT))
tx0 = min(max(cx - side / 2, 0.0), LAT - side)
ty0 = min(max(cy - side / 2, 0.0), LAT - side)
FULL = (0.0, 0.0, float(LAT), float(LAT))
TCROP = (tx0, ty0, tx0 + side, ty0 + side)

def render(crop):
    x0, y0, x1, y1 = crop
    # Derive the buffer from the slice, never from the float crop: the two
    # disagree by one cell whenever the float boundary straddles a cell.
    sx0, sy0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    sx1, sy1 = min(LAT, int(math.ceil(x1))), min(LAT, int(math.ceil(y1)))
    s = max(sx1 - sx0, sy1 - sy0)
    buf = np.zeros((s, s, 3), np.uint8)
    buf[:] = BG
    if sx1 > sx0 and sy1 > sy0:
        buf[sy0 - sy0:sy1 - sy0, sx0 - sx0:sx1 - sx0] = vis[sy0:sy1, sx0:sx1]
    return Image.fromarray(buf).resize((VIEW, VIEW), Image.NEAREST)

def inset_img(crop):
    small = Image.fromarray(vis).resize((120, 120), Image.NEAREST)
    d = ImageDraw.Draw(small)
    x0, y0, x1, y1 = crop
    sc = 120.0 / LAT
    d.rectangle([x0 * sc, y0 * sc, x1 * sc, y1 * sc], outline=WHITE)
    d.rectangle([0, 0, 119, 119], outline=DIM)
    return small

def compose(img, l1, l2, inset=None):
    fr = Image.new("RGB", (W, H), BG)
    fr.paste(img, (VIEW_X, HEAD))
    dr = ImageDraw.Draw(fr)
    dr.rectangle([VIEW_X, HEAD, VIEW_X + VIEW - 1, HEAD + VIEW - 1], outline=DIM)
    y = wrap_text(dr, (14, 10), l1, max_w=W - 28, size=16)
    wrap_text(dr, (14, max(38, y + 2)), l2, max_w=W - 28, size=12, fill=DIM)
    if inset is not None:
        fr.paste(inset, (W - 132, H - 128))
    return fr

def cell_screen(crop, i, j):
    x0, y0, x1, _ = crop
    s = x1 - x0
    return (VIEW_X + (i - x0) / s * VIEW, HEAD + (j - y0) / s * VIEW)

frames, dur = [], []

# Act 1a: the full GPU window, anchor marked.
vis[:] = base_arr
intro = compose(render(FULL),
    "The Flood Fill: Measuring One Contiguous Region",
    "The GPU fill sees a ±8,000-block window; the CPU re-measures out to ±48,000")
dr = ImageDraw.Draw(intro)
axp, ayp = cell_screen(FULL, HALF, HALF)
dr.ellipse([axp - 6, ayp - 6, axp + 6, ayp + 6], outline=WHITE, width=2)
dr.line([axp - 12, ayp, axp + 12, ayp], fill=WHITE, width=1)
dr.line([axp, ayp - 12, axp, ayp + 12], fill=WHITE, width=1)
dr.text((axp + 10, ayp - 18), "Anchor (%s, %s)" % (format(AX, ","), format(AZ, ",")),
        font=font(12), fill=WHITE)
frames.append(intro)
dur.append(2500)

# Act 1b: smooth zoom into the blob's bbox.
ZFRAMES = 12
for f in range(ZFRAMES):
    t = (f + 1) / ZFRAMES
    e = t * t * (3 - 2 * t)
    crop = tuple(FULL[k] + (TCROP[k] - FULL[k]) * e for k in range(4))
    frames.append(compose(render(crop),
        "Zooming to the connected region...",
        "Background = Erosion, octave 0A + 0B; each cell is 64 × 64 blocks"))
    dur.append(60)
frames[-1] = compose(render(TCROP),
    "Cells join the region if they pass the shared climate field", "")
dur[-1] = 2000

# Act 2: the fill. Vectorized: colors come straight from the visit-order array.
step = max(24, N // 44)
shown = 0
crossed = False
while shown < N:
    new = min(N, shown + step)
    vis[:] = base_arr
    m_seen = (order >= 0) & (order < new)
    vis[m_seen] = SETTLE
    vis[m_seen & (order >= new - 420)] = ACCENT        # amber wavefront band
    vis[m_seen & (order >= new - 48)] = WHITE          # white-hot leading edge
    if not crossed and shown < TARGET_CELLS <= new:
        crossed = True
        for _ in range(4):                             # ~1.6 s hold on the flash
            frames.append(compose(render(TCROP),
                "GPU early exit at %s cells; candidate emitted"
                % format(TARGET_CELLS, ","),
                "A float32 prefilter; the CPU re-fills in double precision out to ±48,000 blocks",
                inset_img(TCROP)))
            dur.append(400)
    if crossed:
        l1 = "Past the GPU target; the CPU re-measures out to ±48,000 blocks"
        l2 = "Connected cells: %s (%s blocks²)" % (format(new, ","), format(new * 4096, ","))
    else:
        l1 = "Breadth-first fill over the shared Erosion noisemap"
        l2 = "Connected cells: %s of %s needed for early exit" % (format(new, ","), format(TARGET_CELLS, ","))
    frames.append(compose(render(TCROP), l1, l2))
    dur.append(55)
    shown = new

frames.append(compose(render(TCROP),
    "Fill Complete: %s cells (%s blocks²)" % (format(N, ","), format(N * 4096, ",")),
    "Next: prevent one-cell river gorges from splitting the region", inset_img(TCROP)))
dur.append(2500)

# Act 3: closing.
vis[:] = base_arr
vis[blob] = SETTLE
vis[dil & ~blob] = blend(BG, DIM, 0.9)
frames.append(compose(render(TCROP),
    "Morphological Closing, pt. 1: Dilation",
    "%s gap cells touching the region join it" % format(int((dil & ~blob).sum()), ","),
    inset_img(TCROP)))
dur.append(2500)

vis[:] = base_arr
vis[closed] = SETTLE
vis[bridges] = (235, 120, 200)  # bridged cells: pink, so they don't read as water
frames.append(compose(render(TCROP),
    "Morphological Closing, pt. 2: Erosion",
    "The outer shell falls away; %s bridged cells (pink) remain, for %s blocks²"
    % (format(int(bridges.sum()), ","), format(AREA_CLOSED, ",")),
    inset_img(TCROP)))
dur.append(2500)

# Act 4: chamfer heatmap, then the shell peel.
t = np.clip(dt / max(CORE, 1), 0, 1)
heatcol = (np.array(BG, float) * (1 - t[..., None])
           + np.array(ACCENT, float) * t[..., None]).astype(np.uint8)
vis[:] = base_arr
vis[closed] = heatcol[closed]
frames.append(compose(render(TCROP),
    "Chamfer Distance Transform: each cell's distance to the edge",
    "(brighter means deeper inland)", inset_img(TCROP)))
dur.append(2500)

for s in range(1, CORE + 1):
    vis[:] = base_arr
    vis[closed & (dt <= s)] = SHALLOW
    vis[closed & (dt > s)] = ACCENT
    frames.append(compose(render(TCROP),
        "Peeling inward, one shell at a time...",
        "Cells deeper than %s blocks remain" % format(s * 64, ","),
        inset_img(TCROP)))
    dur.append(140 if s < CORE else 600)

vis[:] = base_arr
vis[closed] = SHALLOW
vis[closed & (dt >= CORE)] = WHITE
frames.append(compose(render(TCROP),
    "Inscribed Core Radius: %s blocks"
    % format(CORE * 64, ","),
    "The gate requires at least %s blocks" % format(MIN_CORE_CELLS * 64, ","),
    inset_img(TCROP)))
dur.append(2500)

# Act 5: the verdict card.
card_img = (render(TCROP)).point(lambda p: int(p * 0.30))
card = compose(card_img, "", "")
dr = ImageDraw.Draw(card)
x0, y0, x1, y1 = 80, HEAD + VIEW // 2 - 118, W - 80, HEAD + VIEW // 2 + 118
dr.rectangle([x0, y0, x1, y1], fill=PANEL, outline=DIM, width=2)
lines = [
    ("CPU Measurement of the Low-Erosion Region", font(18), WHITE),
    ("", font(10), TXT),
    ("Area: %s blocks² (gate: %s) — pass"
        % (format(AREA_CLOSED, ","), format(MIN_BLOB_AREA, ",")), font(15), HIT),
    ("Core radius: %s blocks (gate: %s) — pass"
        % (format(CORE * 64, ","), format(MIN_CORE_CELLS * 64, ",")), font(15), HIT),
    ("", font(10), TXT),
    ("Matches the pipeline's own measurement exactly" if MATCH
        else "Differs from the pipeline's measurement; investigate", font(12),
        ACCENT if MATCH else MISS),
    ("Next clips: the CPU verifier's three checks", font(13), DIM),
]
ty = y0 + 16
for text, f, col in lines:
    tw = dr.textlength(text, font=f)
    dr.text(((W - tw) / 2, ty), text, font=f, fill=col)
    ty += f.size + 8
frames.append(card)
dur.append(3750)

frames = [with_rail(f, 8) for f in frames]

# Make everything ~25% slower globally so the text has time to be read
dur = [round(d * 1.25) for d in dur]

save_gif(frames, dur, "g06_floodfill.gif")
