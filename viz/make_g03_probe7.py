#!/usr/bin/env python3
# viz/make_g03_probe7.py -- the shared 7-point probe (KernelCoverage, gpu.cu).
#
# One continuous camera move: full +-16k map -> zoom into the demo anchor ->
# the 7-point stencil assembles as the zoom settles -> the sharing argument
# drawn in the same zoomed neighborhood (no hard scene breaks).
#
# Fidelity notes: the kernel loads all seven points at once and sums their
# hits -- there is no sequential order in the code. The stencil assembles here
# in the fields' declaration order (vC, vE, vUR, vUL, vW, vDR, vDL). The
# kernel's "up" is +row = +z = screen-down, so captions avoid compass names.

import json, os
import numpy as np
from PIL import Image, ImageDraw

from geom import (hex_anchors_blocks, hex_lattice_edges, PROBE_MIDS,
                  draw_probe_lattice, HEX_ROWS, HEX_COLS, phyllotaxis)
from fields import climate_grid
from style import (BG, TXT, DIM, HIT, MISS, ACCENT, WHITE,
                   font, blend, save_gif, title, with_rail)

SEED = 6696478651374553046
MAGIC = (-16064, -15044)      # recall anchor; the demo orbits the magic blob
PROBE_GATE, PROBE_NEED = -0.30, 6
FIELD_LABEL = "Field: Erosion, octaves 0A + 0B; a point passes at ≤ %.2f" % PROBE_GATE

ANCHORS_RC = hex_anchors_blocks()                 # (x, z, r, c)
ANCHORS = [(x, z) for x, z, r, c in ANCHORS_RC]
EDGES = hex_lattice_edges()

MAP = 640
HEAD = 74              # room for the two caption lines above the map
W, H = MAP, HEAD + MAP

def at(grid, x0, z0, pitch, x, z):
    i = min(max(int(round((x - x0) / pitch)), 0), grid.shape[1] - 1)
    j = min(max(int(round((z - z0) / pitch)), 0), grid.shape[0] - 1)
    return grid[j, i]

def render(grid):
    lo, hi = np.percentile(grid, [2, 98])
    grey = np.clip((grid - lo) / (hi - lo), 0, 1)
    arr = (grey[..., None] * np.array([70.0, 90.0, 120.0]) * 0.55).astype(np.uint8)
    return Image.fromarray(arr).resize((MAP, MAP), Image.NEAREST)

# --- full map: vote at every anchor (coarse field; the GPU is exact) ---------
R = 16000
PITCH = 320
N = 2 * R // PITCH + 1
ero = climate_grid(SEED, "erosion", 2, -R // 4, -R // 4, N, N, PITCH // 4)
ero_img = render(ero)

def mpx(x, z):
    return ((x + R) * (MAP / (2 * R)), HEAD + (z + R) * (MAP / (2 * R)))

votes = [sum(at(ero, -R, -R, PITCH, ax + dx, az + dz) <= PROBE_GATE
             for dx, dz in PROBE_MIDS) for ax, az in ANCHORS]
passing = [a for a, v in zip(ANCHORS, votes) if v >= PROBE_NEED]

def dist2(a):
    return (a[0] - MAGIC[0]) ** 2 + (a[1] - MAGIC[1]) ** 2

# Prefer an interior anchor (rim anchors are missing lattice neighbors, which
# used to leave a phantom spoke / a gap in the home hexagon), and require it
# to also survive the two KernelExtrema disc checks: the downstream clips
# (g04/g05) follow this anchor through the whole gauntlet, so it needs to be
# a real survivor, not just a probe passer. (Coarse-grid rehearsal, hence the
# safety margin over the real 77/64 thresholds -- nudge them up a point or
# two if g05's verdict card ever shows a FAIL.)
interior = {(x, z) for (x, z, r, c) in ANCHORS_RC
            if 1 <= r <= HEX_ROWS - 2 and 1 <= c <= HEX_COLS - 2}
passing_i = [a for a in passing if a in interior]
near = [a for a in (passing_i or passing) if dist2(a) <= 6000 ** 2]

PHYL = phyllotaxis() * 1600   # the 96-sample disc pattern (blocks)
cont = climate_grid(SEED, "cont", 4, -R // 4, -R // 4, N, N, PITCH // 4)

def gauntlet_ok(ax, az):
    ero_hits  = sum(at(ero,  -R, -R, PITCH, ax + dx, az + dz) <= -0.40
                    for dx, dz in PHYL)
    cont_hits = sum(at(cont, -R, -R, PITCH, ax + dx, az + dz) >= 0.0
                    for dx, dz in PHYL)
    return ero_hits >= 79 and cont_hits >= 67

strong = [a for a in near if gauntlet_ok(*a)]
pool = strong or near or passing_i or passing
CAX, CAZ = min(pool, key=dist2) if pool else min(ANCHORS, key=dist2)

# Hand the chosen anchor to the follow-up clips (g04/g05 read this file), so
# the disc scan starts exactly where this clip's zoom ends.
os.makedirs("viz/out", exist_ok=True)
with open("viz/out/demo_anchor.json", "w") as fp:
    json.dump({"seed": SEED, "x": int(CAX), "z": int(CAZ)}, fp)

# --- zoomed field -------------------------------------------------------------
ZR = 3200
ZP = 64
ZN = 2 * ZR // ZP + 1
zf = climate_grid(SEED, "erosion", 2, (CAX - ZR) // 4, (CAZ - ZR) // 4,
                  ZN, ZN, ZP // 4)
zf_img = render(zf)

# Orientation locator: a mini full-map with the current view rectangle.
locator = ero_img.resize((90, 90), Image.NEAREST)

def paste_locator(fr, half_blocks=ZR):
    m = locator.copy()
    d = ImageDraw.Draw(m)
    s = 90.0 / (2 * R)
    d.rectangle([(CAX - half_blocks + R) * s, (CAZ - half_blocks + R) * s,
                 (CAX + half_blocks + R) * s, (CAZ + half_blocks + R) * s],
                outline=WHITE)
    d.rectangle([0, 0, 89, 89], outline=DIM)
    fr.paste(m, (W - 100, HEAD + 10))

# zoomed-frame screen position (includes the head paste offset)
def zsx(x):
    return (x - (CAX - ZR)) / (2 * ZR) * MAP
def zsy(z):
    return HEAD + (z - (CAZ - ZR)) / (2 * ZR) * MAP
def zpx(x, z):
    return (zsx(x), zsy(z))

def lattice(dr, emphasize_home=False):
    draw_probe_lattice(dr, zpx, (0, W), (HEAD, H))
    if emphasize_home:
        # the six spokes around the demo anchor, a touch brighter: the probe
        # dots sit exactly on these lines (they are edge midpoints)
        for (dx, dz) in ((1920, 0), (960, 1663), (-960, 1663),
                         (-1920, 0), (-960, -1663), (960, -1663)):
            dr.line([zpx(CAX, CAZ), zpx(CAX + dx, CAZ + dz)],
                    fill=blend(BG, DIM, 0.85), width=2)

frames, dur = [], []

# Act 1: full map, lattice lines + all anchors' votes
fr = Image.new("RGB", (W, H), BG)
fr.paste(ero_img, (0, HEAD))
dr = ImageDraw.Draw(fr)
for (x1, z1, x2, z2) in EDGES:
    dr.line([mpx(x1, z1), mpx(x2, z2)], fill=blend(BG, DIM, 0.35))
dr.rectangle([0, HEAD, W - 1, H - 1], outline=DIM)
for (ax, az), v in zip(ANCHORS, votes):
    col = (blend(BG, HIT, 0.85) if v >= PROBE_NEED else
           blend(BG, DIM, 0.9) if v >= 3 else blend(BG, MISS, 0.55))
    cx, cy = mpx(ax, az)
    dr.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=col)
cx, cy = mpx(CAX, CAZ)
dr.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], outline=WHITE, width=2)
title(dr, (12, 8), "All 399 Anchors Probed for Low Erosion",
      max_w=W - 24)
dr.text((12, 36), FIELD_LABEL, font=font(11), fill=DIM)
dr.text((12, 50), "Green anchors advance; the white ring marks the one this clip follows", font=font(11), fill=DIM)
frames.append(fr)
dur.append(2200)

# Act 2: the zoom itself (backdrop crossfades as the camera moves)
NZ = 20
for f in range(NZ):
    t = (f + 1) / NZ
    e = t * t * (3 - 2 * t)
    img = Image.blend(ero_img, zf_img, e)
    fr = Image.new("RGB", (W, H), BG)
    fr.paste(img, (0, HEAD))
    dr = ImageDraw.Draw(fr)
    dr.rectangle([0, HEAD, W - 1, H - 1], outline=DIM)
    # the lattice edges morph along with the camera, so the grid never vanishes
    for (x1, z1, x2, z2) in EDGES:
        ax0, ay0 = mpx(x1, z1); ax1, ay1 = mpx(x2, z2)
        bx0, by0 = zpx(x1, z1); bx1, by1 = zpx(x2, z2)
        dr.line([ax0 + (bx0 - ax0) * e, ay0 + (by0 - ay0) * e,
                 ax1 + (bx1 - ax1) * e, ay1 + (by1 - ay1) * e],
                fill=blend(BG, DIM, 0.35))
    for (ax, az), v in zip(ANCHORS, votes):
        col0 = (blend(BG, HIT, 0.85) if v >= PROBE_NEED else
                blend(BG, DIM, 0.9) if v >= 3 else blend(BG, MISS, 0.55))
        col = blend(BG, col0, 1 - 0.45 * e)   # the crowd dims, never vanishes
        sx0, sy0 = mpx(ax, az)          # map-space start
        ex, ey = zsx(ax), zsy(az)       # zoomed-space end (may leave the canvas)
        pxx = sx0 + (ex - sx0) * e
        pyy = sy0 + (ey - sy0) * e
        if -8 <= pxx <= W + 8 and HEAD - 8 <= pyy <= H + 8:
            dr.ellipse([pxx - 3, pyy - 3, pxx + 3, pyy + 3], fill=col)
    sx0, sy0 = mpx(CAX, CAZ)
    pxx = sx0 + (zsx(CAX) - sx0) * e
    pyy = sy0 + (zsy(CAZ) - sy0) * e
    rr = 7 + 6 * e
    dr.ellipse([pxx - rr, pyy - rr, pxx + rr, pyy + rr], outline=WHITE, width=2)
    dr.text((12, 8), "Zooming to the selected anchor...", font=font(17), fill=TXT)
    dr.text((12, 36), FIELD_LABEL, font=font(11), fill=DIM)
    paste_locator(fr, R + (ZR - R) * e)
    frames.append(fr)
    dur.append(70)

# Act 3: the 7-point stencil assembles (kernel declaration order), each point
# landing exactly on its lattice edge
vals = [at(zf, CAX - ZR, CAZ - ZR, ZP, CAX + dx, CAZ + dz) for dx, dz in PROBE_MIDS]
hits = sum(v <= PROBE_GATE for v in vals)
verdict = hits >= PROBE_NEED
print("Selected Anchor: (%d, %d) -- %d/7 probe hits" % (CAX, CAZ, hits))

cx0, cy0 = zsx(CAX), zsy(CAZ)
for k in range(1, 8):
    fr = Image.new("RGB", (W, H), BG)
    fr.paste(zf_img, (0, HEAD))
    dr = ImageDraw.Draw(fr)
    dr.rectangle([0, HEAD, W - 1, H - 1], outline=DIM)
    lattice(dr, emphasize_home=True)
    for idx in range(k):
        dx, dz = PROBE_MIDS[idx]
        pxx, pyy = zsx(CAX + dx), zsy(CAZ + dz)
        ok = vals[idx] <= PROBE_GATE
        dr.ellipse([pxx - 8, pyy - 8, pxx + 8, pyy + 8],
                   fill=HIT if ok else MISS, outline=WHITE)
        lbl = "%.2f" % vals[idx]
        lw = dr.textlength(lbl, font=font(12))
        lx = pxx + 11 if pxx + 11 + lw <= W - 4 else pxx - 11 - lw
        ly = min(max(pyy - 9, HEAD + 2), H - 16)
        dr.text((lx, ly), lbl, font=font(12), fill=TXT)
    dr.text((12, 8), "The 7-Point Erosion Probe",
            font=font(17), fill=TXT)
    dr.text((12, 36), FIELD_LABEL, font=font(11), fill=DIM)
    dr.text((12, 50), "Point %d of 7" % k, font=font(12), fill=DIM)
    paste_locator(fr)
    frames.append(fr)
    dur.append(430)

# Act 4: verdict (lattice stays!) then the sharing argument -- same view
fr = Image.new("RGB", (W, H), BG)
fr.paste(zf_img, (0, HEAD))
dr = ImageDraw.Draw(fr)
dr.rectangle([0, HEAD, W - 1, H - 1], outline=DIM)
lattice(dr, emphasize_home=True)
for idx, (dx, dz) in enumerate(PROBE_MIDS):
    pxx, pyy = zsx(CAX + dx), zsy(CAZ + dz)
    ok = vals[idx] <= PROBE_GATE
    dr.ellipse([pxx - 8, pyy - 8, pxx + 8, pyy + 8],
               fill=HIT if ok else MISS, outline=WHITE)
    dr.text((pxx + 11, pyy - 9), "%.2f" % vals[idx], font=font(12), fill=TXT)
dr.text((12, 8), "Probe: %d of 7 have Low Erosion (Need %d): %s"
        % (hits, PROBE_NEED, "Pass" if verdict else "Fail"), font=font(17),
        fill=HIT if verdict else MISS)
dr.text((12, 36), FIELD_LABEL, font=font(11), fill=DIM)
dr.text((12, 50), "Surviving anchors are scanned with the 96-sample sunflower spiral (next clip)",
        font=font(12), fill=DIM)
paste_locator(fr)
frames.append(fr)
dur.append(2600)

fr = Image.new("RGB", (W, H), BG)
fr.paste(zf_img, (0, HEAD))
dr = ImageDraw.Draw(fr)
dr.rectangle([0, HEAD, W - 1, H - 1], outline=DIM)
lattice(dr)
# The demo anchor, its east neighbor, and their shared edge midpoint. Each
# anchor owns the hexagon of its six surrounding midpoints; the shared one
# sits exactly where the two hexagons touch.
axp, azp = cx0, cy0
exp, eyp = zpx(CAX + 1920, CAZ)
mid = zsx(CAX + 960)
HEX_MID_OFFSETS = [(960, 0), (480, -832), (-480, -832),
                   (-960, 0), (-480, 832), (480, 832)]
for hx0, hy0 in ((axp, azp), (exp, eyp)):
    ring = [(hx0 + zsx(CAX + dx) - zsx(CAX), hy0 + zsy(CAZ + dz) - zsy(CAZ))
            for (dx, dz) in HEX_MID_OFFSETS]
    dr.polygon(ring, outline=blend(BG, ACCENT, 0.7))
dr.ellipse([axp - 9, azp - 9, axp + 9, azp + 9], outline=ACCENT, width=3)
dr.ellipse([exp - 9, eyp - 9, exp + 9, eyp + 9], outline=ACCENT, width=3)
dr.ellipse([mid - 7, azp - 7, mid + 7, azp + 7], outline=WHITE, width=2)
dr.text((axp - dr.textlength("anchor", font=font(12)) / 2, azp + 14),
        "anchor", font=font(12), fill=ACCENT)
dr.text((exp - dr.textlength("anchor", font=font(12)) / 2, eyp + 14),
        "anchor", font=font(12), fill=ACCENT)
dr.text((mid - dr.textlength("shared midpoint", font=font(12)) / 2, azp - 32),
        "shared midpoint", font=font(12), fill=WHITE)
dr.text((12, 8), "Every midpoint is shared by two anchors",
        font=font(17), fill=TXT)
title(dr, (12, 36), "1,596 shared samples probe all 399 anchors: "
                    "4 samples per anchor instead of 7", max_w=W - 24, size=12, fill=DIM)
paste_locator(fr)
frames.append(fr)
dur.append(3800)

frames = [with_rail(f, 4) for f in frames]
save_gif(frames, dur, "g03_probe7.gif")
