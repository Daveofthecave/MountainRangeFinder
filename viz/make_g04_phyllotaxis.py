#!/usr/bin/env python3
# viz/make_g04_phyllotaxis.py -- KernelCoverage's disc scan, on the real seed.
# Continuity: opens on exactly the +-3,200-block view g03 ended with (stencil
# dots still shown), then pushes into the r=1,600 core disc. The probe lattice
# stays as a dim ghost so the disc work never loses its frame of reference.

import json
import numpy as np
from PIL import Image, ImageDraw

from geom import phyllotaxis, PROBE_MIDS, draw_probe_lattice
from fields import climate_grid
from style import (BG, HIT, MISS, TXT, DIM, ACCENT, WHITE,
                   font, blend, save_gif, with_rail)

SEED = 6696478651374553046
# Continuity: reuse the anchor the probe clip (g03) chose, so this clip
# starts exactly where that one ended. Falls back to the recall anchor.
try:
    with open("viz/out/demo_anchor.json") as fp:
        _a = json.load(fp)
    AX, AZ = int(_a["x"]), int(_a["z"])
except Exception:
    AX, AZ = -16064, -15044      # recall anchor (blocks)
R = 1600                         # core disc radius (blocks)
NEED, TOTAL, THRESH = 77, 96, -0.40
PROBE_GATE = -0.30

W = 760
HEAD, FOOT = 64, 56
MAP = 700
MX, MY = (W - MAP) // 2, HEAD
H = HEAD + MAP + FOOT

EXT2, PITCH2 = 3200, 64 # exactly g03's closing view (continuity)
EXT, PITCH = 2100, 32     # main view: the disc occupies ~76% of the map

ero2 = climate_grid(SEED, "erosion", 2, (AX - EXT2) // 4, (AZ - EXT2) // 4,
                    2 * EXT2 // PITCH2 + 1, 2 * EXT2 // PITCH2 + 1, PITCH2 // 4)
ero = climate_grid(SEED, "erosion", 2, (AX - EXT) // 4, (AZ - EXT) // 4,
                   2 * EXT // PITCH + 1, 2 * EXT // PITCH + 1, PITCH // 4)
NF = ero.shape[0]

def render(grid):
    lo, hi = np.percentile(grid, [2, 98])
    grey = np.clip((grid - lo) / (hi - lo), 0, 1)
    arr = (grey[..., None] * np.array([70.0, 90.0, 120.0]) * 0.55).astype(np.uint8)
    return Image.fromarray(arr).resize((MAP, MAP), Image.NEAREST)

img_coarse = render(ero2)
img_fine = render(ero)

def px2(bx, bz):   # coarse (continuity) view
    return (MX + (bx - (AX - EXT2)) / (2 * EXT2) * MAP,
            MY + (bz - (AZ - EXT2)) / (2 * EXT2) * MAP)
def pxf(bx, bz):   # fine (disc) view
    return (MX + (bx - (AX - EXT)) / (2 * EXT) * MAP,
            MY + (bz - (AZ - EXT)) / (2 * EXT) * MAP)

def field_at(bx, bz):
    gi = int(round((bx - (AX - EXT)) / PITCH))
    gj = int(round((bz - (AZ - EXT)) / PITCH))
    return ero[np.clip(gj, 0, NF - 1), np.clip(gi, 0, NF - 1)]

def coarse_at(bx, bz):
    gi = int(round((bx - (AX - EXT2)) / PITCH2))
    gj = int(round((bz - (AZ - EXT2)) / PITCH2))
    return ero2[np.clip(gj, 0, ero2.shape[0] - 1), np.clip(gi, 0, ero2.shape[1] - 1)]

pts = phyllotaxis()              # (96, 2), visit order

def compose(img, l1, l2, foot="", zoom_frac=None):
    """Base frame: backdrop + border + title strip + footer strip."""
    fr = Image.new("RGB", (W, H), BG)
    fr.paste(img, (MX, MY))
    dr = ImageDraw.Draw(fr)
    dr.rectangle([MX, MY, MX + MAP - 1, MY + MAP - 1], outline=DIM)
    dr.text((14, 10), l1, font=font(17), fill=TXT)
    dr.text((14, 38), l2, font=font(12), fill=DIM)
    if foot:
        dr.text((14, H - FOOT + 16), foot, font=font(13), fill=TXT)
    return fr

def draw_stencil(dr, px):
    """g02's closing image: the 7 probe dots with their pass/fail colors."""
    for (dx, dz) in PROBE_MIDS:
        ok = coarse_at(AX + dx, AZ + dz) <= PROBE_GATE
        pxx, pyy = px(AX + dx, AZ + dz)
        dr.ellipse([pxx - 6, pyy - 6, pxx + 6, pyy + 6],
                   fill=HIT if ok else MISS, outline=WHITE)

def disc_outline(dr, px, ext):
    r_px = R / (2 * ext) * MAP
    cx0, cy0 = px(AX, AZ)
    dr.ellipse([cx0 - r_px, cy0 - r_px, cx0 + r_px, cy0 + r_px],
               outline=blend(BG, WHITE, 0.45), width=2)

frames, dur = [], []

def lattice_layer(px, home=None, opacity=1.0):
    """The probe lattice on its own layer, clipped to the map square, so the
    underlay can never bleed past the frame. `opacity` lets the lattice
    recede during the zoom and the disc scan, so it doesn't compete with
    the phyllotaxis dots."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    draw_probe_lattice(d, px, (MX, MX + MAP), (MY, MY + MAP), home=home)
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rectangle([MX, MY, MX + MAP - 1, MY + MAP - 1], fill=255)
    alpha = layer.split()[3].point(lambda v: int(v * opacity))
    layer.putalpha(Image.composite(alpha, Image.new("L", (W, H), 0), mask))
    return layer

def with_lattice(fr, px, home=None, opacity=1.0):
    return Image.alpha_composite(fr.convert("RGBA"),
                                 lattice_layer(px, home, opacity)).convert("RGB")

# --- intro: g03's closing frame, verbatim --------------------------------------
fr = compose(img_coarse,
             "This anchor passed the 7-point erosion probe",
             "Field: Erosion, octaves 0A + 0B")
fr = with_lattice(fr, px2, home=(AX, AZ))
draw_stencil(ImageDraw.Draw(fr), px2)
frames.append(fr)
dur.append(1700)

# --- the stencil bows out BEFORE the camera moves ------------------------------
fr = compose(img_coarse, "The probe points step aside",
             "Field: Erosion, octaves 0A + 0B")
fr = with_lattice(fr, px2, home=(AX, AZ))
dr = ImageDraw.Draw(fr)
for (dx, dz) in PROBE_MIDS:
    pxx, pyy = px2(AX + dx, AZ + dz)
    dr.ellipse([pxx - 6, pyy - 6, pxx + 6, pyy + 6], outline=DIM)   # neutral now
frames.append(fr)
dur.append(1000)

# --- smooth push-in: +-3,200 -> +-2,100. The lattice rides along with the
#     camera; the disc outline only appears once the zoom has landed. -----------
ZF = 12
for f in range(ZF):
    t = (f + 1) / ZF
    e = t * t * (3 - 2 * t)
    ext = EXT2 + (EXT - EXT2) * e
    def pxt(bx, bz, ext=ext):
        return (MX + (bx - (AX - ext)) / (2 * ext) * MAP,
                MY + (bz - (AZ - ext)) / (2 * ext) * MAP)
    fr = compose(Image.blend(img_coarse, img_fine, e),
                 "Zooming to the core sunflower disc (radius = 1,600 blocks)...",
                 "Field: Erosion, octaves 0A + 0B")
    fr = with_lattice(fr, pxt, home=(AX, AZ), opacity=1.0 - 0.5 * e)
    frames.append(fr)
    dur.append(65)

fr = compose(img_fine, "The Sunflower Disc Scan: 96 Samples in a Phyllotaxis Spiral",
             "Erosion ≤ %.2f at %d or more of %d samples" % (THRESH, NEED, TOTAL),
             foot="Samples are checked in the GPU's fail-fast batch order")
fr = with_lattice(fr, pxf, home=(AX, AZ), opacity=0.5)
disc_outline(ImageDraw.Draw(fr), pxf, EXT)
frames.append(fr)
dur.append(900)

# --- the scan ---------------------------------------------------------------
hits = 0
for k in range(0, TOTAL, 2):                     # two samples per frame
    fr = compose(img_fine, "Erosion ≤ %.2f: %d of %d Needed" % (THRESH, hits, NEED),
                 "Field: Erosion, octaves 0A + 0B",
                 foot="Sample %d of %d" % (min(k + 2, TOTAL), TOTAL))
    fr = with_lattice(fr, pxf, home=(AX, AZ), opacity=0.5)
    dr = ImageDraw.Draw(fr)
    disc_outline(dr, pxf, EXT)
    for i in range(min(k + 2, TOTAL)):
        bx = AX + pts[i, 0] * R
        bz = AZ + pts[i, 1] * R
        ok = field_at(bx, bz) <= THRESH
        if i >= k:
            hits += ok
        pxx, pyy = pxf(bx, bz)
        rr = 7 if i >= k else 3                  # newest samples pop
        dr.ellipse([pxx - rr, pyy - rr, pxx + rr, pyy + rr],
                   fill=HIT if ok else MISS)
    frames.append(fr)
    dur.append(70)

verdict = hits >= NEED
end = frames[-1].copy()
d = ImageDraw.Draw(end)
d.rectangle([MX, H - FOOT, MX + MAP, H - 1], fill=BG)
d.text((14, H - FOOT + 8),
       "Disc scan passed: %d of %d needed" % (hits, NEED) if verdict
       else "Disc scan failed: %d of %d needed" % (hits, NEED),
       font=font(20), fill=HIT if verdict else MISS)
d.text((14, H - FOOT + 32),
       "Passing anchors proceed to the window checks (next clip)",
       font=font(11), fill=DIM)
frames += [end] * 2
dur += [1500, 1500]

frames = [with_rail(f, 5) for f in frames]
save_gif(frames, dur, "g04_phyllotaxis.gif")
print("%d/%d hits -> %s" % (hits, TOTAL, "PASS" if verdict else "FAIL"))
