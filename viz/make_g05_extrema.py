#!/usr/bin/env python3
# viz/make_g05_extrema.py -- KernelExtrema: the surviving anchor's windowed
# gauntlet (gpu.cu stage 4). Picks up exactly where g04's disc scan ended.
#
# Fidelity notes: the kernel walks each grid in row-major order, 32 points per
# warp iteration, and the extrema checks (3)/(5) early-exit at the first hit.
# The sweeps are animated as continuous row-major passes (a little slower and
# more complete than reality) because that's how you watch a scan.

import json
import numpy as np
from PIL import Image, ImageDraw

from geom import phyllotaxis
from fields import climate_grid, height_grid
from style import (BG, PANEL, TXT, DIM, HIT, MISS, ACCENT, WHITE,
                   font, blend, save_gif, title, with_rail)

SEED = 6696478651374553046
try:
    with open("viz/out/demo_anchor.json") as fp:
        _a = json.load(fp)
    AX, AZ = int(_a["x"]), int(_a["z"])
except Exception:
    AX, AZ = -16064, -15044      # recall anchor (blocks)

# Check constants (gpu.cu KernelExtrema; comments note the recipe step).
TEMP_HALF, TEMP_STEP, TEMP_NEED = 500, 100, 79        # (6) 11x11 window
TEMP_LO, TEMP_HI = -0.40, 0.20
CONT_DISC_R, CONT_NEED = 1600, 64                     # (4) 96-sample disc, 1B
WIN_HALF, WIN_STEP = 1600, 64                         # (3)/(5) 51x51, full octaves
ERO_MIN_REQ, CONT_MAX_REQ = -1.15, 0.70
W_BAND_LO, W_BAND_HI, W_RIDGE_MIN = 0.55, 0.82, 0.012 # (w) (2,2) weirdness
H_HALF, H_STEP, H_GATE = 1536, 128, 215               # (h) 25x25, depth spline

W, H = 1020, 880
MAP = 700
MAPX, MAPY = 20, 100
EXT = WIN_HALF
PITCH = 32
N = 2 * EXT // PITCH + 1

print("sampling fields (7 grids; cached after first run)...")
eroB  = climate_grid(SEED, "erosion", 2,  (AX - EXT)//4, (AZ - EXT)//4, N, N, PITCH//4)
eroF  = climate_grid(SEED, "erosion", -1, (AX - EXT)//4, (AZ - EXT)//4, N, N, PITCH//4)
contF = climate_grid(SEED, "cont",    -1, (AX - EXT)//4, (AZ - EXT)//4, N, N, PITCH//4)
contB = climate_grid(SEED, "cont",     4, (AX - EXT)//4, (AZ - EXT)//4, N, N, PITCH//4)
tempF = climate_grid(SEED, "temp",    -1, (AX - EXT)//4, (AZ - EXT)//4, N, N, PITCH//4)
weird = climate_grid(SEED, "weird",     4, (AX - EXT)//4, (AZ - EXT)//4, N, N, PITCH//4)
HN = 2 * H_HALF // H_STEP + 1
hg = height_grid(SEED, AX - H_HALF, AZ - H_HALF, HN, HN, H_STEP)

def at(g, x, z):
    i = min(max(int(round((x - (AX - EXT)) / PITCH)), 0), N - 1)
    j = min(max(int(round((z - (AZ - EXT)) / PITCH)), 0), N - 1)
    return float(g[j, i])

def render_bg(grid):
    lo, hi = np.percentile(grid, [2, 98])
    grey = np.clip((grid - lo) / (hi - lo), 0, 1)
    return Image.fromarray((grey[..., None] * np.array([70, 90, 120]) * 0.6)
                           .astype(np.uint8)).resize((MAP, MAP), Image.NEAREST)

def height_bg():
    t = np.clip((hg - 60.0) / 200.0, 0, 1)
    arr = (np.array([60.0, 75.0, 95.0]) * (1 - t[..., None])
           + np.array([225.0, 230.0, 240.0]) * t[..., None])
    return Image.fromarray((arr * 0.6).astype(np.uint8)).resize((MAP, MAP), Image.NEAREST)

BACKDROPS = {
    "intro": (render_bg(eroB),  "Field: Erosion, octaves 0A + 0B"),
    "temp":  (render_bg(tempF), "Field: Temperature, all octaves"),
    "cont":  (render_bg(contB), "Field: Continentalness, octaves 0A to 1B"),
    "weird": (render_bg(weird), "Field: Weirdness, octaves 0A to 1B"),
    "eroF":  (render_bg(eroF),  "Field: Erosion, all octaves"),
    "contF": (render_bg(contF), "Field: Continentalness, all octaves"),
    "h":     (height_bg(),      "Approximate surface height"),
}
base, _base_tag = BACKDROPS["intro"]

def px(x, z):
    return (MAPX + (x - (AX - EXT)) / (2 * EXT) * MAP,
            MAPY + (z - (AZ - EXT)) / (2 * EXT) * MAP)

CHECKS = ["Temperature window",
          "Continentalness disc",
          "Weirdness ridge band",
          "Deep erosion point",
          "High continentalness point",
          "Approximate height gate"]
state = ["wait"] * 6

def compose(pts, note, marks=(), dot_r=3, arena=("rect", WIN_HALF), bdrop="intro",
            dim_dots=False):
    img, field_tag = BACKDROPS[bdrop]
    fr = Image.new("RGB", (W, H), BG)
    fr.paste(img, (MAPX, MAPY))
    dr = ImageDraw.Draw(fr)
    dr.rectangle([MAPX, MAPY, MAPX + MAP - 1, MAPY + MAP - 1], outline=DIM)
    # the arena outline for the current check
    x0, y0 = px(AX - arena[1], AZ - arena[1])
    x1, y1 = px(AX + arena[1], AZ + arena[1])
    if arena[0] == "rect":
        dr.rectangle([x0, y0, x1, y1], outline=blend(BG, WHITE, 0.35))
    else:
        dr.ellipse([x0, y0, x1, y1], outline=blend(BG, WHITE, 0.35))
    for pt in pts:
        x, z, ok = pt[0], pt[1], pt[2]
        cx, cy = px(x, z)
        if len(pt) > 3 and pt[3] is not None:
            col = pt[3]                       # explicit per-point color
        elif ok is None:
            col = blend(BG, WHITE, 0.18) if dim_dots else DIM
        else:
            col = HIT if ok else MISS
        dr.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r], fill=col)
    for (x, z, col, lbl, val) in marks:
        cx, cy = px(x, z)
        dr.line([cx - 8, cy - 8, cx + 8, cy + 8], fill=col, width=2)
        dr.line([cx - 8, cy + 8, cx + 8, cy - 8], fill=col, width=2)
        dr.text((cx + 11, cy - 8), lbl + " " + val, font=font(13), fill=col)
    title(dr, (16, 10), "Window Checks - Cheapest to Costliest",
          max_w=W - 32)
    title(dr, (16, 36), field_tag, max_w=W - 32, size=12, fill=ACCENT)
    title(dr, (16, 58), note, max_w=W - 32, size=13, fill=DIM)
    y = MAPY + 130
    for name, st in zip(CHECKS, state):
        mark, col = {"wait": ("[  ]", DIM), "run": ("[..]", ACCENT),
                     "pass": ("[ok]", HIT), "fail": ("[XX]", MISS)}[st]
        dr.text((MAPX + MAP + 26, y), mark, font=font(14), fill=col)
        dr.text((MAPX + MAP + 26, y + 26), name, font=font(13), fill=col)
        y += 78
    return fr

def row_major(half, step):
    """Grid points in the kernel's row-major order."""
    n = 2 * (half // step) + 1
    return [(AX - half + i * step, AZ - half + j * step)
            for j in range(n) for i in range(n)], n

def sweep(frames, dur, pts, oks, per_frame, note_fn, frame_dur=60, **kw):
    """Animate a row-major sweep `per_frame` points at a time."""
    shown = 0
    while shown < len(pts):
        shown = min(len(pts), shown + per_frame)
        view = [(pts[i][0], pts[i][1], oks[i]) for i in range(shown)]
        frames.append(compose(view, note_fn(shown, len(pts)), **kw))
        dur.append(frame_dur)
    return shown

frames, dur = [], []

# Rail stage 4 (late table init) gets its one nod here: the disc survivors pay
# for the remaining 32 noise tables before any window check runs.
fr = compose([], "Late Init: The seed's remaining climate noisemaps are built",
             arena=("rect", WIN_HALF))
frames.append(fr)
dur.append(1700)

frames.append(compose([], "Each check proceeds only if the previous one passed",
                      arena=("rect", WIN_HALF)))
dur.append(1200)

# --- (6) temperature coverage: 11x11, one row at a time -----------------------
state[0] = "run"
tpts, tn = row_major(TEMP_HALF, TEMP_STEP)
toks = [TEMP_LO <= at(tempF, x, z) <= TEMP_HI for x, z in tpts]
sweep(frames, dur, tpts, toks, tn // 2,
      lambda s, tot: "Temperature in [%.2f, %.2f]: %d of %d sampled, %d in range (need %d)"
                     % (TEMP_LO, TEMP_HI, s, tot, sum(toks[:s]), TEMP_NEED),
      frame_dur=80, arena=("rect", TEMP_HALF), bdrop="temp")
th = sum(toks)
state[0] = "pass" if th >= TEMP_NEED else "fail"
frames.append(compose([(x, z, o) for (x, z), o in zip(tpts, toks)],
                      "Temperature: %d of 121 in range: %s" % (th, state[0].capitalize()),
                      arena=("rect", TEMP_HALF), bdrop="temp"))
dur.append(1200)

# --- (4) continentalness disc: phyllotaxis, unchanged (it reads beautifully) --
state[1] = "run"
dpts = phyllotaxis() * CONT_DISC_R
cok = [(AX + dx, AZ + dz, at(contB, AX + dx, AZ + dz) >= 0.0) for dx, dz in dpts]
for k in range(0, 96, 4):
    part = cok[:k + 4]
    hits = sum(o for _, _, o in part)
    frames.append(compose(part, "Continentalness ≥ 0: %d hits (%d needed)"
                          % (hits, CONT_NEED), arena=("disc", CONT_DISC_R), bdrop="cont"))
    dur.append(70)
ch = sum(o for _, _, o in cok)
state[1] = "pass" if ch >= CONT_NEED else "fail"
frames.append(compose(cok, "Continentalness: %d of 96 inland: %s"
                      % (ch, state[1].capitalize()), arena=("disc", CONT_DISC_R), bdrop="cont"))
dur.append(1200)

# --- (w) weirdness ridge band: row-major, every sample shown ------------------
state[2] = "run"
wpts, wn = row_major(WIN_HALF, 100)
wok = [W_BAND_LO <= abs(at(weird, x, z)) <= W_BAND_HI for x, z in wpts]
sweep(frames, dur, wpts, wok, wn,
      lambda s, tot: "Weirdness fraction: %.3f (need %.3f)"
                     % (sum(wok[:s]) / s, W_RIDGE_MIN),
      arena=("rect", WIN_HALF), bdrop="weird", dot_r=2, dim_dots=True)
wfr = sum(wok) / len(wok)
state[2] = "pass" if wfr >= W_RIDGE_MIN else "fail"
frames.append(compose([(x, z, o) for (x, z), o in zip(wpts, wok) if o],
                      "Weirdness fraction %.3f: %s"
                      % (wfr, state[2].capitalize()),
                      arena=("rect", WIN_HALF), bdrop="weird"))
dur.append(1300)

# --- (3) deep erosion point over the 51x51 window, row-major ------------------
xs = [AX - WIN_HALF + i * WIN_STEP for i in range(51)]
zs = [AZ - WIN_HALF + j * WIN_STEP for j in range(51)]
E = np.array([[at(eroF, x, z) for x in xs] for z in zs])
C = np.array([[at(contF, x, z) for x in xs] for z in zs])
epts, en = row_major(WIN_HALF, WIN_STEP)
state[3] = "run"
run_e = np.inf
me_pt = None
shown = 0
while shown < len(epts):
    shown = min(len(epts), shown + 2 * en)   # two rows per frame
    for i in range(max(0, shown - 2 * en), shown):
        x, z = epts[i]
        ev = E[(z - zs[0]) // WIN_STEP, (x - xs[0]) // WIN_STEP]
        if ev < run_e: run_e, me_pt = float(ev), (x, z)
    view = [(x, z, None) for (x, z) in epts[:shown]]
    marks = [(me_pt[0], me_pt[1], WHITE, "erosion min", "%.2f" % run_e)] if me_pt else []
    frames.append(compose(view,
        "Deep erosion point: minimum %.2f so far (need %.2f)"
        % (run_e, ERO_MIN_REQ),
        marks, dot_r=1, arena=("rect", WIN_HALF), bdrop="eroF", dim_dots=True))
    dur.append(90)
state[3] = "pass" if run_e <= ERO_MIN_REQ else "fail"
frames.append(compose([], "Erosion minimum %.2f: %s" % (run_e, state[3].capitalize()),
                      [(me_pt[0], me_pt[1], WHITE, "erosion min", "%.2f" % run_e)],
                      arena=("rect", WIN_HALF), bdrop="eroF"))
dur.append(1300)

# --- (5) high continentalness point over the same window ----------------------
state[4] = "run"
run_c = -np.inf
mc_pt = None
shown = 0
while shown < len(epts):
    shown = min(len(epts), shown + 2 * en)
    for i in range(max(0, shown - 2 * en), shown):
        x, z = epts[i]
        cv = C[(z - zs[0]) // WIN_STEP, (x - xs[0]) // WIN_STEP]
        if cv > run_c: run_c, mc_pt = float(cv), (x, z)
    view = [(x, z, None) for (x, z) in epts[:shown]]
    marks = [(mc_pt[0], mc_pt[1], ACCENT, "continentalness max", "%.2f" % run_c)] if mc_pt else []
    frames.append(compose(view,
        "High continentalness point: maximum %.2f so far (need %.2f)"
        % (run_c, CONT_MAX_REQ),
        marks, dot_r=1, arena=("rect", WIN_HALF), bdrop="contF", dim_dots=True))
    dur.append(90)
state[4] = "pass" if run_c >= CONT_MAX_REQ else "fail"
frames.append(compose([], "Continentalness maximum %.2f: %s" % (run_c, state[4].capitalize()),
                      [(mc_pt[0], mc_pt[1], ACCENT, "continentalness max", "%.2f" % run_c)],
                      arena=("rect", WIN_HALF), bdrop="contF"))
dur.append(1300)

# --- (h) approx height gate: 25x25, one row per frame ---------------------------
state[5] = "run"
hpts, hn = row_major(H_HALF, H_STEP)

def hdot(x, z):
    """Whitish height-ramp dot so the scan reads clearly over the heightmap."""
    v = float(hg[(z - (AZ - H_HALF)) // H_STEP, (x - (AX - H_HALF)) // H_STEP])
    if v <= 62:
        return blend(BG, (70, 110, 170), 0.95)
    tt = min(1.0, max(0.0, (v - 62) / 190.0))
    return blend((110, 125, 150), WHITE, tt)

run_h = -1e9
hp = None
shown = 0

def hm_col(v):
    """red -> amber -> green with the running max (green once it clears 215)"""
    t = max(0.0, min(1.0, (v - 150.0) / 100.0))
    return blend(MISS, HIT, t)

while shown < len(hpts):
    shown = min(len(hpts), shown + hn)
    for i in range(max(0, shown - hn), shown):
        x, z = hpts[i]
        hv = float(hg[(z - (AZ - H_HALF)) // H_STEP, (x - (AX - H_HALF)) // H_STEP])
        if hv > run_h: run_h, hp = hv, (x, z)
    view = [(x, z, None, hdot(x, z)) for (x, z) in hpts[:shown]]
    marks = [(hp[0], hp[1], hm_col(run_h), "max height", "%.0f" % run_h)] if hp else []
    frames.append(compose(view, "Approximate height: maximum %.0f so far (need %d)"
                          % (run_h, H_GATE), marks, dot_r=2,
                          arena=("rect", H_HALF), bdrop="h"))
    dur.append(85)
state[5] = "pass" if run_h >= H_GATE else "fail"
marks = [(hp[0], hp[1], hm_col(run_h), "max height", "%.0f" % run_h)]
frames.append(compose([], "Approximate maximum height %.0f: %s" % (run_h, state[5].capitalize()),
                      marks, arena=("rect", H_HALF), bdrop="h"))
dur.append(1300)

# --- verdict card ---------------------------------------------------------------
ok_all = all(s == "pass" for s in state)
card_img = base.point(lambda p: int(p * 0.30))
fr = Image.new("RGB", (W, H), BG)
fr.paste(card_img, (MAPX, MAPY))
dr = ImageDraw.Draw(fr)
dr.rectangle([MAPX, MAPY, MAPX + MAP - 1, MAPY + MAP - 1], outline=DIM)
bx0, by0, bx1, by1 = 70, MAPY + 170, W - 70, MAPY + 430
dr.rectangle([bx0, by0, bx1, by1], fill=PANEL, outline=DIM, width=2)
lines = [
    ("Climate Window Checks: Results", font(18), WHITE),
    ("", font(10), TXT),
    ("Temperature %d of 121  ·  Continentalness %d of 96  ·  Weirdness %.3f" % (th, ch, wfr),
     font(14), TXT),
    ("Erosion min %.2f  ·  Continentalness max %.2f  ·  Max Height %.0f" % (run_e, run_c, run_h),
     font(14), TXT),
    ("", font(10), TXT),
    ("Core emitted; the flood fill is next" if ok_all
     else "Anchor rejected; no further work spent",
     font(16), HIT if ok_all else MISS),
]
ty = by0 + 16
for text, f, col in lines:
    tw = dr.textlength(text, font=f)
    dr.text(((W - tw) / 2, ty), text, font=f, fill=col)
    ty += f.size + 10
frames.append(fr)
dur.append(4000)

# Hold the previous rail item (5) for the first two frames,
# then advance to the current rail item (6) for the rest of the animation.
frames = [with_rail(f, 6) for f in frames[:2]] + [with_rail(f, 7) for f in frames[2:]]
save_gif(frames, dur, "g05_extrema.gif")
print("temp %d/121 | cont %d/96 | weird %.4f | ero %.2f | cont %.2f | maxy %.0f"
      % (th, ch, wfr, run_e, run_c, run_h))
