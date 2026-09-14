#!/usr/bin/env python3
# viz/make_g00_preamble.py -- concept prelude: a Minecraft world is decided by
# five climate noise fields, and filtering seeds on the fields is far cheaper
# than computing biomes (a biome lookup needs all five fields at full depth
# plus the biome tree). Everything here is full-octave; octave truncation is
# the next clip (g01).

import numpy as np
from PIL import Image, ImageDraw

from fields import climate_grid, biome_grid, BIO_COLORS
from style import (BG, TXT, DIM, WHITE, font, blend, save_gif, title, with_rail)

SEED = 6696478651374553046
CX, CZ = -14500, -15000     # the magic region: peaks, forests, and sea in view
EXT = 4800                  # blocks (half-extent)
PITCH = 48                  # blocks per cell
N = 2 * EXT // PITCH + 1    # 201

# Slowest-varying to fastest-varying -- which also builds the world in story
# order: climate zones, then land vs sea, then relief, then vegetation, then
# the carved details.
FIELDS = [
    ("temp", "Temperature", (215, 175, 120),
     "From frozen taigas to hot deserts",
     "The slowest-varying field"),
    ("cont", "Continentalness", (135, 170, 225),
     "From deep ocean to far inland",
     "Features thousands of blocks wide"),
    ("erosion", "Erosion", (200, 200, 210),
     "From jagged peaks to flat plains and swamps",
     "Detail down to about 130 blocks"),
    ("humidity", "Humidity", (140, 200, 130),
     "From sparse savannas to dense jungles",
     "Features hundreds of blocks wide"),
    ("weird", "Weirdness", (200, 140, 215),
     "Ridges, valleys, and rivers",
     "The fastest-varying field"),
]

MAPX, MAPY, MAP = 20, 70, 700
SIDEX = MAPX + MAP + 26
W = SIDEX + 340
H = MAPY + MAP + 66

print("sampling the biome map + five full-octave fields (cached after the first run)...")
biome = biome_grid(SEED, CX - EXT, CZ - EXT, N, N, PITCH)
bio_arr = np.zeros((N, N, 3), np.uint8)
for bid, col in BIO_COLORS.items():
    bio_arr[biome == bid] = col
bio_arr[~np.isin(biome, list(BIO_COLORS))] = (70, 75, 80)
bio_img = Image.fromarray(bio_arr).resize((MAP, MAP), Image.NEAREST)
bio_dim_arr = (bio_arr.astype(float) * 0.30).astype(np.uint8)
bio_dim_img = Image.fromarray(bio_dim_arr).resize((MAP, MAP), Image.NEAREST)

def field_base(key, tint):
    """Full-octave field, percentile-stretched, tinted; float array N x N x 3."""
    f = climate_grid(SEED, key, -1, (CX - EXT) // 4, (CZ - EXT) // 4,
                     N, N, PITCH // 4)
    lo, hi = np.percentile(f, [2, 98])
    span = hi - lo if hi > lo else 1.0
    g = np.clip((f - lo) / span, 0, 1)
    return g[..., None] * np.array(tint, float)

FIELD_ALPHA = 0.95   # overlay opacity of the climate fields (bump toward 1.0
                     # for a stronger overlay, trim toward 0.9 to taste)

def composite(fbase):
    return Image.fromarray(
        np.clip(bio_dim_arr.astype(float) + fbase * FIELD_ALPHA, 0, 255)
        .astype(np.uint8)).resize((MAP, MAP), Image.NEAREST)

def draw_field_list(dr, upto, active=-1):
    dr.text((SIDEX, MAPY), "The five climate noisemaps:", font=font(13), fill=TXT)
    y = MAPY + 36
    for k, (key, name, tint, role, scale) in enumerate(FIELDS):
        if k >= upto:
            break
        dimmed = active >= 0 and k != active
        tcol = blend(BG, tint, 0.30 if dimmed else 1.0)
        dr.rectangle([SIDEX, y + 2, SIDEX + 14, y + 16], fill=tcol, outline=DIM)
        dr.text((SIDEX + 22, y), name, font=font(14),
                fill=DIM if dimmed else TXT)
        y += 24
        if k == active:
            dr.text((SIDEX + 22, y), role, font=font(11), fill=tint)
            y += 16
            dr.text((SIDEX + 22, y), scale, font=font(10), fill=DIM)
            y += 16
        y += 6

def compose(img, hdr, sub, upto=0, active=-1, foot=None):
    fr = Image.new("RGB", (W, H), BG)
    fr.paste(img, (MAPX, MAPY))
    dr = ImageDraw.Draw(fr)
    dr.rectangle([MAPX, MAPY, MAPX + MAP - 1, MAPY + MAP - 1], outline=DIM)
    title(dr, (14, 10), hdr, max_w=W - 28)
    title(dr, (14, 38), sub, max_w=W - 28, size=12, fill=DIM)
    if upto:
        draw_field_list(dr, upto, active)
    if foot:
        dr.text((MAPX + 4, MAPY + MAP + 14), foot, font=font(12), fill=TXT)
    return fr

frames, dur = [], []

# --- Act 1: the world as players know it --------------------------------------
frames.append(compose(bio_img,
    "A Minecraft World (Seed %s)" % SEED,
    "Biomes around (%s, %s)"
    % (format(CX, ","), format(CZ, ","))))
dur.append(2400)

frames.append(compose(bio_img,
    "Every biome is decided by five hidden climate noisemaps",
    "They are sampled every four blocks, before any terrain exists"))
dur.append(3000)

# --- Act 2: dim the world, bring in the five names ----------------------------
for f in range(3):
    frames.append(compose(Image.blend(bio_img, bio_dim_img, (f + 1) / 3),
                          "The Five Climate Noisemaps",
                          "Each noisemap influences certain aspects of the terrain"))
    dur.append(70)

for k in range(len(FIELDS)):
    frames.append(compose(bio_dim_img, "The Five Climate Noisemaps",
                          "Each noisemap influences certain aspects of the terrain",
                          upto=k + 1))
    dur.append(430)
frames.append(compose(bio_dim_img, "The Five Climate Noisemaps",
                      "Each noisemap overlaid on the same base map", upto=5))
dur.append(2000)

# --- Act 3: the tour of the five fields ---------------------------------------
fbases, comps = [], []
prev = bio_dim_img
for k, (key, name, tint, role, scale) in enumerate(FIELDS):
    fb = field_base(key, tint)
    fbases.append(fb)
    comp = composite(fb)
    comps.append(comp)
    for t in (0.4, 0.75):
        frames.append(compose(Image.blend(prev, comp, t),
                              name.capitalize(),
                              "Bright = high values; Dark = low values",
                              upto=5, active=k))
        dur.append(60)
    frames.append(compose(comp, name.capitalize(),
                          "Bright = high values; Dark = low values",
                          upto=5, active=k))
    dur.append(2000)
    prev = comp

# --- Act 4: the punchline -- fields are cheap, biomes are expensive -----------
punch = Image.new("RGB", (MAP, MAP), BG)
dr = ImageDraw.Draw(punch)
dr.text((16, 20), "At every point, the game evaluates all five noisemaps",
        font=font(13), fill=TXT)
tb = 68
for k, (key, name, tint, role, scale) in enumerate(FIELDS):
    thumb = Image.fromarray(np.clip(fbases[k], 0, 255).astype(np.uint8)) \
        .resize((96, 96), Image.NEAREST)
    punch.paste(thumb, (24, tb + k * 106))
    dr.rectangle([24, tb + k * 106, 119, tb + k * 106 + 95], outline=DIM)
    dr.text((132, tb + k * 106 + 40), name, font=font(11), fill=tint)
ay = tb + 2 * 106 + 48
dr.line([250, ay, 312, ay], fill=WHITE, width=4)
dr.line([298, ay - 10, 314, ay], fill=WHITE, width=4)
dr.line([298, ay + 10, 314, ay], fill=WHITE, width=4)
punch.paste(bio_img.resize((360, 360), Image.NEAREST), (324, ay - 180))
dr.rectangle([324, ay - 180, 683, ay + 179], outline=DIM)
dr.text((324, ay + 190), "and only then picks a biome.", font=font(13), fill=TXT)

frames.append(compose(punch,
    "The five climate noisemaps shape all biomes",
    "A biome lookup needs all five noisemaps at full depth, plus the biome tree",
    foot="Filtering on one noisemap needs only a few noise samples"))
dur.append(2600)

frames.append(compose(punch,
    "The pipeline filters out seeds based on these noisemap patterns",
    "Biomes are computed only for the few seeds that survive the climate noise filters",
    foot="Roughly 1 in 10 million seeds passes all climate checks and subsequent filters"))
dur.append(4000)

frames = [with_rail(f, 0) for f in frames]

save_gif(frames, dur, "g00_preamble.gif")
