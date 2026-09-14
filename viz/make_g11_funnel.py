#!/usr/bin/env python3
# viz/make_g11_funnel.py -- the pipeline funnel with real numbers from one
# 5-second --debug window (your hardware's numbers will vary).

import math
from PIL import Image, ImageDraw
from style import (BG, TXT, DIM, HIT, ACCENT, WATER, WHITE,
                   font, blend, shade, save_gif, with_rail)

W, H = 980, 470
STAGES = [  # (name, value for bar length, count label, note, color)
    ("Init",       103.02e6, "103.02M seeds",
     "2 erosion tables (0A/0B) per seed", shade(TXT, 0.55)),
    ("Coverage",   1.51e6,   "1.51M anchors",
     "Seven-point erosion check and 96-sample disc (1 in 27,225 of 41.11G)", WATER),
    ("Late init",  1.42e6,   "1.42M seeds",
     "The other 32 noise tables, once per surviving seed", shade(WATER, 0.7)),
    ("Extrema",    30.3e3,   "30.3k cores",
     "Window checks: temperature, continentalness, texture, height (1 in 50)", ACCENT),
    ("Blob fill",  116.0,    "116 candidates",
     "Connected-area flood fill with early exit (1 in 261)", blend(ACCENT, HIT, 0.5)),
    ("CPU verify", 9.2,      "~9 verified / 5 s",
     "Dark-forest veto, area and core gates, height window", HIT),
]
LOGMAX = math.log10(STAGES[0][1])
BAR_X = 150
BAR_MAXW = 430                 # bars: x in [150, 580]
LABEL_X = 600                  # all text in a fixed right column

frames, dur = [], []

def compose(rows, partial=None):
    fr = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(fr)
    dr.text((16, 10), "The Pipeline Funnel: 5 Seconds of a Sample Run",
            font=font(17), fill=TXT)
    dr.text((16, 36), "Arguments: --threads 32 --debug --min-blob-area 20000000 "
                      "(rates vary with hardware and settings)",
            font=font(12), fill=DIM)
    y = 56
    for (name, val, count, note, col) in rows:
        wbar = max(10, int(BAR_MAXW * math.log10(max(val, 1)) / LOGMAX))
        dr.rectangle([BAR_X, y, BAR_X + wbar, y + 34], fill=col)
        dr.text((12, y + 9), name, font=font(14), fill=TXT)
        dr.text((LABEL_X, y + 2), count, font=font(15), fill=TXT)
        dr.text((LABEL_X, y + 21), note, font=font(11), fill=DIM)
        y += 52
    if partial:
        name, val, col, frac = partial
        wbar = max(10, int(BAR_MAXW * math.log10(max(val, 1)) / LOGMAX * frac))
        dr.rectangle([BAR_X, y, BAR_X + wbar, y + 34], fill=col)
        dr.text((12, y + 9), name, font=font(14), fill=TXT)
    return fr

frames.append(compose([]))
dur.append(1000)

done = []
for st in STAGES:
    for k in range(5):
        f = (k + 1) / 5
        frames.append(compose(done, (st[0], st[1], st[4], f * f * (3 - 2 * f))))
        dur.append(55)
    done.append(st)
    frames.append(compose(done))
    dur.append(500)

end = compose(done)
dr = ImageDraw.Draw(end)
y = 56 + 52 * len(STAGES) + 10
dr.text((16, y),      "Net: 103,020,000 seeds, ~9 verified regions (1 in ~11.4 million)",
        font=font(16), fill=WHITE)
dr.text((16, y + 26), "Each phase is tighter than the one before it, and rejects most of what remains",
        font=font(13), fill=DIM)
frames.append(end)
dur.append(9000)   # let the summary breathe (it's the densest frame of the set)

frames = [with_rail(f, 12) for f in frames]
save_gif(frames, dur, "g11_funnel.gif")

