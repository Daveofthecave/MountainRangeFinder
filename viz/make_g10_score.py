#!/usr/bin/env python3
# viz/make_g10_score.py -- the composite score as an animated waterfall, for
# the magic seed's real recall row. Uses seedlab.py's mirror of compute_score(),
# so the final bar lands on the exact number in output.txt.

import os
import sys
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import seedlab  # noqa: E402

from style import (BG, PANEL, TXT, DIM, HIT, MISS, ACCENT, WATER, WHITE,  # noqa: E402
                   font, blend, save_gif, with_rail)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
rows = [r for r in seedlab.load_rows([os.path.join(ROOT, "magic_seed.txt")])
        if seedlab.g(r, "score") is not None]
if not rows:
    raise SystemExit("no scored rows in magic_seed.txt")
r = max(rows, key=lambda rr: rr["score"])
p = seedlab.score_full(r)
print("mirror score %.4f  | file score %.4f" % (p["score"], r["score"]))

# The scenery term is the unified, core-gated `setting` bonus (probe.cpp
# setting_parts). Derive it as the residual vs the file's own score, so the
# waterfall always sums to the printed total even if the seedlab mirror drifts.
_head = (2.0 * p["key"]
         + 0.30 * (p["amp"] + p["tex"] + p["frag"] + p["depth"] + p["darkasp"])
         + 1.10 * p["ext"] + 1.25 * p["leg"]
         - p["dark"] - p["massif"])
p_setting = r["score"] - _head

W, H = 980, 560
frames, dur = [], []

# --- panel 1: the gated aspects -------------------------------------------------
ASPECTS = [("Amplitude", p["amp"]), ("Texture", p["tex"]),
           ("Peak packing", p["frag"]), ("Valley depth", p["depth"]),
           ("Dark forest", p["darkasp"])]
key = p["key"]

fr = Image.new("RGB", (W, H), BG)
dr = ImageDraw.Draw(fr)
dr.text((24, 20), "How a Seed's Score Is Calculated (Seed %s)" % r["_seed"],
        font=font(20), fill=TXT)
dr.text((24, 52), "Five quality aspects; the weakest one dominates, "
                  "so a great region must be good at all of them",
        font=font(13), fill=DIM)
bx, bw = 90, 120
y0 = 420
scale = 46
dr.line([60, y0, 60, 110], fill=DIM, width=1)
dr.line([40, y0, W - 60, y0], fill=DIM, width=1)
for k, (name, v) in enumerate(ASPECTS):
    x = bx + k * (bw + 30)
    is_min = abs(v - key) < 1e-9
    col = ACCENT if is_min else blend(BG, WATER, 0.9)
    hh = max(2, int(v * scale))
    dr.rectangle([x, y0 - hh, x + bw, y0], fill=col)
    dr.text((x + 8, y0 - hh - 22), "%.2f" % v, font=font(15),
            fill=ACCENT if is_min else TXT)
    # full-word labels, split across two lines when needed
    words = name.split(" ")
    if len(words) == 2:
        dr.text((x + 8, y0 + 8), words[0], font=font(13), fill=TXT)
        dr.text((x + 8, y0 + 24), words[1], font=font(13), fill=TXT)
    else:
        dr.text((x + 8, y0 + 10), name, font=font(13), fill=TXT)
    if is_min:
        dr.text((x - 10, y0 + 44), "Weakest Aspect", font=font(12), fill=ACCENT)
dr.text((40, y0 + 66), "Quality aspects", font=font(12), fill=DIM)
dr.text((24, H - 30), "Next up: how the terms stack into the final score",
        font=font(13), fill=DIM)
frames.append(fr)
dur.append(5000)

# --- panel 2: the waterfall -------------------------------------------------------
COMP = [
    ("Weakest x2.00",  "the weakest aspect, doubled", 2.0 * key),
    ("Aspects x0.30",  "the aspect sum, a tiebreak",
     0.30 * (p["amp"] + p["tex"] + p["frag"] + p["depth"] + p["darkasp"])),
    ("Extent x1.10",   "the expansiveness of the mountain pattern", 1.10 * p["ext"]),
    ("Consensus x1.25","39 custom-calibrated weights",       1.25 * p["leg"]),
    ("Dark forest",    "penalty for dark forests in the window",   -p["dark"]),
    ("Massif tax",     "a penalty for massive plateaus",           -p["massif"]),
    ("Setting",        "scenery bonus, gated by core quality",      p_setting),
]
cum = [0.0]
for _, _, v in COMP:
    cum.append(cum[-1] + v)
total = cum[-1]

X0, X1 = 70, W - 60
YT = H - 40
vlo = min(0.0, min(cum)) - 0.4
vhi = max(cum + [total]) + 0.6

def vy(v):
    return YT - (v - vlo) / (vhi - vlo) * (YT - 90 - 40)

def draw_waterfall(upto, frac=1.0):
    fr = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(fr)
    dr.text((24, 16), "How the Aspects Contribute to the Final Score", font=font(20), fill=TXT)
    dr.text((24, 46), "Each term adds to the running total",
            font=font(13), fill=DIM)
    dr.line([40, vy(0), W - 40, vy(0)], fill=DIM, width=1)
    n = len(COMP)
    slot = (X1 - X0) / (n + 1)
    bw_ = slot * 0.7
    for k in range(min(upto, n)):
        name, desc, v = COMP[k]
        a, b = cum[k], cum[k + 1]
        x = X0 + k * slot
        col = HIT if v >= 0 else MISS
        dr.rectangle([x, vy(max(a, b)), x + bw_, vy(min(a, b))], fill=col)
        dr.line([x + bw_, vy(b), x + slot, vy(b)], fill=DIM)
        dr.text((x, YT + 6), name, font=font(10), fill=DIM)
        dr.text((x, vy(max(a, b)) - 18), "%+.2f" % v, font=font(11), fill=col)
    if upto < n and frac > 0.0:   # the in-progress bar
        name, desc, v = COMP[upto]
        a, b = cum[upto], cum[upto] + v * frac
        x = X0 + upto * slot
        col = HIT if v >= 0 else MISS
        dr.rectangle([x, vy(max(a, b)), x + bw_, vy(min(a, b))],
                     fill=blend(col, ACCENT, 0.5))
        dr.text((x, YT + 6), name, font=font(10), fill=DIM)
        dr.text((x, vy(max(a, b)) - 18), "%+.2f" % v, font=font(11), fill=col)
    if upto >= n:                 # TOTAL only once everything else is done
        x = X0 + n * slot
        dr.rectangle([x, vy(total), x + bw_, vy(0)], fill=ACCENT)
        lbl = "score %.2f" % total
        lx = min(x - 6, W - 20 - dr.textlength(lbl, font=font(15)))
        dr.text((lx, vy(total) - 24), lbl, font=font(15), fill=ACCENT)
        dr.text((x - 2, YT + 6), "TOTAL", font=font(11), fill=TXT)
    return fr

for k in range(len(COMP)):
    for f in range(8):
        frames.append(draw_waterfall(k, (f + 1) / 8))
        dur.append(45)
    frames.append(draw_waterfall(k + 1, 0.0))   # hold: bar k complete
    dur.append(600)

frames.append(draw_waterfall(len(COMP)))
dur.append(700)
frames.append(draw_waterfall(len(COMP)))
dur.append(3000)

# --- epilogue A: 4 sublattice phases converge to the best one ------------------
def draw_phase_frame(e, merged=False):
    fr = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(fr)
    dr.text((24, 40), "Each candidate is measured on 4 interleaved sublattices",
            font=font(17), fill=TXT)
    dr.text((24, 68), "Only the best-scoring phase is kept (illustrative values)",
            font=font(13), fill=DIM)
    xt = 470
    vals = ("6.9", "8.30", "7.7", "8.1")
    # Boxes never fade, and every label stays put during the slide; the losing
    # phases' labels drop only once the merge lands (merged=True). The winner
    # (p1) is drawn last so its text stays on top.
    for k in (0, 2, 3, 1):
        x0 = 170 + k * 160
        x = x0 + (xt - x0) * e
        col = blend(WATER, ACCENT, e)
        dr.rectangle([x, 200, x + 60, 200 + 60], outline=col, width=3)
        if not merged or k == 1:
            dr.text((x + 6, 224), "p%d" % k, font=font(13), fill=TXT)
            dr.text((x - 2, 268), vals[k], font=font(11), fill=DIM)
    if merged:
        dr.text((400, 300), "The best phase wins", font=font(15), fill=ACCENT)
    return fr

frames.append(draw_phase_frame(0.0))
dur.append(2200)                       # let the four phases introduce themselves
for f in range(20):
    t = f / 19
    e = t * t * (3 - 2 * t)
    frames.append(draw_phase_frame(e, merged=(e >= 0.995)))
    dur.append(70)
frames.append(draw_phase_frame(1.0, merged=True))
dur.append(2800)                       # and let the result sink in

# --- epilogue B: same-seed candidate rows merge into the best one ----------------
ROWS_EPI = [("Candidate at (-12,608, -14,656)", 7.91),
            ("Candidate at (-12,992, -15,040)", 8.30),
            ("Candidate at (-13,632, -14,784)", 7.12)]

def draw_cluster_frame(e, merged=False):
    fr = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(fr)
    dr.text((24, 40), "One Seed, Several Anchors: Nearby Rows Cluster",
            font=font(17), fill=TXT)
    dr.text((24, 68), "Only the best-scoring row of each cluster is written out",
            font=font(13), fill=DIM)
    # Boxes never fade; every label rides its box through the slide, and the
    # losing rows' labels drop only once the merge lands (merged=True). The
    # winning row is drawn last so its text stays on top.
    for k in (0, 2, 1):
        label, sc = ROWS_EPI[k]
        y = 180 + k * 70
        ym = 180 + 70          # the best row's slot
        yy = y + (ym - y) * e
        col = ACCENT if k == 1 else DIM
        dr.rectangle([250, yy, 730, yy + 40], outline=col, width=2)
        if k == 1 or not merged:
            dr.text((262, yy + 11), "%s   score %.2f" % (label, sc), font=font(13),
                    fill=TXT if k == 1 else DIM)
    if merged:
        dr.text((340, 330), "One row per mountainous megaregion", font=font(14), fill=ACCENT)
    return fr

frames.append(draw_cluster_frame(0.0))
dur.append(2200)
for f in range(20):
    t = f / 19
    e = t * t * (3 - 2 * t)
    frames.append(draw_cluster_frame(e, merged=(e >= 0.995)))
    dur.append(70)
frames.append(draw_cluster_frame(1.0, merged=True))
dur.append(4200)                       # closing frame: linger

frames = [with_rail(f, 11) for f in frames]
save_gif(frames, dur, "g10_score.gif")
