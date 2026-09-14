# viz/style.py -- shared palette, fonts, and small drawing helpers.
# Every GIF pulls from this file so the docs read as one explainer.
import os
from PIL import Image, ImageDraw, ImageFont

# GitHub-dark native palette -- sits naturally on a README rendered in dark mode.
BG     = (13, 17, 23)      # canvas
PANEL  = (22, 27, 34)      # cards / overlays
TXT    = (230, 237, 243)
DIM    = (125, 140, 155)
HIT    = (63, 185, 80)     # passes the gate
MISS   = (248, 81, 73)     # fails
ACCENT = (210, 153, 34)    # frontier / highlight
WATER  = (88, 166, 255)    # ocean / informational blue
WHITE  = (245, 245, 250)

OUT_DIR = "viz/out"        # all GIFs land here (relative to the repo root)

def blend(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))

def shade(c, f):
    return tuple(int(round(v * f)) for v in c)

_font_cache = {}
_font_path = None
_font_searched = False

def _find_font():
    candidates = []
    try:  # matplotlib bundles DejaVuSansMono; use it if present (no hard dep)
        import matplotlib
        candidates.append(os.path.join(matplotlib.get_data_path(),
                                       "fonts", "ttf", "DejaVuSansMono.ttf"))
    except Exception:
        pass
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def font(size):
    global _font_path, _font_searched
    if size in _font_cache:
        return _font_cache[size]
    if not _font_searched:
        _font_path = _find_font()
        _font_searched = True
    try:
        _font_cache[size] = ImageFont.truetype(_font_path, size) if _font_path \
                            else ImageFont.load_default()
    except Exception:
        _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]

def wrap_text(dr, xy, text, max_w, size=12, fill=None, leading=None, max_lines=2):
    """Word-wrap `text` into lines no wider than max_w, drawn top-down from xy.
    Returns the y just past the last line. Unlike title(), lines never shrink
    to unreadable sizes: a single overlong word shrinks, then the text wraps."""
    fill = TXT if fill is None else fill
    words = text.split()
    f = font(size)
    longest = max(words, key=len) if words else ""
    while size > 9 and dr.textlength(longest, font=f) > max_w:
        size -= 1
        f = font(size)
    lh = leading if leading is not None else size + 3
    x, y = xy
    lines, cur = [], ""
    for wd in words:
        trial = wd if not cur else cur + " " + wd
        if dr.textlength(trial, font=f) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    overflow = len(lines) > max_lines
    for i, ln in enumerate(lines[:max_lines]):
        if overflow and i == max_lines - 1:
            ln += " ..."
        dr.text((x, y), ln, font=f, fill=fill)
        y += lh
    return y

def title(dr, xy, text, max_w, size=17, fill=None):
    """Draw text at `size`, shrinking one point at a time until it fits
    max_w. Returns the drawn width."""
    fill = TXT if fill is None else fill
    f = font(size)
    while size > 10 and dr.textlength(text, font=f) > max_w:
        size -= 1
        f = font(size)
    dr.text(xy, text, font=f, fill=fill)
    return dr.textlength(text, font=f)

def dashed_circle(dr, cx, cy, r, color, on=16, off=11, width=2):
    a = 0
    while a < 360:
        dr.arc([cx - r, cy - r, cx + r, cy + r], a, min(a + on, 360),
               fill=color, width=width)
        a += on + off

def xmark(dr, x, y, s, color, width=2):
    dr.line([x - s, y - s, x + s, y + s], fill=color, width=width)
    dr.line([x - s, y + s, x + s, y - s], fill=color, width=width)

# Global pacing knob: every clip's frame delays are scaled by TEMPO, so the
# whole set breathes at one rate. 1.0 = as-authored, 1.25 = 25% slower.
# Nudge it here if you want the set slower or snappier overall.
TEMPO = 1.25

def save_gif(frames, durations, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    # GIF delays are centiseconds; keep a 20 ms floor for browser sanity.
    durations = [max(20, int(round(d * TEMPO))) for d in durations]
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    print("wrote %s (%d frames, %.0f KiB)"
          % (path, len(frames), os.path.getsize(path) / 1024))


# ---------------------------------------------------------------------------
# Pipeline rail: the shared "where are we" sidebar. The stage list mirrors the
# real pipeline order -- gpu.cu's kernels, then cpu.cpp's verifier, then
# probe.cpp's enrichment/scoring -- so the clips can never drift from the code.
# Wrap frames with with_rail() right before save_gif.
# ---------------------------------------------------------------------------
RAIL = [
    ("Climate noisemaps",     []),  # g00 preamble
    ("Octave stacks",         []),  # g01
    ("Erosion table init",    []),  # KernelInit: erosion 0A/0B tables only
    ("Anchor lattice setup",  []),  # KernelCoverage geometry      (g02)
    ("Low Erosion check 1",   []),  # KernelCoverage probe         (g03)
    ("Low Erosion check 2",   []),  # KernelCoverage disc scan     (g04)
    ("Late table init",       []),  # KernelLateInit (one beat in g05's intro)
    ("Window checks",         []),  # KernelExtrema (g05 draws its own sub-list)
    ("Flood fill",            []),  # KernelBlob                   (g06)
    ("CPU gates",             ["Dark forest veto",
                               "Low Erosion re-measure",
                               "Height window re-check"]),  # cpu.cpp       (g07/g08)
    ("Pattern windows",       []),  # probe.cpp best-pattern windows (g09)
    ("Composite score",       []),  # probe.cpp compute_score      (g10)
    ("Score distribution",    []),  # funnel + constellation       (g11/g12)
]
RAIL_W = 224   # pixels added to the left of every frame

def with_rail(frame, active, sub=-1, aside=None):
    """Return `frame` with the pipeline rail attached on the left.
    active = index into RAIL (done = green, current = gold with an arrow,
    upcoming = dim). sub = active substep when a stage has them (g07).
    aside = optional note for explainer detours (g05, g08)."""
    w, h = frame.size
    out = Image.new("RGB", (w + RAIL_W, h), BG)
    out.paste(frame, (RAIL_W, 0))
    dr = ImageDraw.Draw(out)
    dr.rectangle([0, 0, RAIL_W - 1, h - 1], fill=PANEL, outline=DIM)
    dr.text((14, 10), "Pipeline", font=font(13), fill=DIM)
    marks = ("✓", "▶", "·", "▸") if _font_path else ("x", ">", ".", "-")
    y = 40
    for i, (name, subs) in enumerate(RAIL):
        if i < active:
            mark, col = marks[0], HIT
        elif i == active:
            mark, col = marks[1], ACCENT
        else:
            mark, col = marks[2], DIM
        dr.text((12, y), mark, font=font(12), fill=col)
        f = font(12)
        for _ in range(3):  # shrink long labels until they fit the rail
            if not hasattr(f, "size") or dr.textlength(name, font=f) <= RAIL_W - 46:
                break
            f = font(max(9, f.size - 1))
        dr.text((28, y), name, font=f,
                fill=col if i == active else (TXT if i < active else DIM))
        # (substage rows are drawn below by the `sub >= 0` branch)
        y += 24
        if i == active and sub >= 0:
            for si, s in enumerate(subs):
                scol = HIT if si < sub else (ACCENT if si == sub else DIM)
                smark = marks[3] if si == sub else (marks[0] if si < sub else marks[2])
                dr.text((28, y), smark, font=font(10), fill=scol)
                dr.text((42, y), s, font=font(10), fill=scol)
                y += 15
    if aside:
        dr.text((12, h - 42), "aside:", font=font(11), fill=DIM)
        dr.text((12, h - 26), aside, font=font(11), fill=DIM)
    return out


# Gate chips for the CPU-verifier clips (g07/g08): three little status pills
# drawn into a footer strip. states: 0 upcoming, 1 running, 2 passed, 3 failed.
GATE_CHIPS = ["Dark forest veto", "Erosion re-measure", "Height re-check"]

def gate_chips(dr, x, y, states):
    for name, st in zip(GATE_CHIPS, states):
        col = (DIM, ACCENT, HIT, MISS)[st]
        wbox = 158
        dr.rectangle([x, y, x + wbox, y + 22], outline=col, width=2)
        mark = {0: " ", 1: "~", 2: ("✓" if _font_path else "x"), 3: "X"}[st]
        dr.text((x + 6, y + 5), mark, font=font(12), fill=col)
        dr.text((x + 22, y + 5), name, font=font(11), fill=col)
        x += wbox + 14