# viz/geom.py
import math
import numpy as np

from style import BG, DIM, WHITE, blend

HEX_D, HEX_ROWS, HEX_COLS = 480, 21, 19            # quarts; gpu.cu KernelCoverage
HEX_ROW_Z = np.round((np.arange(HEX_ROWS) - 10) * HEX_D * math.sqrt(3) / 2).astype(int)

def hex_anchors_blocks():
    """All 399 anchor centers in block coords (row, col tagged)."""
    pts = []
    for r in range(HEX_ROWS):
        for c in range(HEX_COLS):
            xq = (c - (HEX_COLS - 1) // 2) * HEX_D + (HEX_D // 2 if r & 1 else 0)
            pts.append((xq * 4, int(HEX_ROW_Z[r]) * 4, r, c))
    return pts

def phyllotaxis(covers=3, pts_per=32):
    """The 96-point disc table, same construction as make_disc_offsets()."""
    golden = math.pi * (3 - math.sqrt(5))
    pts = np.empty((covers * pts_per, 2))
    for g in range(covers):
        for j in range(pts_per):
            k = g * pts_per + j
            r = math.sqrt((j + 0.5) / pts_per)
            a = k * golden
            pts[k] = (r * math.cos(a), r * math.sin(a))
    return pts


# --- lattice underlays -------------------------------------------------------
# Block-coordinate geometry of the anchor lattice (KernelCoverage): spacing
# 1,920 blocks; diagonal neighbors at (+-960, ~1,663). PROBE_MIDS is the
# 7-point stencil in the kernel's read order (vC, vE, vUR, vUL, vW, vDR, vDL;
# the code's "+z" is screen-DOWN).
PROBE_MIDS = [(0, 0), (960, 0), (480, 832), (-480, 832), (-960, 0),
              (480, -832), (-480, -832)]
FWD_EDGES = [(1920, 0), (960, 1663), (-960, 1663)]  # each lattice edge, once
HEX6 = [(1920, 0), (960, 1663), (-960, 1663),
        (-1920, 0), (-960, -1663), (960, -1663)]

_ANCHOR_XY = {(r, c): (x, z) for x, z, r, c in hex_anchors_blocks()}
ANCHOR_RC = {(x, z): (r, c) for x, z, r, c in hex_anchors_blocks()}

def hex_neighbor_xy(r, c, k):
    """Coords of the k-th neighbor (0 = E, then clockwise) of anchor (r, c),
    or (None, None) if that neighbor is off the lattice. Mirrors
    KernelCoverage's hex connectivity: even rows lean left, odd rows right."""
    if   k == 0: q = (r, c + 1)
    elif k == 1: q = (r + 1, c + 1) if (r & 1) else (r + 1, c)
    elif k == 2: q = (r + 1, c)     if (r & 1) else (r + 1, c - 1)
    elif k == 3: q = (r, c - 1)
    elif k == 4: q = (r - 1, c)     if (r & 1) else (r - 1, c - 1)
    else:        q = (r - 1, c + 1) if (r & 1) else (r - 1, c)
    return _ANCHOR_XY.get(q, (None, None))

def square_lattice_edges(step=1600, half=16000, n=21):
    """Edges of the n x n square anchor grid (blocks), for underlay lines."""
    coords = [-half + step * i for i in range(n)]
    edges = []
    for r in range(n):
        for c in range(n):
            if c + 1 < n:
                edges.append((coords[c], coords[r], coords[c + 1], coords[r]))
            if r + 1 < n:
                edges.append((coords[c], coords[r], coords[c], coords[r + 1]))
    return edges

def hex_lattice_edges():
    """Edges of the hex anchor lattice (blocks): each anchor to its east
    neighbor and its two next-row neighbors -- every edge exactly once.
    (Even rows link to (r+1, c) and (r+1, c-1); odd rows to (r+1, c)
    and (r+1, c+1) -- mirroring KernelCoverage's midpoint bookkeeping.)"""
    pts = {(r, c): (x, z) for x, z, r, c in hex_anchors_blocks()}
    edges = []
    for (r, c), (x, z) in pts.items():
        nbrs = [(r, c + 1)]
        if r + 1 < HEX_ROWS:
            nbrs.append((r + 1, c))
            nbrs.append((r + 1, c + 1) if (r & 1) else (r + 1, c - 1))
        for q in nbrs:
            if q in pts:
                edges.append((x, z, pts[q][0], pts[q][1]))
    return edges

def draw_probe_lattice(dr, px, xlim, ylim, home=None):
    """The shared probe lattice as a dim underlay: anchor rings, lattice
    edges (anchor -> shared midpoint -> neighbor anchor), and the midpoint
    dots themselves. `home` = an anchor whose six-neighbor hexagon gets
    emphasized. px = world->screen mapper; xlim/ylim = map area in px."""
    lc = blend(BG, DIM, 0.4)
    for (ax, az, r, c) in hex_anchors_blocks():
        x0, y0 = px(ax, az)
        if xlim[0] - 12 <= x0 <= xlim[1] + 12 and ylim[0] - 12 <= y0 <= ylim[1] + 12:
            dr.ellipse([x0 - 3, y0 - 3, x0 + 3, y0 + 3], outline=blend(BG, DIM, 0.7))
        for (dx, dz) in FWD_EDGES:
            x1, y1 = px(ax + dx // 2, az + dz // 2)
            x2, y2 = px(ax + dx, az + dz)
            dr.line([x0, y0, x1, y1], fill=lc)
            dr.line([x1, y1, x2, y2], fill=lc)
            dr.ellipse([x1 - 2, y1 - 2, x1 + 2, y1 + 2], fill=blend(BG, WHITE, 0.35))
    if home is not None:
        hx, hz = home
        # Outline only edges whose neighbor anchor actually exists, so rim
        # anchors never grow phantom spokes (the old g03/g04 "gap" bug).
        rc = ANCHOR_RC.get((hx, hz))
        if rc is not None:
            ring = [hex_neighbor_xy(rc[0], rc[1], k) for k in range(6)]
            ring = [px(x, z) for (x, z) in ring if x is not None]
            if len(ring) >= 2:
                dr.line(ring + [ring[0]], fill=DIM, width=2)
        else:
            pts = [px(hx + dx, hz + dz) for (dx, dz) in HEX6]
            dr.line(pts + [pts[0]], fill=DIM, width=2)
