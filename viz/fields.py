# viz/fields.py
import ctypes, math, os
import numpy as np

_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libmrfshim.so"))
_lib.mrf_climate_grid.argtypes = [
    ctypes.c_uint64, ctypes.c_int, ctypes.c_int,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_int32, ctypes.POINTER(ctypes.c_double)]
NP = dict(temp=0, humidity=1, cont=2, erosion=3, weird=5)

def climate_grid(seed, name, nmax, x0q, z0q, nx, nz, stepq=1, cache="viz/cache"):
    """Real climate field as an (nz, nx) float64 array. Cached to .npy."""
    os.makedirs(cache, exist_ok=True)
    key = f"{seed}_{name}{nmax}_{x0q}_{z0q}_{nx}x{nz}_{stepq}.npy"
    path = os.path.join(cache, key)
    if os.path.exists(path):
        return np.load(path)
    out = np.empty(nz * nx, dtype=np.float64)
    _lib.mrf_climate_grid(seed & 0xFFFFFFFFFFFFFFFF, NP[name], nmax,
                          x0q, z0q, nx, nz, stepq,
                          out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    out = out.reshape(nz, nx)
    np.save(path, out)
    return out


# ---------------------------------------------------------------------------
# Terrain / biome access (through the same shim; results cached like fields)
# ---------------------------------------------------------------------------
_lib.mrf_gen_seed.argtypes = [ctypes.c_int, ctypes.c_uint64]
_lib.mrf_height_grid.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                                 ctypes.c_int32, ctypes.c_int32,
                                 ctypes.POINTER(ctypes.c_float)]
_lib.mrf_biome.argtypes = [ctypes.c_int32, ctypes.c_int32]
_lib.mrf_biome.restype = ctypes.c_int

MC_1_21_3 = 27   # cubiomes MCVersion enum order (biomes.h); update if cubiomes changes
_genned = None

def ensure_gen(seed, mc=MC_1_21_3):
    global _genned
    if _genned != seed:
        _lib.mrf_gen_seed(mc, seed & 0xFFFFFFFFFFFFFFFF)
        _genned = seed

def height_grid(seed, x0, z0, nx, nz, step, mc=MC_1_21_3, cache="viz/cache"):
    """mapApproxHeight grid in BLOCKS, (nz, nx) float32. Same call the CPU
    verifier's height gate uses; one-time cost, cached to .npy."""
    os.makedirs(cache, exist_ok=True)
    key = f"h_{seed}_{mc}_{x0}_{z0}_{nx}x{nz}_{step}.npy"
    path = os.path.join(cache, key)
    if os.path.exists(path):
        return np.load(path)
    ensure_gen(seed, mc)
    out = np.empty(nz * nx, dtype=np.float32)
    _lib.mrf_height_grid(x0, z0, nx, nz, step,
                         out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    out = out.reshape(nz, nx)
    np.save(path, out)
    return out

def biome_grid(seed, x0, z0, nx, nz, step, mc=MC_1_21_3, cache="viz/cache"):
    """getBiomeAt(..., y=256) per grid cell (BLOCK coords), (nz, nx) int32."""
    os.makedirs(cache, exist_ok=True)
    key = f"b_{seed}_{mc}_{x0}_{z0}_{nx}x{nz}_{step}.npy"
    path = os.path.join(cache, key)
    if os.path.exists(path):
        return np.load(path)
    ensure_gen(seed, mc)
    out = np.empty((nz, nx), dtype=np.int32)
    for j in range(nz):
        zb = z0 + j * step
        row = out[j]
        for i in range(nx):
            row[i] = _lib.mrf_biome(x0 + i * step, zb)
    np.save(path, out)
    return out

def biome_at(seed, x, z, mc=MC_1_21_3):
    ensure_gen(seed, mc)
    return _lib.mrf_biome(int(x), int(z))


# ---------------------------------------------------------------------------
# Shared blob fill -- the same field, lattice, order and closing as
# probe.cpp's blob_fill() / gpu.cu's KernelBlob (GPU reach +-8,000 blocks;
# recall-verified against the pipeline's own numbers by make_g06).
# ---------------------------------------------------------------------------
BLOB_ERO_MAX, BLOB_CONT_MIN = -0.35, -0.12
BLOB_TEMP_MIN, BLOB_TEMP_MAX = -0.45, 0.20
BLOB_STEP, BLOB_HALF = 64, 125

def blob_cells(seed, ax, az):
    """Connected BLOB_* flood fill from the anchor on the world-aligned
    64-block lattice. Returns (closed_mask[251,251] bool, (bx0, bz0))."""
    from collections import deque
    bx0, bz0 = ax & ~63, az & ~63
    n = 2 * BLOB_HALF + 1
    x0q, z0q = (bx0 - 8000) // 4, (bz0 - 8000) // 4
    ero  = climate_grid(seed, "erosion", 2, x0q, z0q, n, n, BLOB_STEP // 4)
    cont = climate_grid(seed, "cont",    4, x0q, z0q, n, n, BLOB_STEP // 4)
    temp = climate_grid(seed, "temp",    2, x0q, z0q, n, n, BLOB_STEP // 4)
    mask = ((ero <= BLOB_ERO_MAX) & (cont >= BLOB_CONT_MIN)
            & (temp >= BLOB_TEMP_MIN) & (temp <= BLOB_TEMP_MAX))
    c = BLOB_HALF
    seedcell = None
    if mask[c, c]:
        seedcell = (c, c)
    else:
        for r in (1, 2):
            for dj in range(-r, r + 1):
                for di in range(-r, r + 1):
                    if mask[c + dj, c + di]:
                        seedcell = (c + di, c + dj); break
                if seedcell: break
            if seedcell: break
    if seedcell is None:
        return np.zeros((n, n), bool), (bx0, bz0)
    vis = np.zeros((n, n), bool)
    vis[seedcell[1], seedcell[0]] = True
    q = deque([seedcell])
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n and not vis[nj, ni] and mask[nj, ni]:
                vis[nj, ni] = True
                q.append((ni, nj))
    # 1-cell closing, dilate-any then erode-all (probe.cpp parity)
    dil = vis.copy()
    dil[1:, :] |= vis[:-1, :]; dil[:-1, :] |= vis[1:, :]
    dil[:, 1:] |= vis[:, :-1]; dil[:, :-1] |= vis[:, 1:]
    closed = dil.copy()
    closed[0, :] = closed[-1, :] = False
    closed[:, 0] = closed[:, -1] = False
    closed[1:-1, 1:-1] &= (dil[:-2, 1:-1] & dil[2:, 1:-1]
                           & dil[1:-1, :-2] & dil[1:-1, 2:])
    return closed, (bx0, bz0)

def blob_core_radius(closed):
    """Chamfer DT max in lattice cells -- the same two sweeps as probe.cpp."""
    n = closed.shape[0]
    dt = np.where(closed, 1e9, 0.0)
    SQ2 = math.sqrt(2.0)
    for j in range(n):
        row = dt[j]
        for i in range(n):
            v = row[i]
            if v == 0.0: continue
            if i > 0: v = min(v, row[i - 1] + 1.0)
            if j > 0: v = min(v, dt[j - 1][i] + 1.0)
            if i > 0 and j > 0: v = min(v, dt[j - 1][i - 1] + SQ2)
            if i < n - 1 and j > 0: v = min(v, dt[j - 1][i + 1] + SQ2)
            row[i] = v
    for j in range(n - 1, -1, -1):
        row = dt[j]
        for i in range(n - 1, -1, -1):
            v = row[i]
            if v == 0.0: continue
            if i < n - 1: v = min(v, row[i + 1] + 1.0)
            if j < n - 1: v = min(v, dt[j + 1][i] + 1.0)
            if i < n - 1 and j < n - 1: v = min(v, dt[j + 1][i + 1] + SQ2)
            if i > 0 and j < n - 1: v = min(v, dt[j + 1][i - 1] + SQ2)
            row[i] = v
    return int(round(float(dt.max())))


# Surface-biome colors for map renders (1.21.3 ids, cubiomes biomes.h) --
# shared so every clip's biome backdrop uses the same palette.
BIO_COLORS = {
    0: (40, 75, 130), 10: (60, 90, 140), 24: (30, 60, 110),
    44: (45, 90, 150), 45: (45, 90, 150), 46: (45, 90, 150),
    47: (35, 75, 130), 48: (35, 75, 130), 49: (35, 75, 130), 50: (35, 75, 130),
    7: (70, 120, 190), 11: (120, 150, 200),
    1: (140, 165, 85), 129: (170, 190, 90), 177: (150, 170, 95),
    185: (230, 160, 190), 4: (55, 110, 50), 132: (90, 140, 70),
    27: (85, 130, 65), 155: (110, 150, 80), 29: (25, 55, 28),
    5: (50, 100, 85), 30: (90, 120, 110), 32: (45, 90, 75), 160: (55, 100, 85),
    12: (215, 220, 225), 140: (180, 200, 225),
    178: (205, 210, 215), 179: (200, 208, 213),
    180: (190, 190, 200), 181: (225, 230, 240), 182: (150, 150, 155),
    2: (225, 210, 140), 35: (180, 160, 85), 36: (190, 170, 95), 163: (160, 130, 95),
    6: (80, 100, 65), 184: (70, 95, 75), 21: (70, 130, 60), 168: (85, 140, 70),
    37: (200, 115, 75), 165: (210, 125, 80), 38: (190, 105, 70),
    3: (110, 125, 110), 34: (120, 130, 110), 131: (115, 125, 105),
    16: (215, 205, 165), 26: (200, 195, 160), 25: (150, 150, 150),
}