// gpu.cu
// The GPU side of the search. Everything here is 2D climate-noise filtering;
// the expensive 3D work (approximate surface heights, biome lookups) happens
// later, on the CPU verifier threads, and only for the few seeds that make
// it through this pipeline.
//
// Some terminology used throughout this file:
//
//   * Each climate field (erosion, continentalness, temperature, weirdness)
//     is a stack of octaves, and every octave has two variants, A and B,
//     sampled at slightly different frequencies and summed. "0B" means the
//     stack truncated after octave 0 (only 0A + 0B), "1B" includes octave 1
//     (0A + 0B + 1A + 1B), and "full" means every octave. Early stages read
//     mostly truncations, which are much cheaper and accurate enough for
//     filtering; survivors get the full stacks.
//   * All kernel coordinates are in quart units (blocks / 4), the resolution
//     Minecraft itself samples these fields at. Conversion back to blocks
//     happens exactly once, when a candidate is emitted in KernelBlob.
//   * A warp is 32 threads executing in lockstep; the __ballot_sync and
//     __shfl intrinsic family lets a warp exchange values without touching
//     memory.
//
// The stages, in order:
//
//   Stage 1  KernelInit:      Builds the two erosion octave-0 tables (0A and
//                             0B) for a whole batch of seeds, one thread per
//                             table. Those are the only tables stage 2 reads,
//                             so each seed's other 32 tables are deferred to
//                             stage 3, which builds them only for survivors.
//   Stage 2  KernelCoverage:  Checks a hexagonal lattice of 399 anchor points
//                             spread over the search square (+-16,000 blocks
//                             around the origin). Each anchor first gets a
//                             cheap 7-sample erosion probe; anchors that pass
//                             get a 96-sample disc scan requiring roughly 80%
//                             low-erosion coverage. Disc coverage, not any
//                             single extreme value, is what defines a
//                             mountain-forming region here.
//   Stage 3  KernelLateInit:  Builds the remaining 32 noise tables once per
//                             surviving seed (the rest of erosion, all of
//                             continentalness and temperature, and the
//                             truncated weirdness), one warp per anchor with
//                             the 32 tables spread across the 32 lanes.
//   Stage 4  KernelExtrema:   Six windowed checks around each surviving
//                             anchor, cheapest and most selective first:
//                             temperature coverage, continentalness disc
//                             coverage, weirdness ridge texture, the erosion
//                             minimum, the continentalness maximum, and an
//                             approximate-height gate. Any check can kick out
//                             the anchor early, so the expensive full-stack
//                             scans only run on whichever seeds made it
//                             through all six checks.
//   Stage 5  KernelBlob:      The megaregion measurement. One thread block
//                             per surviving anchor flood-fills the connected
//                             low-erosion, inland, temperate field on a
//                             64-block lattice out to +-8,000 blocks, with an
//                             early exit the moment the connected area
//                             crosses a target size (g_blobcfg.target_cells,
//                             pushed from main() as 0.88x the CPU gate). The
//                             GPU works in float32 as a prefilter; the CPU
//                             re-measures the same field in double precision
//                             against the real gate. Each passing anchor
//                             emits one candidate, at most PER_SEED_CAP per
//                             seed per batch.

#include "random.h"
#include "noise_common.h"
#include "common.h"
#include "gpu.h"

#include "biomenoise.h" // initBiomeNoise() spline tree

#include <cfloat>
#include <memory>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>


#define PANIC(...)                      \
  {                                     \
    std::fprintf(stderr, __VA_ARGS__);  \
    std::abort();                       \
  }

#define TRY_CUDA(expr) try_cuda(expr, __FILE__, __LINE__)

static void try_cuda(cudaError_t error, const char *file, uint64_t line) {
    if (error == cudaSuccess) return;

    PANIC("CUDA error at %s:%" PRIu64 ": %s (%d)\n", file, line, cudaGetErrorString(error), error);
}

// A per-stage timing and funnel table is printed every STATS_INTERVAL
// batches (about a second of work at typical speeds), plus a final partial
// table when the pipeline drains, so short --seeds runs still get one.
#ifndef STATS_INTERVAL
#define STATS_INTERVAL 128
#endif

// ---------------------------------------------------------------------------
// GPU-side copies of the shared climate configs. They are built from the
// same expressions as the host copies in noise_common.h, so the GPU and CPU
// can never disagree about how any octave stack is scaled.
// ---------------------------------------------------------------------------
__device__ constexpr ClimateConfig<4> D_EROSION_CFG         = make_climate_config<5, 4>(-9,  EROSION_AMPS,         HASH_EROSION);
__device__ constexpr ClimateConfig<9> D_CONTINENTALNESS_CFG = make_climate_config<9, 9>(-9,  CONTINENTALNESS_AMPS, HASH_CONTINENTALNESS);
__device__ constexpr ClimateConfig<2> D_TEMPERATURE_CFG     = make_climate_config<6, 2>(-10, TEMPERATURE_AMPS,     HASH_TEMPERATURE);
__device__ constexpr ClimateConfig<3> D_WEIRDNESS_CFG       = make_climate_config<6, 3>(-7,  WEIRDNESS_AMPS,       HASH_WEIRDNESS);

__device__ GradDotTable g_grad_dot_table;
__device__ DiscOffsets  g_disc;

// The blob-field thresholds and the flood fill's early-exit target, pushed to
// the GPU from main() before any worker thread starts. The GPU measures in
// float32 as a prefilter and must always be the looser side, so main() sets
// the target about 12% below the CPU's double-precision area gate
// (target_cells ~ 0.88 * area / 4096). A borderline region then always
// reaches the CPU, which has the final say.
struct DeviceBlobCfg {
    float ero_max, cont_min, temp_min, temp_max;
    uint32_t target_cells;
};
__device__ DeviceBlobCfg g_blobcfg;
static DeviceBlobCfg h_blobcfg = {
    BLOB_ERO_MAX, BLOB_CONT_MIN, BLOB_TEMP_MIN, BLOB_TEMP_MAX, TARGET_BLOB_CELLS
};
static int g_mc_host = MC_1_21_3;
void gpu_set_blob_cfg(float ero_max, float cont_min, float temp_min, float temp_max,
                      uint32_t target_cells) {
    h_blobcfg.ero_max = ero_max; h_blobcfg.cont_min = cont_min;
    h_blobcfg.temp_min = temp_min; h_blobcfg.temp_max = temp_max;
    h_blobcfg.target_cells = target_cells;
}
void gpu_set_mc(int mc_version) { g_mc_host = mc_version; }

static void upload_grad_dot_table() {
    const GradDotTable t = GRAD_DOT_TABLE;
    TRY_CUDA(cudaMemcpyToSymbol(g_grad_dot_table, &t, sizeof(t)));
}

static void upload_disc_offsets() {
    const DiscOffsets d = make_disc_offsets();
    TRY_CUDA(cudaMemcpyToSymbol(g_disc, &d, sizeof(d)));
}

// ---------------------------------------------------------------------------
// Climate thresholds for the core checks, in raw noise units (the values
// Cubiomes Viewer displays divided by 10,000). If the candidate rate ever
// drops too low, CORE_COV_ERO_MAX and CORE_ERO_MIN_REQ are the first knobs
// to relax. The flood fill's own field thresholds (BLOB_*) live in common.h,
// so the GPU and the CPU verifier always sample the same field.
// ---------------------------------------------------------------------------
constexpr float CORE_COV_ERO_MAX  = -0.40f;  // disc scan: 0B erosion ceiling
constexpr float CORE_ERO_MIN_REQ  = -1.15f;  // window scan: full-stack erosion minimum
constexpr float CORE_COV_CONT_MIN =  0.00f;  // disc scan: 1B continentalness floor (inland)
constexpr float CORE_CONT_MAX_REQ =  0.70f;  // window scan: full-stack continentalness max

// Core temperature window: at least 65% of a 1000x1000-block window must sit
// inside [-0.40, +0.20] (full stack). Deliberately a little tighter than the
// flood fill's temperature limits, so a region's core is solidly temperate
// even where the periphery is allowed to relax.
constexpr float CORE_TEMP_MIN     = -0.40f;
constexpr float CORE_TEMP_MAX     =  0.20f;

// The stage-2 pre-probe: a cheap vote that decides whether an anchor is
// worth the 96-sample disc scan at all. Each anchor reads seven shared probe
// values (see KernelCoverage): its own center plus the six points halfway to
// its neighbors, 960 blocks out. Octave-0 erosion is smooth at this scale
// (its features are wider than the 3,200-block disc), so six out of seven
// samples below PROBE_GATE is a good forecast of the disc passing its ~80%
// coverage requirement. The gate sits looser than the disc's own -0.40
// ceiling on purpose: one unlucky sample should not disqualify a borderline
// area. The midpoints also happen to sit near the disc's area-weighted mean
// radius (2R/3 ~ 1067 blocks), which makes them better disc predictors than
// rim points.
// If a recall run ever loses known-good seeds, loosen in this order:
// PROBE_NEED to 5, then 4, then PROBE_GATE from -0.30 to -0.25.
constexpr float PROBE_GATE = -0.30f;
constexpr int   PROBE_NEED = 6;   // required hits out of 7 probe points

// The search square (+-SEARCH_RADIUS_Q) and the default megaregion size gate
// (TARGET_BLOB_CELLS) live in common.h; the anchor lattice geometry is in
// KernelCoverage below.

// ---------------------------------------------------------------------------
// Buffer utilities
// ---------------------------------------------------------------------------
struct DeviceBuffer {
    void *data;
    size_t size;
    explicit DeviceBuffer(size_t size) : size(size) { TRY_CUDA(cudaMalloc(&data, size)); }
    ~DeviceBuffer() { TRY_CUDA(cudaFree(data)); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};

template <typename T> struct OutputBuffer {
    T *data;
    uint32_t *len;
    uint32_t max_len;
    OutputBuffer(T *data, uint32_t *len, uint32_t max_len) : data(data), len(len), max_len(max_len) {}
    OutputBuffer(const DeviceBuffer &buffer, uint32_t *len)
        : data((T *)buffer.data), len(len), max_len((uint32_t)(buffer.size / sizeof(T))) {}
    OutputBuffer(const OutputBuffer<T> &other)
        : data(other.data), len(other.len), max_len(other.max_len) {}
};

// Overflow convention: producers keep incrementing len past max_len, so the
// CPU can detect and warn about a full buffer, and consumers must therefore
// clamp with size(). Reading up to the raw len would run out of bounds.
template <typename T> struct InputBuffer {
    const T *data;
    const uint32_t *len;
    uint32_t max_len;
    InputBuffer(const T *data, const uint32_t *len, uint32_t max_len)
        : data(data), len(len), max_len(max_len) {}
    InputBuffer(const OutputBuffer<T> &buffer)
        : data(buffer.data), len(buffer.len), max_len(buffer.max_len) {}
    InputBuffer(const DeviceBuffer &buffer, const uint32_t *len)
        : data((const T *)buffer.data), len(len), max_len((uint32_t)(buffer.size / sizeof(T))) {}

    __device__ uint32_t size() const { return min(*len, max_len); }
};

// Strided copy of the Perlin gradient table (16 directions x 3 axes) into
// shared memory. Works for any block size; the 48 words are split across
// however many threads the block has.
__device__ inline void load_grad_table_shared(GradDotTable &s_grad) {
    for (uint32_t i = threadIdx.x; i < sizeof(GradDotTable) / sizeof(uint32_t); i += blockDim.x)
        reinterpret_cast<uint32_t *>(&s_grad)[i] =
            reinterpret_cast<const uint32_t *>(&g_grad_dot_table)[i];
    __syncthreads();
}

// Strided copy of one noise table into shared memory. The caller must
// __syncthreads() before the table is sampled.
__device__ inline void load_noise_shared(ImprovedNoise &dst, const ImprovedNoise &src) {
    for (uint32_t i = threadIdx.x; i < sizeof(ImprovedNoise) / sizeof(uint32_t); i += blockDim.x)
        reinterpret_cast<uint32_t *>(&dst)[i] =
            reinterpret_cast<const uint32_t *>(&src)[i];
}

// Sum a value across all 32 lanes of a warp and return the total on every
// lane (XOR butterfly), so the result can safely drive warp-uniform control
// flow. A shfl_down reduction would leave the total on lane 0 only, and
// lanes branching on different partial values around a later __ballot_sync
// would be undefined behavior.
__device__ inline uint32_t warp_reduce_add(uint32_t v) {
    v += __shfl_xor_sync(~0u, v, 16);
    v += __shfl_xor_sync(~0u, v, 8);
    v += __shfl_xor_sync(~0u, v, 4);
    v += __shfl_xor_sync(~0u, v, 2);
    v += __shfl_xor_sync(~0u, v, 1);
    return v;
}

// init_climate_table with the shuffled permutation staged through a
// caller-provided shared-memory scratch row (65 words: 64 of permutation
// plus one of padding, which keeps concurrent rows on distinct shared-memory
// banks). The finished table is written to global memory as whole words
// instead of 256 single-byte stores. The RNG chain and swap order are
// unchanged, so the result is bit-identical to the plain version.
template <size_t M>
__device__ inline void init_table_scratch(ImprovedNoise &out, uint64_t seed,
        const ClimateConfig<M> &cfg, uint32_t i, bool b_side, uint8_t *scratch) {
    const XrsrRandomFork base = xrsr_seed_fork(seed);
    XrsrRandom rng = base.from(cfg.fork_hash);
    XrsrRandomFork fa, fb;
    xrsr_double_fork(rng, fa, fb);
    XrsrRandom trng = (b_side ? fb : fa).from((b_side ? cfg.b[i] : cfg.a[i]).fork_hash);

    out.xo = trng.nextFloat() * 256.0f;
    out.yo = trng.nextFloat() * 256.0f;
    out.zo = trng.nextFloat() * 256.0f;
    out.pad = 0.0f;

    for (uint32_t k = 0; k < 256; k++) scratch[k] = (uint8_t)k;
    for (uint32_t k = 0; k < 256; k++) {
        const uint32_t j = trng.nextInt(256u - k);
        const uint8_t b = scratch[k];
        scratch[k] = scratch[k + j];
        scratch[k + j] = b;
    }
    uint32_t *dst = reinterpret_cast<uint32_t *>(out.p);
    const uint32_t *src = reinterpret_cast<const uint32_t *>(scratch);
    #pragma unroll 16
    for (uint32_t k = 0; k < 64; k++) dst[k] = src[k];
}

// ---------------------------------------------------------------------------
// Per-stage statistics. CUDA events bracket each kernel launch, and after a
// batch syncs, the stage timings and funnel counts accumulate into StageStat
// rows that print every STATS_INTERVAL batches.
// ---------------------------------------------------------------------------
struct CudaEvent {
    cudaEvent_t ev = nullptr;
    CudaEvent() { TRY_CUDA(cudaEventCreate(&ev)); }
    ~CudaEvent() { if (ev) cudaEventDestroy(ev); }
    CudaEvent(const CudaEvent &) = delete;
    CudaEvent &operator=(const CudaEvent &) = delete;
    void record(cudaStream_t s) { TRY_CUDA(cudaEventRecord(ev, s)); }
    float ms_since(const CudaEvent &prev) const {
        float ms = 0.0f;
        TRY_CUDA(cudaEventElapsedTime(&ms, prev.ev, ev));
        return ms;
    }
};

struct StageStat {
    const char *name;    // stage label
    const char *in_unit; // what the input count measures (for the printout)
    double time_s = 0.0;
    uint64_t in  = 0;
    uint64_t out = 0;
};

static void fmt_si(uint64_t v, char *buf, size_t n) {
    static const char *units[] = { "", "k", "M", "G", "T" };
    double d = (double)v;
    size_t u = 0;
    while (d >= 1000.0 && u < 4) { d /= 1000.0; u++; }
    std::snprintf(buf, n, "%.2f%s", d, units[u]);
}

// Funnel counters mirrored from the device (per-batch BatchBuf::counts).
static void print_funnel_stats(int device, const uint64_t h_counts[4]) {
    if (h_counts[0] == 0) return;
    std::printf("  [gpu%d]   funnel: %" PRIu64 " probed -> %" PRIu64 " probe7 (surv %.2f%%) -> %" PRIu64 " emitted (surv %.2f%%)\n",
            device, h_counts[0], h_counts[1],
            100.0 * (double)h_counts[1] / (double)h_counts[0],
            h_counts[2],
            h_counts[1] ? 100.0 * (double)h_counts[2] / (double)h_counts[1] : 0.0);
}

static void print_stage_stats(int device, uint64_t batches, double wall_s,
        uint64_t seeds, const std::vector<StageStat> &stages) {
    double busy_ms = 0.0;
    for (const auto &s : stages) busy_ms += s.time_s * 1e3;

    char a[32], b[32];
    fmt_si(seeds, a, sizeof(a));
    fmt_si((uint64_t)(seeds / (wall_s > 0 ? wall_s : 1e-9)), b, sizeof(b));
    std::printf("\n[gpu%d] --- pipeline stats: %" PRIu64 " batches, %s seeds in %.1fs (%s seeds/s) ---\n",
            device, batches, a, wall_s, b);

    for (const auto &s : stages) {
        fmt_si(s.in, a, sizeof(a));
        fmt_si(s.out, b, sizeof(b));
        std::printf("  %-10s %9.1f ms (%5.1f%%) | %s %s -> %s",
                s.name, s.time_s * 1e3,
                busy_ms > 0.0 ? s.time_s * 1e3 / busy_ms * 100.0 : 0.0,
                a, s.in_unit, b);
        if (s.out > 0 && s.in > s.out)
            std::printf(" | 1 in %.3g", (double)s.in / (double)s.out);
        std::printf("\n");
    }
    std::printf("  %-10s %9.1f ms\n", "TOTAL", busy_ms);
    std::fflush(stdout);
}

// ===========================================================================
// Stage 1: build the erosion octave-0 tables (0A and 0B) for an entire batch
// of seeds. These are the only noise tables stage 2 ever reads, so this is
// all the initialization most seeds will ever get: one thread per table, two
// threads per seed. The remaining 32 tables per seed (the rest of erosion,
// continentalness, temperature, and the truncated weirdness pair) are built
// later by KernelLateInit, and only for the small share of seeds that
// survive stage 2 (roughly 1.5 surviving anchors per 100 seeds).
//
// Memory: the batch array is 131072 seeds * 9248 bytes = 1156 MiB, allocated
// in full, twice over while double-buffering is active. Only the erosion
// octave-0 slots are populated here.
// ===========================================================================
namespace KernelInit {
constexpr uint32_t SEEDS_PER_RUN      = 1u << 17; // 131072 seeds per batch
constexpr uint32_t TABLES_PER_SEED    = 2;        // the two octave-0 tables (0A, 0B)
constexpr uint32_t THREADS_PER_BLOCK  = 128;      // 1 thread per table -> 128 tables per block

// Each permutation table starts as the identity and is shuffled with 256
// Fisher-Yates steps, every step drawing from that octave's RNG stream. The
// shuffle is inherently serial, so the win here is in where the window
// lives. In per-thread local memory the whole warp accesses the same shuffle
// index at once, and because each window is 256 bytes, those accesses land
// 256 bytes apart: one distinct 128-byte sector per lane, about 32
// serialized sector reads per warp access. Staging the window in shared
// memory with one padding word per row (a 65-word stride) spreads the lanes
// over all 32 shared-memory banks instead, and the finished table is written
// out as coalesced 4-byte words. The RNG draws and swap order are untouched,
// so the tables are bit-identical to the straightforward version. (A
// warp-cooperative variant that shuffled via lane exchanges measured no
// faster; the serial RNG chain is the bottleneck either way.)
__global__ __launch_bounds__(THREADS_PER_BLOCK) void kernel(
        InputBuffer<uint64_t> seeds, MountainNoiseResult *__restrict__ results) {
    constexpr uint32_t STRIDE = 65; // 64 words of permutation + 1 word pad
    __shared__ uint32_t s_perm[THREADS_PER_BLOCK * STRIDE]; // 32.5 KiB

    const uint32_t n_tables = seeds.size() * TABLES_PER_SEED;
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n_tables) {
        const uint32_t seed_index = index >> 1; // / TABLES_PER_SEED
        const bool     b_side = (index & 1u) != 0;
        const uint64_t seed = seeds.data[seed_index];

        // Same fork chain as init_climate_table(..., D_EROSION_CFG, 0, b_side).
        const XrsrRandomFork base = xrsr_seed_fork(seed);
        XrsrRandom rng = base.from(D_EROSION_CFG.fork_hash);
        XrsrRandomFork fa, fb;
        xrsr_double_fork(rng, fa, fb);
        XrsrRandom trng = (b_side ? fb : fa).from(b_side ? D_EROSION_CFG.b[0].fork_hash
                                                         : D_EROSION_CFG.a[0].fork_hash);

        const float xo = trng.nextFloat() * 256.0f;
        const float yo = trng.nextFloat() * 256.0f;
        const float zo = trng.nextFloat() * 256.0f;

        uint8_t *p = reinterpret_cast<uint8_t *>(&s_perm[threadIdx.x * STRIDE]);
        for (uint32_t i = 0; i < 256; i++) p[i] = (uint8_t)i;
        for (uint32_t i = 0; i < 256; i++) {
            const uint32_t j = trng.nextInt(256u - i);
            const uint8_t b = p[i];
            p[i] = p[i + j];
            p[i + j] = b;
        }

        ImprovedNoise &out = b_side ? results[seed_index].erosion_b[0]
                                    : results[seed_index].erosion_a[0];
        out.xo = xo; out.yo = yo; out.zo = zo; out.pad = 0.0f;
    }
    __syncthreads();

    // Coalesced write-out of the permutation words: consecutive k cover
    // consecutive (table, word) pairs, so each warp instruction writes a
    // contiguous 128-byte run.
    for (uint32_t k = threadIdx.x; k < THREADS_PER_BLOCK * 64; k += THREADS_PER_BLOCK) {
        const uint32_t gi = blockIdx.x * THREADS_PER_BLOCK + (k >> 6);
        if (gi >= n_tables) break;
        ImprovedNoise &out = (gi & 1u) ? results[gi >> 1].erosion_b[0]
                                       : results[gi >> 1].erosion_a[0];
        reinterpret_cast<uint32_t *>(out.p)[k & 63] = s_perm[(k >> 6) * STRIDE + (k & 63)];
    }
}
} // namespace KernelInit

// ===========================================================================
// Stage 2: core disc coverage.
//
// ANCHOR LATTICE. Sampling every point of the +-16,000-block search square
// is not feasible, so the search tests a fixed lattice of anchor points and
// takes a closer look wherever an anchor reports low erosion. The lattice is
// hexagonal: for a given nearest-neighbor spacing it covers the plane with
// fewer anchors than a square grid, and with a smaller worst-case distance
// from any point to the nearest anchor. At 1,920-block spacing that worst
// case is 1108.5 blocks (d / sqrt(3)); a square grid with 1,600-block steps
// would sit at 1131.4 blocks (800 * sqrt(2)) while needing about a third
// more anchors per unit area (hex density ratio 4/(3*sqrt(3)) ~ 0.77). The
// layout is 21 rows x 19 columns = 399 anchors, with alternating rows
// shifted by half a spacing and some padding past the +-4,000-quart search
// square so the rim is covered as well as the interior (x out to +-4,560,
// z to +-4,157, in quarts):
//   row r:  z = HEX_ROW_Z[r] = round((r-10) * 480 * sqrt(3)/2)
//   col c:  x = (c-9)*480 + (r odd ? 240 : 0)
// To change the spacing, adjust HEX_D and regenerate HEX_ROW_Z with the same
// formula, keeping the lattice padded past the search square.
//
// SHARED PROBE LATTICE. The 7-point probe described above the PROBE_GATE
// definition would cost 7 noise evaluations per anchor if every anchor
// computed its own. Instead, the kernel samples a shared lattice once per
// seed: every anchor center plus the midpoint of every hex edge. Each edge
// midpoint serves exactly the two anchors it lies between, so the whole
// pre-probe costs 4 * N_ANCHORS noise evaluations per seed, and probing one
// anchor is then seven shared-memory reads. s_pts layout (all erosion 0B):
//   [0, NA)      anchor centers
//   [NA, 2NA)    east edge midpoints
//   [2NA, 3NA)   up-right edge midpoints
//   [3NA, 4NA)   up-left edge midpoints
// The three backward directions (west, down-left, down-right) are the
// forward edges of the anchors behind, so they need no samples of their own.
// Anchors on the lattice rim lack some neighbors and reuse anchor-local
// values for the missing midpoints, which only softens probes anchored
// outside the search square.
//
// DISC SCAN. Anchors that pass the probe get the real test: 96 samples
// arranged in a phyllotaxis (golden-angle) spiral over a disc of radius
// 1,600 blocks, counted in three fail-fast batches of 32. The anchor passes
// with at least 77 samples (~80%) reading erosion 0B at or below -0.40, and
// it emits the centroid of its passing samples, so later checks center on
// the low-erosion heart of the area rather than the lattice point that
// happened to detect it. One warp per surviving anchor.
// ===========================================================================
namespace KernelCoverage {
constexpr uint32_t THREADS_PER_BLOCK = 256;
constexpr uint32_t WARPS_PER_BLOCK   = THREADS_PER_BLOCK / 32;

// Hex anchor lattice geometry (quarts; 480 quarts = 1920 blocks).
constexpr int32_t HEX_D    = 480;
constexpr int32_t HEX_ROWS = 21;
constexpr int32_t HEX_COLS = 19;
// z_r = round((r - 10) * HEX_D * sqrt(3)/2), precomputed exactly:
__device__ constexpr int32_t HEX_ROW_Z[HEX_ROWS] = {
    -4157, -3741, -3326, -2910, -2494, -2078, -1663, -1247, -831, -416,
        0,   416,   831,  1247,  1663,  2078,  2494,  2910,  3326,  3741,
     4157
};
constexpr uint32_t N_ANCHORS = (uint32_t)(HEX_ROWS * HEX_COLS); // 399

constexpr int32_t  DISC_R_Q    = CORE_BLOB_RADIUS / 4;  // disc radius in quarts (400)
constexpr int      COV_N       = CORE_COV_SAMPLES;      // 96: three phyllotaxis batches of 32
constexpr int      COV_NEED    = CORE_COV_NEED;         // 77: ~80% must pass
static_assert(N_ANCHORS == 399, "hex lattice coverage");
static_assert(COV_N % DISC_COVER_POINTS == 0, "coverage batches line up with disc covers");

struct AnchorHit {
    uint32_t seed_index;
    int32_t xq, zq; // QUART coordinates, centroid-adjusted
};

// Per-batch funnel counters live in a device buffer passed by the host
// (BatchBuf::counts), rather than a __device__ global. With two streams in flight,
// resetting a global would race with the other stream's in-flight coverage
// kernel and silently drop counts.
// [0] = anchors probed, [1] = survived the shared 7-point probe,
// [2] = emitted (passed the disc coverage).

// Anchor coordinates from lattice index (quarts).
__device__ inline void hex_anchor_xz(uint32_t a, int32_t &xq, int32_t &zq) {
    const uint32_t r = a / (uint32_t)HEX_COLS;
    const uint32_t c = a % (uint32_t)HEX_COLS;
    xq = ((int32_t)c - (HEX_COLS - 1) / 2) * HEX_D + ((r & 1u) ? HEX_D / 2 : 0);
    zq = HEX_ROW_Z[r];
}

__global__ __launch_bounds__(THREADS_PER_BLOCK) void kernel(
        InputBuffer<uint64_t> seeds,
        OutputBuffer<AnchorHit> outputs,
        const MountainNoiseResult *__restrict__ results,
        uint32_t *__restrict__ stage_counts) {
    __shared__ GradDotTable s_grad;
    __shared__ ImprovedNoise s_ero0A, s_ero0B;
    __shared__ float s_pts[4 * N_ANCHORS]; // shared probe lattice (see above)

    const uint32_t seed_index = blockIdx.x;
    // The whole block exits together here, so the __syncthreads() below
    // cannot deadlock.
    if (seed_index >= seeds.size()) return;
    const MountainNoiseResult &noise = results[seed_index];

    load_grad_table_shared(s_grad); // includes __syncthreads()
    load_noise_shared(s_ero0A, noise.erosion_a[0]);
    load_noise_shared(s_ero0B, noise.erosion_b[0]);
    __syncthreads();

    // --- Fill the shared probe lattice: 4 * N_ANCHORS independent noise
    // evaluations, the only ones the vast majority of seeds will ever get.
    for (uint32_t j = threadIdx.x; j < 4 * N_ANCHORS; j += THREADS_PER_BLOCK) {
        const uint32_t kind = j / N_ANCHORS;      // 0 center, 1 E, 2 UR, 3 UL
        const uint32_t a    = j - kind * N_ANCHORS;
        const uint32_t r    = a / (uint32_t)HEX_COLS;
        int32_t xq, zq;
        hex_anchor_xz(a, xq, zq);
        if (kind == 1) {                          // E-edge midpoint: (+240, 0)
            xq += HEX_D / 2;
        } else if (kind >= 2) {                   // diagonal-edge midpoint: (+-120, up/2)
            const uint32_t rup = (r + 1u < (uint32_t)HEX_ROWS) ? r + 1u : r;
            zq = (zq + HEX_ROW_Z[rup] + 1) >> 1;  // midpoint, rounded half-up so both anchors agree
            xq += (kind == 2) ? (HEX_D / 4) : -(HEX_D / 4);
        }
        s_pts[j] = sample_climate_trunc(s_grad, &s_ero0A, &s_ero0B, D_EROSION_CFG,
                                        1, 1, xq, zq);
    }
    __syncthreads();

    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;

    // Preload this lane's disc offsets: lane L carries sample L of each
    // 32-sample batch, so the table loads stay coalesced.
    int32_t ddx[CORE_COVERS], ddz[CORE_COVERS];
    #pragma unroll
    for (int b = 0; b < CORE_COVERS; b++) {
        const int k = b * DISC_COVER_POINTS + (int)lane;
        ddx[b] = __float2int_rn((float)g_disc.x[k] * (DISC_R_Q / 32767.0f));
        ddz[b] = __float2int_rn((float)g_disc.z[k] * (DISC_R_Q / 32767.0f));
    }

    // --- Probe phase: one anchor per lane and seven shared-memory lookups
    // each, with no noise evaluation at all. The warp then disc-scans each
    // surviving anchor one at a time.
    for (uint32_t base = warp * 32; base < N_ANCHORS; base += WARPS_PER_BLOCK * 32) {
        const uint32_t a = base + lane;
        bool probe_ok = false;
        if (a < N_ANCHORS) {
            const uint32_t r = a / (uint32_t)HEX_COLS;
            const uint32_t c = a % (uint32_t)HEX_COLS;
            const float vC  = s_pts[a];                 // center
            const float vE  = s_pts[N_ANCHORS + a];     // E midpoint (mine)
            const float vUR = s_pts[2 * N_ANCHORS + a]; // up-right midpoint (mine)
            const float vUL = s_pts[3 * N_ANCHORS + a]; // up-left midpoint (mine)
            const float vW  = s_pts[N_ANCHORS + (c > 0 ? a - 1 : a)]; // E midpoint of the west anchor
            // The two downward midpoints are stored as the up-midpoints of
            // the anchors one row back; which columns they sit in flips with
            // the row's parity. Row 0 has no row behind it and reuses its
            // own values (rim anchors only, all outside the search square).
            const uint32_t rd = (r > 0) ? (a - (uint32_t)HEX_COLS) : a;
            float vDR, vDL;
            if ((r & 1u) == 0) {
                vDR = s_pts[3 * N_ANCHORS + rd];                            // UL of (r-1, c)
                vDL = s_pts[2 * N_ANCHORS + (c > 0 ? rd - 1 : rd)];         // UR of (r-1, c-1)
            } else {
                vDL = s_pts[2 * N_ANCHORS + rd];                            // UR of (r-1, c)
                vDR = s_pts[3 * N_ANCHORS + (c + 1 < (uint32_t)HEX_COLS ? rd + 1 : rd)]; // UL of (r-1, c+1)
            }
            const int hits = (int)(vC  <= PROBE_GATE) + (int)(vE  <= PROBE_GATE)
                           + (int)(vUR <= PROBE_GATE) + (int)(vUL <= PROBE_GATE)
                           + (int)(vW  <= PROBE_GATE) + (int)(vDR <= PROBE_GATE)
                           + (int)(vDL <= PROBE_GATE);
            probe_ok = (hits >= PROBE_NEED);
        }
        uint32_t pass = __ballot_sync(~0u, probe_ok);
        if (lane == 0) {
            atomicAdd(&stage_counts[0], min(32u, N_ANCHORS - base));        // probed
            if (pass) atomicAdd(&stage_counts[1], (uint32_t)__popc(pass));  // probe-passed
        }

        // --- Disc coverage: one surviving anchor at a time, the whole warp on it. ---
        while (pass) {
            const uint32_t b = (uint32_t)(__ffs((int)pass) - 1);
            pass &= pass - 1u;
            const uint32_t anchor = base + b;
            int32_t axq, azq;
            hex_anchor_xz(anchor, axq, azq);

            int hits = 0, sdx = 0, sdz = 0;
            bool alive = true;
            #pragma unroll
            for (int b2 = 0; b2 < CORE_COVERS; b2++) {
                const float e = sample_climate_trunc(s_grad, &s_ero0A, &s_ero0B, D_EROSION_CFG,
                                                     1, 1, axq + ddx[b2], azq + ddz[b2]);
                const bool hit = (e <= CORE_COV_ERO_MAX);
                hits += __popc(__ballot_sync(~0u, hit));
                if (hit) { sdx += ddx[b2]; sdz += ddz[b2]; }
                // hits is warp-uniform (ballot), so these branches never diverge:
                if (hits + (COV_N - DISC_COVER_POINTS * (b2 + 1)) < COV_NEED) { alive = false; break; }
                if (hits >= COV_NEED) break;
            }
            if (!alive || hits < COV_NEED) continue;

            // Centroid of the passing samples (warp reduction; only lane 0
            // needs the sums).
            for (int o = 16; o; o >>= 1) {
                sdx += __shfl_down_sync(~0u, sdx, o);
                sdz += __shfl_down_sync(~0u, sdz, o);
            }
            if (lane == 0) {
                atomicAdd(&stage_counts[2], 1u); // emitted
                const uint32_t out = atomicAdd(outputs.len, 1);
                if (out < outputs.max_len)
                    outputs.data[out] = { seed_index, axq + sdx / hits, azq + sdz / hits };
            }
        }
    }
}
} // namespace KernelCoverage

// ===========================================================================
// Stage 3: build the remaining noise tables for seeds that survived stage 2:
// the rest of erosion (6 tables), all of continentalness (18) and
// temperature (4), and the truncated weirdness pair (4). One warp takes one
// surviving anchor; lane 0 claims the anchor's seed for this batch with a
// compare-and-swap on init_flags, then the 32 lanes build the 32 tables in
// parallel. Anchors whose seed was already claimed by any warp skip, so each
// seed is initialized exactly once per batch. Both kernels run in the same
// stream, so the tables are guaranteed to be visible to the next kernel.
//
// Lane assignments:  0-8    continentalness A
//                    9-17   continentalness B
//                    18-19  temperature A
//                    20-21  temperature B
//                    22-24  erosion A[1..3]
//                    25-27  erosion B[1..3]
//                    28-29  weirdness A[0..1]
//                    30-31  weirdness B[0..1]
// ===========================================================================
namespace KernelLateInit {
constexpr uint32_t THREADS_PER_BLOCK = 128; // 4 warps per block

// The same shared-memory shuffle scratch as KernelInit: one padded row per
// thread (64 words of permutation plus one pad word, a 65-word stride), so
// a warp's 32 lanes land on 32 different shared-memory banks at every
// shuffle step instead of serializing on one. The tables come out
// bit-identical to the straightforward path.
__global__ __launch_bounds__(THREADS_PER_BLOCK) void kernel(
        InputBuffer<uint64_t> seeds,
        InputBuffer<KernelCoverage::AnchorHit> anchors,
        MountainNoiseResult *__restrict__ results,
        uint32_t *__restrict__ init_flags,
        uint32_t *__restrict__ init_count) {
    constexpr uint32_t STRIDE = 65;
    __shared__ uint32_t s_scratch[THREADS_PER_BLOCK * STRIDE]; // 32.5 KiB

    const uint32_t lane        = threadIdx.x & 31u;
    const uint32_t warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t num_warps   = (gridDim.x * blockDim.x) >> 5;
    const uint32_t anchors_len = anchors.size();
    uint8_t *scratch = reinterpret_cast<uint8_t *>(&s_scratch[threadIdx.x * STRIDE]);

    for (uint32_t i = warp_global; i < anchors_len; i += num_warps) {
        const uint32_t seed_index = anchors.data[i].seed_index;

        uint32_t mine = 0;
        if (lane == 0)
            mine = (atomicCAS(&init_flags[seed_index], 0u, 1u) == 0u);
        mine = __shfl_sync(~0u, mine, 0);
        if (!mine) continue;

        const uint64_t seed = seeds.data[seed_index];
        MountainNoiseResult &r = results[seed_index];
        if (lane < 9)
            init_table_scratch(r.cont_a[lane],      seed, D_CONTINENTALNESS_CFG, lane,      false, scratch);
        else if (lane < 18)
            init_table_scratch(r.cont_b[lane - 9],  seed, D_CONTINENTALNESS_CFG, lane - 9,  true,  scratch);
        else if (lane < 20)
            init_table_scratch(r.temp_a[lane - 18], seed, D_TEMPERATURE_CFG,     lane - 18, false, scratch);
        else if (lane < 22)
            init_table_scratch(r.temp_b[lane - 20], seed, D_TEMPERATURE_CFG,     lane - 20, true,  scratch);
        else if (lane < 25)
            init_table_scratch(r.erosion_a[lane - 21], seed, D_EROSION_CFG, lane - 21, false, scratch); // a[1..3]
        else if (lane < 28)
            init_table_scratch(r.erosion_b[lane - 24], seed, D_EROSION_CFG, lane - 24, true,  scratch); // b[1..3]
        else if (lane < 30)
            init_table_scratch(r.weird_a[lane - 28], seed, D_WEIRDNESS_CFG, lane - 28, false, scratch); // a[0..1]
        else
            init_table_scratch(r.weird_b[lane - 30], seed, D_WEIRDNESS_CFG, lane - 30, true,  scratch); // b[0..1]

        if (lane == 0) atomicAdd(init_count, 1u);
    }
}
} // namespace KernelLateInit

// ---------------------------------------------------------------------------
// Approximate surface height from the depth spline (1.18+). Minecraft turns
// continentalness, erosion, and weirdness into a "depth" value through a
// small tree of cubic splines, and that depth maps to terrain height. The
// tree is fixed for a given game version, so it is baked once at startup
// from cubiomes' initBiomeNoise() output into the flat __constant__ arrays
// below, and each sample is then
//   approx_h = (10000/76) * (0.48125 + spline(c, e, w) + 0.015)
// matching cubiomes' sampleClimatePara(NP_DEPTH, y=0) plus mapApproxHeight.
// The shift noise (a small positional jitter Minecraft applies to samples)
// is skipped; at the sampling pitches used here it would move a sample by at
// most ~20 blocks. The CPU verifier's height check stays authoritative.
// ---------------------------------------------------------------------------
struct DevSplineNode { float loc, der, leaf; int16_t child, pad; };
struct DevSplineHead { int32_t typ, nnodes, node_off, pad; };
constexpr int MAX_DEV_SPLINES = 64;
constexpr int MAX_DEV_NODES   = 768;
__constant__ DevSplineHead g_splines[MAX_DEV_SPLINES];
__constant__ DevSplineNode g_nodes[MAX_DEV_NODES];
__constant__ int32_t g_spline_root = 0;

__device__ float spline_eval_node(int si, const float vals[4]) {
    const DevSplineHead &sp = g_splines[si];
    const DevSplineNode *nodes = g_nodes + sp.node_off;
    const float f = vals[sp.typ];
    int i;
    for (i = 0; i < sp.nnodes; i++)
        if (nodes[i].loc >= f) break;
    if (i == 0 || i == sp.nnodes) {
        if (i) i--;
        const float v = (nodes[i].child < 0) ? nodes[i].leaf
                                             : spline_eval_node(nodes[i].child, vals);
        return v + nodes[i].der * (f - nodes[i].loc);
    }
    const DevSplineNode &a = nodes[i - 1];
    const DevSplineNode &b = nodes[i];
    const float n0 = (a.child < 0) ? a.leaf : spline_eval_node(a.child, vals);
    const float n1 = (b.child < 0) ? b.leaf : spline_eval_node(b.child, vals);
    const float g = a.loc, h = b.loc;
    const float k = (f - g) / (h - g);
    const float pterm =  a.der * (h - g) - (n1 - n0);
    const float qterm = -b.der * (h - g) + (n1 - n0);
    return lerp1(k, n0, n1) + k * (1.0f - k) * lerp1(k, pterm, qterm);
}

__device__ float approx_height(float c, float e, float w) {
    // Minecraft folds weirdness into a "peaks and valleys" value: highest on
    // ridge lines (|w| = 2/3), lowest on the valley axis (w = 0).
    const float pv = -3.0f * (fabsf(fabsf(w) - 0.6666667f) - 0.33333334f);
    const float vals[4] = { c, e, pv, w }; // spline input order: continentalness, erosion, peaks-and-valleys, weirdness
    const float off = spline_eval_node(g_spline_root, vals) + 0.015f;
    return (0.48125f + off) * (10000.0f / 76.0f);
}

// Flatten cubiomes' pointer-linked spline tree into the flat head/node
// arrays the device code walks, and upload them to __constant__ memory.
static void bake_and_upload_splines(int mc) {
    BiomeNoise bn;
    initBiomeNoise(&bn, mc);
    static DevSplineHead heads[MAX_DEV_SPLINES];
    static DevSplineNode nodes[MAX_DEV_NODES];
    memset(heads, 0, sizeof(heads));
    memset(nodes, 0, sizeof(nodes));
    int nh = 0, nn = 0;
    for (int si = 0; si < bn.ss.len; si++) {
        const Spline *sp = &bn.ss.stack[si];
        if (nh >= MAX_DEV_SPLINES || nn + sp->len > MAX_DEV_NODES)
            PANIC("spline bake overflow (%d splines, %d nodes)\n", nh, nn);
        DevSplineHead &H = heads[nh++];
        H.typ = sp->typ; H.nnodes = sp->len; H.node_off = nn; H.pad = 0;
        for (int i = 0; i < sp->len; i++) {
            DevSplineNode &N = nodes[nn++];
            N.loc = sp->loc[i];
            N.der = sp->der[i];
            if (sp->val[i]->len == 1) { // FixSpline leaf
                N.child = -1;
                N.leaf = ((FixSpline *)sp->val[i])->val;
            } else {
                N.child = (int16_t)(sp->val[i] - bn.ss.stack);
                N.leaf = 0;
            }
            N.pad = 0;
        }
    }
    int32_t root = (int32_t)(bn.sp - bn.ss.stack);
    TRY_CUDA(cudaMemcpyToSymbol(g_splines, heads, sizeof(DevSplineHead) * nh));
    TRY_CUDA(cudaMemcpyToSymbol(g_nodes,   nodes, sizeof(DevSplineNode) * nn));
    TRY_CUDA(cudaMemcpyToSymbol(g_spline_root, &root, sizeof(root)));
}

// ===========================================================================
// Stage 4: windowed checks around each surviving anchor, one warp per
// anchor. The checks run cheapest and most-selective first, so the
// expensive full-stack window scans only run for near-certain blobs:
//
//   1. Temperature coverage:  >= 65% of a 1000x1000-block window inside
//                              [CORE_TEMP_MIN, CORE_TEMP_MAX] (full stack).
//   2. Continentalness disc:  >= ~67% of the radius-1600 disc reading
//                              inland (1B truncation, same phyllotaxis
//                              pattern as stage 2).
//   3. Weirdness ridge check: enough of the window inside the ridge band;
//                              rejects windows too flat for a real mountain
//                              pattern to form.
//   4. Erosion minimum:       some point in a 3200x3200-block window at or
//                              below CORE_ERO_MIN_REQ (full stack).
//   5. Continentalness max:   some point in that window at or above
//                              CORE_CONT_MAX_REQ (full stack).
//   6. Height gate:           the highest approximate surface height in the
//                              window reaches H_GATE_MIN.
//
// Noise tables are read straight from global memory in this kernel: the
// anchors packed into a block belong to arbitrary seeds, so staging them in
// shared memory would not pay off.
// ===========================================================================
namespace KernelExtrema {
constexpr uint32_t THREADS_PER_BLOCK = 256;
constexpr uint32_t WARPS_PER_BLOCK   = THREADS_PER_BLOCK / 32;
constexpr int      CONT_COV_N    = CORE_COV_SAMPLES; // the same 96-sample disc as stage 2
constexpr int      CONT_COV_NEED = 64;               // ~67% must read inland
constexpr int32_t  TEMP_HALF_Q = 125;   // half-width of the 1000-block window, in quarts
constexpr int32_t  TEMP_STEP_Q = 25;    // 100-block sample pitch
constexpr uint32_t TEMP_N      = 2 * (TEMP_HALF_Q / TEMP_STEP_Q) + 1;  // 11
constexpr uint32_t TEMP_POINTS = TEMP_N * TEMP_N;                      // 121
constexpr uint32_t TEMP_NEED   = 79;                                   // ~65% must be temperate
constexpr int32_t  WIN_HALF_Q  = 400;   // half-width of the 3200-block window, in quarts
constexpr int32_t  WIN_STEP_Q  = 16;    // 64-block pitch, fine enough for the shortest
                                        // erosion octave (128-block wavelength)
constexpr uint32_t WIN_N       = 2 * (WIN_HALF_Q / WIN_STEP_Q) + 1;    // 51
constexpr uint32_t WIN_POINTS  = WIN_N * WIN_N;                        // 2601

// The weirdness texture check, on the two-octave truncation (octaves 0 and
// 1, sides A and B). Weirdness drives Minecraft's peaks-and-valleys pattern:
// |w| near 2/3 marks ridge lines, while w near 0 marks the valley and river
// axis. A window with no ridge-band values at all is too flat for a real
// mountain pattern to form, so a small minimum ridge fraction is enforced.
//
// The threshold comes from measuring ~57,000 verified output rows with
// seedlab.py lens and comparing the best-scoring regions (score >= 8) with
// the rest. Valley frequency did not separate the two groups (the best
// regions, if anything, showed slightly less valley-axis texture), so only
// the ridge term is enforced, and even that is set well below the corpus
// value of 0.0188 to leave room for the coarser window placement and the
// truncated octaves. Set W_TEX_RIDGE_MIN to 0 to disable the check entirely.
constexpr float    W_TEX_BAND_LO  =  0.55f;   // the ridge band: |w| in [0.55, 0.82]
constexpr float    W_TEX_BAND_HI  =  0.82f;
constexpr float    W_TEX_RIDGE_MIN = 0.012f;  // minimum ridge-band fraction of the window
constexpr float    W_TEX_VALLEY_TH =  0.17f;  // the valley axis: |w| at or below this
constexpr float    W_TEX_VALLEY_MIN = 0.0f;   // disabled: the corpus showed no signal here
constexpr int32_t  W_TEX_STEP_Q   = 25;       // 100-block pitch
constexpr int32_t  W_TEX_HALF_Q   = 400;
constexpr uint32_t W_TEX_N        = 2 * (W_TEX_HALF_Q / W_TEX_STEP_Q) + 1; // 33
constexpr uint32_t W_TEX_POINTS   = W_TEX_N * W_TEX_N;                     // 1089

// Check 6, the height gate: the best approximate surface height in the
// window (from the depth spline above) must reach H_GATE_MIN. The threshold
// comes from the verifier's measured peak heights: gating at 238.7 would
// keep 98% of the best-scoring regions while discarding 42% of the rest, and
// the CPU-side maxY column agrees almost exactly (237). The GPU measurement
// is coarser (wider pitch, no shift noise), so the gate sits at 215, about
// 0.9x, to avoid cutting candidates the CPU would have passed. It runs last
// so it only ever runs on the few dozen anchors per batch that get this far.
// The CPU verifier's height check has the final word.
constexpr float    H_GATE_MIN  = 215.0f;
constexpr int32_t  H_STEP_Q    = 32;    // 128-block pitch
constexpr int32_t  H_HALF_Q    = 384;
constexpr uint32_t H_N         = 2 * (H_HALF_Q / H_STEP_Q) + 1; // 25
constexpr uint32_t H_POINTS    = H_N * H_N;                     // 625
static_assert(CONT_COV_NEED * 3 >= CONT_COV_N * 2, "continentalness coverage is at least 67%");
static_assert(TEMP_NEED * 20 >= TEMP_POINTS * 13, "temperature coverage is at least 65%");

__global__ __launch_bounds__(THREADS_PER_BLOCK) void kernel(
        InputBuffer<KernelCoverage::AnchorHit> anchors,
        OutputBuffer<KernelCoverage::AnchorHit> cores,
        const MountainNoiseResult *__restrict__ results) {
    __shared__ GradDotTable s_grad;
    load_grad_table_shared(s_grad);

    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;

    int32_t ddx[CORE_COVERS], ddz[CORE_COVERS];
    #pragma unroll
    for (int b = 0; b < CORE_COVERS; b++) {
        const int k = b * DISC_COVER_POINTS + (int)lane;
        ddx[b] = __float2int_rn((float)g_disc.x[k] * (KernelCoverage::DISC_R_Q / 32767.0f));
        ddz[b] = __float2int_rn((float)g_disc.z[k] * (KernelCoverage::DISC_R_Q / 32767.0f));
    }

    const uint32_t anchors_len = anchors.size();
    for (uint32_t base = blockIdx.x * WARPS_PER_BLOCK;
         base < anchors_len;
         base += gridDim.x * WARPS_PER_BLOCK) {
        const uint32_t anchor_index = base + warp;
        if (anchor_index >= anchors_len) continue;
        const KernelCoverage::AnchorHit anchor = anchors.data[anchor_index];
        const MountainNoiseResult &noise = results[anchor.seed_index];

        // Check 1: temperature coverage over the window.
        uint32_t temp_hits = 0;
        for (uint32_t p = lane; p < TEMP_POINTS; p += 32) {
            const int32_t xq = anchor.xq + (int32_t)(p % TEMP_N) * TEMP_STEP_Q - TEMP_HALF_Q;
            const int32_t zq = anchor.zq + (int32_t)(p / TEMP_N) * TEMP_STEP_Q - TEMP_HALF_Q;
            const float t = sample_climate(s_grad, noise.temp_a, noise.temp_b,
                                           D_TEMPERATURE_CFG, xq, zq);
            temp_hits += (t >= CORE_TEMP_MIN && t <= CORE_TEMP_MAX);
        }
        if (warp_reduce_add(temp_hits) < TEMP_NEED) continue;

        // Check 2: continentalness disc coverage (the 1B truncation), in
        // the same fail-fast batches as stage 2's disc scan.
        int hits = 0;
        bool alive = true;
        #pragma unroll
        for (int b = 0; b < CORE_COVERS; b++) {
            const float c = sample_climate_trunc(s_grad, noise.cont_a, noise.cont_b,
                                                 D_CONTINENTALNESS_CFG, 2, 2,
                                                 anchor.xq + ddx[b], anchor.zq + ddz[b]);
            const bool hit = (c >= CORE_COV_CONT_MIN);
            hits += __popc(__ballot_sync(~0u, hit));
            if (hits + (CONT_COV_N - DISC_COVER_POINTS * (b + 1)) < CONT_COV_NEED) { alive = false; break; }
            if (hits >= CONT_COV_NEED) break;
        }
        if (!alive || hits < CONT_COV_NEED) continue;

        // Check 3: weirdness ridge-band presence (the valley term is off).
        {
            uint32_t wridge = 0, wvalley = 0;
            for (uint32_t p = lane; p < W_TEX_POINTS; p += 32) {
                const int32_t xq = anchor.xq + (int32_t)(p % W_TEX_N) * W_TEX_STEP_Q - W_TEX_HALF_Q;
                const int32_t zq = anchor.zq + (int32_t)(p / W_TEX_N) * W_TEX_STEP_Q - W_TEX_HALF_Q;
                const float w = sample_climate_trunc(s_grad, noise.weird_a, noise.weird_b,
                                                     D_WEIRDNESS_CFG, 2, 2, xq, zq);
                const float aw = fabsf(w);
                wridge  += (aw >= W_TEX_BAND_LO && aw <= W_TEX_BAND_HI);
                wvalley += (aw <= W_TEX_VALLEY_TH);
            }
            wridge  = warp_reduce_add(wridge);   // uniform across the warp
            wvalley = warp_reduce_add(wvalley);
            if ((float)wridge  < W_TEX_RIDGE_MIN  * W_TEX_POINTS) continue;
            if ((float)wvalley < W_TEX_VALLEY_MIN * W_TEX_POINTS) continue;
        }

        // Check 4: the deep-erosion point. Full-stack minimum over the
        // window; a single point below the threshold passes, so the warp
        // stops at the first 32-sample batch that contains one.
        bool ero_ok = false;
        for (uint32_t base_p = 0; base_p < WIN_POINTS && !ero_ok; base_p += 32) {
            const uint32_t p = base_p + lane;
            bool hit = false;
            if (p < WIN_POINTS) {
                const int32_t xq = anchor.xq + (int32_t)(p % WIN_N) * WIN_STEP_Q - WIN_HALF_Q;
                const int32_t zq = anchor.zq + (int32_t)(p / WIN_N) * WIN_STEP_Q - WIN_HALF_Q;
                const float e = sample_climate(s_grad, noise.erosion_a, noise.erosion_b,
                                               D_EROSION_CFG, xq, zq);
                hit = (e <= CORE_ERO_MIN_REQ);
            }
            ero_ok = __ballot_sync(~0u, hit) != 0;
        }
        if (!ero_ok) continue;

        // Check 5: the high-continentalness point: a full-stack maximum over
        // the same window, with the same early exit.
        bool cont_ok = false;
        for (uint32_t base_p = 0; base_p < WIN_POINTS && !cont_ok; base_p += 32) {
            const uint32_t p = base_p + lane;
            bool hit = false;
            if (p < WIN_POINTS) {
                const int32_t xq = anchor.xq + (int32_t)(p % WIN_N) * WIN_STEP_Q - WIN_HALF_Q;
                const int32_t zq = anchor.zq + (int32_t)(p / WIN_N) * WIN_STEP_Q - WIN_HALF_Q;
                const float c = sample_climate(s_grad, noise.cont_a, noise.cont_b,
                                               D_CONTINENTALNESS_CFG, xq, zq);
                hit = (c >= CORE_CONT_MAX_REQ);
            }
            cont_ok = __ballot_sync(~0u, hit) != 0;
        }
        if (!cont_ok) continue;

        // Check 6: approximate peak height from the depth spline (full
        // continentalness and erosion, truncated weirdness).
        {
            float h_max = -FLT_MAX;
            for (uint32_t p = lane; p < H_POINTS; p += 32) {
                const int32_t xq = anchor.xq + (int32_t)(p % H_N) * H_STEP_Q - H_HALF_Q;
                const int32_t zq = anchor.zq + (int32_t)(p / H_N) * H_STEP_Q - H_HALF_Q;
                const float c = sample_climate(s_grad, noise.cont_a, noise.cont_b,
                                               D_CONTINENTALNESS_CFG, xq, zq);
                const float e = sample_climate(s_grad, noise.erosion_a, noise.erosion_b,
                                               D_EROSION_CFG, xq, zq);
                const float w = sample_climate_trunc(s_grad, noise.weird_a, noise.weird_b,
                                                     D_WEIRDNESS_CFG, 2, 2, xq, zq);
                h_max = fmaxf(h_max, approx_height(c, e, w));
            }
            for (int o = 16; o; o >>= 1)
                h_max = fmaxf(h_max, __shfl_xor_sync(~0u, h_max, o));
            if (h_max < H_GATE_MIN) continue;
        }

        if (lane == 0) {
            const uint32_t out = atomicAdd(cores.len, 1);
            if (out < cores.max_len) cores.data[out] = anchor;
        }
    }
}
} // namespace KernelExtrema

// ===========================================================================
// Stage 5: the flood fill, the actual megaregion measurement.
//
// One thread block takes one surviving anchor and flood-fills the connected
// region around it. A cell of the 64-block lattice joins the region if it
// passes all three shared field tests from common.h (BLOB_*): erosion 0B at
// or below the ceiling, continentalness 1B at or above the floor, and
// temperature 0B inside the window. The block keeps the noise tables and
// three lattice bitmaps in shared memory and expands the frontier one ring
// per round, with all 512 threads claiming new cells through atomic bitmap
// operations. The fill ends the moment the connected area crosses the
// runtime target (a pass), when the frontier runs out of qualifying cells
// (a fail: the area is a puddle), or at a generous round cap that exists
// only as a safety net. The fill reaches +-8,000 blocks from the anchor;
// regions that run off that edge are flagged and forwarded, since the CPU's
// +-48,000-block re-measurement can still see their full extent.
// ===========================================================================
namespace KernelBlob {
constexpr uint32_t THREADS_PER_BLOCK = 512;  // 512 threads halve the per-round
                                             // wavefront latency vs 256; the
                                             // shared-memory budget is unchanged.
constexpr uint32_t GRID_BLOCKS       = 1024; // blocks drawing anchors off the work counter

// Lattice geometry: 64 blocks (16 quarts) per cell, +-125 cells = +-8,000 blocks.
constexpr int32_t  STEP_Q = 16;
constexpr int32_t  HALF   = 8000 / 64;   // 125
constexpr int32_t  DIM    = 2 * HALF + 1; // 251
constexpr uint32_t NCELLS = (uint32_t)(DIM * DIM);   // 63001
constexpr uint32_t WORDS  = (NCELLS + 31u) / 32u;    // 1969 (~7.7 KB per bitmap)

// Tunables:
//   Early-exit target: g_blobcfg.target_cells, pushed from main() as 0.88x
//     the CPU area gate (in 4096-block cells), so the float32 prefilter is
//     always looser than the CPU's double-precision re-measurement.
//   MAX_ROUNDS: a safety net only, far beyond the lattice's 500-round
//     diameter and generous enough for long snaking regions.
//   PER_SEED_CAP: at most this many candidates per seed per batch.
//   The tier-2 texture gates run after the fill, on passing fills only, and
//     are currently measure-only (0 disables them): calibrate against the
//     gpuH/gpuDh output columns before enabling either. Gating on height
//     inside the fill was tried at 230 and abandoned: it lost 4 of 200
//     recall-test seeds to peaks just outside the blob's field edge, and it
//     made this stage about 6x slower by disabling the area early exit for
//     big-but-flat fills.
constexpr uint32_t MAX_ROUNDS      = 2048;
constexpr uint32_t BLOB2_H_MIN_I   = 0;     // tier-2 max-height gate (0 = off)
constexpr float    BLOB2_DH_MIN    = 0.0f;  // tier-2 mean neighbor-height-difference gate (0 = off)
constexpr uint32_t PER_SEED_CAP    = 16;
// One-cell morphological closing on the blob mask (dilate, then erode):
// bridges single-cell gaps such as river gorges without inflating the area.
constexpr int      BLOB_CLOSE1     = 1;

__global__ __launch_bounds__(THREADS_PER_BLOCK) void kernel(
        InputBuffer<uint64_t> seeds,
        InputBuffer<KernelCoverage::AnchorHit> cores,
        OutputBuffer<GpuOutput> outputs,
        const MountainNoiseResult *__restrict__ results,
        uint32_t *__restrict__ emit_counts,
        uint32_t *__restrict__ work_counter) {
    __shared__ GradDotTable s_grad;
    __shared__ ImprovedNoise s_ero_a[4], s_ero_b[4];      // full erosion (for the height stats)
    __shared__ ImprovedNoise s_cont_a[9], s_cont_b[9];    // full continentalness
    __shared__ ImprovedNoise s_temp0A, s_temp0B;          // truncated temperature (the blob field)
    __shared__ ImprovedNoise s_weird_a[2], s_weird_b[2];  // truncated weirdness (for the height spline)
    __shared__ uint32_t s_visited[WORDS];
    __shared__ uint32_t s_front[2][WORDS];
    __shared__ uint32_t s_count;   // connected blob-field cells claimed so far
    __shared__ uint32_t s_next;    // cells pushed into the next frontier this round
    __shared__ uint32_t s_seed;    // chosen seed cell index (or ~0u if none)
    __shared__ uint32_t s_state;   // 0 = running, 1 = pass, 2 = fail
    __shared__ uint32_t s_edge;    // 1 = fill touched the +-8000-block lattice rim
    // Tier-2 texture stats, measured after the fill and only for fills that
    // passed the area gate (currently measure-only, see the tunables above):
    //   s_hmax:  the region's maximum approximate height, whole blocks
    //   s_dh_*:  mean height difference between neighboring samples at the
    //            128-block pitch, the same measurement domain the scoring
    //            system uses for its steepness stats
    __shared__ uint32_t s_hmax;    // max approx height in whole blocks (atomicMax)
    __shared__ float    s_dh_sum;  // sum of |dh| over measured pairs
    __shared__ uint32_t s_dh_cnt;  // measured pair count

    load_grad_table_shared(s_grad);
    const uint32_t tid = threadIdx.x;

    const uint32_t cores_len = cores.size();
    while (true) {
        // Work-stealing: thread 0 claims the next anchor index off an atomic
        // counter and shares it with the rest of the block through shared
        // memory, so a block that draws two large fills in a row cannot hold
        // up the others.
        __shared__ uint32_t s_core_idx;
        if (tid == 0)
            s_core_idx = atomicAdd(work_counter, 1u);
        __syncthreads();
        const uint32_t core_index = s_core_idx;
        __syncthreads(); // ensure s_core_idx is safe to overwrite next iter
        if (core_index >= cores_len) break;
        const KernelCoverage::AnchorHit core = cores.data[core_index];
        const MountainNoiseResult &noise = results[core.seed_index];

        if (tid == 0) { s_count = 0; s_next = 0; s_seed = ~0u; s_state = 0;
                        s_edge = 0;
                        s_hmax = 0; s_dh_sum = 0.0f; s_dh_cnt = 0; }
        for (uint32_t w = tid; w < WORDS; w += THREADS_PER_BLOCK) {
            s_visited[w]   = 0u;
            s_front[0][w]  = 0u;
            s_front[1][w]  = 0u;
        }
        __syncthreads();
        // The fill itself reads only 8 tables (the truncated erosion pair,
        // the four truncated continentalness tables, and the truncated
        // temperature pair). The other 24 are only needed by the tier-2
        // texture pass, which runs on a small minority of fills, so those
        // are loaded there instead of paying their global-memory latency on
        // every fill.
        load_noise_shared(s_ero_a[0], noise.erosion_a[0]);
        load_noise_shared(s_ero_b[0], noise.erosion_b[0]);
        load_noise_shared(s_cont_a[0], noise.cont_a[0]);
        load_noise_shared(s_cont_a[1], noise.cont_a[1]);
        load_noise_shared(s_cont_b[0], noise.cont_b[0]);
        load_noise_shared(s_cont_b[1], noise.cont_b[1]);
        load_noise_shared(s_temp0A, noise.temp_a[0]);
        load_noise_shared(s_temp0B, noise.temp_b[0]);
        __syncthreads();

        // Full-stack approximate height at one lattice cell (tier-2 only).
        auto blob_h_at = [&](int32_t i, int32_t j) {
            const int32_t xq = core.xq + i * STEP_Q;
            const int32_t zq = core.zq + j * STEP_Q;
            const float cf = sample_climate(s_grad, s_cont_a, s_cont_b, D_CONTINENTALNESS_CFG, xq, zq);
            const float ef = sample_climate(s_grad, s_ero_a, s_ero_b, D_EROSION_CFG, xq, zq);
            const float wf = sample_climate_trunc(s_grad, s_weird_a, s_weird_b, D_WEIRDNESS_CFG, 2, 2, xq, zq);
            return approx_height(cf, ef, wf);
        };

        // -- Choose the fill's starting cell (thread 0). The anchor centroid
        // is normally already inside the region; for a concave region it
        // might not be, so search outward up to two lattice rings for a
        // valid cell.
        if (tid == 0) {
            auto low_at = [&](int32_t i, int32_t j) {
                int32_t xq = core.xq + i * STEP_Q;
                int32_t zq = core.zq + j * STEP_Q;
                if (sample_climate_trunc(s_grad, &s_ero_a[0], &s_ero_b[0], D_EROSION_CFG, 1, 1, xq, zq) > g_blobcfg.ero_max) return false;
                if (sample_climate_trunc(s_grad, s_cont_a, s_cont_b, D_CONTINENTALNESS_CFG, 2, 2, xq, zq) < g_blobcfg.cont_min) return false;
                float t = sample_climate_trunc(s_grad, &s_temp0A, &s_temp0B, D_TEMPERATURE_CFG, 1, 1, xq, zq);
                if (t < g_blobcfg.temp_min || t > g_blobcfg.temp_max) return false;
                return true;
            };
            int32_t si = INT32_MAX, sj = INT32_MAX;
            if (low_at(0, 0)) { si = 0; sj = 0; }
            else {
                for (int32_t r = 1; r <= 2 && si == INT32_MAX; r++)
                    for (int32_t j = -r; j <= r && si == INT32_MAX; j++)
                        for (int32_t i = -r; i <= r && si == INT32_MAX; i++)
                            if (low_at(i, j)) { si = i; sj = j; }
            }
            if (si != INT32_MAX) {
                const uint32_t idx = (uint32_t)((sj + HALF) * DIM + (si + HALF));
                s_seed = idx;
                s_visited[s_seed >> 5]  |= 1u << (s_seed & 31);
                s_front[0][s_seed >> 5] |= 1u << (s_seed & 31);
                s_count = 1;
            } else {
                s_state = 2; // no valid cell near the anchor: fail immediately
            }
        }
        __syncthreads();

        // -- Wavefront BFS: expand the frontier one ring per round until the
        // area target is hit or the region stops growing.
        uint32_t parity = 0;
        for (uint32_t round = 0; round < MAX_ROUNDS && s_state == 0; round++) {
            if (tid == 0) s_next = 0;
            __syncthreads();
            for (uint32_t w = tid; w < WORDS; w += THREADS_PER_BLOCK)
                s_front[parity ^ 1][w] = 0u;
            __syncthreads();

            for (uint32_t w = tid; w < WORDS; w += THREADS_PER_BLOCK) {
                uint32_t bits = s_front[parity][w];
                while (bits) {
                    const uint32_t b = (uint32_t)(__ffs((int)bits) - 1);
                    bits &= bits - 1u;
                    const uint32_t idx = w * 32u + b;
                    const int32_t i = (int32_t)(idx % (uint32_t)DIM) - HALF;
                    const int32_t j = (int32_t)(idx / (uint32_t)DIM) - HALF;

                    const int32_t nb[4][2] = {{i-1,j},{i+1,j},{i,j-1},{i,j+1}};
                    #pragma unroll
                    for (int k = 0; k < 4; k++) {
                        const int32_t ni = nb[k][0], nj = nb[k][1];
                        if (ni < -HALF || ni > HALF || nj < -HALF || nj > HALF) continue;
                        const uint32_t nidx = (uint32_t)((nj + HALF) * DIM + (ni + HALF));
                        const uint32_t word = nidx >> 5;
                        const uint32_t bit  = 1u << (nidx & 31);
                        if (s_visited[word] & bit) continue;

                        int32_t xq = core.xq + ni * STEP_Q;
                        int32_t zq = core.zq + nj * STEP_Q;
                        const float e = sample_climate_trunc(s_grad, &s_ero_a[0], &s_ero_b[0], D_EROSION_CFG, 1, 1, xq, zq);
                        if (e > g_blobcfg.ero_max) continue;
                        const float c = sample_climate_trunc(s_grad, s_cont_a, s_cont_b, D_CONTINENTALNESS_CFG, 2, 2, xq, zq);
                        if (c < g_blobcfg.cont_min) continue;
                        const float t = sample_climate_trunc(s_grad, &s_temp0A, &s_temp0B, D_TEMPERATURE_CFG, 1, 1, xq, zq);
                        if (t < g_blobcfg.temp_min || t > g_blobcfg.temp_max) continue;

                        const uint32_t old = atomicOr(&s_visited[word], bit);
                        if (!(old & bit)) {
                            atomicOr(&s_front[parity ^ 1][word], bit);
                            atomicAdd(&s_count, 1u);
                            atomicAdd(&s_next, 1u);
                            if (ni == -HALF || ni == HALF || nj == -HALF || nj == HALF)
                                s_edge = 1; // benign race: only ever 0 -> 1
                        }
                    }
                }
            }
            __syncthreads();
            if (tid == 0) {
                if (s_count >= g_blobcfg.target_cells) s_state = 1;
                // The frontier died: normally a fail (the region is a
                // puddle). The exception is a fill that reached the
                // +-8,000-block lattice rim: it may be an elongated range
                // whose true area only the CPU's +-48,000-block fill can
                // see, so forward it if it reached at least half the target.
                // Raising that fraction (to 3/4, say) emits fewer of these
                // for the CPU to reject.
                else if (s_next == 0u)
                    s_state = (s_edge && s_count >= g_blobcfg.target_cells / 2) ? 1 : 2;
            }
            __syncthreads();
            parity ^= 1;
        }

        // One-cell morphological closing, applied to passing fills. Dilation
        // adds every unset cell that touches the region; erosion then strips
        // the added ring back off. The net effect is that one-cell gaps
        // (river gorges, mostly) get bridged, while the original region
        // always survives and the measured area can never shrink. The two
        // frontier bitmaps are idle at this point and serve as scratch.
        if (BLOB_CLOSE1 && s_state == 1) {
            uint32_t *s_dilate = s_front[0]; // dilated mask
            uint32_t *s_erode  = s_front[1]; // closed mask
            
            for (uint32_t w = tid; w < WORDS; w += THREADS_PER_BLOCK) {
                s_dilate[w] = 0u;
                s_erode[w]  = 0u; // clear the output buffer
            }
            __syncthreads();
            // Dilate: this has to check every cell; the gap cells we are
            // looking for are exactly the unset ones.
            for (uint32_t c = tid; c < NCELLS; c += THREADS_PER_BLOCK) {
                const uint32_t w = c >> 5, b = c & 31;
                uint32_t v = (s_visited[w] >> b) & 1u;
                if (!v) {
                    const int32_t i = (int32_t)(c % (uint32_t)DIM) - HALF;
                    const int32_t j = (int32_t)(c / (uint32_t)DIM) - HALF;
                    if (i > -HALF) v |= (s_visited[(c - 1)   >> 5] >> ((c - 1)   & 31)) & 1u;
                    if (i <  HALF) v |= (s_visited[(c + 1)   >> 5] >> ((c + 1)   & 31)) & 1u;
                    if (j > -HALF) v |= (s_visited[(c - DIM) >> 5] >> ((c - DIM) & 31)) & 1u;
                    if (j <  HALF) v |= (s_visited[(c + DIM) >> 5] >> ((c + DIM) & 31)) & 1u;
                }
                if (v) atomicOr(&s_dilate[w], 1u << b);
            }
            __syncthreads();
            // Erode: drop every dilated cell that still touches an unset
            // neighbor (cells past the lattice rim count as unset, which
            // only matters for edge-clipped fills). This removes the dilation ring.
            // 
            // Race fix: Read exclusively from s_dilate, write survivors to s_erode.
            // Writing in-place to s_dilate causes a data race. Thread A might
            // read cell X+1 to check its neighbors at the exact moment Thread B
            // is eroding cell X+1 via atomicAnd. A plain read happening at the
            // same time as an atomic write is "undefined behavior" (the GPU
            // is allowed to return a garbled value). We avoid this by
            // writing to a separate buffer. s_front[1] is currently idle (the
            // BFS is over), so we recycle it as s_erode for free.
            for (uint32_t c = tid; c < NCELLS; c += THREADS_PER_BLOCK) {
                const uint32_t w = c >> 5, b = c & 31;
                // Read from s_dilate
                if (!((s_dilate[w] >> b) & 1u)) continue; 
                const int32_t i = (int32_t)(c % (uint32_t)DIM) - HALF;
                const int32_t j = (int32_t)(c / (uint32_t)DIM) - HALF;
                int keep = 1;
                // Read neighbors from s_dilate (safe cuz it's never modified in this loop)
                if (i > -HALF) keep &= (int)((s_dilate[(c - 1)   >> 5] >> ((c - 1)   & 31)) & 1u); else keep = 0;
                if (i <  HALF) keep &= (int)((s_dilate[(c + 1)   >> 5] >> ((c + 1)   & 31)) & 1u); else keep = 0;
                if (j > -HALF) keep &= (int)((s_dilate[(c - DIM) >> 5] >> ((c - DIM) & 31)) & 1u); else keep = 0;
                if (j <  HALF) keep &= (int)((s_dilate[(c + DIM) >> 5] >> ((c + DIM) & 31)) & 1u); else keep = 0;
                // Write survivors to s_erode using atomicOr
                if (keep) atomicOr(&s_erode[w], 1u << b);
            }
            __syncthreads();
            // Copy back to visited and recount the area.
            if (tid == 0) s_count = 0;
            __syncthreads();
            for (uint32_t w = tid; w < WORDS; w += THREADS_PER_BLOCK) {
                // Read final state from s_erode
                const uint32_t v = s_erode[w]; 
                s_visited[w] = v;
                if (v) atomicAdd(&s_count, __popc(v));
            }
            __syncthreads();
        }

        // -- Tier 2, for passing fills only: region-scale texture stats over
        // the visited bitmap, measured every other lattice cell (a 128-block
        // pitch, the same spacing the scoring system's steepness stats use).
        // First a bounding box is computed in shared memory, then heights
        // are sampled only inside it. Neighbor cells are recomputed rather
        // than cached (about 3x the samples), which is cheap at the observed
        // rate of passing fills and keeps the block's shared-memory
        // footprint under 48 KiB. The deferred noise tables are also loaded
        // now; s_state is block-uniform here, so the strided loads are safe.
        if (s_state == 1) {
            for (int i = 1; i < 4; i++) { load_noise_shared(s_ero_a[i], noise.erosion_a[i]);
                                          load_noise_shared(s_ero_b[i], noise.erosion_b[i]); }
            for (int i = 2; i < 9; i++) { load_noise_shared(s_cont_a[i], noise.cont_a[i]);
                                          load_noise_shared(s_cont_b[i], noise.cont_b[i]); }
            load_noise_shared(s_weird_a[0], noise.weird_a[0]);
            load_noise_shared(s_weird_a[1], noise.weird_a[1]);
            load_noise_shared(s_weird_b[0], noise.weird_b[0]);
            load_noise_shared(s_weird_b[1], noise.weird_b[1]);
            __syncthreads();
        }
        if (s_state == 1) {
            __shared__ int32_t s_minI, s_maxI, s_minJ, s_maxJ;
            if (tid == 0) { s_minI = HALF; s_maxI = -HALF; s_minJ = HALF; s_maxJ = -HALF; }
            __syncthreads();
            for (uint32_t w = tid; w < WORDS; w += THREADS_PER_BLOCK) {
                uint32_t bits = s_visited[w];
                while (bits) {
                    const uint32_t b = (uint32_t)(__ffs((int)bits) - 1);
                    bits &= bits - 1u;
                    const uint32_t idx = w * 32u + b;
                    const int32_t i = (int32_t)(idx % (uint32_t)DIM) - HALF;
                    const int32_t j = (int32_t)(idx / (uint32_t)DIM) - HALF;
                    if ((i & 1) || (j & 1)) continue;
                    // Signed atomics: i and j span +-HALF, and unsigned
                    // atomicMin/Max would order negative offsets as huge
                    // positive values.
                    atomicMin(&s_minI, i);
                    atomicMax(&s_maxI, i);
                    atomicMin(&s_minJ, j);
                    atomicMax(&s_maxJ, j);
                }
            }
            __syncthreads();

            const int32_t minI2 = s_minI;
            const int32_t maxI2 = s_maxI;
            const int32_t minJ2 = s_minJ;
            const int32_t maxJ2 = s_maxJ;
            const int32_t tileW = (maxI2 - minI2) / 2 + 1;
            const int32_t tileH = (maxJ2 - minJ2) / 2 + 1;

            for (int32_t tj = 0; tj < tileH; tj++) {
                const int32_t j = minJ2 + tj * 2;
                for (int32_t ti = tid; ti < tileW; ti += THREADS_PER_BLOCK) {
                    const int32_t i = minI2 + ti * 2;
                    const uint32_t idx = (uint32_t)((j + HALF) * DIM + (i + HALF));
                    if (!(s_visited[idx >> 5] & (1u << (idx & 31)))) continue;
                    const float h0 = blob_h_at(i, j);
                    atomicMax(&s_hmax, (uint32_t)fmaxf(h0, 0.0f));
                    if (i + 2 <= maxI2) {
                        const uint32_t nidx = (uint32_t)((j + HALF) * DIM + (i + 2 + HALF));
                        if (s_visited[nidx >> 5] & (1u << (nidx & 31))) {
                            atomicAdd(&s_dh_sum, fabsf(h0 - blob_h_at(i + 2, j)));
                            atomicAdd(&s_dh_cnt, 1u);
                        }
                    }
                    if (j + 2 <= maxJ2) {
                        const uint32_t nidx = (uint32_t)((j + 2 + HALF) * DIM + (i + HALF));
                        if (s_visited[nidx >> 5] & (1u << (nidx & 31))) {
                            atomicAdd(&s_dh_sum, fabsf(h0 - blob_h_at(i, j + 2)));
                            atomicAdd(&s_dh_cnt, 1u);
                        }
                    }
                }
            }
            __syncthreads();
            // Texture gates, both currently disabled: flat-blob veto + dh veto.
            if (tid == 0) {
                const float dh = s_dh_cnt ? s_dh_sum / (float)s_dh_cnt : 0.0f;
                if (BLOB2_H_MIN_I > 0 && s_hmax < BLOB2_H_MIN_I) s_state = 2;
                if (BLOB2_DH_MIN > 0.0f && dh < BLOB2_DH_MIN) s_state = 2;
            }
            __syncthreads();
        }

        if (s_state == 1 && tid == 0) {
            if (atomicAdd(&emit_counts[core.seed_index], 1u) < PER_SEED_CAP) {
                const uint32_t out = atomicAdd(outputs.len, 1);
                if (out < outputs.max_len) {
                    const uint64_t seed = seeds.data[core.seed_index];
                    const float dh = s_dh_cnt ? s_dh_sum / (float)s_dh_cnt : 0.0f;
                    // coordinates become blocks here (x4)
                    outputs.data[out] = { seed, core.xq * 4, core.zq * 4,
                        (int32_t)s_hmax, (int32_t)lrintf(dh * 8.0f) };
                }
            }
        }
        __syncthreads(); // tables and bitmaps are reused next iteration
    }
}
} // namespace KernelBlob

// ===========================================================================
// GPU worker thread
// ===========================================================================

// Batch resource sizes at namespace scope, so the batch-buffer struct below
// can size its allocations.
constexpr uint32_t GPU_NUM_SEEDS   = KernelInit::SEEDS_PER_RUN;
constexpr uint32_t GPU_MAX_ANCHORS = 1u << 20; // 12 MiB of AnchorHit
constexpr uint32_t GPU_MAX_CORES   = 1u << 18; //  3 MiB of AnchorHit
constexpr uint32_t GPU_MAX_OUTPUTS = 1u << 18;

// Everything one batch needs on the device: buffers, counters, staging
// memory, and timing events. Two of these structures alternate across two
// CUDA streams, so while one batch runs the compute-heavy stages (coverage,
// extrema, blob fill), the next batch's KernelInit, which is latency-bound
// on its serial per-table shuffles, can already start. The host stays one
// batch ahead and collects the older batch's results while the GPU keeps
// working. If there isn't enough VRAM for two batches, the code falls back
// to one and the batches simply run back to back.
struct BatchBuf {
    void *seeds = nullptr, *results = nullptr, *anchors = nullptr, *cores = nullptr;
    void *fin = nullptr, *flags = nullptr, *emits = nullptr, *work = nullptr;
    uint32_t *lens = nullptr; // [0]=seeds [1]=anchors [2]=cores [3]=final [4]=late-init
    uint32_t *counts = nullptr; // per-batch funnel: probed / probe7 / emitted
    uint32_t batch_n = 0;
    std::vector<uint64_t> h_seeds;
    CudaEvent e_start, e_init, e_cov, e_late, e_ext, e_blob;

    bool alloc() {
        bool ok = true;
        ok = ok && cudaMalloc(&seeds,   sizeof(uint64_t) * GPU_NUM_SEEDS)            == cudaSuccess;
        ok = ok && cudaMalloc(&results, sizeof(MountainNoiseResult) * GPU_NUM_SEEDS) == cudaSuccess;
        ok = ok && cudaMalloc(&anchors, sizeof(KernelCoverage::AnchorHit) * GPU_MAX_ANCHORS) == cudaSuccess;
        ok = ok && cudaMalloc(&cores,   sizeof(KernelCoverage::AnchorHit) * GPU_MAX_CORES)   == cudaSuccess;
        ok = ok && cudaMalloc(&fin,     sizeof(GpuOutput) * GPU_MAX_OUTPUTS)         == cudaSuccess;
        ok = ok && cudaMalloc(&flags,   sizeof(uint32_t) * GPU_NUM_SEEDS)            == cudaSuccess;
        ok = ok && cudaMalloc(&emits,   sizeof(uint32_t) * GPU_NUM_SEEDS)            == cudaSuccess;
        ok = ok && cudaMalloc(&work,    sizeof(uint32_t))                            == cudaSuccess;
        ok = ok && cudaMalloc((void **)&lens, sizeof(uint32_t) * 5)                  == cudaSuccess;
        ok = ok && cudaMalloc((void **)&counts, sizeof(uint32_t) * 4)                == cudaSuccess;
        if (!ok) {
            free_all();
            (void)cudaGetLastError(); // swallow the OOM so later CUDA calls are unaffected
            return false;
        }
        h_seeds.resize(GPU_NUM_SEEDS);
        return true;
    }
    void free_all() {
        (void)cudaFree(seeds);   (void)cudaFree(results); (void)cudaFree(anchors);
        (void)cudaFree(cores);   (void)cudaFree(fin);     (void)cudaFree(flags);
        (void)cudaFree(emits);   (void)cudaFree(work);    (void)cudaFree(lens);
        (void)cudaFree(counts);
        seeds = results = anchors = cores = fin = flags = emits = work = nullptr;
        lens = nullptr;
        counts = nullptr;
    }
    ~BatchBuf() { free_all(); }
};

void GpuThread::run() {
    std::printf("Initializing GPU device %d...\n", device);
    TRY_CUDA(cudaSetDevice(device));
    upload_grad_dot_table();
    upload_disc_offsets();
    TRY_CUDA(cudaMemcpyToSymbol(g_blobcfg, &h_blobcfg, sizeof(h_blobcfg)));
    bake_and_upload_splines(g_mc_host);

    cudaStream_t streams[2];
    TRY_CUDA(cudaStreamCreate(&streams[0]));
    TRY_CUDA(cudaStreamCreate(&streams[1]));

    BatchBuf bufs[2];
    int nbuf = 0;
    for (int s = 0; s < 2; s++) {
        if (bufs[s].alloc()) nbuf++;
        else break;
    }
    if (nbuf == 0) PANIC("GPU %d: failed to allocate device buffers\n", device);
    std::printf("GPU %d ready: %u seeds/batch, %.1f MiB noise tables x%d%s.\n",
                device, GPU_NUM_SEEDS,
                (double)(sizeof(MountainNoiseResult) * GPU_NUM_SEEDS) / 1048576.0,
                nbuf, nbuf == 2 ? " (batch overlap on)" : " (single-buffer mode)");

    std::vector<GpuOutput> h_outputs(GPU_MAX_OUTPUTS);
    uint64_t h_stage_counts[4] = {0, 0, 0, 0}; // 64-bit: anchor counts over a stats window overflow 32 bits

    // Per-stage statistics (printed every STATS_INTERVAL batches and once
    // more at shutdown so short --seeds runs still get a table).
    std::vector<StageStat> stats = {
        { "init",      "seeds"   },
        { "coverage",  "anchors" },
        { "late-init", "anchors" },
        { "extrema",   "anchors" },
        { "blobfill",  "cores"   },
    };
    uint64_t stats_batches = 0;
    uint64_t stats_seeds = 0;
    auto stats_t0 = std::chrono::steady_clock::now();
    auto stats_last_print = stats_t0;

    uint64_t launched  = 0; // batches launched
    uint64_t harvested = 0; // batches synced + drained

    // Synchronize one finished batch and collect its counters, stage
    // timings, and emitted candidates. This runs at most one batch behind
    // the launcher, so the other stream keeps the GPU busy while the host
    // works.
    auto harvest = [&](int s) {
        BatchBuf &b = bufs[s];
        cudaStream_t stream = streams[s];
        const uint32_t n = b.batch_n;

        TRY_CUDA(cudaStreamSynchronize(stream));
        input.mark_done(n);

        uint32_t h_lens[5] = {};
        TRY_CUDA(cudaMemcpy(h_lens, b.lens, sizeof(h_lens), cudaMemcpyDeviceToHost));
        const uint32_t h_anchors_len = h_lens[1];
        const uint32_t h_cores_len   = h_lens[2];
        uint32_t       h_final_len   = h_lens[3];
        const uint32_t h_late_count  = h_lens[4];

        // Funnel counters: per-batch device buffer, read after the stream
        // sync -- no cross-stream race, no reset needed.
        {
            uint32_t batch_counts[4] = {};
            TRY_CUDA(cudaMemcpy(batch_counts, b.counts, sizeof(batch_counts), cudaMemcpyDeviceToHost));
            for (int k = 0; k < 4; k++) h_stage_counts[k] += batch_counts[k];
        }

        // Accumulate per-stage time and funnel counts.
        stats[0].time_s += b.e_init.ms_since(b.e_start) / 1e3;
        stats[1].time_s += b.e_cov.ms_since(b.e_init) / 1e3;
        stats[2].time_s += b.e_late.ms_since(b.e_cov) / 1e3;
        stats[3].time_s += b.e_ext.ms_since(b.e_late) / 1e3;
        stats[4].time_s += b.e_blob.ms_since(b.e_ext) / 1e3;

        stats[0].in  += n;                              stats[0].out += n;
        stats[1].in  += (uint64_t)n * KernelCoverage::N_ANCHORS;
        stats[1].out += h_anchors_len;
        stats[2].in  += h_anchors_len;                  stats[2].out += h_late_count;
        stats[3].in  += h_anchors_len;                  stats[3].out += h_cores_len;
        stats[4].in  += h_cores_len;                    stats[4].out += h_final_len;
        stats_seeds += n;
        stats_batches++;

        // Funnel tables: counters accumulate silently in normal mode; --debug
        // prints them throttled to one table per ~5 seconds of wall time.
        if (g_ui_level.load(std::memory_order_relaxed) >= UI_DEBUG) {
            const auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - stats_last_print).count() >= 5.0) {
                const double wall = std::chrono::duration<double>(now - stats_t0).count();
                ui_block_begin();
                print_stage_stats(device, stats_batches, wall, stats_seeds, stats);
                print_funnel_stats(device, h_stage_counts);
                ui_block_end();
                for (int k = 0; k < 4; k++) h_stage_counts[k] = 0;
                for (auto &st : stats) { st.time_s = 0.0; st.in = 0; st.out = 0; }
                stats_batches = 0;
                stats_seeds = 0;
                stats_t0 = now;
                stats_last_print = now;
            }
        }

        if (h_anchors_len > GPU_MAX_ANCHORS)
            ui_eventf("GPU %d WARNING: anchor buffer overflow (%u > %u), candidates dropped.",
                      device, h_anchors_len, GPU_MAX_ANCHORS);
        if (h_cores_len > GPU_MAX_CORES)
            ui_eventf("GPU %d WARNING: core-point buffer overflow (%u > %u), candidates dropped.",
                      device, h_cores_len, GPU_MAX_CORES);
        if (h_final_len > GPU_MAX_OUTPUTS) {
            ui_eventf("GPU %d WARNING: output buffer overflow (%u > %u), candidates dropped.",
                      device, h_final_len, GPU_MAX_OUTPUTS);
            h_final_len = GPU_MAX_OUTPUTS;
        }

        completed.fetch_add(n, std::memory_order_relaxed);
        if (h_final_len > 0) {
            TRY_CUDA(cudaMemcpy(h_outputs.data(), b.fin,
                                sizeof(GpuOutput) * h_final_len, cudaMemcpyDeviceToHost));
            std::lock_guard<std::mutex> lock(outputs.mutex);
            for (uint32_t i = 0; i < h_final_len; i++)
                outputs.queue.push(h_outputs[i]);
            outputs.total_pushed.fetch_add(h_final_len, std::memory_order_relaxed);
        }
        harvested++;
    };

    while (!should_stop()) {
        // Backpressure: if the CPU verifiers are falling behind, pause
        // before starting the next batch so the candidate queue (and host
        // memory) cannot grow without bound.
        {
            bool noticed = false;
            while (!should_stop()) {
                size_t q;
                {
                    std::lock_guard<std::mutex> lock(outputs.mutex);
                    q = outputs.queue.size();
                }
                if (q <= 65536) break;
                if (!noticed) {
                    ui_eventf("GPU %d: CPU backlog (%zu candidates), throttling...", device, q);
                    noticed = true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        const int s = (int)(launched % (uint64_t)nbuf);
        BatchBuf &b = bufs[s];
        cudaStream_t stream = streams[s];
        uint32_t *d_seeds_len   = b.lens + 0;
        uint32_t *d_anchors_len = b.lens + 1;
        uint32_t *d_cores_len   = b.lens + 2;
        uint32_t *d_final_len   = b.lens + 3;
        uint32_t *d_late_count  = b.lens + 4;

        // Grab the next batch; a finite seed list (--seeds) returns 0 when drained.
        const uint32_t n = input.next_batch(b.h_seeds.data(), GPU_NUM_SEEDS);
        if (n == 0) break;
        b.batch_n = n;

        TRY_CUDA(cudaMemsetAsync(d_anchors_len, 0, sizeof(uint32_t) * 4, stream));
        TRY_CUDA(cudaMemsetAsync(b.flags, 0, sizeof(uint32_t) * GPU_NUM_SEEDS, stream));
        TRY_CUDA(cudaMemsetAsync(b.emits, 0, sizeof(uint32_t) * GPU_NUM_SEEDS, stream));
        TRY_CUDA(cudaMemsetAsync(b.work, 0, sizeof(uint32_t), stream));
        TRY_CUDA(cudaMemsetAsync(b.counts, 0, sizeof(uint32_t) * 4, stream));
        TRY_CUDA(cudaMemcpyAsync(d_seeds_len, &n, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
        TRY_CUDA(cudaMemcpyAsync(b.seeds, b.h_seeds.data(), sizeof(uint64_t) * n, cudaMemcpyHostToDevice, stream));

        b.e_start.record(stream);

        KernelInit::kernel<<<(n * KernelInit::TABLES_PER_SEED + KernelInit::THREADS_PER_BLOCK - 1) / KernelInit::THREADS_PER_BLOCK, KernelInit::THREADS_PER_BLOCK, 0, stream>>>(
            InputBuffer<uint64_t>((const uint64_t *)b.seeds, d_seeds_len, GPU_NUM_SEEDS),
            (MountainNoiseResult *)b.results);
        TRY_CUDA(cudaGetLastError());
        b.e_init.record(stream);

        KernelCoverage::kernel<<<n, KernelCoverage::THREADS_PER_BLOCK, 0, stream>>>(
            InputBuffer<uint64_t>((const uint64_t *)b.seeds, d_seeds_len, GPU_NUM_SEEDS),
            OutputBuffer<KernelCoverage::AnchorHit>((KernelCoverage::AnchorHit *)b.anchors, d_anchors_len, GPU_MAX_ANCHORS),
            (const MountainNoiseResult *)b.results,
            b.counts);
        TRY_CUDA(cudaGetLastError());
        b.e_cov.record(stream);

        KernelLateInit::kernel<<<1024, KernelLateInit::THREADS_PER_BLOCK, 0, stream>>>(
            InputBuffer<uint64_t>((const uint64_t *)b.seeds, d_seeds_len, GPU_NUM_SEEDS),
            InputBuffer<KernelCoverage::AnchorHit>((const KernelCoverage::AnchorHit *)b.anchors, d_anchors_len, GPU_MAX_ANCHORS),
            (MountainNoiseResult *)b.results,
            (uint32_t *)b.flags,
            d_late_count);
        TRY_CUDA(cudaGetLastError());
        b.e_late.record(stream);

        KernelExtrema::kernel<<<2048, KernelExtrema::THREADS_PER_BLOCK, 0, stream>>>(
            InputBuffer<KernelCoverage::AnchorHit>((const KernelCoverage::AnchorHit *)b.anchors, d_anchors_len, GPU_MAX_ANCHORS),
            OutputBuffer<KernelCoverage::AnchorHit>((KernelCoverage::AnchorHit *)b.cores, d_cores_len, GPU_MAX_CORES),
            (const MountainNoiseResult *)b.results);
        TRY_CUDA(cudaGetLastError());
        b.e_ext.record(stream);

        KernelBlob::kernel<<<KernelBlob::GRID_BLOCKS, KernelBlob::THREADS_PER_BLOCK, 0, stream>>>(
            InputBuffer<uint64_t>((const uint64_t *)b.seeds, d_seeds_len, GPU_NUM_SEEDS),
            InputBuffer<KernelCoverage::AnchorHit>((const KernelCoverage::AnchorHit *)b.cores, d_cores_len, GPU_MAX_CORES),
            OutputBuffer<GpuOutput>((GpuOutput *)b.fin, d_final_len, GPU_MAX_OUTPUTS),
            (const MountainNoiseResult *)b.results,
            (uint32_t *)b.emits,
            (uint32_t *)b.work);
        TRY_CUDA(cudaGetLastError());
        b.e_blob.record(stream);

        launched++;

        // Stay one batch ahead: collect the older batch's results while the
        // one we just launched keeps the GPU busy.
        if (launched - harvested >= (uint64_t)nbuf)
            harvest((int)(harvested % (uint64_t)nbuf));
    }

    // Collect any launched-but-uncollected batches (end of a list, or shutdown).
    while (harvested < launched)
        harvest((int)(harvested % (uint64_t)nbuf));

    // Final partial stats table (covers short --seeds runs and shutdowns).
    if (stats_batches > 0 && g_ui_level.load(std::memory_order_relaxed) >= UI_DEBUG) {
        const auto now = std::chrono::steady_clock::now();
        const double wall = std::chrono::duration<double>(now - stats_t0).count();
        ui_block_begin();
        print_stage_stats(device, stats_batches, wall, stats_seeds, stats);
        print_funnel_stats(device, h_stage_counts);
        ui_block_end();
    }

    finished.store(true, std::memory_order_relaxed);
    TRY_CUDA(cudaStreamDestroy(streams[0]));
    TRY_CUDA(cudaStreamDestroy(streams[1]));
    ui_eventf("GPU %d shutting down.", device);
}
