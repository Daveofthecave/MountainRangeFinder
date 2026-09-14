// cpu.cpp
// CPU verifier: receives GPU pre-filtered candidates (BLOCK coordinates) and
// applies the slow exact checks the GPU skips, in cheap-to-expensive order:
//
//   * Climate stats (informational, default on): re-measures the GPU
//     pipeline's claims with cubiomes in double precision -- core 0B-erosion
//     disc coverage, windowed full-octave erosion minimum and continentalness
//     maximum. Uses the same phyllotaxis disc samples and window grids as the
//     GPU kernels, so the numbers are directly comparable.
//   * NOT Dark Forest exclusion (recipe steps 58-59), evaluated with the
//     biome tree of the target MC version (--mc; 1.21.4's pale_garden takes
//     over part of dark_forest's climate niche, so the version matters).
//   * Contiguous multi-climate blob fill (shared engine from probe.cpp:
//     blob_fill over the BLOB_* field, 64-block lattice, +-48,000-block
//     reach): the headline "megaregion size" score AND a hard gate at
//     MIN_BLOB_AREA, plus the measured inscribed core radius and the
//     edge-clipped flag.
//   * Approximate surface height (recipe steps 7-8) via cubiomes'
//     mapApproxHeight -- the same depth-based heuristic behind CV's "approx
//     surface height".
//   * Enrichment + score (shared engine: blob_enrich on the same fill):
//     full-octave climates, weirdness texture, approx heights, biome census,
//     neighbor-pair crossings, windowed best-subregion stats, and the
//     composite score. This is the expensive bundle, so it runs last -- only
//     for candidates that passed every gate. It also produces the headline
//     x/z of the output row: the center of the best-pattern 1664-block
//     window (the heart of the densest-packed peaks).

#include "cpu.h"
#include "common.h"
#include "noise_common.h" // make_disc_offsets(): the same disc pattern the GPU uses
#include "probe.h"        // the shared measurement engine (MeasureCtx, blob_fill, blob_enrich)

#include "finders.h"
#include "generator.h"
#include "biomenoise.h"
#include "biomes.h"

#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cinttypes>
#include <limits>
#include <thread>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <mutex>

// Climate thresholds mirrored from gpu.cu (raw noise units = CV display / 10000).
// Keep these in sync with the GPU side when tuning.
constexpr double CORE_COV_ERO_MAX  = -0.40;  // (2)  disc coverage, 0B erosion

// Geometry mirrored from gpu.cu (quart units).
constexpr int32_t CORE_DISC_R_Q   = CORE_BLOB_RADIUS / 4;      // 400
constexpr int32_t WIN_HALF_Q      = 400;                       // centered square side 3200
constexpr int32_t WIN_STEP_Q      = 16;                        // 64 blocks

// The GPU's disc sample table, shared from noise_common.h. Offsets are scaled
// with the same float expression and round-to-nearest as the device code
// (__float2int_rn), so CPU stats and GPU decisions sample the same points.
static const DiscOffsets &disc_table() {
    static const DiscOffsets d = make_disc_offsets();
    return d;
}
static inline int32_t disc_off_x(int k, int32_t r_q) {
    return (int32_t)std::lrintf(disc_table().x[k] * (r_q / 32767.0f));
}
static inline int32_t disc_off_z(int k, int32_t r_q) {
    return (int32_t)std::lrintf(disc_table().z[k] * (r_q / 32767.0f));
}

// ---------------------------------------------------------------------------
// Climate stats: re-measure the GPU's filter claims with cubiomes (double
// precision), reusing the pre-seeded samplers of the shared MeasureCtx
// (eroB = 0B truncation, eroF/contF = full octaves). The core-disc stat
// samples the same CORE_COV_SAMPLES points as the GPU's KernelCoverage
// (see noise_common.h), so the numbers line up exactly.
// ---------------------------------------------------------------------------
struct ClimateStats {
    float ero_cov  = -1.0f; // fraction of the core disc with 0B erosion <= -0.4
    float ero_min  = 0.0f;  // full-octave erosion window minimum
    float cont_max = 0.0f;  // full-octave continentalness window maximum
};

static ClimateStats compute_climate_stats(MeasureCtx &ctx, int32_t bx, int32_t bz) {
    ClimateStats s;
    const int32_t cx = bx >> 2, cz = bz >> 2; // block -> quart

    // (2) core disc coverage: same samples and threshold as KernelCoverage.
    int hits = 0;
    for (int k = 0; k < CORE_COV_SAMPLES; k++) {
        const double v = sampleClimatePara(&ctx.eroB, nullptr,
            cx + disc_off_x(k, CORE_DISC_R_Q), cz + disc_off_z(k, CORE_DISC_R_Q));
        hits += (v <= CORE_COV_ERO_MAX);
    }
    s.ero_cov = (float)hits / (float)CORE_COV_SAMPLES;

    // (3) deep erosion point: full-octave minimum over the window.
    double emin = DBL_MAX;
    for (int32_t dz = -WIN_HALF_Q; dz <= WIN_HALF_Q; dz += WIN_STEP_Q)
        for (int32_t dx = -WIN_HALF_Q; dx <= WIN_HALF_Q; dx += WIN_STEP_Q)
            emin = std::min(emin, sampleClimatePara(&ctx.eroF, nullptr, cx + dx, cz + dz));
    s.ero_min = (float)emin;

    // (5) high continentalness point: full-octave maximum over the window.
    double cmax = -DBL_MAX;
    for (int32_t dz = -WIN_HALF_Q; dz <= WIN_HALF_Q; dz += WIN_STEP_Q)
        for (int32_t dx = -WIN_HALF_Q; dx <= WIN_HALF_Q; dx += WIN_STEP_Q)
            cmax = std::max(cmax, sampleClimatePara(&ctx.contF, nullptr, cx + dx, cz + dz));
    s.cont_max = (float)cmax;

    return s;
}

// ---------------------------------------------------------------------------
// Approximate surface height (recipe steps 7-8).
// ---------------------------------------------------------------------------
std::atomic_uint64_t g_rej_dark{0}, g_rej_area{0}, g_rej_core{0}, g_rej_height{0}, g_rej_dedup{0};

static bool height_scan(Generator *g, int32_t bx, int32_t bz, const VerifyConfig &cfg,
                        int32_t &out_min, int32_t &out_max) {
    constexpr int32_t R_B    = CORE_BLOB_RADIUS; // 1600 blocks (recipe radius)
    constexpr int32_t STEP_B = 64;               // coarse spiral; mountain features
                                                 // are O(100s) of blocks wide
    const int64_t R2 = (int64_t)R_B * R_B;

    int32_t ymin = INT32_MAX, ymax = INT32_MIN;
    bool window_hit = false;

    auto probe = [&](int32_t x, int32_t z) {
        const int64_t dx = (int64_t)x - bx, dz = (int64_t)z - bz;
        if (dx * dx + dz * dz > R2) return; // radial clip (the recipe uses a disc)
        float y;
        if (mapApproxHeight(&y, nullptr, g, nullptr, x >> 2, z >> 2, 1, 1) != 0) return;
        const int32_t yi = (int32_t)std::lrintf(y);
        ymin = std::min(ymin, yi);
        ymax = std::max(ymax, yi);
        if (yi >= cfg.min_height && yi <= cfg.max_height) window_hit = true;
    };

    // Outward square spiral with the radial clip, same pattern as the recipe.
    // Full scan: no early exit, so ymin/ymax are the true extremes over the
    // radius (the old version stopped at the first in-window sample, which
    // made the reported "maxY" a first-hit value rather than a maximum).
    // Affordable at candidate rates, and it makes both the pass rule and the
    // report honest.
    for (int32_t r = 0; r <= R_B; r += STEP_B) {
        for (int32_t t = -r; t <= r; t += STEP_B) {
            probe(bx + t, bz - r);
            if (r > 0) {
                probe(bx + t, bz + r);
                if (t > -r && t < r) { // side edges, corners already covered
                    probe(bx - r, bz + t);
                    probe(bx + r, bz + t);
                }
            }
        }
    }

    out_min = ymin;
    out_max = ymax;
    // Pass on a direct window hit, or when the observed range straddles the
    // window: terrain height is continuous, so a max above the window with
    // lower ground nearby implies slopes crossing the window between samples.
    return window_hit || (ymax >= cfg.min_height && ymin <= cfg.max_height);
}

// ---------------------------------------------------------------------------
// Full verification of one GPU candidate. Returns true iff the seed is a
// keeper. On success, `out` carries the enrichment bundle + composite score
// and the headline (window-center) coordinates.
// ---------------------------------------------------------------------------
static bool verify_candidate(MeasureCtx &ctx, const VerifyConfig &cfg,
                             const GpuOutput &candidate, CpuOutput &out) {
    measure_ctx_seed(ctx, candidate.seed);

    out.anchor_x = candidate.x;
    out.anchor_z = candidate.z;
    out.gpu_h    = candidate.blob_h;
    out.gpu_dh8  = candidate.blob_dh;
    out.x = candidate.x; // fallbacks; the enrichment may replace these with
    out.z = candidate.z; // the best-window center
    out.score = std::numeric_limits<double>::quiet_NaN();
    out.blob_edge = 0;

    // (Climate stats moved below the gates: only keepers ever print these,
    // so computing them for rejects is wasted work.)

    // Recipe steps 58-59: NOT Dark Forest. Sample every 64 blocks inside a
    // radius-1365 circle around the core blob; reject at >= 7% coverage.
    if (cfg.check_dark_forest) {
        const int32_t EXCL_RADIUS = ADJACENT_BLOB_RADIUS;
        const int32_t SAMPLE_STEP = 64;
        const int64_t ER2 = (int64_t)EXCL_RADIUS * EXCL_RADIUS;

        int32_t dark = 0, total = 0;
        for (int32_t x = candidate.x - EXCL_RADIUS; x <= candidate.x + EXCL_RADIUS; x += SAMPLE_STEP) {
            for (int32_t z = candidate.z - EXCL_RADIUS; z <= candidate.z + EXCL_RADIUS; z += SAMPLE_STEP) {
                const int64_t dx = (int64_t)x - candidate.x;
                const int64_t dz = (int64_t)z - candidate.z;
                if (dx * dx + dz * dz > ER2) continue;
                total++;
                if (getBiomeAt(&ctx.g, 1, x, 256, z) == dark_forest) dark++;
            }
        }
        if (total == 0 || (double)dark / (double)total >= cfg.dark_forest_frac) {
            g_rej_dark.fetch_add(1, std::memory_order_relaxed);
            if (g_ui_level.load(std::memory_order_relaxed) >= UI_VERBOSE)
                ui_eventf("  [dark-forest] rejected %" PRIi64 " at (%" PRIi32 ", %" PRIi32 "): "
                          "%" PRIi32 "/%" PRIi32 " dark samples",
                          (int64_t)candidate.seed, candidate.x, candidate.z, dark, total);
            return false;
        }    }

    // Contiguous multi-climate area: the megaregion size score and the hard
    // gates. Shared engine (probe.cpp blob_fill): the same field definition
    // and fill the GPU's KernelBlob used to earn emission, re-measured here
    // in double precision.
    BlobFill blob;
    bool have_blob = false;
    if (cfg.measure_blob) {
        blob = blob_fill(ctx, cfg, candidate.x, candidate.z);
        out.blob_area   = blob.area;
        out.core_radius = blob.core_cells;
        out.blob_edge   = blob.edge;

        if (blob.area < cfg.min_blob_area) {
            g_rej_area.fetch_add(1, std::memory_order_relaxed);
            if (g_ui_level.load(std::memory_order_relaxed) >= UI_VERBOSE)
                ui_eventf("  [blob-area] rejected %" PRIi64 " at (%" PRIi32 ", %" PRIi32 "): "
                          "area %" PRIi64 " < gate %" PRIi64 " (cells=%zu, edge=%d)",
                          (int64_t)candidate.seed, candidate.x, candidate.z,
                          (int64_t)blob.area, (int64_t)cfg.min_blob_area,
                          blob.cells.size(), (int)blob.edge);
            return false;
        }
        if (blob.core_cells < cfg.min_core_width) {
            g_rej_core.fetch_add(1, std::memory_order_relaxed);
            if (g_ui_level.load(std::memory_order_relaxed) >= UI_VERBOSE)
                ui_eventf("  [blob-core] rejected %" PRIi64 " at (%" PRIi32 ", %" PRIi32 "): "
                          "core %" PRIi32 " < gate %" PRIi32,
                          (int64_t)candidate.seed, candidate.x, candidate.z,
                          blob.core_cells, cfg.min_core_width);
            return false;
        }
        have_blob = true;
    }

    // Recipe steps 7-8: approx surface height window.
    if (cfg.check_height) {
        int32_t ymin = 0, ymax = 0;
        if (!height_scan(&ctx.g, candidate.x, candidate.z, cfg, ymin, ymax)) {
            g_rej_height.fetch_add(1, std::memory_order_relaxed);
            if (g_ui_level.load(std::memory_order_relaxed) >= UI_VERBOSE)
                ui_eventf("  [height] rejected %" PRIi64 " at (%" PRIi32 ", %" PRIi32 "): "
                          "approx height range [%" PRIi32 ", %" PRIi32 "], window [%d, %d]",
                          (int64_t)candidate.seed, candidate.x, candidate.z,
                          ymin, ymax, cfg.min_height, cfg.max_height);
            return false;
        }
        out.y_height = ymax;
    }

    // Informational re-measurement (printed with verified seeds). Runs
    // after the gates: only keepers ever print these.
    if (cfg.climate_stats) {
        const ClimateStats cs = compute_climate_stats(ctx, candidate.x, candidate.z);
        out.ero_cov  = cs.ero_cov;
        out.ero_min  = cs.ero_min;
        out.cont_max = cs.cont_max;
    }

    // All gates passed: the enrichment bundle (full-octave climates,
    // weirdness texture, approx heights, biome census, neighbor-pair
    // crossings, windowed best-subregion stats) and the composite score.
    // This is the expensive part -- running it last means it only ever runs
    // on keepers. It also yields the headline coordinates: the center of the
    // best-pattern 1664-block window inside the blob.
    if (have_blob) {
        blob_enrich_best(ctx, blob, out.stats, cfg.enrich_phases);
        out.score = out.stats.score;
        // Headline = center of the coherent best-pattern 1664-block window
        // (the most magic-like neighborhood), not the best-erosion window.
        if (!std::isnan(out.stats.bw1664_sc)) {
            out.x = out.stats.bw1664_x;
            out.z = out.stats.bw1664_z;
        } else if (!std::isnan(out.stats.bw896_sc)) {
            out.x = out.stats.bw896_x;
            out.z = out.stats.bw896_z;
        } else if (!std::isnan(out.stats.w1664_ero_min)) {
            out.x = out.stats.w1664_ero_x;
            out.z = out.stats.w1664_ero_z;
        } else if (!std::isnan(out.stats.w896_ero_min)) {
            out.x = out.stats.w896_ero_x;
            out.z = out.stats.w896_ero_z;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Result deduplication.
//
// KernelBlob can emit several candidates for the same seed when adjacent core
// anchors sit on one megaregion: each fill measures the same connected field
// and reports it from a slightly different centroid (observed pairs ~1.5k and
// ~3.4k blocks apart with near-identical areas). We keep one point per seed
// per dedup radius.
//
// The authoritative check happens after verification and atomically with the
// recording. The old code checked before verification and recorded after, so
// two threads verifying same-seed candidates concurrently both passed the
// check and both recorded -- a check-then-act race. Verification is cheap
// relative to the idle CPU pool, so an occasional wasted verify beats a dupe.
//
// Dedup keys on the candidate anchor, while the output file stores the
// window-center headline coords; the 8000-block default radius absorbs the
// (few-km) shift between the two when resuming from an existing output file.
// ---------------------------------------------------------------------------
static std::mutex g_dedup_mutex;
static std::unordered_map<uint64_t, std::vector<std::pair<int32_t, int32_t>>> g_emitted_seeds;

// Caller must hold g_dedup_mutex.
static bool dedup_is_dup_locked(int32_t radius, uint64_t seed, int32_t x, int32_t z) {
    if (radius <= 0) return false;
    const int64_t r2 = (int64_t)radius * radius;
    auto it = g_emitted_seeds.find(seed);
    if (it == g_emitted_seeds.end()) return false;
    for (const auto &p : it->second) {
        const int64_t dx = (int64_t)x - p.first;
        const int64_t dz = (int64_t)z - p.second;
        if (dx * dx + dz * dz < r2) return true;
    }
    return false;
}

size_t dedup_preload(const std::vector<GpuOutput> &prior) {
    std::lock_guard<std::mutex> lock(g_dedup_mutex);
    for (const auto &o : prior)
        g_emitted_seeds[o.seed].push_back({o.x, o.z});
    return prior.size();
}

// Decide-and-record atomically. Returns true if this point is new (kept).
static bool dedup_claim(int32_t radius, uint64_t seed, int32_t x, int32_t z) {
    std::lock_guard<std::mutex> lock(g_dedup_mutex);
    if (dedup_is_dup_locked(radius, seed, x, z)) return false;
    g_emitted_seeds[seed].push_back({x, z});
    return true;
}

void CpuThread::run() {
    if (g_ui_level.load(std::memory_order_relaxed) >= UI_DEBUG)
        std::printf("Started CPU verifier thread %d\n", id);
    MeasureCtx *ctxp = new MeasureCtx(); // ~100+ KB of generator/noise tables
    measure_ctx_setup(*ctxp, cfg.mc_version);
    MeasureCtx &ctx = *ctxp;

    while (true) {
        GpuOutput candidate;
        {
            std::unique_lock<std::mutex> lock(inputs.mutex);
            if (inputs.queue.empty()) {
                if (should_stop()) break;
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
            candidate = inputs.queue.front();
            inputs.queue.pop();
        }

        CpuOutput out{};
        out.seed    = candidate.seed;
        out.ero_cov = -1.0f; // marks "stats disabled"
        out.adj_dirs = -1;   // adjacency probe is gone (GPU blob fill instead)
        out.core_radius = 0;

        if (!verify_candidate(ctx, cfg, candidate, out)) {
            processed.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        // Authoritative dedup, post-verification. Points that fail
        // verification are never recorded, so a later nearby candidate of the
        // same seed still gets its chance.
        // In aggregate mode (search / --seeds) this is skipped: every verified
        // row flows to main.cpp's aggregator, which clusters same-seed rows
        // and keeps the highest-scoring one per cluster. That subsumes
        // first-claim-wins dedup -- the runner-up rows were already fully
        // verified anyway, so the CPU cost is identical and the output rows
        // get strictly better (and reproducible across reruns).
        if (!cfg.aggregate &&
            !dedup_claim(cfg.dedup_radius, candidate.seed, candidate.x, candidate.z)) {
            g_rej_dedup.fetch_add(1, std::memory_order_relaxed);
            if (cfg.dedup_radius > 0 &&
                g_ui_level.load(std::memory_order_relaxed) >= UI_VERBOSE)
                ui_eventf("  [dedup] dropped %" PRIi64 " at (%" PRIi32 ", %" PRIi32
                          "): within %d blocks of an already-reported point",
                          (int64_t)candidate.seed, candidate.x, candidate.z,
                          cfg.dedup_radius);
            processed.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(outputs.mutex);
            outputs.queue.push(out);
            outputs.total_pushed.fetch_add(1, std::memory_order_relaxed);
        }
        // Incremented after the output push, so processed == total_pushed
        // implies every popped candidate is fully accounted for.
        processed.fetch_add(1, std::memory_order_relaxed);
    }

    delete ctxp;
    if (g_ui_level.load(std::memory_order_relaxed) >= UI_DEBUG)
        std::printf("CPU verifier thread %d shutting down.\n", id);
}
